# MoeDistributeDispatchV2 (<<<>>> Kernel直调版本)

## 简介

本算子为MoeDistributeDispatchV2 算子的A5（ascend950）Kernel直调实现，采用 <<<>>> 调用方式直接启动Ascend C Kernel。Kernel实现直接引用仓内 `mc2/moe_distribute_dispatch_v2/op_kernel/arch35/` 头文件（经仓库根include路径解析，无本地副本）。

关于Dispatch算子的核心功能、计算公式及通信原理，详见 [aclnnMoeDistributeDispatchV3](../../../../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV3.md)文档介绍。

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:--------:|
| Atlas A5系列产品 | √ |

## 约束说明

想要完整开启功能，需要满足以下版本要求：
- gcc 9.4.0+
- python 3.9+
- torch>=2.7.1
- cann 9.0.0+

该算子支持的平台为ascend950，默认构建环境为ascend910b，在ascend950 构建环境下构建，请先执行命令：
```sh
export NPU_SOC_VERSION=ascend950
```

当前支持范围：非量化（quant_mode=0）与 perToken 动态量化（quant_mode=2，int8 输出）、BF16/FP16输入、tp_world_size=1、仅 MTE 通信。

## <<<>>> Kernel直调调用流程

### 整体调用链路

```
Python层(测试脚本)
    │
    ├── 初始化torch.distributed(hccl)通信域
    │
    └── 调用torch.ops.ascend_ops.MoeDistributeDispatchV2
            │
            ↓
        C++ Extension Layer (npu_moe_distribute_dispatch_v2)
            │
            ├── 输入校验(ValidateMoeDistributeDispatchV2Input)
            │
            ├── 输出Tensor Shape计算与创建
            │
            ├── Tiling数据构造(calculate_tilingdata_a5，含mc2通信tiling)
            │
            ├── 获取NPU Stream
            │
            └── 调用moe_distribute_dispatch_v2_api
                    │
                    ├── Tensor地址转换
                    │
                    ├── TilingKey计算(calculate_tilingkey_a5)
                    │
                    ├── Tiling数据H2D拷贝(aclrtMemcpyAsync至tilingGM)
                    │
                    ├── Profiling集成
                    │
                    └── 调用moe_distribute_dispatch_v2_entry
                            │
                            ├── <<<>>> Kernel Launch
                            │       │
                            │       ├── moe_distribute_dispatch_v2_generic
                            │       │
                            │       └── SK模式(moe_distribute_dispatch_v2_generic_sk)
                            │
                            └── TilingKey分支选择(moe_dispatch_a5_switch)
                                    │
                                    ↓
                                Ascend C Kernel执行
```

### 目录结构

```
moe_distribute_dispatch_v2/ascend950/
├── moe_distribute_dispatch_v2_torch.cpp      # C++ Extension主实现
├── moe_distribute_dispatch_v2_torch.h        # Tensor地址获取辅助函数
├── moe_distribute_dispatch_v2_torch_validate.cpp  # 输入参数校验
├── moe_distribute_dispatch_v2_torch_validate.h
├── moe_distribute_dispatch_v2_entry.cpp      # <<<>>>入口函数与SK适配
├── moe_distribute_dispatch_v2_entry.h
├── mc2_cc_tiling_shim.cpp                    # mc2通信tiling shim（旧ABI链接垫片）
├── mc2_cc_tiling_shim.h
└── CMakeLists.txt                            # 构建配置
```

## 调用流程详解

### 1. Python层

A5的Kernel经updateContext流程构建通信上下文（对齐A3），Python层需要：

- 通过torch.distributed初始化EP通信域（backend=hccl），获取通信域名称
- 根据通信参数计算HCCL Buffer Size并设置HCCL_BUFFSIZE环境变量
- 构造MoeDistributeBuffer实例（其内部经updateContext完成HCCL通信域初始化）
- 调用torch.ops.ascend_ops.MoeDistributeDispatchV2，mc2_context由包装类自动填充

### 2. C++ Extension层

#### Torch接口注册

通过TORCH_LIBRARY_FRAGMENT注册算子签名，定义输入输出参数类型，通过TORCH_LIBRARY_IMPL进行绑定。

#### 主函数

主函数npu_moe_distribute_dispatch_v2 执行以下流程：

**阶段1：输入校验**

将参数打包到MoeDistributeDispatchV2ValidateParams结构体，调用ValidateMoeDistributeDispatchV2Input执行校验。校验完成后，从validate_params获取计算后的维度信息（bs、h、k）。支持quant_mode=0（非量化）与quant_mode=2（perToken动态量化，int8输出）。

**阶段2：输出Tensor Shape计算**

根据rank类型（共享专家卡或MoE专家卡）计算输出Shape：

- 共享专家卡：local_moe_expert_num=1，输出行数a为max_bs * max_shared_group_num
- MoE专家卡：local_moe_expert_num为moe_expert_num除以MoE专家卡数，输出行数a为global_bs * min(local_moe_expert_num, k)

**阶段3：输出Tensor创建**

根据计算出的Shape和配置创建输出Tensor：expand_x、dynamic_scales、assist_info_forcombine、expert_token_nums、ep_recv_counts、expand_scales、new_workspace（16MB，核间同步），另创建tiling_buffer（TilingData结构体大小的Byte Tensor，H2D目标）。

**阶段4：Tiling数据构造**

调用calculate_tilingdata_a5构造`MoeDistributeDispatchV2TilingData`（含mc2InitTiling/mc2CcTiling1/moeDistributeDispatchV2Info三部分）：

- mc2通信tiling：经`AscendC::Mc2CcTilingConfig`（HCCL_CMD_ALLTOALLV + AIV_ENGINE）填充mc2InitTiling与mc2CcTiling1
- info字段：配置参数直接赋值（epWorldSize、epRankId、moeExpertNum、globalBs等）；A5不支持TP通信（tpWorldSize=1）；totalWinSizeEp = HCCL_BUFFSIZE(MB)×1024²×2（对齐mc2 GetMaxWindowSize×2）；hasExpertScales由expert_scales输入是否存在决定；从平台API获取AIV核数和UB大小

**阶段5：获取Stream并异步调用**

获取当前NPU Stream，构造lambda函数调用moe_distribute_dispatch_v2_api，通过OpCommand::RunOpApiV2 异步执行。

### 3. API层

API层函数moe_distribute_dispatch_v2_api负责将Tensor转换为<<<>>>调用所需的地址参数，集成Profiling，并调用下层Entry函数。

#### Tensor地址转换

通过get_first_tensor_address函数获取各Tensor的设备地址指针：必选Tensor直接获取data_ptr；可选Tensor先判断has_value，不存在则传nullptr。

#### TilingKey计算

calculate_tilingkey_a5根据运行配置计算4位十进制TilingKey：

| 基础值 | 20000 |
|-------|-------|
| quantMode | 加到个位（0 或 2） |
| xType为BF16 | 加10（到十位） |
| scales存在（smoothScale） | 加100（到百位） |

#### Tiling数据H2D

TilingData结构体经`aclrtMemcpyAsync`（HostToDevice）拷贝至tiling_buffer，Kernel内通过tilingGM指针以`GET_TILING_DATA_WITH_STRUCT`读取。

#### Profiling集成

通过`aclprofRangePushEx`和`aclprofRangePop`包裹<<<>>>调用形成profiling区间，记录算子元信息与Tensor信息。

### 4. Entry层

Entry层包含<<<>>>调用入口和SK适配，定义在moe_distribute_dispatch_v2_entry.cpp和moe_distribute_dispatch_v2_entry.h文件中。

#### <<<>>>调用入口

moe_distribute_dispatch_v2_entry函数使用 <<<>>> 语法启动moe_distribute_dispatch_v2_generic Kernel，传入TilingKey、Block数量（AIV核数）、Stream、各Tensor地址和tilingGM。

#### SK适配

SK（Super Kernel）是Ascend C提供的一种算子融合机制，通过将同一个模型中的多个算子融合为一个整体算子，避免host下发开销。

**SK模式说明**：
- `__sk__`标记的函数为SK kernel，支持算子融合
- `split_num`模板参数表示分割模式，支持0-3 四种模式，对应不同的核间任务分配策略
- `SK_BIND`宏将SK kernel的不同分割模式绑定到主kernel，由底层调度器自动选择最优模式

**参数封装**：`MoeDistributeDispatchV2SkArgs`结构体将所有kernel参数打包，便于SK调用时传递。结构体包含tilingKey、所有输入输出tensor地址、workspace、tilingGM等字段。

**执行流程**：当调度器选择SK模式执行时，调用`moe_distribute_dispatch_v2_generic_sk`函数，从SkArgs结构体解析参数，然后执行与普通kernel相同的逻辑（调用`moe_dispatch_a5_switch`）。

### 5. Kernel层

#### TilingKey分支选择

Kernel入口moe_dispatch_a5_switch通过switch语句根据TilingKey选择对应的模板实例化（仅MTE非fullmesh）：

- **20000/20010**：非量化（quant_mode=0），fp16/bf16，无smoothScale
- **20002/20012**：perToken动态量化（quant_mode=2，int8输出），fp16/bf16，无smoothScale
- **20102/20112**：perToken动态量化 + smoothScale，fp16/bf16

模板参数（输入类型、输出类型、量化模式、smoothScale存在与否）由TilingKey的各位值隐含确定。

#### Kernel总体功能

moe_distribute_dispatch_v2_generic Kernel基于HCCL Window机制实现MoE分发功能：根据expert_ids将Token数据发送到各目标卡的Window，等待所有卡发送完成后从Window读取数据写入输出Tensor。

Kernel实现直接引用仓内mc2 arch35头文件（`moe_distribute_dispatch_v2.h`，经仓库根include路径解析），无本地副本。关于Kernel的详细算法原理、通信流程、数据排布等，详见 [aclnnMoeDistributeDispatchV3](../../../../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV3.md)。

## MC2 Context说明

A5 Kernel经updateContext流程构建HCCL通信上下文（对齐A3）：updateContext产出内核直接读取的`HcclCombinOpParam`布局上下文（workSpace/winSize/windowsIn/windowsOut/rankId/rankDim），作为mc2Context参数传入内核，内核类`Init(mc2Context, ...)`直接使用。

## 与aclnnMoeDistributeDispatchV3的区别

### 参数对比

| 参数 | aclnn版本 | <<<>>>版本 |
|-----|----------|-----------|
| epRankId, epWorldSize, groupEp | 用户传入 | 调用方传入 |
| tp域参数 | 支持 | 不支持（A5当前不支持TP通信） |
| totalWinsizeEp大小 | 用户自行配置环境变量 | 可按Buffer Size计算或用户自行配置（Kernel侧按HCCL_BUFFSIZE×2语义使用） |
| mc2_context | 不支持、无需传入 | 必传参数、由updateContext构建（对齐A3） |

### 量化模式支持

支持：非量化(0)、PerToken动态量化(2，int8输出)

### commAlg支持

支持：""（默认MTE，非fullmesh）

## 调用示例

基本流程：

1. 初始化分布式环境，创建EP通信域
2. 设置HCCL_BUFFSIZE
3. 调用torch.ops.ascend_ops.MoeDistributeDispatchV2 执行分发
4. 调用torch.ops.ascend_ops.MoeDistributeCombineV2 执行组合（使用Dispatch输出）

## 构建说明

构建前设置环境变量：export NPU_SOC_VERSION=ascend950，source cann包环境并激活conda

编译方法：进入examples\fast_kernel_launch_example目录，开启命令：python3 -m build --wheel -n
编译完成后，开启命令
python3 -m pip install ./dist/*.whl --force-reinstall --no-deps
以安装ascend_ops

详细构建流程可参考 [README.md](../../../../README.md)。

通过CMake构建，输出共享库，由PyTorch Extension加载。
