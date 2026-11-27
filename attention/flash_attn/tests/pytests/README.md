# pytests：flash_attn固定用例精度测试

## 1. 概述

`pytests/`是flash_attn的固定用例精度测试工具，基于pytest执行JSON定义的case，逐条完成输入生成、参考计算、NPU执行和精度判定。

主要功能：

- 固定case的精度回归；
- 按case_id、tag或skip列表筛选；
- 两方/三方精度判定；
- 保存输入、参考输出和NPU输出，便于定位问题。

不适用：

- 性能采集：使用`perf_benchmark`；
- 随机测试和长跑：使用`random_fuzz`；
- 生成随机用例：使用`random_fuzz`。

## 2. 文件结构

```text
pytests/
├── conftest.py            # 参数定义、case收集和测试fixture
├── pytest.ini             # marker注册
├── test_flash_attn.py     # pytest入口
└── cases/                 # JSON用例
```

## 3. 环境配置

从`attention/flash_attn/tests/pytests`目录执行：

```bash
cd attention/flash_attn/tests/pytests
source <CANN>/set_env.sh
```

需要可用的TorchNPU环境；graph模式还需要torchair。

## 4. 使用方法

### 4.1 常见用法与指令

#### 4.1.1 执行精度测试

```bash
python3 -m pytest -q --cases=cases/
```

`--cases`必填，可重复传入多个JSON文件或目录；过滤后没有case会直接报错。

#### 4.1.2 筛选用例

```bash
python3 -m pytest -q --cases=cases/ --case-id=decode_BSND
```

- `--case-id`：逗号分隔；无通配符按case_id子串匹配，有通配符时匹配全名、短名或可读前缀；
- `--testset`：逗号分隔，匹配case的`tags`；
- `--skip-cases`：`.txt`文件或逗号分隔模式，`#`开头为注释。

### 4.2 常用参数

| 参数 | 默认值 | 可用取值/说明 |
|---|---|---|
| `--cases` | 无，必填 | JSON文件或目录，可重复 |
| `--case-id` | 不过滤 | 逗号分隔的模式；支持子串和通配符 |
| `--testset` | 不过滤 | 逗号分隔的tag |
| `--skip-cases` | 不跳过 | `.txt`文件或逗号分隔的模式 |
| `--stages` | `all` | `all`或`gen`/`golden`/`run`/`compare`的逗号组合；见4.3 |
| `--golden` | `device=npu,compute_dtype=fp32,p_dtype=out_dtype` | 逗号分隔的`key=value`；字段和默认值见4.4 |
| `--compare-mode` | `two_way` | `two_way`或`three_way` |
| `--out-dir` | `tests/out/functional-<timestamp>/` | 任意可写目录 |
| `--save-data` | 关闭 | 布尔开关；保存`inputs.pt`、`golden.pt`、`npu_out.pt` |
| `--mem-guard-gb` | `32` | 浮点≥0；0只关闭host守卫，HBM×0.9守卫仍生效 |
| `--graph` | 关闭 | 布尔开关；需要torchair |
| `--device-id` | `0` | 整数≥0 |

资源与执行控制：

- `--mem-guard-gb=16`：host峰值超过16 GiB的case记为`skip`；
- `--graph`：图模式执行，首次执行包含建图/编译开销；缺少torchair时直接报错，不回退eager。

### 4.3 `--stages`阶段组合

| 组合 | 是否支持 | 执行所需数据 | 说明 |
|---|---|---|---|
| 空 | ❌ 不支持 | - | 无有效阶段，不建议使用 |
| `gen` | ✅ 支持 | 无需已有数据 | 只生成输入 |
| `golden` | ✅ 支持 | `inputs.pt` | 只计算参考 |
| `run` | ✅ 支持 | `inputs.pt` | 只执行算子 |
| `compare` | ❌ 不支持 | - | 缺少`run`，且当前不加载`npu_out.pt` |
| `gen,golden` | ✅ 支持 | 无需已有数据 | 生成输入并计算参考 |
| `gen,run` | ✅ 支持 | 无需已有数据 | 生成输入并执行 |
| `gen,compare` | ❌ 不支持 | - | 缺少参考和`run` |
| `golden,run` | ✅ 支持 | `inputs.pt` | 计算参考并执行 |
| `golden,compare` | ❌ 不支持 | - | 缺少`run` |
| `run,compare` | ✅ 支持 | `inputs.pt`、`golden.pt` | 执行并比对，复用已有参考 |
| `gen,golden,run` | ✅ 支持 | 无需已有数据 | 输入、参考和执行齐全，尚未比对 |
| `gen,golden,compare` | ❌ 不支持 | - | 缺少`run` |
| `gen,run,compare` | ✅ 支持 | `golden.pt` | 生成输入、执行并比对 |
| `golden,run,compare` | ✅ 支持 | `inputs.pt` | 计算参考、执行并比对 |
| `gen,golden,run,compare` | ✅ 支持 | 无需已有数据 | 完整流程 |

`--stages`默认`all`。未包含`gen`时从`<out-dir>/<case_id>/inputs.pt`加载输入；未包含`golden`但包含`compare`时从`golden.pt`加载参考。跨命令复用阶段数据时配合`--save-data`。

### 4.4 `--golden`配置

`--golden`定义参考计算方式。默认在NPU上计算参考，fp32累加，P落盘到V的dtype；需要高精度真值时使用`device=cpu,compute_dtype=fp64,p_dtype=fp32`。

`--golden`整体默认值为`device=npu,compute_dtype=fp32,p_dtype=out_dtype`；下表“默认值”指显式传入`--golden`但省略该字段时的取值。

| 字段 | 默认值 | 可用取值 | 说明 |
|---|---|---|---|
| `device` | `cpu` | `cpu`、`npu`、`gpu` | 参考计算设备 |
| `compute_dtype` | `fp32` | `fp32`、`fp64` | `fp64`仅CPU，配合`p_dtype=fp32`使用 |
| `p_dtype` | `out_dtype` | `out_dtype`、`fp32` | P落盘精度 |
| `p_block` | `0`（auto） | 整数，0=auto | P落盘/在线归一化的块大小 |
| `softmax_mode` | `auto` | `auto`、`full`、`online` | softmax实现方式 |
| `softmax_block_size` | `0`（auto） | 整数≥0，0=auto | online模式的KV块大小 |
| `kv_loader` | `dense` | `dense`、`paged` | `paged`仅PA使用 |
| `provider` | `default` | `default`、`peer` | 参考实现口径 |

未识别的字段会直接报错。

## 5. 输出

结果写入`<out-dir>/results.csv`，每个case一行：

- `pass`：精度通过；
- `fail`：精度失败；
- `skip`：算子拒绝或守卫跳过；
- `error`：执行或比对异常。

`--save-data`会在`<out-dir>/<case_id>/`下保存`inputs.pt`、`golden.pt`和`npu_out.pt`；默认不保存。

`case_id`从`results.csv`读取；同一组`params`的case_id稳定，不受机器、路径和时间影响。

## 6. 精度用例的构造

用例格式（两条case）：

```json
{
  "schema": "case/v1",
  "count": 2,
  "cases": [
    {
      "op": "flash_attn",
      "params": {
        "dtype": "fp16",
        "layout_q": "BNSD",
        "B": 2,
        "N1": 8,
        "S1": 256,
        "S2": 256,
        "D": 128,
        "mask_mode": 0
      },
      "tags": ["ci"]
    },
    {
      "op": "flash_attn",
      "params": {
        "dtype": "bf16",
        "layout_q": "BSND",
        "layout_kv": "BSND",
        "layout_out": "BSND",
        "B": 1,
        "N1": 8,
        "N2": 2,
        "S1": 145,
        "S2": 257,
        "D": 64,
        "DV": 64,
        "mask_mode": 3,
        "return_softmax_lse": true,
        "seed": 13
      },
      "tags": ["func_rdv"]
    }
  ]
}
```

顶层也可以直接是case数组。`cases`数组的每个元素是一条case；单条case对象不能单独作为文件顶层。

文件字段如下：

| 字段 | 含义 | 必填 | 不填时 |
|---|---|---|---|
| `schema` | case文件格式版本 | 否 | 按`case/v1`处理；填写其他值会报错 |
| `count` | `cases`数量 | 否 | 不校验；填写时必须等于数组长度 |
| `cases` | 单条case的数组 | 是 | 文件校验失败 |
| `op` | 单条case的算子名 | 否 | 默认`flash_attn` |
| `params` | 单条case的算子参数对象 | 是 | 文件校验失败 |
| `tags` | pytest marker | 否 | 默认为空 |

`params`字段如下；参数含义和取值范围以算子规格为准：

| 参数 | 含义 | 必填 | 不填时 |
|---|---|---|---|
| `dtype` | 数据类型 | 否 | 默认`fp16` |
| `N1` | Q头数 | 是 | 归一化时报错 |
| `N2` | KV头数 | 否 | 默认等于`N1` |
| `D` | Q/K头维 | 是 | 执行构造时报错 |
| `DV` | V头维 | 否 | 默认等于`D` |
| `B` | batch大小 | 非TND必填 | TND由`cu_seqlens`推导；dense无法构造 |
| `S1` | Q序列长度 | 非TND必填 | TND由`cu_seqlens_q`推导；dense无法构造 |
| `S2` | KV序列长度 | 否 | 默认等于`S1`；TND由`cu_seqlens_kv`推导 |
| `layout_q` | Q layout | 否 | 默认`BNSD` |
| `layout_kv` | KV layout | 否 | 默认等于`layout_q` |
| `layout_out` | 输出layout | 否 | 默认等于`layout_q` |
| `mask_mode` | mask类型 | 否 | 默认`0` |
| `scale` | attention scale | 否 | 默认`1/sqrt(D)` |
| `win_left`/`win_right` | window边界 | 否 | 默认INT_MAX |
| `cu_seqlens_q` | TND Q段长 | `layout_q=TND`时必填 | 非TND忽略；TND缺失报错 |
| `cu_seqlens_kv` | TND KV段长 | 否 | 默认复用`cu_seqlens_q`；非TND忽略 |
| `seqused_q`/`seqused_kv` | 有效长度 | 否 | TND由段长推导；PA的`seqused_kv`默认`[S2]*B` |
| `max_seqlen_q`/`max_seqlen_kv` | 最大长度 | 否 | 由`seqused`、`cu_seqlens`或`S1`/`S2`推导 |
| `return_softmax_lse` | 是否返回lse | 否 | 默认`false` |
| `sinks` | 是否生成sinks | 否 | 默认不生成 |
| `block_size` | PA page大小 | 否 | PA默认`128` |
| `seed` | 输入随机种子 | 否 | 默认由`params`哈希生成 |

`tags`用于pytest marker；`params.seed`用于固定输入。

## 7. 已知说明与约束

- 只做功能/精度，不采集性能；
- PA的K/V在参考侧按dense展开，内存估算已计入，超限case记为`skip`；
- 显式`scale<=0`按非法输入记为`skip`；算子自报不支持或tiling失败也记为`skip`，不计入`fail`；
- 设备/队列不可恢复错误会立即中止整轮，并保留已写结果；
- `--stages`不是任意组合：`golden`需要已保存输入，`compare`需要本次执行结果；
- 不提供xlsx/zip/CSV输入，用例统一使用JSON；CSV仅用于结果输出；
- 随机用例请使用`random_fuzz`，本工具只执行已有JSON。
