# aclnnMergeChunk

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

MergeChunk 对每一路 token 数据独立执行按专家合并。每个 chunk 的输入行已按专家编号分段。对同一专家，算子按 chunk 的输入顺序拼接各段，保留段内的行序；再按专家编号依次输出。同一个 chunk 的各路输入共用一份专家计数，各 chunk 的计数按专家相加，得到各路输出共用的一份计数。

设数据路数为 `R`，每路的 chunk 数为 `C`。对第 `r` 路（`0 ≤ r < R`）的专家 `e`，`X_{r,c,e}` 表示该专家在第 `c` 个 chunk 中的行段，`g_{c,e}` 表示该段的行数。合并后的行段和计数分别为

$$
Y_{r,e}=\operatorname{Concat}_{\mathrm{rows}}(X_{r,0,e},\ldots,X_{r,C-1,e}),\qquad
G_e=\sum_{c=0}^{C-1}g_{c,e}.
$$

## 函数原型

本接口采用[两段式调用](../../../docs/zh/context/two_phase_api.md)。先调用 `aclnnMergeChunkGetWorkspaceSize` 获取 workspace 大小和执行器，再调用 `aclnnMergeChunk` 提交计算任务。

```cpp
aclnnStatus aclnnMergeChunkGetWorkspaceSize(
    const aclTensorList *inputChunks,
    const aclTensorList *groupListChunk,
    int64_t              numPaths,
    int64_t              numChunks,
    const aclTensorList *outChunks,
    const aclTensor     *outGroupList,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor);
```

```cpp
aclnnStatus aclnnMergeChunk(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnMergeChunkGetWorkspaceSize

**参数说明**

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） | 非连续 Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `inputChunks` (`const aclTensorList*`) | 输入 | 各路、各 chunk 的 token 数据，列表长度为 `R × C`。 | 第 `r * C + c` 项对应第 `r` 路、第 `c` 块。 | FLOAT16、BFLOAT16 | ND | `[t_c, h_r]` | × |
| `groupListChunk` (`const aclTensorList*`) | 输入 | 每个 chunk 一份专家计数，各路共用，列表长度为 `C`。 | 第 `c` 项中，下标 `e` 对应专家 `e` 的 token 数。 | INT32 | ND | `[E]` | × |
| `numPaths` (`int64_t`) | 输入 | 数据路数 `R`。 | 必选。 | INT64 | - | - | - |
| `numChunks` (`int64_t`) | 输入 | 每路的 chunk 数 `C`。 | 必选。 | INT64 | - | - | - |
| `outChunks` (`const aclTensorList*`) | 输出 | 每路一个合并结果，列表长度为 `R`。 | 调用方按路序提供输出 Tensor，第 `r` 项对应第 `r` 路。 | FLOAT16、BFLOAT16 | ND | `[T, h_r]` | × |
| `outGroupList` (`const aclTensor*`) | 输出 | 各专家合并后的 token 数。 | 调用方提供长度为 `E` 的输出 Tensor。 | INT32 | ND | `[E]` | × |
| `workspaceSize` (`uint64_t*`) | 输出 | 所需 workspace 字节数。 | 用于准备第二阶段所需的设备内存。 | - | - | - | - |
| `executor` (`aclOpExecutor**`) | 输出 | 算子执行器。 | 传给第二阶段接口。 | - | - | - | - |

**返回值**

返回 `aclnnStatus`，详见[aclnn 返回码](../../../docs/zh/context/aclnn_return_code.md)。当路数或 chunk 数超出当前支持范围，或列表长度与参数不匹配时，返回 `ACLNN_ERR_PARAM_INVALID`。

## aclnnMergeChunk

**参数说明**

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） | 非连续 Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `workspace` (`void*`) | 输入 | 设备侧临时内存。 | 大小为 0 时可传空指针。 | - | - | - | - |
| `workspaceSize` (`uint64_t`) | 输入 | 临时内存字节数。 | 使用第一阶段返回值。 | - | - | - | - |
| `executor` (`aclOpExecutor*`) | 输入 | 算子执行器。 | 使用第一阶段返回的执行器。 | - | - | - | - |
| `stream` (`aclrtStream`) | 输入 | 执行任务的流。 | 读取输出前须等待任务完成。 | - | - | - | - |

**返回值**

返回 `aclnnStatus`，详见[aclnn 返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 当前版本仅支持两路数据、每路两个 chunk（`R = C = 2`）。
- 输入列表先按数据路排列，每路内按 chunk 顺序排列。第 `r` 路、第 `c` 个 chunk 的形状为 `[t_c, h_r]`，其中 `t_c > 0`。同一 chunk 在各路中的行数和专家分段一致。同一路内各 chunk 的宽度相同，不同路的宽度可以不同。
- 每份计数是长度为 `E` 的一维 Tensor，`1 ≤ E ≤ 1024`。调用方须确保各项非负且每份计数之和等于对应的 `t_c`。
- 所有数据 Tensor 统一使用 FLOAT16 或 BFLOAT16。`h_r` 是 16 的倍数，范围为 `[16, 90112]`。设 `T` 为每路所有 chunk 的 token 行数之和，每路输出形状为 `[T, h_r]`，`T × h_r` 不超过 INT32 最大值。
- 所有 Tensor 均为连续 ND 格式，位于同一设备。各输出缓冲区相互独立，与输入缓冲区无重叠。

## 调用示例

完整的 C++ 调用及输出校验见[test_aclnn_merge_chunk.cpp](../examples/test_aclnn_merge_chunk.cpp)。
