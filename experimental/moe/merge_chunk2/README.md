# MergeChunk2（Experimental）

本目录为原创算子 `merge_chunk2`（分类：moe）。

对每个 expert，把两个 chunk 的 token 依次拼接，并合并两个 chunk 的 group list。

## 功能与接口

给定两路 chunk：

```text
input1_chunk1: [t1, h1]
input1_chunk2: [t2, h1]
input2_chunk1: [t1, h2]
input2_chunk2: [t2, h2]
group_list_chunk1: [num_expert]
group_list_chunk2: [num_expert]
```

对每个 expert `e`：

```text
out1 = concat(input1_chunk1[e], input1_chunk2[e])
out2 = concat(input2_chunk1[e], input2_chunk2[e])
group_list[e] = group_list_chunk1[e] + group_list_chunk2[e]
```

Torch 接口：

```python
torch.ops.cann_ops_transformer.merge_chunk2(
    input1_chunk1, input1_chunk2,
    input2_chunk1, input2_chunk2,
    group_list_chunk1, group_list_chunk2,
)
```

| 参数 | dtype / shape | 说明 |
| --- | --- | --- |
| input1_chunk1 | FP16/BF16；`[t1, h1]`，ND | 第一路 chunk1 |
| input1_chunk2 | FP16/BF16；`[t2, h1]`，ND | 第一路 chunk2 |
| input2_chunk1 | FP16/BF16；`[t1, h2]`，ND | 第二路 chunk1 |
| input2_chunk2 | FP16/BF16；`[t2, h2]`，ND | 第二路 chunk2 |
| group_list_chunk1 | INT32；`[num_expert]` | 第一路每个 expert 的 token 数 |
| group_list_chunk2 | INT32；`[num_expert]` | 第二路每个 expert 的 token 数 |
| out1 | 与输入同 dtype；`[t1 + t2, h1]` | 第一路合并结果 |
| out2 | 与输入同 dtype；`[t1 + t2, h2]` | 第二路合并结果 |
| group_list | INT32；`[num_expert]` | 两路 group list 之和 |

## 约束

- `input1_chunk1` 与 `input2_chunk1` 的 token 数必须一致（`t1`）；
- `input1_chunk2` 与 `input2_chunk2` 的 token 数必须一致（`t2`）；
- 同一路的两个 chunk 的 hidden 必须一致（`h1` 或 `h2`）；
- `h1`、`h2` 必须大于 0，并且按 16 对齐；
- `num_expert <= 1024`；
- `h1`、`h2` 不超过 `MAX_TOKEN_BYTES / 2`；
- 支持 `float16`、`bfloat16` 两种数据 dtype，group list 固定 `int32`。

## 目录与实现

```text
merge_chunk2/
  CMakeLists.txt
  op_host/                       # OpDef、InferShape、Tiling
    merge_chunk2_def.cpp
    merge_chunk2_infershape.cpp
    merge_chunk2_tiling.cpp
    merge_chunk2_tiling.h
  op_kernel/                     # Ascend C 内核（向量）
    merge_chunk2.cpp
    merge_chunk2_impl.hpp
    merge_chunk2_tiling_def.h
  tests/
    pytest/test_aclnn_merge_chunk2.py
    pytest/aclnn_utils.py
    pytest/conftest.py
  torch_extension/               # Python schema/Meta + C++ ACLNN 桥接
    merge_chunk2.py
    csrc/merge_chunk2.cpp
```

实现要点：按 expert 切分 token，按核切分 token；每个核把属于当前 expert 的
chunk1/chunk2 token 依次拷贝到输出；`group_list` 由 0 号核把两路 group list 相加。

## 调用示例

```python
import torch
import torch_npu  # noqa: F401
import cann_ops_transformer  # 触发 torch_extension 注册

t1, t2, h1, h2, e = 8, 6, 16, 16, 4
in1c1 = torch.randn(t1, h1, dtype=torch.bfloat16, device="npu:0")
in1c2 = torch.randn(t2, h1, dtype=torch.bfloat16, device="npu:0")
in2c1 = torch.randn(t1, h2, dtype=torch.bfloat16, device="npu:0")
in2c2 = torch.randn(t2, h2, dtype=torch.bfloat16, device="npu:0")
glc1 = torch.tensor([2, 2, 2, 2], dtype=torch.int32, device="npu:0")
glc2 = torch.tensor([1, 2, 1, 2], dtype=torch.int32, device="npu:0")

out1, out2, gl = torch.ops.cann_ops_transformer.merge_chunk2(
    in1c1, in1c2, in2c1, in2c2, glc1, glc2
)
```

## 支持平台

当前迁移版本注册 `ascend910b`、`ascend910_93`。
