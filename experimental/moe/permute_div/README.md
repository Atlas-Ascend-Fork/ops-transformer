# PermuteDiv（Experimental）

本目录为原创算子 `permute_div`（分类：moe）。

对一维 int32 序列做 `floor(indice / divisor)`。

## 功能与接口

```text
out[i] = floor(indice[i] / divisor)
```

Torch 接口：

```python
torch.ops.cann_ops_transformer.permute_div(indice, divisor)
```

| 参数 | dtype / shape | 说明 |
| --- | --- | --- |
| indice | INT32；`[N]`，ND | 一维输入序列 |
| divisor | int | 除数，必须大于 0 |
| out | INT32；`[N]` | `floor(indice / divisor)` |

## 约束

- `indice` 必须是一维、dtype 固定 `int32`；
- `divisor > 0`；
- 单核处理的数据量不能超过 UB 限制（`MAX_UB = 184KB`）。

## 目录与实现

```text
permute_div/
  CMakeLists.txt
  op_host/
    permute_div_def.cpp
    permute_div_infershape.cpp
    permute_div_tiling.cpp
    permute_div_tiling.h
  op_kernel/
    permute_div.cpp
    permute_div_impl.hpp
    permute_div_tiling_def.h
  tests/
    pytest/test_aclnn_permute_div.py
    pytest/aclnn_utils.py
    pytest/conftest.py
  torch_extension/
    permute_div.py
    csrc/permute_div.cpp
```

实现要点：按核切分 indice；每个核把 int32 Cast 到 fp32，乘以 `1/divisor`，
加 `0.4/divisor` 修正舍入，再 `CAST_FLOOR` 回 int32。

## 调用示例

```python
import torch
import torch_npu  # noqa: F401
import cann_ops_transformer  # 触发 torch_extension 注册

indice = torch.arange(1024, dtype=torch.int32, device="npu:0")
out = torch.ops.cann_ops_transformer.permute_div(indice, 12)
```

## 支持平台

当前迁移版本注册 `ascend910b`、`ascend910_93`。
