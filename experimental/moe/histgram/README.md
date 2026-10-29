# Histgram（Experimental） <!-- codespell:ignore histgram -->

本目录提供算子 `histgram`（分类：moe）。 <!-- codespell:ignore histgram -->
统计一维 token 序列中每个 expert 出现的次数（直方图），常用于 MoE 场景下的 token 分发计数。

## 功能与接口

给定一维 int32 token 序列，统计 `[0, numExpert)` 内每个取值的出现次数：

```
out[e] = count(input == e),  e in [0, numExpert)
```

Torch 接口：

```python
torch.ops.cann_ops_transformer.histgram(input, num_expert)  # codespell:ignore histgram
```

| 参数 | dtype / shape | 说明 |
| --- | --- | --- |
| input | INT32；`[num_token]`，`ND` | 一维 token 序列，取值属于 `[0, num_expert)` |
| num_expert | int | expert 个数（算子属性），需大于 0 |
| out | INT32；`[num_expert]` | 每个 expert 的出现次数 |

## 约束

- `input` 必须为一维、dtype 固定 `int32`。
- `num_expert > 0`。

## 目录与实现

```text
histgram/  # codespell:ignore histgram
  CMakeLists.txt
  op_host/                       # OpDef、InferShape/InferDataType、Tiling
    histgram_def.cpp
    histgram_infershape.cpp
    histgram_tiling.cpp
    histgram_tiling.h
  op_kernel/                     # Ascend C 内核（向量）
    histgram.cpp  # codespell:ignore histgram
    histgram_impl.hpp
    histgram_tiling_def.h
  torch_extension/               # Python schema/Meta + C++ ACLNN 桥接
    histgram.py  # codespell:ignore histgram
    csrc/histgram.cpp  # codespell:ignore histgram
```

实现要点：纯向量算子（AIV）。按 expert 分块、token 分块，各向量核分别统计局部直方图并写入
workspace，再汇总（`tokenStartPos == 0` 直接写回，否则原子加）得到最终输出。

## 调用示例

```python
import torch
import torch_npu  # noqa: F401
import cann_ops_transformer  # 触发 torch_extension 注册

num_token, num_expert = 32000 * 3, 32
input = torch.randint(0, num_expert, (num_token,), dtype=torch.int32, device="npu:0")
out = torch.ops.cann_ops_transformer.histgram(input, num_expert)  # codespell:ignore histgram
```
