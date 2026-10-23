# aclnnSplitQkvRmsnormMrope

## 功能说明

拆分融合 QKV/Gate 输入，对 Q/K 执行 RMSNorm、仿射和三轴 MRoPE，并输出保持原始 BF16
位表示的 V/Gate。

## 接口原型

```cpp
aclnnStatus aclnnSplitQkvRmsnormMropeGetWorkspaceSize(
    const aclTensor *qkv,
    const aclTensor *qWeight,
    const aclTensor *kWeight,
    const aclTensor *cosSin,
    const aclTensor *qBiasOptional,
    const aclTensor *kBiasOptional,
    int64_t numQHeads,
    int64_t numKvHeads,
    double eps,
    bool interleaved,
    bool hasGate,
    const aclTensor *qOutput,
    const aclTensor *kOutput,
    const aclTensor *vOutput,
    const aclTensor *gateOutput,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnSplitQkvRmsnormMrope(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 参数说明

- `qkv`：融合输入，BFLOAT16，shape 见上级 README。`hasGate=true` 时 Q/Gate 按 head 交错排列为
  `[q0, gate0, q1, gate1, ...]`，随后为连续的 K、V 区域。
- `qWeight`、`kWeight`：Q/K RMSNorm 仿射权重，BFLOAT16 `[256]`。
- `cosSin`：三轴 MRoPE 系数，BFLOAT16 `[3,T,64]`。
- `qBiasOptional`、`kBiasOptional`：可选 BFLOAT16 `[256]`。
- `numQHeads`：当前固定为 16。
- `numKvHeads`：当前支持 2 或 4。
- `eps`：RMSNorm epsilon，有限正数。
- `interleaved`：三轴 MRoPE 映射模式。对前 64 维组成的第 `j` 个旋转维度对，`true` 时使用轴
  `j % 3`；`false` 时按连续 section `[11,11,10]` 依次使用轴 0、1、2。两种模式的 cos/sin
  频率下标均使用全局 `j`，不会在 section 内重新从 0 计数。
- `hasGate`：融合输入是否包含 Gate。
- `qOutput`、`kOutput`：归一化、仿射并旋转后的 Q/K。
- `vOutput`、`gateOutput`：原始 BF16 位表示直通结果。
- `workspaceSize`、`executor`：第一阶段返回的 workspace 大小和执行器。
- `stream`：执行使用的 ACL stream。

## 调用流程

必须先调用 `aclnnSplitQkvRmsnormMropeGetWorkspaceSize`，按返回值分配 workspace，再调用执行接口。
完整资源管理代码参见
[`examples/test_aclnn_split_qkv_rmsnorm_mrope.cpp`](../examples/test_aclnn_split_qkv_rmsnorm_mrope.cpp)。
