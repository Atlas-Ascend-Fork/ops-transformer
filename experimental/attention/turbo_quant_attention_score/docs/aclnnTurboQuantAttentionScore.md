# aclnnTurboQuantAttentionScore

## 功能说明

使用 2/3/4-bit 分页量化 Key cache 计算 Query-Key attention score。接口支持 Decode 和单 batch
Prefill，输出 FLOAT32 score，并将每个 batch 的 `seq_lens` 之外的位置填充为负无穷。

## 接口原型

```cpp
aclnnStatus aclnnTurboQuantAttentionScoreGetWorkspaceSize(
    const aclTensor *queryRotated,
    const aclTensor *queryQjl,
    const aclTensor *keyCacheIdx,
    const aclTensor *keyCacheQjl,
    const aclTensor *keyCacheNorm,
    const aclTensor *keyCacheGamma,
    const aclTensor *blockTable,
    const aclTensor *seqLens,
    int64_t mseBits,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnTurboQuantAttentionScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 参数说明

- `queryRotated`：必选输入，BFLOAT16，`[T, Nq, 128]`。
- `queryQjl`：必选输入，BFLOAT16，shape 与 `queryRotated` 相同。
- `keyCacheIdx`：必选输入，UINT8，`[P, 128, Nkv, 16 * mseBits]`。
- `keyCacheQjl`：必选输入，UINT8，`[P, 128, Nkv, 16]`。
- `keyCacheNorm`：必选输入，BFLOAT16 或 FLOAT16，`[P, 128, Nkv]`。
- `keyCacheGamma`：必选输入，shape 和 dtype 与 `keyCacheNorm` 相同。
- `blockTable`：必选输入，INT32，`[B, M]`，保存逻辑页到物理页的映射。有效物理页号应位于 `[0, P)`；运行时的无效页号会被安全屏蔽，对应页输出为负无穷。
- `seqLens`：必选输入，INT32，`[B]`，保存每个 batch 的有效 Key 长度。
- `mseBits`：量化 bit 数，仅支持 2、3、4。
- `out`：必选输出，FLOAT32，`[T, Nq, L]`；接口从输出 Tensor 的最后一维取得 `L`。
- `workspaceSize`：返回执行所需 workspace 字节数。
- `executor`：返回算子执行器。
- `stream`：执行使用的 ACL stream。

所有 Tensor 均要求连续 ND 布局。更完整的 Shape 约束见上级目录 `README.md`。

## 调用流程

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus status = aclnnTurboQuantAttentionScoreGetWorkspaceSize(
    queryRotated, queryQjl, keyCacheIdx, keyCacheQjl,
    keyCacheNorm, keyCacheGamma, blockTable, seqLens,
    mseBits, out, &workspaceSize, &executor);

void *workspace = nullptr;
if (status == ACLNN_SUCCESS && workspaceSize > 0) {
    status = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}
if (status == ACLNN_SUCCESS) {
    status = aclnnTurboQuantAttentionScore(workspace, workspaceSize, executor, stream);
}
```

调用方必须使用第一阶段返回的 `workspaceSize`，不能硬编码。完整资源创建和释放示例参见
[`examples/test_aclnn_turbo_quant_attention_score.cpp`](../examples/test_aclnn_turbo_quant_attention_score.cpp)。
