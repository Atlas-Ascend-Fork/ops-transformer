# aclnnMergeChunk

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

MergeChunk merges token rows by expert independently for each data path. Rows within each input chunk are already grouped by expert. For each expert, the operator concatenates its segments in input chunk order, preserving row order within each segment. Each path outputs the resulting segments in expert order. All paths share one expert count vector per chunk. Summing these vectors across chunks gives one count vector shared by all outputs.

Let `R` be the number of paths and `C` the number of chunks per path. For expert `e` on path `r` (`0 ≤ r < R`), `X_{r,c,e}` denotes its row segment in chunk `c`, and `g_{c,e}` denotes the number of rows in that segment. The merged segment and count are

$$
Y_{r,e}=\operatorname{Concat}_{\mathrm{rows}}(X_{r,0,e},\ldots,X_{r,C-1,e}),\qquad
G_e=\sum_{c=0}^{C-1}g_{c,e}.
$$

## Function Prototypes

Use the [two-stage API](../../../docs/en/context/two_phase_api.md). Call `aclnnMergeChunkGetWorkspaceSize` to obtain the workspace size and executor, then call `aclnnMergeChunk` to submit the operation.

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

**Parameters**

| Parameter | Input/Output | Description | Usage | Data Type | Format | Shape | Non-contiguous Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `inputChunks` (`const aclTensorList*`) | Input | Token data for each path and chunk. List length: `R × C`. | Item `r * C + c` is chunk `c` of path `r`. | FLOAT16, BFLOAT16 | ND | `[t_c, h_r]` | × |
| `groupListChunk` (`const aclTensorList*`) | Input | One expert count tensor per chunk, shared by all paths. List length: `C`. | In item `c`, entry `e` gives the token count for expert `e`. | INT32 | ND | `[E]` | × |
| `numPaths` (`int64_t`) | Input | Number of data paths (`R`). | Required. | INT64 | - | - | - |
| `numChunks` (`int64_t`) | Input | Number of chunks per path (`C`). | Required. | INT64 | - | - | - |
| `outChunks` (`const aclTensorList*`) | Output | One merged tensor per path. List length: `R`. | Provide output tensors in path order. Item `r` corresponds to path `r`. | FLOAT16, BFLOAT16 | ND | `[T, h_r]` | × |
| `outGroupList` (`const aclTensor*`) | Output | Merged token count per expert. | Provide an output tensor of length `E`. | INT32 | ND | `[E]` | × |
| `workspaceSize` (`uint64_t*`) | Output | Required workspace size in bytes. | Use the returned size to allocate device memory for the second call. | - | - | - | - |
| `executor` (`aclOpExecutor**`) | Output | Operator executor. | Pass the returned executor to the second call. | - | - | - | - |

**Return Value**

Returns `aclnnStatus`. See [aclnn return codes](../../../docs/en/context/aclnn_return_code.md). Returns `ACLNN_ERR_PARAM_INVALID` if the path or chunk count is unsupported or a list length does not match the supplied counts.

## aclnnMergeChunk

**Parameters**

| Parameter | Input/Output | Description | Usage | Data Type | Format | Shape | Non-contiguous Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `workspace` (`void*`) | Input | Temporary device memory. | May be null if the required size is zero. | - | - | - | - |
| `workspaceSize` (`uint64_t`) | Input | Workspace size in bytes. | Use the size returned by the first call. | - | - | - | - |
| `executor` (`aclOpExecutor*`) | Input | Operator executor. | Use the executor returned by the first call. | - | - | - | - |
| `stream` (`aclrtStream`) | Input | Execution stream. | Wait for completion before reading the outputs. | - | - | - | - |

**Return Value**

Returns `aclnnStatus`. See [aclnn return codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- This version requires two paths and two chunks per path (`R = C = 2`).
- Order input tensors by path, then by chunk within each path. Chunk `c` of path `r` has shape `[t_c, h_r]`, where `t_c > 0`. All paths share the row count and expert partition for each chunk. Width must match within a path and may differ between paths.
- Each count tensor is a vector of length `E`, with `1 ≤ E ≤ 1024`. The caller must supply nonnegative counts that sum to `t_c` for each chunk.
- Use the same data type, FLOAT16 or BFLOAT16, for all token data tensors. Each `h_r` must be a multiple of 16 in `[16, 90112]`. Let `T` be the sum of the chunk row counts for one path. Each output has shape `[T, h_r]`, and `T × h_r` must fit in INT32.
- All tensors must be contiguous, use ND format and reside on the same device. Output buffers must be mutually disjoint and must not overlap any input buffer.

## Example

For a complete C++ call and output check, see [test_aclnn_merge_chunk.cpp](../examples/test_aclnn_merge_chunk.cpp).
