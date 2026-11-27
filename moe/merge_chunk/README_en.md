# MergeChunk

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR&950DT products</term> | × |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor products</term> | × |
| <term>Kirin 9030 processor products</term> | × |

## Function Description

MergeChunk merges token rows by expert independently for each data path. Rows within each input chunk are already grouped by expert. For each expert, the operator concatenates its segments in input chunk order, preserving row order within each segment. Each path outputs the resulting segments in expert order. All paths share one expert count vector per chunk. Summing these vectors across chunks gives one count vector shared by all outputs.

Let `R` be the number of paths and `C` the number of chunks per path. For expert `e` on path `r` (`0 ≤ r < R`), `X_{r,c,e}` denotes its row segment in chunk `c`, and `g_{c,e}` denotes the number of rows in that segment. The merged segment and count are

$$
Y_{r,e}=\operatorname{Concat}_{\mathrm{rows}}(X_{r,0,e},\ldots,X_{r,C-1,e}),\qquad
G_e=\sum_{c=0}^{C-1}g_{c,e}.
$$

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| --- | --- | --- | --- | --- |
| input_chunks | Input | Token data for each path and chunk. List length: `R × C`. | FLOAT16, BFLOAT16 | ND |
| group_list_chunk | Input | One expert count tensor per chunk, shared by all paths. List length: `C`. | INT32 | ND |
| num_paths | Attribute | Number of data paths (`R`). | INT64 | - |
| num_chunks | Attribute | Number of chunks per path (`C`). | INT64 | - |
| out_chunks | Output | One merged tensor per path. List length: `R`. | FLOAT16, BFLOAT16 | ND |
| out_group_list | Output | Merged token count per expert. | INT32 | ND |

## Constraints

- This version requires two paths and two chunks per path (`R = C = 2`).
- Order input tensors by path, then by chunk within each path. Chunk `c` of path `r` has shape `[t_c, h_r]`, where `t_c > 0`. All paths share the row count and expert partition for each chunk. Width must match within a path and may differ between paths.
- Each count tensor is a vector of length `E`, with `1 ≤ E ≤ 1024`. The caller must supply nonnegative counts that sum to `t_c` for each chunk.
- Use the same data type, FLOAT16 or BFLOAT16, for all token data tensors. Each `h_r` must be a multiple of 16 in `[16, 90112]`. Let `T` be the sum of the chunk row counts for one path. Each output has shape `[T, h_r]`, and `T × h_r` must fit in INT32.
- All tensors must be contiguous, use ND format and reside on the same device. Output buffers must be mutually disjoint and must not overlap any input buffer.

## Calling Instructions

| Calling Method | Example | Description |
| --- | --- | --- |
| aclnn | [test_aclnn_merge_chunk.cpp](examples/test_aclnn_merge_chunk.cpp) | [aclnnMergeChunk](docs/aclnnMergeChunk_en.md) |
