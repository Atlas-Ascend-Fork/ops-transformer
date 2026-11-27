# MergeChunk

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR&950DT系列产品</term> | × |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |
| <term>Kirin X90处理器系列产品</term> | × |
| <term>Kirin 9030处理器系列产品</term> | × |

## 功能说明

MergeChunk 对每一路 token 数据独立执行按专家合并。每个 chunk 的输入行已按专家编号分段。对同一专家，算子按 chunk 的输入顺序拼接各段，保留段内的行序；再按专家编号依次输出。同一个 chunk 的各路输入共用一份专家计数，各 chunk 的计数按专家相加，得到各路输出共用的一份计数。

设数据路数为 `R`，每路的 chunk 数为 `C`。对第 `r` 路（`0 ≤ r < R`）的专家 `e`，`X_{r,c,e}` 表示该专家在第 `c` 个 chunk 中的行段，`g_{c,e}` 表示该段的行数。合并后的行段和计数分别为

$$
Y_{r,e}=\operatorname{Concat}_{\mathrm{rows}}(X_{r,0,e},\ldots,X_{r,C-1,e}),\qquad
G_e=\sum_{c=0}^{C-1}g_{c,e}.
$$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| input_chunks | 输入 | 各路、各 chunk 的 token 数据，列表长度为 `R × C`。 | FLOAT16、BFLOAT16 | ND |
| group_list_chunk | 输入 | 每个 chunk 一份专家计数，各路共用，列表长度为 `C`。 | INT32 | ND |
| num_paths | 属性 | 数据路数 `R`。 | INT64 | - |
| num_chunks | 属性 | 每路的 chunk 数 `C`。 | INT64 | - |
| out_chunks | 输出 | 每路一个合并结果，列表长度为 `R`。 | FLOAT16、BFLOAT16 | ND |
| out_group_list | 输出 | 各专家合并后的 token 数。 | INT32 | ND |

## 约束说明

- 当前版本仅支持两路数据、每路两个 chunk（`R = C = 2`）。
- 输入列表先按数据路排列，每路内按 chunk 顺序排列。第 `r` 路、第 `c` 个 chunk 的形状为 `[t_c, h_r]`，其中 `t_c > 0`。同一 chunk 在各路中的行数和专家分段一致。同一路内各 chunk 的宽度相同，不同路的宽度可以不同。
- 每份计数是长度为 `E` 的一维 Tensor，`1 ≤ E ≤ 1024`。调用方须确保各项非负且每份计数之和等于对应的 `t_c`。
- 所有数据 Tensor 统一使用 FLOAT16 或 BFLOAT16。`h_r` 是 16 的倍数，范围为 `[16, 90112]`。设 `T` 为每路所有 chunk 的 token 行数之和，每路输出形状为 `[T, h_r]`，`T × h_r` 不超过 INT32 最大值。
- 所有 Tensor 均为连续 ND 格式，位于同一设备。各输出缓冲区相互独立，与输入缓冲区无重叠。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn | [test_aclnn_merge_chunk.cpp](examples/test_aclnn_merge_chunk.cpp) | [aclnnMergeChunk](docs/aclnnMergeChunk.md) |
