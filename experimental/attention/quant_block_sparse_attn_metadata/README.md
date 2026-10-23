# QuantBlockSparseAttnMetadata

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

产品支持范围与配套的 [QuantBlockSparseAttn（FP8）](../quant_block_sparse_attn/docs/QuantBlockSparseAttn.md) 和 [QuantBlockSparseAttnMx（MXFP8）](../quant_block_sparse_attn/docs/QuantBlockSparseAttnMx.md) 保持一致。代码中的平台分支用于选择调度表格式，不表示额外产品已受支持。

## 功能说明

- 算子功能：`QuantBlockSparseAttnMetadata`（qbsa_metadata）是 `QuantBlockSparseAttn` 的前置算子，根据稀疏块数量在 AI CPU 上生成分段、分核的任务划分方案，供后续块稀疏 Attention 计算使用。
- 算子根据各 Batch、各 Q 头、各 Q 块行的有效 KV 块数估算负载，按访存成本划分 section，再将每个 section 的任务分配给 AI Core。
- 本算子不执行 Attention 计算，也不生成稀疏索引；输出 `metadata` 应传给配套的 `QuantBlockSparseAttn` 算子。

### 调度计算过程

记 `L[b,n,q] = max(sparse_seq_len[b,n,q], 0)`，表示第 b 个 Batch、第 n 个 Q 头、第 q 个 Q 块行的有效 KV 块数，则该 Batch/Head 的稀疏任务块总数为：

$$
W_{b,n} = \sum_{q=0}^{Qb-1} L[b,n,q]
$$

主要计算过程为：

1. 统计每个 Batch/Head 的任务块总数、非空 Q 块行数及单行最大 KV 块数；二维输入 `(B,Qb)` 的块数在各 Q 头间共享。
2. 按 `bn = b * Nq + n` 的顺序遍历 Batch/Head，估算 Q/KV 访存成本。当累计成本超过实现中设定的 120 MiB 阈值时，在 Batch/Head 边界划分新的 section。
3. 对每个 section，根据剩余任务块数和剩余核数计算目标负载，优先分配完整 Batch/Head，再按完整 Q 块行分配任务，跳过无有效 KV 块的行。
4. 将各核使能标记及任务起止位置写入 metadata。起止 Q/KV 位置采用稀疏块坐标，供主算子读取。

上述块数用于调度负载估算，输出是任务划分信息。由于当前实现按完整 Q 块行分配任务，各核的实际块数可能不同。

## 接口说明

### PyTorch 接口原型

```python
torch.ops.custom.npu_quant_block_sparse_attn_metadata(
    sparse_seq_len: Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    seqused_q: Optional[Tensor] = None,
    seqused_kv: Optional[Tensor] = None,
    batch_size: int = 0,
    sparse_block_size_q: int = 128,
    sparse_block_size_k: int = 128,
    quant_mode: int = 1,
    mask_mode: int = 3,
    layout_q: str = "TND",
    layout_kv: str = "PA_BNBD",
    layout_sparse_indices: str = "B_N_Qb_Kb",
) -> Tensor
```

`batch_size` 虽有默认值 0，实际调用必须显式传入正数；当前实现不会根据输入张量自动推导 Batch 数量。MXFP8 场景必须显式设置 `quant_mode=2`。

## 参数说明

以下使用输入和属性的 snake_case 名称，默认值以 PyTorch 接入层为准。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| sparse_seq_len | 必选输入 | 每个 Q 块行参与计算的有效 KV 稀疏块数，推荐 shape 为 `(B, Nq, Qb)`；变长布局兼容 `(B, Qb)`，表示各 Q 头共享块数。 | INT32 | ND |
| cu_seqlens_q | 可选输入 | Q 的累积序列长度，通常为 `(B+1,)`。当前调度不读取其数值；BSND/BNSD 场景必须为空。 | INT32 | ND |
| cu_seqlens_kv | 可选输入 | KV 的累积序列长度，通常为 `(B+1,)`。当前调度不读取其数值。 | INT32 | ND |
| seqused_q | 可选输入 | 各 Batch 的有效 Q 长度，通常为 `(B,)`。当前调度不读取其数值；BSND/BNSD 场景必须为空。 | INT32 | ND |
| seqused_kv | 可选输入 | 各 Batch 的有效 KV 长度，通常为 `(B,)`。当前调度不读取其数值。 | INT32 | ND |
| batch_size | 属性 | Batch 数量 B，必须大于 0，并与 sparse_seq_len 的第 0 维一致。 | INT64 | - |
| num_heads_q | 属性 | Q 头数 Nq，必须大于 0；三维 sparse_seq_len 的第 1 维必须等于 Nq。 | INT64 | - |
| num_heads_kv | 属性 | KV 头数。非 0 时必须为正数，且能整除 num_heads_q；为 0 时内部按 num_heads_q 处理。 | INT64 | - |
| head_dim | 属性 | 注意力头维度，用于估算访存成本，需与配套 Attention 一致； | INT64 | - |
| sparse_block_size_q | 属性 | Q 稀疏块大小，默认 128。quant_mode=1 时仅支持 128；quant_mode=2 时支持 64、128。 | INT64 | - |
| sparse_block_size_k | 属性 | KV 稀疏块大小，默认 128。quant_mode=1 时仅支持 128；quant_mode=2 时支持 64、128。 | INT64 | - |
| quant_mode | 属性 | 量化模式，1 表示 FP8，2 表示 MXFP8，默认 1。 | INT64 | - |
| mask_mode | 属性 | 支持 0（No mask）、3（rightDownCausal），默认 3；需与配套 Attention 一致。 | INT64 | - |
| layout_q | 属性 | Q 的布局，默认 TND。FP8 支持 TND、NTD；MXFP8 支持 TND、BSND、BNSD，另见平台约束。 | STRING | - |
| layout_kv | 属性 | KV 的布局，仅支持 PA_BNBD，默认 PA_BNBD。 | STRING | - |
| layout_sparse_indices | 属性 | 稀疏索引布局，仅支持 B_N_Qb_Kb，默认 B_N_Qb_Kb。 | STRING | - |
| metadata | 输出 | 一维任务调度表，包含 section 数量、各核任务起止位置等信息；长度见下文。 | INT32 | ND |

其中，B 为 Batch 数量，Nq 为 Q 头数，Qb 为输入稀疏块行维度的大小。上述布局描述的是配套 Attention 输入的逻辑布局，metadata 算子本身不接收 Q、K、V 张量。

与主算子文档中的维度符号对应关系如下：

| 本文符号 | 主算子文档符号 | 含义 |
| --- | --- | --- |
| Nq | N1 | Query head 数量 |
| Nkv | N2 | KV head 数量 |
| Qb | max_Qb | 每个 Batch 的 Query block 最大数量 |
| head_dim | D | Query/Key 的 head dim |

## 返回值

PyTorch 接口返回一个与 `sparse_seq_len` 位于相同设备上的一维 INT32 `metadata` Tensor，shape 为 `(584 + 576 * B * Nq,)`。其中 `metadata[0]` 保存实际 section 数量，预分配容量与实际写入的调度表长度不同，详见下文。

## 约束说明

- 本算子需与 `QuantBlockSparseAttn` 配套使用，量化模式、头数、头维度、布局、块大小及稀疏块信息应保持一致。
- `sparse_seq_len` 的元素表示块数，不是 token 数。调用方需保证其与配套 Attention 的 `sparse_indices` 一致，无效或补齐的 Q 块行应填 0。当前调度实现将非正数按 0 处理。
- BSND/BNSD 场景要求 `sparse_seq_len` 为 `(B, Nq, Qb)` 且 Qb 大于 0，`cu_seqlens_q`、`seqused_q` 必须为空。
- 当前分核主要依据 `sparse_seq_len`；序列长度输入不会自动修正该张量中的有效块数，`mask_mode` 也不在 metadata 算子中执行实际的 mask 计算。

### 与 MXFP8 主算子配套的约束

- MXFP8 主算子必须传入本算子生成的有效 `metadata`，不能传 None 或空 Tensor。
- `head_dim` 必须与 Query 的 D 一致。在主算子文档描述的 Ascend 950 场景下支持 64、128、256；
- `sparse_block_size_q`、`sparse_block_size_k` 分别对应主算子的 `sparse_q_block_size`、`sparse_kv_block_size`；MXFP8 配套使用时两者必须相等，均为 64 或 128。
- 配套 MXFP8 Attention 使用三维 `sparse_seq_len=(B,Nq,Qb)`，元素必须在 `[0,max_Kb]` 内，其中 max_Kb 为 `sparse_indices` 最后一维的大小。metadata 接口兼容的二维输入不代表主算子支持二维输入。
- BSND/BNSD 场景下，Qb 应至少为 `ceil(Sq/sparse_block_size_q)`，Sq 为每个 Batch 的 Q 长度；两处调用均须设置相同的 `layout_q`，主算子的 `layout_out` 也须与之相同。
- TND 场景下，主算子必须传入 `cu_seqlens_q`、`seqused_kv`；BSND/BNSD 场景下，主算子必须传入 `seqused_kv`，且 `cu_seqlens_q`、`seqused_q` 必须为 None。主算子的 `cu_seqlens_kv` 为预留参数，必须为 None。这些是配套 Attention 的要求，metadata 调度本身不读取长度张量的数值。
- `sparse_seq_len` 或其他任务划分参数发生变化后，应重新生成 metadata，确保调度范围与本次 Attention 的稀疏任务一致。

### metadata 长度

| 平台分支 | aic_slots | aiv_slots | 最小元素数 |
| --- | --- | --- | --- |
| 950PR&950DT | 36 | 72 | `584 + 288 * B * Nq` |

PyTorch 接入层统一按 `584 + 576 * B * Nq` 分配容量。这里的 slot 数表示调度表容量，不代表实际启动的核数。

实际调度表使用的元素数为 `8 + section_num * aic_slots * 8 + aiv_slots * 8`，其中 `section_num=metadata[0]`。调用方分配输出时应使用上表中的容量公式，为最多 `B*Nq` 个 section 预留空间。

输出依次包含 8 个元素的头部、按实际 section 数排列的 AIC 任务表和 AIV 规约信息区。每个 AIC 任务记录占 8 个元素，包括使能标记、起止 Batch/Head 合并索引、Q/KV 块位置及规约工作区索引。调用方应将输出直接交给配套算子使用，避免自行修改内部字段。

## 调用说明

| 调用方式 | 接口与说明 |
| --- | --- |
| PyTorch API | 通过配套算子的 [torch_ops_extension](../quant_block_sparse_attn/torch_ops_extension/README.md) 注册 `torch.ops.custom.npu_quant_block_sparse_attn_metadata`，接入层自动分配输出。 |

### 环境准备

- 按 [PyTorch 接入层说明](../quant_block_sparse_attn/torch_ops_extension/README.md) 准备 Linux、CANN Toolkit、匹配的 PyTorch 与 TorchNPU，并构建安装 `custom_ops`。
- 将输入张量放在目标 NPU 上；先导入 `custom_ops` 完成自定义算子注册。

### FP8 实现参考

实现参考：[参数校验](./op_host/quant_block_sparse_attn_metadata_check.h)、[AI CPU 调度实现](./op_kernel_aicpu/quant_block_sparse_attn_metadata_aicpu.cpp)、[metadata 结构定义](./op_kernel_aicpu/quant_block_sparse_attn_metadata.h)。

### MXFP8 实现参考

实现参考: [QuantBlockSparseAttnMx 调用示例](../quant_block_sparse_attn/docs/QuantBlockSparseAttnMx.md#调用示例)。
