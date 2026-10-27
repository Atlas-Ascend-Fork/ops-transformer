# BlockSparseAttention Torch HIF8 支持

### quant_mode=HIF8_QKVP_PERTENSOR_QUANT（5，HIF8 per-tensor量化）相关约束

- 仅<term>Ascend 950PR&950DT系列产品</term>支持。query、key、value必须同时为uint8，并按HIFLOAT8编码解释；HIFLOAT8输入必须使用`quant_mode=HIF8_QKVP_PERTENSOR_QUANT`（5）。
- q_input_layout与kv_input_layout必须一致，仅支持"TND"、"BNSD"、"BSND"；query、key、value的D轴仅支持64或128。
- q_dequant_scale、k_dequant_scale、v_dequant_scale必须同时传入，均为float32、shape为[1]，分别作为整个query、key、value张量共用的反量化缩放因子；不支持per-token或per-channel缩放因子。
- p_quant_scale可选。传入时必须为float32、shape为[1]，且为有限正数，用于softmax结果P的量化；传入None时默认使用16.0。其他量化模式必须为None。
- q_dequant_scale、k_dequant_scale、v_dequant_scale及scale_value必须为有限数；有效缩放因子q_dequant_scale × k_dequant_scale × scale_value必须为有限非负数。上述数值约束由调用者保证，接口仅在算子侧校验缩放因子的类型和shape。
- inner_precise必须为0，online softmax和rescale均采用float32计算。
- block_sparse_mask必须传入；block_x和block_y均须为正数且为16的倍数。Q/KV序列长度无需被稀疏块大小整除。
- mask_type必须为0，block_table必须为None，block_size必须为0，pre_tokens和next_tokens必须为2147483647。
- attention_out_dtype必须为float16或bfloat16；输出shape与query一致。`return_softmax_lse=True`时可返回float32类型的softmax_lse。
- dst_type_max在此模式下不参与计算，建议传入0.0。

HIF8 通过 `QuantMode.HIF8_QKVP_PERTENSOR_QUANT`（5）选择；调用参数与本目录 `torch_extension/block_sparse_attention.py` 中的 `block_sparse_attention` 接口一致。

### HIF8 块内有效行 mask

`quant_mode=5` 支持通过 `atten_mask` 传入 INT32 的 `blockEffRows`，形状为 `[batch, qHeads, maxBlockNum, 2]`，其中 `maxBlockNum >= max(Q块数, KV块数)`。最后一维的 0/1 分别表示对应 Q/KV 块从首行开始的有效行数。它与 `block_sparse_mask` 共同决定参与计算的数据，不表示任意逐元素 mask。

有效 Q 行数按实际序列长度和块大小裁剪；选中 KV 块的有效前缀按原块顺序紧凑拼接，采用既有 per-tensor scale。零有效 Q 或零有效 KV 的行保留调用方 O/LSE；Torch 接口使用未初始化输出，调用方不能读取被屏蔽行作为有效数值。
