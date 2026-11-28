# block_sparse_attention

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
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

> [!NOTE]
>
> <!-- npu="950" id10 -->
> <term>Ascend 950PR&950DT products</term> supports `quant_mode=0` (non-quantized float16/bfloat16), `quant_mode=1` (FP8 quantization), and `quant_mode=2/3` (MXFP4 quantization).
> <!-- end id10 -->
> <!-- npu="910b,A3" id11 -->
> <term>Atlas A2 products</term> and <term>Atlas A3 products</term> support only `quant_mode=0` (non-quantized float16/bfloat16 input).
> <!-- end id11 -->

## Description

- **API function**:

  `block_sparse_attention` is an extended `cann_ops_transformer` API based on `TorchNPU`. It is used to call the `BlockSparseAttention` operator to complete block-level sparse attention calculation. Use `block_sparse_mask` to specify the KV blocks that need to participate in the computation of each Q block to implement efficient sparse attention computation and training inference normalization.

  Currently, the following quantization scenarios are supported:

  - **Non-quantization scenario** (`quant_mode=0`, default): float16 or bfloat16 for Q, K, and V;
  - **FP8 scenario** (`quant_mode=1`): The Q, K, and V vectors are all of the type float8_e4m3fn, the FP32 dequantization scaling factor is carried, and the output is dequantized to float16 or bfloat16.
  - **MXFP4 scenario** (`quant_mode=2/3`): Q/K/V = float4_e2m1fn_x2 (passed as uint8), carries the float8_e8m0fnu descale tensor, and the output is dequantized to float16 or bfloat16.

- **Formula**:

  Self-attention is a mechanism used in machine learning, particularly in natural language processing (NLP) and computer vision tasks, to capture dependencies and relationships within input sequences. The principle is as follows: Assume that there is an input sample sequence $x$ whose length is $n$. Each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be considered as a token embedding, and the sequence is transformed by using three weight matrices to obtain three matrices whose dimensions are $n \times d$.

  The computation formula for self-attention is generally defined as follows, where $Q$, $K$, and $V$ are key attribute elements of the input sample, obtained through spatial transformation and unified into a single feature space. "Attention" in the formula and operator name is an abbreviation for "self-attention." The sparse block size is specified by $blockShapeX \times blockShapeY$, and the sparse mode is specified by `block_sparse_mask`.

  $$
  attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T + atten\_mask) \cdot value_{sparse}
  $$

  $scale$ is the scaling coefficient. Generally, $scale = D^{-0.5}$, $key_{sparse}$ is used. $value_{sparse}$ is the KV block selected by `block_sparse_mask`.

> [!NOTE]
>
> The query, key, and value data layout formats can be interpreted from multiple dimensions. B (Batch) indicates the batch size of the input sample, S (Seq-Length) indicates the length of the input sample sequence, and H (Hidden-Size) indicates the size of the hidden layer. N (Head-Num) represents a quantity of heads, D (Head-Dim) represents headdim, the minimum unit size of the hidden layer, and D=H/N is satisfied, and T represents a length (Total tokens) of closely arranged B and S on a combined axis. Q_S indicates the sequence length of the input query tensor, Q_N indicates the number of headers of the input query tensor, KV_S indicates the sequence length of the input key/value tensor, and KV_N indicates the number of headers of the input key/value tensor.

## Prototype

```python
cann_ops_transformer.block_sparse_attention(
    query,
    key,
    value,
    block_sparse_mask,
    block_shape,
    *,
    q_input_layout="TND",
    kv_input_layout="TND",
    num_key_value_heads=1,
    scale_value=1.0,
    inner_precise=1,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    return_softmax_lse=False,
    mask_type=0,
    quant_mode=0,
    block_size=0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    dst_type_max=0.0,
    atten_mask=None,
    block_table=None,
    q_dequant_scale=None,
    k_dequant_scale=None,
    v_dequant_scale=None,
    p_quant_scale=None,
    attention_out_dtype=None
) -> (Tensor, Tensor)
```

> [!NOTE]
>
> All parameters following `*` can be transferred only in keyword mode.

## Parameter description

| Parameter| Type| Optional/Required| Description| Data Type| Data Format| Dimension| Non-contiguous Tensor|
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| query | Tensor | Required| Query in the formula| float16/bfloat16/float8_e4m3fn/uint8 | ND | <ul><li>TND: [Tq, Nq, D]</li><li>BNSD: [B, Nq, Sq, D]</li><li>BSND: [B, Sq, Nq, D]</li></ul> (D is the logical headDim for MXFP4 quantization. The actual storage is uint8, and the last dimension of the shape is D//2.)| × |
| key | Tensor | Required| Key in the formula| float16/bfloat16/float8_e4m3fn/uint8 | ND | <ul><li>TND: [Tkv, Nkv, D]</li><li>BNSD: [B, Nkv, Skv, D]</li><li>BSND: [B, Skv, Nkv, D]</li></ul> (D is the logical headDim for MXFP4 quantization. The actual storage is uint8, and the last dimension of the shape is D//2.)| × |
| value | Tensor | Required| Value in the formula| float16/bfloat16/float8_e4m3fn/uint8 | ND | <ul><li>TND: [Tkv, Nkv, D]</li><li>BNSD: [B, Nkv, Skv, D]</li><li>BSND: [B, Skv, Nkv, D]</li></ul> (D is the logical headDim for MXFP4 quantization. The actual storage is uint8, and the last dimension of the shape is D//2.)| × |
| block_sparse_mask | Tensor | Required| Block-level sparsity mask. The value 1 indicates that the block is reserved for calculation, and the value 0 indicates that the block is skipped.| int8 | ND | [B, N, ceil(Sq/block_x), ceil(Skv/block_y)] | × |
| block_shape | list[int] | Required| Sparse block shape, which must contain two elements [block_x, block_y].| int64 | - | The length is 2.| - |
| q_input_layout | string | Optional| Data layout format of the input query tensor. The default value is TND.| string | - | - | - |
| kv_input_layout | string | Optional| Data layout format of the input key-value tensor. The default value is TND.| string | - | - | - |
| num_key_value_heads | int | Optional| Number of key/value heads. The default value is 1.| int64 | - | - | - |
| scale_value | float | Optional| scale in the formula. The default value is 1.0. Generally, the value is D^-0.5.| double | - | - | - |
| inner_precise | int | Optional| Precision level used for Softmax calculation. The value can be 0, 1, or 4. The default value is 1.| int64 | - | - | - |
| actual_seq_lengths | list[int] | Mandatory for TND/Optional for others| Actual length of the query sequence corresponding to each batch| int64 | - | The length is B.| - |
| actual_seq_lengths_kv | list[int] | Mandatory for TND/Optional for others| Actual sequence length of the key/value corresponding to each batch| int64 | - | The length is B.| - |
| return_softmax_lse | bool | Optional| Whether to enable the softmaxLse output. The default value is False.| BOOL | - | - | - |
| mask_type | int | Optional| Mask type in the attention calculation. The default value is 0. Currently, only 0 is supported.| int64 | - | - | - |
| quant_mode | int | Optional| Quantization mode. The default value is 0. The options are as follows: 0: non-quantization; 1: FP8 quantization (float8_e4m3fn for Q/K/V); 2: MXFP4 OCP quantization (scale down). 3: MXFP4 CX quantization. The quantization scale is user-defined, and the quantization scale is truncated upwards. For details about the MXFP4 quantization formula, see "Constraints".| int64 | - | - | - |
| block_size | int | Optional| Block size of PagedAttention. The default value is 0. Currently, PagedAttention is not supported. Only 0 is supported.| int64 | - | - | - |
| pre_tokens | int | Optional| Number of tokens to be included in the sliding window attention. The default value is 2147483647. Currently, only 2147483647 is supported.| int64 | - | - | - |
| next_tokens | int | Optional| Number of tokens to be included in the sliding window attention. The default value is 2147483647. Currently, only 2147483647 is supported.| int64 | - | - | - |
| dst_type_max | float | Optional| User-defined quantization range input during MXFP4 CX quantization. The default value is 0.0.| double | - | - | - |
| atten_mask | Tensor | Optional| atten_mask in the formula is not supported currently and must be None.| int8 | ND | 2 | × |
| block_table | Tensor | Optional| Block table, which is used for PagedAttention and is not supported currently. The value must be None.| int32 | ND | 2 | × |
| q_dequant_scale | Tensor | Optional| Dequantization scaling factor of a query| float32/float8_e8m0fnu | ND | See "Constraints".| × |
| k_dequant_scale | Tensor | Optional| Dequantization scaling factor of the key| float32/float8_e8m0fnu | ND | See "Constraints".| × |
| v_dequant_scale | Tensor | Optional| Dequantization scaling factor of value| float32/float8_e8m0fnu | ND | See "Constraints".| × |
| p_quant_scale | Tensor | Optional| Quantization scaling factor of the softmax result p. This parameter is reserved and is not supported currently. The value must be None.| float32 | ND | - | × |
| attention_out_dtype | ScalarType | Optional| Data type of attentionOut. The default value is None. When quant_mode is 0, the value is the same as that of query by default. When quant_mode is not 0, this parameter is mandatory. Only float16 and bfloat16 are supported.| - | - | - | - |

## Return Values

| Parameter| Type| Mandatory (Yes/No)| Description| Data Type| Data Format| Dimension| Non-contiguous Tensor|
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | Required| attentionOut in the formula| float16/bfloat16 | ND | Same as the query. (During MXFP4 quantization, the query is masqueraded as uint8 and the storage dimension is D/2. attentionOut is the floating-point output after dequantization. The last dimension is the logical headDim, that is, 2 x (D/2)=D.)| × |
| softmax_lse | Tensor | Optional| Intermediate result of log-sum-exp computed by Softmax. The result is output when `return_softmax_lse` is set to True. Otherwise, an empty tensor {0} is output.| float32 | ND | <ul><li>TND: [Tq, Nq, 1]</li><li>BNSD: [B, Nq, Sq, 1]</li><li>BSND: [B, Sq, Nq, 1]</li></ul> | × |

## Constraints

- The values of q_input_layout and kv_input_layout must be the same. Only TND, BNSD, and BSND are supported.
- The data types of query, key, and value must be the same. Both float16 and bfloat16 are supported. If FP8 (`quant_mode=1`) is used, the data types of key, value, and value must be float8_e4m3fn. If MXFP4 (`quant_mode=2/3`) is used, the data types of key, value, and value must be uint8. (float4_e2m1fn_x2 masquerading. Each byte is packed with two fp4 values.)
- Currently, the D axis of the query, key, and value can only be 64 or 128.
- The number of query headers N1 (determined by the query shape and not explicitly transferred by the interface) and num_key_value_heads (N2) must meet the N1 > = N2 && N1% N2 = = 0 requirement.
- block_shape must contain two elements [block_x, block_y]. The value must be greater than 0. The restrictions on the multiples of each product are as follows:
  <!-- npu="950" id7 -->
  - For <term>Ascend 950PR&950DT products</term>, block_y must be a multiple of 16. For MXFP4 quantization (`quant_mode=2/3`), both block_x and block_y must be a multiple of 64.
  <!-- end id7 -->
  <!-- npu="910b,A3" id8 -->
  - In <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The value of block_y must be a multiple of 128.
  <!-- end id8 -->
- block_sparse_mask must be passed, and the shape must be [batch, headNum, ceilDiv(maxQS, block_x), ceilDiv(maxKVS, block_y)].
- actual_seq_lengths is mandatory when q_input_layout is set to TND. actual_seq_lengths_kv is mandatory when kv_input_layout is set to TND. The two parameters must be configured at the same time or not configured at the same time. If only one of them is configured, the request will be intercepted.
- qSeqlen and kvSeqlen do not need to be exactly divided by block_shape. Non-alignment scenarios are supported. The actual number of blocks is rounded up.
- inner_precise can only be set to 0, 1, or 4.
  - 0: The online softmax and rescale use the fp32 data type.
  - 1: Only query is supported. The fp16 data type is used when both the key and value are fp16. The performance is better but the precision is lower.
  - 4: mixed precision operation. fp16/bf16 is used for online softmax, and fp32 is used for rescale.
  <!-- npu="950,A3,910b" id9 -->
  - <term>Ascend 950PR&950DT products</term>: The value can only be 4; <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The value can only be set to 0 or 1.
  <!-- end id9 -->
- Currently, mask_type can only be set to 0, indicating that no mask is added.
- Currently, block_size can only be set to 0, indicating that paged cache is not supported.
- Currently, pre_tokens and next_tokens support only 2147483647, indicating that all tokens before and after the current token participate in the attention operation. That is, the sliding window attention is not supported.
- atten_mask, block_table, and p_quant_scale are reserved parameters and must be set to None.
- When quant_mode is 0 (non-quantization), q_dequant_scale, k_dequant_scale and v_dequant_scale must be None, and attention_out_dtype must be the same as that of the query by default.
- If quant_mode is not 0 (quantization), attention_out_dtype is required and can only be float16 or bfloat16.

### Restrictions when quant_mode is 1 (FP8 quantization)

- Only <term>Ascend 950PR&950DT products</term> is supported.
- The query, key, and value must be float8_e4m3fn at the same time, and the three dequantization scaling factors q_dequant_scale, k_dequant_scale and v_dequant_scale must be provided.
- The data type of the dequantization scaling factor must be float32.
  - q_dequant_scale: dequantizes the query during QK matrix multiplication. The shape is [Batch, HeadNum, ceilDiv(maxQSeqLength, 128), 1].
  - k_dequant_scale: The shape is [Batch, KVHeadNum, ceilDiv(maxKVSeqLength, 256), 1] or [Batch, KVHeadNum, ceilDiv(maxKVSeqLength, 512), 1]. The key is dequantized during QK matrix multiplication.
  - v_dequant_scale: The shape is the same as that of k_dequant_scale. The value is dequantized during PV matrix multiplication.
- block_shape must be passed, and the sizes of the quantization blocks of q and kv must be the same as those of the two elements of block_shape.

### Restrictions on quant_mode=2/3 (MXFP4 quantization)

- Only <term>Ascend 950PR&950DT products</term> is supported.
- query: The key and value are float4_e2m1fn_x2 (passed in uint8 masquerading mode, the D axis is the number of uint8 elements, and the logical headDim is 2*D). The descale tensor of the float8_e8m0fnu type needs to be provided.
  - q_dequant_scale: MX quantization mode. Only Rowwise E8M0 Micro Scaling is supported. The shape is as follows:
    - BNSD: [B, Nq, Sq, ceilDiv(D, 64), 2];
    - BSND: [B, Sq, Nq, ceilDiv(D, 64), 2];
    - TND: [Tq, Nq, ceilDiv(D, 64), 2].
  - k_dequant_scale: MX quantization mode. Only Rowwise E8M0 Micro Scaling is supported. The shape is as follows:
    - BNSD: [B, Nkv, Skv, ceilDiv(D, 64), 2];
    - BSND: [B, Skv, Nkv, ceilDiv(D, 64), 2];
    - TND: [Tkv, Nkv, ceilDiv(D, 64), 2].
  - v_dequant_scale: MX quantization mode. Only columnwise E8M0 micro scaling is supported. The shape is as follows:
    - BNSD: [B, Nkv, ceilDiv(Skv, 64), D, 2];
    - BSND: [B, ceilDiv(Skv, 64), Nkv, D, 2];
    - TND: [BTkv//64, Nkv, D, 2], where $BTkv = \sum_{b = 0}^{B-1}\lceil\frac{S2_b}{64}\rceil$.
- attention_out_dtype must be of type float16 or bfloat16, and the last dimension of attentionOut is 2 x D (corresponding to the logic headDim, two fp4 values are packed into one uint8 byte).
- The values of block_x and block_y must be multiples of 64.
- dst_type_max can be set to 0.0 or [6.0, 12.0]. 0.0 indicates that Amax (DType) is the maximum value of the quantization result data type. 6.0-12.0 indicates that Amax (DType) is the input value.
- **Quantization formula**: Take the quantization of P (softmax intermediate result) as an example. $P_{\text{max}}$ is the maximum absolute value of elements in a group (32 elements).
  - OCP quantization formula (`quant_mode=2`, truncating down the quantization scale):

    $$
    P_{\text{Scale}} = 2^{\lfloor \log_2 P_{\text{max}} \rfloor - 2}
    $$

  - CX quantization formula (`quant_mode=3`, truncation of the quantization scale upward):

    $$
    P_{\text{Scale}} = 2^{\lceil \log_2 \left( P_{\text{max}} / dst\_type\_max \right) \rceil} = 2^{\lceil \log_2 P_{\text{max}} - \log_2 dst\_type\_max \rceil}
    $$

- Advanced features such as atten_mask, PagedAttention, and softmaxLse are not supported.

## Examples

- Basic example (TND layout, float16 non-quantization)

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    B, Nq, Nkv, Sq, Skv, D = 1, 32, 32, 128, 128, 128
    block_x, block_y = 128, 128

    # TND layout. Tq/Tkv is the sum of batch sequence lengths.
    Tq, Tkv = B * Sq, B * Skv
    query = torch.randn(Tq, Nq, D, dtype=torch.float16, device="npu")
    key = torch.randn(Tkv, Nkv, D, dtype=torch.float16, device="npu")
    value = torch.randn(Tkv, Nkv, D, dtype=torch.float16, device="npu")

    # block_sparse_mask: [B, N, ceil(Sq/block_x), ceil(Skv/block_y)]
    mask = torch.ones(B, Nq, Sq // block_x, Skv // block_y, dtype=torch.int8, device="npu")

    # block_shape / actual_seq_lengths is an int list.
    block_shape = [block_x, block_y]
    actual_seq_lengths = [Sq] * B
    actual_seq_lengths_kv = [Skv] * B

    attention_out, softmax_lse = cann_ops_transformer.block_sparse_attention(
        query, key, value, mask, block_shape,
        q_input_layout="TND",
        kv_input_layout="TND",
        num_key_value_heads=Nkv,
        scale_value=1.0 / (D ** 0.5),
        inner_precise=4, # 950PR/950DT can only be 4. If the value is not transferred (the default value is 1), the system crashes.
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        return_softmax_lse=True,
    )
    torch_npu.npu.synchronize()
    assert attention_out.shape == (Tq, Nq, D)
    assert softmax_lse.shape == (Tq, Nq, 1)
    ```

- MXFP4 quantization example (TND layout, quant_mode=2)

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    # D is the logical headDim. float4_e2m1fn_x2 is passed as uint8, and the storage dimension is D//2.
    B, Nq, Nkv, Sq, Skv, D = 1, 32, 32, 128, 128, 128
    block_x, block_y = 128, 128

    Tq, Tkv = B * Sq, B * Skv
    query = torch.randint(0, 256, (Tq, Nq, D // 2), dtype=torch.uint8, device="npu")
    key = torch.randint(0, 256, (Tkv, Nkv, D // 2), dtype=torch.uint8, device="npu")
    value = torch.randint(0, 256, (Tkv, Nkv, D // 2), dtype=torch.uint8, device="npu")

    mask = torch.ones(B, Nq, Sq // block_x, Skv // block_y, dtype=torch.int8, device="npu")
    block_shape = [block_x, block_y]
    actual_seq_lengths = [Sq] * B
    actual_seq_lengths_kv = [Skv] * B

    # The descale is float8_e8m0fnu, which is disguised as uint8. The number of dimensions is calculated based on the logical headDim D.
    #   Q/K scale: [T, N, D//64, 2]; V scale: [BTkv, N, D, 2], BTkv=B*ceilDiv(Skv,64)
    q_dequant_scale = torch.randint(0, 256, (Tq, Nq, D // 64, 2), dtype=torch.uint8, device="npu")
    k_dequant_scale = torch.randint(0, 256, (Tkv, Nkv, D // 64, 2), dtype=torch.uint8, device="npu")
    v_dequant_scale = torch.randint(0, 256, (Tkv // 64, Nkv, D, 2), dtype=torch.uint8, device="npu")

    attention_out, _ = cann_ops_transformer.block_sparse_attention(
        query, key, value, mask, block_shape,
        q_input_layout="TND",
        kv_input_layout="TND",
        num_key_value_heads=Nkv,
        scale_value=1.0 / (D ** 0.5),
        inner_precise=4,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        quant_mode=2,
        q_dequant_scale=q_dequant_scale,
        k_dequant_scale=k_dequant_scale,
        v_dequant_scale=v_dequant_scale,
        attention_out_dtype=torch.float16,
    )
    torch_npu.npu.synchronize()
    # The output of MXFP4 quantization is logical headDim D (D//2 for Q/K/V storage and D for dequantization).
    assert attention_out.shape == (Tq, Nq, D)
    assert attention_out.dtype == torch.float16
    ```
