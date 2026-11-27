# apply_rotary_pos_emb_grad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:42:38.168Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- **Description**:

    Performs backward computation of the dual-path rotary position embedding [apply_rotary_pos_emb](../../apply_rotary_pos_emb/docs/torchapi_apply_rotary_pos_emb_en.md), fusing the gradient computation of the query and key paths into a single kernel call.

- **Formula**:

    Let `dims` denote the list of axes along which `cos` and `sin` are broadcast in forward computation. When `rotary_mode` is `"half"`:

    $$
    grad\_q_1, grad\_q_2 = chunk(grad\_query\_embed, chunks=2, dim=-1)
    $$

    $$
    grad\_k_1, grad\_k_2 = chunk(grad\_key\_embed, chunks=2, dim=-1)
    $$

    $$
    cos_1, cos_2 = chunk(cos, chunks=2, dim=-1)
    $$

    $$
    sin_1, sin_2 = chunk(sin, chunks=2, dim=-1)
    $$

    $$
    grad\_query = cat(cos_1 \cdot grad\_q_1 + sin_2 \cdot grad\_q_2,\ cos_2 \cdot grad\_q_2 - sin_1 \cdot grad\_q_1,\ dim=-1)
    $$

    $$
    grad\_key = cat(cos_1 \cdot grad\_k_1 + sin_2 \cdot grad\_k_2,\ cos_2 \cdot grad\_k_2 - sin_1 \cdot grad\_k_1,\ dim=-1)
    $$

    When both `query` and `key` are passed in, let:

    $$
    query\_rotate = cat((-query_2, query_1), dim=-1)
    $$

    $$
    key\_rotate = cat((-key_2, key_1), dim=-1)
    $$

    Then the gradients of `cos` and `sin` are:

    $$
    grad\_cos = sum(grad\_query\_embed \cdot query + grad\_key\_embed \cdot key, dims)
    $$

    $$
    grad\_sin = sum(grad\_query\_embed \cdot query\_rotate + grad\_key\_embed \cdot key\_rotate, dims)
    $$

- **NOTE**

  - When `layout=4`, the input is a 3D tensor; for other supported `layout` values, the input is a 4D tensor, where B (Batch) indicates the batch size, S (Seq-Length) indicates the sequence length, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the hidden dimension size of each head, and T indicates the fusion of B and S into one axis.
  - `query` and `key` must be passed in together or omitted together. If neither is passed, only `grad_query` and `grad_key` are computed, and the returned `grad_cos` and `grad_sin` are `None`.

## Function Prototype

```python
cann_ops_transformer.apply_rotary_pos_emb_grad(
    grad_query_embed,
    grad_key_embed,
    cos,
    sin,
    *,
    query=None,
    key=None,
    rotary_mode="half",
    layout=1,
) -> (Tensor, Tensor, Optional[Tensor], Optional[Tensor])
```

## Mandatory

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| grad_query_embed | Tensor | Required | Gradient of the forward output `query_out`. | bfloat16, float16, float32 | (T, Nq, D) when `layout=4`, where Nq is the number of heads (Head-Num) of `grad_query_embed` and can differ from Nk of `grad_key_embed`; 4D tensor for other supported `layout` values |
| grad_key_embed | Tensor | Required | Gradient of the forward output `key_out`. The shape must be the same as that of `grad_query_embed`, except for the N dimension. | Same as grad_query_embed | (T, Nk, D) when `layout=4`; 4D tensor for other supported `layout` values |
| cos | Tensor | Required | Cosine tensor of the forward computation input. The N dimension must be 1. | Same as grad_query_embed | 3D or 4D tensor depending on the input layout |
| sin | Tensor | Required | Sine tensor of the forward computation input. The shape must be the same as that of `cos`. | Same as grad_query_embed | Same as cos |
| query | Tensor | Optional | Forward computation input `query`. If passed in, `grad_cos` and `grad_sin` are computed; must be passed in together with `key` or omitted together. Default value is `None`. | Same as grad_query_embed | Same as grad_query_embed |
| key | Tensor | Optional | Forward computation input `key`. If passed in, `grad_cos` and `grad_sin` are computed; must be passed in together with `query` or omitted together. Default value is `None`. | Same as grad_query_embed | Same as grad_key_embed |
| rotary_mode | str | Optional | Rotary encoding mode. Only supports `"half"`. Default value is `"half"`. | - | - |
| layout | int | Optional | Layout format of the input tensor. 1 indicates BSND, 2 indicates SBND, and 4 indicates TND. Default value is 1. | - | - |

## Return Values

- **grad_query** (`Tensor`): Gradient of the forward input `query`. The shape and data type are the same as those of `grad_query_embed`.
- **grad_key** (`Tensor`): Gradient of the forward input `key`. The shape and data type are the same as those of `grad_key_embed`.
- **grad_cos** (`Optional[Tensor]`): Gradient of the forward input `cos`. If both `query` and `key` are passed in, the shape and data type are the same as those of `cos`; otherwise, the value is `None`.
- **grad_sin** (`Optional[Tensor]`): Gradient of the forward input `sin`. If both `query` and `key` are passed in, the shape and data type are the same as those of `sin`; otherwise, the value is `None`.

## Constraints

- This API can be used in the training use case.
- This API can be called in single-operator mode.
- Empty tensors are not supported. Each dimension of each input tensor must be greater than 0.
- The input tensors `grad_query_embed`, `grad_key_embed`, `cos`, and `sin`, as well as the non-empty `query` and `key`, must have the same data type.
- `rotary_mode` only supports `"half"`.
- `layout` only supports 1, 2, and 4, corresponding to BSND, SBND, and TND respectively.
- The D dimensions of the input tensors must be the same. D must be less than or equal to 1024 and divisible by 2.
- Except for the N dimension, the dimensions of `grad_query_embed` and `grad_key_embed` must be the same.
- `cos` and `sin` must have the same shape, and the N dimension must be 1.
- If `layout=1` (BSND), the shapes of `cos` and `sin` are (cosB, S, 1, D), where cosB can be 1 or equal to the B of `grad_query_embed`.
- If `layout=2` (SBND), the shapes of `cos` and `sin` are (S, cosB, 1, D), where cosB can be 1 or equal to the B of `grad_query_embed`.
- If `layout=4` (TND), the shapes of `cos` and `sin` are (T, 1, D), and T must be the same as the T of `grad_query_embed`.
- The shape of `query` must be the same as that of `grad_query_embed`, and the shape of `key` must be the same as that of `grad_key_embed`. `query` and `key` must be passed in together or omitted together.

## Deterministic Computation

Defaults to a deterministic implementation.

## Related APIs

This operator is the backward operator of [apply_rotary_pos_emb](../../apply_rotary_pos_emb/docs/torchapi_apply_rotary_pos_emb_en.md). When the forward API uses `rotary_mode="half"`, calling `.backward()` on the loss automatically triggers this operator. Call this API manually only when explicit gradient control is required.

## Example

- Call in single-operator mode

    ```python
    # Scenario: single-operator mode calling the RoPE backward operator apply_rotary_pos_emb_grad, layout=1 (BSND), float16 precision
    # Inputs: grad_query_embed, grad_key_embed, cos, and sin are required tensors; query and key are optional tensors. The example uses B=1, S=64, N=8, D=128.
    # Outputs: grad_query and grad_key, plus grad_cos and grad_sin (returned when query and key are passed in).
    # Expected result: the shapes of the four output tensors are printed, which are (1, 64, 8, 128), (1, 64, 8, 128), (1, 64, 1, 128), and (1, 64, 1, 128), respectively.
    import torch
    import torch_npu
    from cann_ops_transformer.ops import apply_rotary_pos_emb_grad

    torch_npu.npu.set_device(0)

    B = 1
    S = 64
    N = 8
    D = 128

    grad_query_embed = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    grad_key_embed = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    cos = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16)
    sin = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16)
    query = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    key = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)

    # Key step: call the apply_rotary_pos_emb_grad backward API to compute gradients.
    grad_query, grad_key, grad_cos, grad_sin = apply_rotary_pos_emb_grad(
        grad_query_embed,
        grad_key_embed,
        cos,
        sin,
        query=query,
        key=key,
        rotary_mode="half",
        layout=1,  # BSND
    )

    # Expected result: grad_query and grad_key have shape (1, 64, 8, 128); grad_cos and grad_sin have shape (1, 64, 1, 128).
    print(f"grad_query shape: {grad_query.shape}")
    print(f"grad_key shape: {grad_key.shape}")
    print(f"grad_cos shape: {grad_cos.shape}")
    print(f"grad_sin shape: {grad_sin.shape}")
    ```
