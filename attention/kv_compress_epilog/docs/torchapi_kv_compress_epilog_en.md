# kv_compress_epilog

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:56:15.099Z -->

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

    Updates the cache in place through quantization compression in the Epilog phase of the KV Cache. Encapsulates aclnnKvCompressEpilog. Quantizes and compresses bfloat16 activation values x, and scatters them into cache based on slot_mapping. Tokens with a value of -1 in slot_mapping are skipped. The last 64 columns of the trailing axis d of x form the rope segment, and the first d-64 columns form the nope segment. Supports three quantization modes:
  - `"fp8_bf16"`/`"fp8_e8m0"` (internal 0/1): per-group dynamic quantization and compression to FP8 (packed as uint8), with the rope segment remaining bfloat16; the two modes differ only in the representation of each group's scale (bf16 / e8m0);
  - `"hifloat8_fp4"` (internal 2): the rope segment uses hifloat8 **static** quantization (multiplied by x_scale and then cast to hifloat8); the nope segment uses per-group **float4_E2M1 dynamic** quantization (scale = in-group amax/6, written out in bfloat16).

- Semantics:

    The underlying aclnn API implements cacheRef in place. This Python API has the same in-place semantics: it updates the input cache directly, and rows not hit by slot_mapping retain their original values. In the schema, cache is marked as `Tensor(a!)`. **This API has no return value**. After the call, use the input cache directly.

## Function Prototype

```python
cann_ops_transformer.kv_compress_epilog(cache, x, slot_mapping, *, quant_group_size=64, quant_mode="fp8_e8m0", round_scale=True, x_scale=1.0) -> None
```

## Parameters

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>cache</td>
        <td>Tensor</td>
        <td>required</td>
        <td>Paged KV Cache of the current layer. <b>Only four-dimensional shapes are supported</b> (num_slots = blockNum × blockSize). <b>headDim must be ≥ kvCacheCol</b> (see Constraints for the calculation of kvCacheCol). <b>Non-contiguous is supported in the blockNum dimension</b> (paged). The data format is $ND$.</td>
        <td>uint8</td>
        <td>(blockNum, blockSize, 1, headDim)</td>
    </tr>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>required</td>
        <td>Activation input to be quantized. d must satisfy d % 64 == 0 and 64 &lt; d ≤ 8192. The data format is $ND$.</td>
        <td>bfloat16</td>
        <td>(bs, d)</td>
    </tr>
    <tr>
        <td>slot_mapping</td>
        <td>Tensor</td>
        <td>required</td>
        <td>Index mapping from token to cache slot. The value range of elements is [-1, num_slots - 1] (num_slots = blockNum × blockSize). -1 indicates that the token is skipped. The number of dimensions must equal that of x minus 1. Data format: $ND$.</td>
        <td>int32, int64</td>
        <td>(bs)</td>
    </tr>
    <tr>
        <td>quant_group_size</td>
        <td>int</td>
        <td>optional</td>
        <td>Quantization group size. Default value: 64. When quant_mode=0/1, the kernel uses a fixed value of 64. When quant_mode=2, only 16/32/64 is supported, and (d-64) % quant_group_size == 0 must be satisfied.</td>
        <td>INT</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quant_mode</td>
        <td>str</td>
        <td>Optional</td>
        <td>Quantization mode (string, case-insensitive; TorchNPU internally maps it to the operator-side int via enumeration). The default value is <code>"fp8_e8m0"</code>. Among the three modes, <code>"fp8_bf16"</code> and <code>"fp8_e8m0"</code> use the same per-group FP8 quantization method (nope/rope as a whole is per-group dynamically quantized to FP8(e4m3) and packed into uint8), and differ only in the scale representation. Optional values and meanings:
            <ul>
                <li><code>"fp8_bf16"</code> (internal 0): per-group FP8 quantization. The group size is determined by quant_group_size (fixed to 64 in the kernel). Each group has one <b>bf16</b> scale.</li>
                <li><code>"fp8_e8m0"</code> (internal 1): per-group FP8 quantization. Each group has one <b>e8m0</b> (power-of-2 exponent) scale, that is, MX-FP8 microscaling.</li>
                <li><code>"hifloat8_fp4"</code> (internal 2): HiFloat8 <b>static</b> quantization for the rope segment (last 64 columns of the tail axis) (cast to HiFloat8 after multiplying by x_scale); per-group <b>float4_E2M1 dynamic</b> quantization for the nope segment (first d-64 columns) (scale = group amax/6, written out in bf16).</li>
            </ul>
            quant_mode=0/1/2 used in the following Constraints section refers to the corresponding internal int values listed above.</td>
        <td>STR</td>
        <td>-</td>
    </tr>
    <tr>
        <td>round_scale</td>
        <td>bool</td>
        <td>optional</td>
        <td>Whether to round up the scale of each group to a power of 2 in group mode. Default value: True. Not effective when quant_mode=2.</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_scale</td>
        <td>float</td>
        <td>optional</td>
        <td>Scaling factor of static hifloat8 quantization in the rope stage when quant_mode=2. Reserved and unused when quant_mode=0/1. Default value: 1.0.</td>
        <td>float</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## Returns

This API has no return value. The computation result is written back to the input tensor `cache` in place. After computation, the shape and dtype of `cache` remain unchanged; rows not hit by `slot_mapping` retain their original values.

## Constraints

- This API supports use in inference scenarios.
- This API supports calls in both single-operator mode and TorchAir graph mode (the GE converter registered via graph_convert is lowered to the KvCompressEpilog operator, and the results in eager mode and graph mode are consistent).
- cache supports only the 4D shape `[blockNum, blockSize, 1, headDim]`. The second-to-last dimension is fixed to 1 (one compressed vector per token). Other dimension counts are not supported.
- cache supports non-contiguous layouts only in the blockNum dimension: blocks do not need to be tightly packed, but data within a block (the blockSize and headDim dimensions) must be contiguous.
- headDim constraint: headDim, the last dimension of cache, must be ≥ the number of bytes written per row kvCacheCol = align(concatCol). concatCol is calculated by quantization mode:
  - quant_mode=0/1: concatCol = (d−64) + 128 + ⌈(d−64)/64⌉ × scaleBytes, where scaleBytes is 2 for mode 0 and 1 for mode 1;
  - quant_mode=2: concatCol = 64 + (d−64)/2 + (d−64)/quant_group_size × 2;
  - Alignment rule: for quant_mode=1, no padding is applied and kvCacheCol = concatCol; for quant_mode=0/2, data is aligned to 32B and kvCacheCol = ⌈concatCol / 32⌉ × 32.
  - Example: d=512, quant_mode=1 → kvCacheCol=583 → headDim ≥ 583 (640 is used in this example).
- The dimension of slot_mapping should be equal to the dimension of x minus 1. That is, slot_mapping is obtained by flattening all dimensions of x except the last one.
- The last dimension of x (the d axis) must satisfy d % 64 == 0 and 64 < d ≤ 8192. Per-group quantization is performed with every 64 elements as a group.
- When quant_mode=2, quant_group_size supports only 16/32/64, and the length of the nope segment (d-64) must be divisible by quant_group_size.
- Tokens with a value of -1 in slot_mapping are skipped and not processed. Other valid elements take values in the range [0, num_slots - 1], and the element values must be unique. The correctness of results is not guaranteed when values are duplicated.

## Deterministic Computation

Defaults to a deterministic implementation.

## Calling Example

- Single-operator call

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import kv_compress_epilog

    # cache is 4D [blockNum, blockSize, 1, headDim], num_slots = blockNum*blockSize
    # headDim must be >= kvCacheCol (kvCacheCol=323 when d=256/quant_mode=1, head_dim=384 meets this requirement)
    block_num, block_size, head_dim = 128, 16, 384
    bs, d = 1024, 256

    cache = torch.zeros(block_num, block_size, 1, head_dim, dtype=torch.uint8).npu()
    x = torch.randn(bs, d, dtype=torch.bfloat16).npu()
    slot_mapping = torch.arange(bs, dtype=torch.int32).npu()

    kv_compress_epilog(
        cache, x, slot_mapping, quant_group_size=64, quant_mode="fp8_e8m0", round_scale=True, x_scale=1.0)
    print(cache.shape, cache.dtype)
    ```

- Graph mode (torchair) call

    Reuse cache/x/slot_mapping constructed in the single-operator example above. Compile with the torchair backend and then call:

    ```python
    import torchair

    class KvCompressEpilogModel(torch.nn.Module):
        def forward(self, cache, x, slot_mapping):
            kv_compress_epilog(
                cache, x, slot_mapping,
                quant_group_size=64, quant_mode="fp8_e8m0", round_scale=True, x_scale=1.0)
            return cache

    model = KvCompressEpilogModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    model(cache, x, slot_mapping)
    print(cache.shape, cache.dtype)
    ```
