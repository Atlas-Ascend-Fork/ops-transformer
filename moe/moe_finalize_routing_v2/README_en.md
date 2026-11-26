# MoeFinalizeRoutingV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                      |     √    |
| <term>Atlas A3 products</term>      |    √    |
| <term>Atlas A2 products</term>      |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    ×     |
| <term>Kirin X90 processor products</term> | √ |
| <term>Kirin 9030 processor products</term> | √ |

## Function Description

- **Operator function**: In MoE (Mixture of Experts) computation, the output of the MoE (Mixture of Experts) FFN (Feed-Forward Network) is finally processed and merged.
- **Calculation Formula**:

  For FLOAT16, BFLOAT16, and FLOAT32 inputs:

  $$
  expertid=expertIdx[i,k]
  $$

  $$
  out(i,j)=x1_{i,j}+x2Optional_{i,j}+\sum_{k=0}^{K}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num_rows},j}+bias_{expertid,j}))
  $$

For `FLOAT4_E2M1` inputs, the following formula dequantizes regular expert inputs. Here r is the physical row, j is the logical column, and E2M1 and E8M0 decode the corresponding formats.

  $$
  \mathrm{dequant}(r,j)=\mathrm{E2M1}(\mathrm{expandedX}[r,j])\cdot
  \mathrm{E8M0}(\mathrm{expandedXScaleOptional}[r,\lfloor j/64\rfloor,\lfloor(j\bmod 64)/32\rfloor])
  $$

For `FLOAT4_E2M1` inputs, ACLNN V5 uses `expandedXScaleOptional` to dequantize `expandedX`. When `scalesOptional` is supplied, each route is multiplied by its routing weight after adding any bias. The operator accumulates in FLOAT32 and returns BFLOAT16.

For `FLOAT8_E4M3FN` or `FLOAT8_E5M2` inputs, the following formula dequantizes regular expert inputs. Here t is the logical route index (t = i*K+k), r = expandedRowIdx[t] is the physical row, j is the logical column, and E4M3FN/E5M2 and E8M0 decode the corresponding formats. Data and dequantization scales share the same physical row.

  $$
  \mathrm{dequant}(t,j)=\mathrm{E4M3orE5M2}(\mathrm{expandedX}[r,j])\cdot
  \mathrm{E8M0}(\mathrm{expandedXScaleOptional}[r,\lfloor j/32\rfloor])
  $$

For `FLOAT8_E4M3FN` or `FLOAT8_E5M2` inputs, ACLNN V5 uses `expandedXScaleOptional` to dequantize `expandedX` in 32-element blocks, with scale factors sharing physical rows with expandedX (same layout as the MXFP8 scale output of `moe_init_routing_v3`). When `scalesOptional` is supplied, each route is multiplied by its routing weight after adding any bias. The operator accumulates in FLOAT32 and returns BFLOAT16.

## Parameter Description

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">Parameter Name</th>
        <th style="white-space: nowrap">Input/Output/Attribute</th>
        <th style="white-space: nowrap">Description</th>
        <th style="white-space: nowrap">Data Type</th>
        <th style="white-space: nowrap">Data Format</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>expandedX</td>
        <td>Input</td>
        <td>`expandedX` in the formula, that is, the FFN (Feed-Forward Network) output of MoE (Mixture of Experts).</td>
        <td>FLOAT16, BFLOAT16, FLOAT32, FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedRowIdx</td>
        <td>Input</td>
        <td>`expandedRowIdx` in the formula.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x1Optional</td>
        <td>Input</td>
        <td>`x1` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32; BFLOAT16 only for quantized inputs (FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2)</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x2Optional</td>
        <td>Input</td>
        <td>`x2Optional` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32; BFLOAT16 only for quantized inputs (FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2)</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>Input</td>
        <td>`bias` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32; BFLOAT16 only for quantized inputs (FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2)</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>scalesOptional</td>
        <td>Input</td>
        <td>Optional routing weights, denoted by `scales` in the formula. When expandedX is a quantized type (FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2), the weights must be FLOAT32.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdxOptional</td>
        <td>Input</td>
        <td>`expertIdx` in the formula.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedXScaleOptional</td>
        <td>Input</td>
        <td>Block dequantization scales for expandedX. Required for FLOAT4_E2M1, FLOAT8_E4M3FN or FLOAT8_E5M2 inputs. For FLOAT4_E2M1, the shape is `[R,ceil(H/64),2]`; for FLOAT8_E4M3FN/FLOAT8_E5M2, the shape is `[R,CeilAlign(CeilDiv(H,32),2)]` (R is the physical row count of expandedX), sharing physical rows with expandedX.</td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dropPadMode</td>
        <td>Attribute</td>
        <td>Indicates whether the drop mode is supported and describes the arrangement of expandedRowIdx.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>k</td>
        <td>Attribute</td>
        <td>Number of top-K experts selected for each token. When expandedX is a non-quantized type and scalesOptional is not empty and k > 1, the system checks that k is consistent with the second dimension of the scalesOptional shape. When scalesOptional is empty, k is directly used to derive the output shape. The default value is 1. For FLOAT4_E2M1, FLOAT8_E4M3FN or FLOAT8_E5M2 inputs, k is a required attribute: it determines NUM_ROWS from the index length, and K is not derived from the scale shape.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>`out` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32; BFLOAT16 only for quantized inputs (FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2)</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

- <term>Kirin X90 processor products</term>, <term>Kirin 9030 processor products</term>: BFLOAT16 is not supported.

## Constraints

- Among ACLNN versions, only V5 supports `FLOAT4_E2M1`, `FLOAT8_E4M3FN`, and `FLOAT8_E5M2` inputs.
- For quantized inputs (`FLOAT4_E2M1`, `FLOAT8_E4M3FN`, `FLOAT8_E5M2`): expandedRowIdx has shape `[NUM_ROWS*K]`; k is a required attribute and must be positive, and the index length must be divisible by k. NUM_ROWS is the index length divided by k; k is independent of the scale shape. x1/x2/bias must be BFLOAT16; x2 requires x1, and bias requires expertIdx. Invalid routes are skipped entirely, including bias. Output requantization is unsupported.
- Quantized inputs do not support zero, copy, or constant experts. Pass no x, alpha1, alpha2, or v tensors and leave all three special expert ranges empty.
- For `FLOAT4_E2M1` inputs, H is the logical width and must be positive and even. R is the physical row count of expandedX (dim0 for 2D, E*C for 3D) and can be smaller than NUM_ROWS*K after dropping. expandedX has shape `[R,H]` for dropPadMode 0/2 or `[E,C,H]` for dropPadMode 1/3. expandedXScaleOptional has shape `[R,ceil(H/64),2]` and uses the same physical rows as expandedX. Each scale dequantizes 32 elements.
- For `FLOAT8_E4M3FN` or `FLOAT8_E5M2` inputs, H must be a positive multiple of 32. expandedX has shape `[R,H]` for dropPadMode 0/2 (R is the physical row count of expandedX and can be smaller than NUM_ROWS*K) or `[E,C,H]` for dropPadMode 1/3 (R=E*C). expandedXScaleOptional has shape `[R,CeilAlign(CeilDiv(H,32),2)]`, sharing physical rows with expandedX (same layout as the MXFP8 scale output of `moe_init_routing_v3`); each scale dequantizes 32 consecutive elements of one row. Routes pointing beyond R are skipped entirely. expandedX and expandedXScaleOptional must be contiguous ND tensors.
- In the PyTorch API, `expanded_x` uses `torch.float4_e2m1fn_x2` or `torch.uint8` carrying E2M1 data, with a packed last dimension of H/2; `FLOAT8_E4M3FN`/`FLOAT8_E5M2` inputs use `torch.float8_e4m3fn` or `torch.float8_e5m2`. `expanded_x_scale` uses `torch.float8_e8m0fnu` or `torch.uint8` carrying E8M0 data. Execution and Meta/FakeTensor inference return BFLOAT16.
- See the [ACLNN V5 documentation](docs/aclnnMoeFinalizeRoutingV5.md) for detailed type, layout, and API constraints.

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_finalize_routing_v2.cpp](examples/test_aclnn_moe_finalize_routing_v2.cpp) | Calls the MoeFinalizeRoutingV2 operator through the [aclnnMoeFinalizeRoutingV2](docs/aclnnMoeFinalizeRoutingV2_en.md) API. |
| aclnn V4 API | - | Calls the operator through [aclnnMoeFinalizeRoutingV4](docs/aclnnMoeFinalizeRoutingV4_en.md) API. |
| aclnn V5 API | - | Calls the operator through the [aclnnMoeFinalizeRoutingV5](docs/aclnnMoeFinalizeRoutingV5.md) API. |
| PyTorch API | - | Calls the moe_finalize_routing operator through the [moe_finalize_routing](docs/torchapi_moe_finalize_routing_en.md) API. |
