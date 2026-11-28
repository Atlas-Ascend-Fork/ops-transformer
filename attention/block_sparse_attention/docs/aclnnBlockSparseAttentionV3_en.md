# aclnnBlockSparseAttentionV3

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

## Function

- **Description**: BlockSparseAttentionV3 sparse attention computation. It supports flexible block-level sparsity patterns. blockSparseMaskOptional specifies the KV blocks selected by each Q block, enabling efficient sparse attention computation.

  Compared with BlockSparseAttentionV2, this API adds the pQuantScaleOptional, quantMode, and dstTypeMax parameters.
- **Formula**: Sparse block size: $blockShapeX \times blockShapeY$, blockSparseMaskOptional specifies the sparse pattern

  $$
  attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T + atten\_mask) \cdot value_{sparse}
  $$

  Here, `scale` is the scaling factor controlled by `scaleValue`. `atten_mask` is the attention mask corresponding to `attenMaskOptional`, which currently must be `nullptr`.

  For BlockSparseAttentionV2, the data layout formats of the inputs query, key, and value support interpretation in multiple dimension layouts and can be passed in through qInputLayout and kvInputLayout.

  - B: Indicates the batch size of the input samples (Batch).
  - T: Indicates the length of B and S concatenated along the axis and tightly packed (Total tokens).
  - S: Indicates the sequence length of the input samples (Seq-Length).
  - H: Indicates the size of the hidden layer (Head-Size).
  - N: Number of heads (Head-Num).
  - D: Minimum unit size of the hidden layer, satisfying D = H/N (Head-Dim).

  Currently supported layouts:

  - qInputLayout: "TND" "BNSD" "BSND"
  - kvInputLayout: "TND" "BNSD" "BSND"

- **MX quantization (supported only on Ascend 950PR&950DT products)**

  This API supports MXFP4 (quantMode=2/3), MXFP8 (quantMode=4), and mixed MXFP8 QK / MXFP4 PV (QK8PV4, quantMode=6/7). Supply FLOAT8_E8M0 dequantization scales for these modes.

- <summary><a id="quantization-scale-factor-of-the-mxfp4-quantization-mode"></a><strong>MX quantization scales (quantMode=2/3/4/6/7)</strong></summary>

  - qDequantScaleOptional:

    - MX quantization mode: only Rowwise E8M0 Micro Scaling is supported
    - Data type: FLOAT8_E8M0
    - shape:
      - BNSD: [B, Nq, Sq, CeilDiv(D, 64), 2];
      - BSND: [B, Sq, Nq, CeilDiv(D, 64), 2];
      - TND: [Tq, Nq, CeilDiv(D, 64), 2].
  - kDequantScaleOptional:

    - MX quantization mode, supporting only Rowwise E8M0 Micro Scaling
    - Data type: FLOAT8_E8M0
    - shape:
      - BNSD:[B, Nkv, Skv, CeilDiv(D, 64), 2];
      - BSND:[B, Skv, Nkv, CeilDiv(D, 64), 2];
      - TND: [Tkv, Nkv, CeilDiv(D, 64), 2].
  - vDequantScaleOptional:

    - MX quantization mode: only Columnwise E8M0 Micro Scaling is supported
    - Data type: FLOAT8_E8M0
    - shape

      - BNSD:[B, Nkv, CeilDiv(Skv, 64), D, 2];
      - BSND:[B, CeilDiv(Skv, 64), Nkv, D, 2];
      - TND: [BTkv, Nkv, D, 2].

      $$
      BTkv = \sum_{b=0}^{B-1} \left\lceil \frac{Skv_b}{64} \right\rceil
      $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnBlockSparseAttentionV3GetWorkspaceSize" to obtain the workspace size required for the computation and the executor containing the computation process of the operator, and then call "aclnnBlockSparseAttentionV3" to execute the computation.

```c++
aclnnStatus aclnnBlockSparseAttentionV3GetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *blockSparseMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *blockShapeOptional,
  const aclIntArray *actualSeqLengthsOptional,
  const aclIntArray *actualSeqLengthsKvOptional,
  const aclTensor   *blockTableOptional,
  const aclTensor   *qDequantScaleOptional,
  const aclTensor   *kDequantScaleOptional,
  const aclTensor   *vDequantScaleOptional,
  const aclTensor   *pQuantScaleOptional,
  char              *qInputLayout,
  char              *kvInputLayout,
  int64_t            numKeyValueHeads,
  int64_t            maskType,
  double             scaleValue,
  int64_t            innerPrecise,
  int64_t            blockSize,
  int64_t            preTokens,
  int64_t            nextTokens,
  int64_t            softmaxLseFlag,
  int64_t            quantMode,
  double             dstTypeMax,
  aclTensor         *attentionOut,
  aclTensor         *softmaxLseOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnBlockSparseAttentionV3(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnBlockSparseAttentionV3GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1300px">
  <colgroup>
  <col style="width: 250px">
  <col style="width: 90px">
  <col style="width: 200px">
  <col style="width: 340px">
  <col style="width: 160px">
  <col style="width: 80px">
  <col style="width: 90px">
  <col style="width: 90px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query (aclTensor*)</td>
      <td>Input</td>
      <td>The query in the formula.</td>
      <td>The supported shapes are as follows:
        <ul><li>TND: [totalQTokens, headNum, headDim].</li>
        <li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li>
        <li>BSND: [batch, maxQSeqLength, headNum, headDim].</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT4_E2M1 </td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>key (aclTensor*)</td>
      <td>Input</td>
      <td>The key in the formula.</td>
      <td>Supported shapes are as follows:
        <ul>
          <li>TND: [totalKTokens, numKeyValueHeads, headDim].</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li>
          <li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim].</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT4_E2M1, </td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>value(aclTensor*)</td>
      <td>Input</td>
      <td>The value in the formula.</td>
      <td>        Supported shapes are as follows:
        <ul>
          <li>TND: [totalVTokens, numKeyValueHeads, headDim].</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li>
          <li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim].</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT4_E2M1,</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockSparseMaskOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the actual sparse pattern.</td>
      <td>        Optional Input (required in the current version)
        <ul>
          <li>The shape is [batch, headNum, qBlockNum, kvBlockNum]. qBlockNum and kvBlockNum must cover the Q and KV blocks obtained by dividing the sequence lengths by blockShapeX and blockShapeY, respectively, and rounding up.</li>
          <li>TND uses the maximum of each actual sequence length array. BNSD and BSND use the S dimensions of Q and K. If both actual sequence length arrays are supplied, the maximum of each array can also be used.</li>
          <li>Indicates which blocks need to participate in the computation (1) and which blocks do not (0) after block-based division.</li>
          <li>Pass a valid blockSparseMaskOptional tensor. The operator rejects nullptr.</li>
        </ul>
      </td>
      <td>INT8</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>attenMaskOptional (aclTensor*)</td>
      <td>Input</td>
      <td>The atten_mask in the formula.</td>
      <td>atten_mask takes effect in combination with the sparse pattern. Currently not supported. Pass in nullptr.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockShapeOptional (aclIntArray*)</td>
      <td>Input</td>
      <td>Sparse block shape array.</td>
      <td>        Used together with blockSparseMaskOptional:
        <ul>
          <li>Specifies the sparse block size. If omitted, the default is [128,128].</li>
        </ul>
        When this input is configured: It must contain two elements [blockShapeX, blockShapeY]
        <ul>
          <li>blockShapeX: Block size in the Q direction. The value must be greater than 0.</li>
          <li>blockShapeY: Block size in the KV direction. The value must be greater than 0.</li>
        </ul>
        For the specific multiple constraints on blockShapeX/blockShapeY for each product, see <a href="#constraint_rule">Constraints</a>.
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqLengthsOptional (aclIntArray*)</td>
      <td>Input</td>
      <td>Query sequence length of each batch.</td>
      <td>        Optional Input, used for variable-length sequence use cases:
        <ul>
          <li>When qInputLayout is "TND": this input must be configured.</li>
          <li>When qInputLayout is "BNSD" or "BSND": if this input is configured, the operator processes the data based on the actual sequence lengths specified by this input; if this input is not configured (nullptr is passed), the operator processes the data based on S in the shape of query.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqLengthsOptional (aclIntArray*)</td>
      <td>Input</td>
      <td>Describes the key/value sequence length of each batch.</td>
      <td>        Optional Input, used for variable-length sequence use cases:
        <ul>
          <li>When kvInputLayout is "TND": this input must be configured.</li>
          <li>When kvInputLayout is "BNSD" or "BSND": if this input is configured, the operator processes the data based on the actual sequence lengths specified by this input; if this input is not configured (nullptr is passed), the operator processes the data based on S in the shape of key/value.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockTableOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Block table used for PagedAttention.</td>
      <td>Currently not supported. Must pass in nullptr.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>qDequantScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Dequantization scaling factor of query.</td>
      <td>        Optional input:
        <ul>
          <li>When quantMode is not 0, this parameter must be passed in.</li>
          <li>When quantMode is 0, this parameter must be set to nullptr.</li>
        </ul>
        When this input is configured:
        <ul>
            <li>When quantMode=1, the shape is [batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), 1].</li>
            <li>quantMode=2/3/4/6/7, see the constraints in <a href="#quantization-scale-factor-of-the-mxfp4-quantization-mode">Quantization Scaling Factor of the MXFP4 Quantization Mode</a>.</li>
        </ul>
      </td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>4/5</td>
      <td>×</td>
    </tr>
    <tr>
      <td>kDequantScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Dequantization scaling factor of key.</td>
      <td>        Optional Input:
        <ul>
          <li>When quantMode is not 0, this parameter is required.</li>
          <li>When quantMode is 0, pass nullptr for this parameter.</li>
        </ul>
        When this input is configured:
        <ul>
            <li>When quantMode=1, the shape is [batch, kvHeadNum, ceilDiv(maxKVSeqLength, blockShapeY), 1].</li>
            <li>For quantMode=2/3/4/6/7, see the constraints in <a href="#quantization-scale-factor-of-the-mxfp4-quantization-mode">Quantization Scaling Factor of the MXFP4 Quantization Mode</a>.</li>
        </ul>
      </td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>4/5</td>
      <td>×</td>
    </tr>
    <tr>
      <td>vDequantScaleOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Dequantization scaling factor of value.</td>
      <td>        Optional input:
        <ul>
          <li>Must be passed in when quantMode is not 0.</li>
          <li>Set to nullptr when quantMode is 0.</li>
        </ul>
        When this input is configured:
        <ul>
          <li>When quantMode=1, the shape is [batch, kvHeadNum, ceilDiv(maxKVSeqLength, blockShapeY), 1].</li>
         <li>When quantMode=2/3/4/6/7, see <a href="#quantization-scale-factor-of-the-mxfp4-quantization-mode">Quantization Scale Factor of the MXFP4 Quantization Mode</a> for constraints.</li>
        </ul>
      </td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>4/5</td>
      <td>×</td>
    </tr>
    <tr>
      <td>pQuantScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Quantization scale factor of the softmax result p.</td>
      <td>        Optional input:
        <ul>
          <li>Reserved parameter, currently not supported, set to nullptr.</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>qInputLayout (char*)</td>
      <td>Input</td>
      <td>Indicates the data layout format of the input query.</td>
      <td>Currently, only "TND", "BNSD", and "BSND" are supported, and qInputLayout must be the same as kvInputLayout.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kvInputLayout (char*)</td>
      <td>Input</td>
      <td>Represents the data layout format of the input key and value.</td>
      <td>Currently only "TND", "BNSD", and "BSND" are supported, and qInputLayout must be consistent with kvInputLayout.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKeyValueHeads (int64_t)</td>
      <td>Input</td>
      <td>Represents the number of heads of key/value.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType (int64_t)</td>
      <td>Input</td>
      <td>Represents the mask type in the attention computation.</td>
      <td>        Currently, only 0 can be passed.
        <ul>
          <li>0: indicates the no-mask use case</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue (double)</td>
      <td>Input</td>
      <td>The scale in the formula, indicating the scaling factor.</td>
      <td>Generally set to D^-0.5.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise (int64_t)</td>
      <td>Input</td>
      <td>Precision level used for the Softmax computation.</td>
      <td>        Controls the data type used for the computation in the online softmax phase and rescale phase. Currently, only 0, 1, or 4 can be passed. For values supported by each product, see <a href="#constraint_rule">Constraints</a>.
        <ul>
          <li>0: Indicates that both online softmax and rescale use the fp32 data type. Suitable for use cases that require high computation precision.</li>
          <li>1: Can be configured only when the input query, key, and value are all of the fp16 data type. Indicates that both online softmax and rescale use the fp16 data type. This value delivers better performance but lower precision, and numeric overflow may occur during computation. Determine whether to use this value based on the value range.</li>
          <li>4: Indicates mixed-precision computation, which strikes a balance between performance and precision. In non-quantized modes, online softmax uses the FP16 or BF16 input data type, and rescale uses FP32. Numeric overflow may occur during online softmax. In MXFP4 modes (quantMode 2/3) and MXFP8 mode (quantMode 4), online softmax uses FP16.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize (int64_t)</td>
      <td>Input</td>
      <td>Block size of PagedAttention.</td>
      <td>Used for the PagedAttention use case. The PagedAttention function is currently not supported; therefore, only 0 can be passed.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens (int64_t)</td>
      <td>Input</td>
      <td>Number of tokens that the sliding window needs to contain before the current token in sliding window attention use cases.</td>
      <td>Used in sliding window attention use cases. Currently, sliding window attention is not supported, and only 2147483647 can be passed in.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens (int64_t)</td>
      <td>Input</td>
      <td>Number of tokens that the sliding window needs to contain after the current token in sliding window attention use cases.</td>
      <td>Used for the sliding window attention use case. Currently, sliding window attention is not supported, and only 2147483647 can be passed in.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxLseFlag (int64_t)</td>
      <td>Input</td>
      <td>Flag indicating whether to enable softmaxLse output.</td>
      <td>        Currently, only 0 or 1 can be passed in.
        <ul>
          <li>0: softmaxLse is not output.</li>
          <li>1: softmaxLse is output. Performance loss may occur compared with not outputting softmaxLse.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Flag for selecting the quantization mode.</td>
      <td>        Supported values are 0 through 7.
        <ul>
          <li>0: no quantization.</li>
          <li>1: FP8 quantization.</li>
          <li>2: MXFP4 OCP quantization (the quantization scale is truncated down).</li>
          <li>3: MXFP4 CX quantization (custom quantization range, the quantization scale is truncated up).</li>
          <li>4: MXFP8 OCP quantization.</li>
          <li>5: HIF8 per-tensor quantization.</li>
          <li>6: QK8PV4 OCP. Query and key use FLOAT8_E4M3FN. Value uses FLOAT4_E2M1.</li>
          <li>7: QK8PV4 CX with the same Q/K/V data types as quantMode=6.</li>
          <li>For details, see <a href="#constraint_rule">Constraints</a>.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax (double)</td>
      <td>Input</td>
      <td>Custom quantization range for MXFP4 CX and QK8PV4 CX.</td>
      <td>        For quantMode 3 or 7, use 0.0 or a value in [6.0, 12.0]. For other modes, pass 0.0.
        <ul>
          <li>0.0: indicates that Amax(DType) is the maximum value of the quantization result data type.</li>
          <li>[6.0, 12.0]: a value in the range 6.0 to 12.0 indicates that Amax(DType) is the input value.</li>
        </ul>
      </td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut (aclTensor*)</td>
      <td>Output</td>
      <td>attentionOut in the formula.</td>
      <td>The shape matches query. In non-quantized mode, the data type matches query. In quantized modes, the output type is FLOAT16 or BFLOAT16.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3/4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxLseOptional (aclTensor*)</td>
      <td>Output</td>
      <td>Intermediate log-sum-exp result of the Softmax computation.</td>
      <td>The supported shapes vary with the shape of query:
        <ul>
          <li>If query is "TND": [totalQTokens, headNum, 1].</li>
          <li>If query is "BNSD": [batch, headNum, maxQSeqLength, 1].</li>
          <li>If query is "BSND": [batch, maxQSeqLength, headNum, 1].</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>3/4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Indicates the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
- **Return Value**:
  aclnnStatus:
  return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).
  The first-phase API performs input parameter validation and reports an error in the following cases:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input query, key, or value is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of query, key, and value are not within the supported range.</td>
    </tr>
    <tr>
      <td>qInputLayout or kvInputLayout is invalid.</td>
    </tr>
    <tr>
      <td>blockShape is invalid (the number of elements is not equal to 2 or the value is less than or equal to 0).</td>
    </tr>
    <tr>
      <td>innerPrecise is invalid (must be 0, 1, or 4).</td>
    </tr>
  </tbody>
  </table>

## aclnnBlockSparseAttentionV3

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by the first-phase API aclnnBlockSparseAttentionV3GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, including the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that executes the task.</td>
    </tr>
  </tbody>
  </table>
- **Return Value**:
  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints<a id="constraint_rule"></a>

- Deterministic computation:

  - aclnnBlockSparseAttentionV3 defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- qInputLayout currently supports only "TND", "BNSD", and "BSND".
- kvInputLayout currently supports only "TND", "BNSD", and "BSND".
- Currently, the InputLayout of query, key, and value must be consistent.
- For quantMode=0–5, query, key, and value must have the same type. For quantMode=6/7, query and key must be FLOAT8_E4M3FN and value must be FLOAT4_E2M1.
- The D axis of query, key, and value currently only supports being configured as 64 or 128.
- If blockShapeOptional is passed in, it must contain two elements [blockShapeX, blockShapeY], and the values must be greater than 0. The multiple constraints on blockShapeX/blockShapeY for different products are as follows:
  <!-- npu="950" id7 -->
  - On <term>Ascend 950PR&950DT products</term>: blockShapeY must be a multiple of 16; in MX quantization (quantMode=2/3/4/6/7), both blockShapeX and blockShapeY support only multiples of 64.
  <!-- end id7 -->
  <!-- npu="910b,A3" id8 -->
  - On <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: blockShapeY must be a multiple of 128.
  <!-- end id8 -->
- blockSparseMaskOptional is required. For the grid shape, see the parameter description.
- attentionMaskOptional currently only supports passing nullptr.
- actualSeqLengthsOptional is mandatory when qInputLayout is "TND"; actualSeqLengthsKvOptional is mandatory when kvInputLayout is "TND".
- actualSeqLengthsOptional and actualSeqLengthsKvOptional currently must be configured together or not configured together. Configuring only one of them will be intercepted by the operator.
- blockTableOptional currently only supports passing nullptr, which indicates that the PagedAttention feature is not enabled.
<!-- npu="950,A3,910b" id9 -->
- innerPrecise must be 0, 1, or 4. <term>Ascend 950PR&950DT products</term> supports only the value 4. <term>Atlas A2 products</term> and <term>Atlas A3 products</term> support only the values 0 or 1.
<!-- end id9 -->
- softmaxLseFlag supports only the values 0 or 1, indicating that the softmaxLse output is disabled or enabled, respectively.
- qSeqlen and kvSeqlen do not need to be divisible by blockShape. Non-aligned scenarios are supported, and the actual number of blocks is calculated by rounding up.
- Let the headNum of the input query be N1 and the headNum of the input key and value be N2. Then N1 >= N2 && N1 % N2 == 0.
- maskType supports only 0, which means no additional attention mask is used.
- blockSize currently only supports input 0, which indicates that paged cache is not supported.
- preTokens and nextTokens currently only support passing in 2147483647, which indicates that all tokens before and after the current token participate in the attention computation, i.e., sliding window attention is not supported.
- **quantMode=1 (FP8 static quantization mode) related constraints**:

  - Supported only on Ascend 950PR&950DT products.
  - query, key, and value inputs are of the FLOAT8_E4M3FN data type. The following quantization scale factor parameters must be provided:
    - qDequantScale (query quantization scale factor)

      - Data type: FLOAT32
      - shape: (Batch, HeadNum, CeilDiv(maxQSeqLength, blockShapeX), 1)
      - Purpose: Performs query dequantization during QK matrix multiplication.
    - kDequantScale (key quantization scale factor)

      - Data type: FLOAT32
      - shape: (Batch, KVHeadNum, CeilDiv(maxKVSeqLength, blockShapeY), 1)
      - Purpose: Perform Dequantization on key during the QK Matrix Multiplication.
    - vDequantScale (value quantization scale factor)

      - Data type: FLOAT32
      - shape: (Batch, KVHeadNum, CeilDiv(maxKVSeqLength, blockShapeY), 1)
      - Purpose: Perform Dequantization on value during the PV Matrix Multiplication.
  - If any of query, key, or value is of the FLOAT8_E4M3FN data type, query, key, and value must all be of the FLOAT8_E4M3FN data type.
  - For FP8 inputs, the corresponding quantization scale factor inputs qDequantScale, kDequantScale, and vDequantScale must be provided.
  - The data type of the quantization scale factors must be FLOAT32.
  - The shape of qDequantScale must be (Batch, HeadNum, CeilDiv(maxQSeqLength, blockShapeX), 1).
  - The shapes of kDequantScale and vDequantScale must be the same: (Batch, KVHeadNum, CeilDiv(maxKVSeqLength, blockShapeY), 1).
  - If the data type of any of query, key, and value is not FLOAT8_E4M3FN, qDequantScale, kDequantScale, and vDequantScale must be set to nullptr.
  - blockShapeOptional must be passed in.
  - The quantization block sizes of q and kv must be consistent with the two elements of blockShapeOptional respectively.
- **Constraints for quantMode=2/3 (MXFP4 Quantization Mode) (new)**

  - Supported only on Ascend 950PR&950DT products
  - D supports only 64/128;
  - attentionOut supports only the FLOAT16 or BFLOAT16 data type;
  - Advanced features such as attenMaskOptional, PagedAttention, and softmaxLse are not supported currently.
  - **Quantization Formula**
    - Taking the quantization of P as an example, where $P_{\text{max}}$ is the maximum absolute value of the elements in a group (32 elements).
    - **OCP Quantization Formula**

      $$ P_{\text{Scale}} = 2^{\lfloor \log_2 P_{\text{max}} \rfloor / 4} = 2^{\lfloor \log_2 P_{\text{max}} \rfloor - 2} $$

    - **CX Quantization Formula**

      $$ P_{\text{Scale}} = 2^{\lceil \log_2 \left( P_{\text{max}} / dstTypeMax \right) \rceil} = 2^{\lceil \log_2 P_{\text{max}} - \log_2 dstTypeMax \rceil} $$

<!-- npu="950" id15 -->
- **Constraints for quantMode=6/7 (QK8PV4 mixed quantization)**

  - Supported only on <term>Ascend 950PR&950DT products</term>.
  - Query and key use FLOAT8_E4M3FN. Value uses FLOAT4_E2M1. The value tensor descriptor uses the logical dimension D, and each D-element vector occupies D/2 bytes.
  - Supply FLOAT8_E8M0 scales for Q, K, and V using the [MX quantization scale shapes](#quantization-scale-factor-of-the-mxfp4-quantization-mode).
  - Set softmaxLseFlag to 0.
  - When a Q block has no valid KV block, its attentionOut region retains the contents supplied by the caller.
<!-- end id15 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_block_sparse_attention_v3.h"

using namespace std;

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // Boilerplate for resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    // Check whether the shape is valid.
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }

    auto size = GetShapeSize(shape) * sizeof(T);

    // Check whether the size of hostData matches.
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n",
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }

    // Call aclrtMalloc to allocate memory on the device.
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
              return ret);

    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n");
            return -1);
    return 0;
}

void FreeResource(aclTensor *query, aclTensor *key, aclTensor *value, aclTensor *blockSparseMask,
                  aclTensor *attentionOut, aclIntArray *actualSeqLengths, aclIntArray *actualSeqLengthsKv,
                  aclIntArray *blockShape, void *queryDeviceAddr, void *keyDeviceAddr, void *valueDeviceAddr,
                  void *blockSparseMaskDeviceAddr,void *attentionOutDeviceAddr, void *workspaceAddr,
                  int32_t deviceId, aclrtStream *stream)
{
    // Release resources.
    if (query) {
        aclDestroyTensor(query);
    }
    if (key) {
        aclDestroyTensor(key);
    }
    if (value != nullptr) {
        aclDestroyTensor(value);
    }
    if (blockSparseMask) {
        aclDestroyTensor(blockSparseMask);
    }
    if (attentionOut) {
        aclDestroyTensor(attentionOut);
    }
    if (actualSeqLengths) {
        aclDestroyIntArray(actualSeqLengths);
    }
    if (actualSeqLengthsKv) {
        aclDestroyIntArray(actualSeqLengthsKv);
    }
    if (blockShape) {
        aclDestroyIntArray(blockShape);
    }

    if (queryDeviceAddr) {
        aclrtFree(queryDeviceAddr);
    }
    if (keyDeviceAddr) {
        aclrtFree(keyDeviceAddr);
    }
    if (valueDeviceAddr) {
        aclrtFree(valueDeviceAddr);
    }
    if (blockSparseMaskDeviceAddr) {
        aclrtFree(blockSparseMaskDeviceAddr);
    }
    if (attentionOutDeviceAddr) {
        aclrtFree(attentionOutDeviceAddr);
    }
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main() {
    // 1. (Boilerplate) device/stream initialization.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Set parameters.
    const char *socName = aclrtGetSocName();
    std::string socVersion = (socName != nullptr) ? std::string(socName) : std::string();
    bool isArch22 = socVersion.find("Ascend910") != std::string::npos;
    int64_t innerPrecise = isArch22 ? 0 : 4;

    int32_t batch = 1;
    int32_t qSeqlen = 128;
    int32_t kvSeqlen = 128;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t headDim = 128;
    int32_t blockShapeX = 128;
    int32_t blockShapeY = 256;

    // Calculate the dimensions in TND format.
    int64_t totalQTokens = batch * qSeqlen;
    int64_t totalKvTokens = batch * kvSeqlen;
    int32_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;  // Number of Q blocks along the X dimension.
    int32_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY;  // Number of KV blocks along the Y dimension.

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *blockSparseMaskTensor = nullptr;
    aclTensor *qDequantScaleTensor = nullptr;
    aclTensor *kDequantScaleTensor = nullptr;
    aclTensor *vDequantScaleTensor = nullptr;
    aclTensor *attentionOutTensor = nullptr;
    aclIntArray *actualSeqLengths = nullptr;
    aclIntArray *actualSeqLengthsKv = nullptr;
    aclIntArray *blockShape = nullptr;

    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *blockSparseMaskDeviceAddr = nullptr;
    void *qDequantScaleDeviceAddr = nullptr;
    void *kDequantScaleDeviceAddr = nullptr;
    void *vDequantScaleDeviceAddr = nullptr;
    void *attentionOutDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;

    // 3. Create the Query tensor (TND format: [totalQTokens, numHeads, headDim])
    std::vector<int64_t> queryShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> queryHostData(GetShapeSize(queryShape), 1.0f);
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 4. Create the Key/Value tensor (TND format: [totalKvTokens, numKvHeads, headDim])
    std::vector<int64_t> kvShape = {totalKvTokens, numKvHeads, headDim};
    std::vector<op::fp16_t> keyHostData(GetShapeSize(kvShape), 1.0f);
    std::vector<op::fp16_t> valueHostData(GetShapeSize(kvShape), 1.0f);
    ret = CreateAclTensor(keyHostData, kvShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create key tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);
    ret = CreateAclTensor(valueHostData, kvShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create value tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 5. Create the blockSparseMask tensor ([batch, numHeads, qBlockNum, kvBlockNum])
    std::vector<int64_t> blockSparseMaskShape = {batch, numHeads, qBlockNum, kvBlockNum};
    std::vector<int8_t> blockSparseMaskHostData(GetShapeSize(blockSparseMaskShape), 1);
    ret = CreateAclTensor(blockSparseMaskHostData, blockSparseMaskShape, &blockSparseMaskDeviceAddr, aclDataType::ACL_INT8, &blockSparseMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create block sparse mask tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 6. Create the output tensor
    std::vector<int64_t> attentionOutShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> attentionOutHostData(GetShapeSize(attentionOutShape), 0.0f);
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create attentionOut tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 7. Create the blockShape array
    std::vector<int64_t> blockShapeData = {blockShapeX, blockShapeY};
    blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_RET(blockShape != nullptr, LOG_PRINT("Failed to create blockShape array\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return -1);

    // 8. Create actualSeqLengths and actualSeqLengthsKv (required parameters)
    std::vector<int64_t> actualSeqLengthsHost(batch, static_cast<int64_t>(qSeqlen));
    std::vector<int64_t> actualSeqLengthsKvHost(batch, static_cast<int64_t>(kvSeqlen));


    actualSeqLengths = aclCreateIntArray(actualSeqLengthsHost.data(), batch);
    actualSeqLengthsKv = aclCreateIntArray(actualSeqLengthsKvHost.data(), batch);
    CHECK_RET(actualSeqLengths != nullptr && actualSeqLengthsKv != nullptr,
              LOG_PRINT("Failed to create actualSeqLengths arrays\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return -1);

    // 9. Prepare the string parameters (ensure the buffer is large enough to include the null terminator)
    const char* qLayoutStr = "TND";
    const char* kvLayoutStr = "TND";
    char qLayoutBuffer[16] = {0};
    char kvLayoutBuffer[16] = {0};
    strncpy(qLayoutBuffer, qLayoutStr, sizeof(qLayoutBuffer) - 1);
    strncpy(kvLayoutBuffer, kvLayoutStr, sizeof(kvLayoutBuffer) - 1);

    // 10. Calculate scaleValue
    float scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));

    // 11. Call the first-phase API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnBlockSparseAttentionV3GetWorkspaceSize(
        queryTensor,              // query
        keyTensor,                // key
        valueTensor,              // value
        blockSparseMaskTensor,    // blockSparseMask
        nullptr,                  // attenMaskOptional
        blockShape,               // blockShape
        actualSeqLengths,         // actualSeqLengthsOptional
        actualSeqLengthsKv,       // actualSeqLengthsKvOptional
        nullptr,                  // blockTableOptional
        nullptr,                  // qDequantScale
        nullptr,                  // kDequantScale
        nullptr,                  // vDequantScale
        nullptr,                  // pDequantScale
        qLayoutBuffer,            // qInputLayout
        kvLayoutBuffer,           // kvInputLayout
        numKvHeads,               // numKeyValueHeads
        0,                        // maskType
        scaleValue,               // scaleValue
        innerPrecise,             // innerPrecise
        0,                        // blockSize
        2147483647,               // preTokens
        2147483647,               // nextTokens
        0,                        // softmaxLseFlag
        0,                        // quantMode
        0.0,                      // dstTypeMax
        attentionOutTensor,       // attentionOut
        nullptr,                  // softmaxLseOptional
        &workspaceSize,           // workspaceSize (out)
        &executor);               // executor (out)

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionV3GetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return -1);

    // 12. Allocate workspace
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                              actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                              valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                              workspaceAddr, deviceId, &stream);
                  return ret);
    }

    // 13. Call the second-phase API
    ret = aclnnBlockSparseAttentionV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionV3 failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 14. Wait for the task execution to complete synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 15. Obtain the output value and copy the result from the device-side memory to the host side.
    int64_t attentionOutSize = GetShapeSize(attentionOutShape);
    std::vector<op::fp16_t> resultData(attentionOutSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(op::fp16_t), attentionOutDeviceAddr,
                      attentionOutSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 16. Print part of the results.
    uint64_t printNum = 10;
    LOG_PRINT("attentionOut results (first %lu elements):\n", printNum);
    for (uint64_t i = 0; i < printNum && i < resultData.size(); i++) {
        LOG_PRINT("  index %lu: %f\n", i, static_cast<float>(resultData[i]));
    }

    // 17. Release resources.
    FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                  actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                  valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                  workspaceAddr, deviceId, &stream);

    LOG_PRINT("Test completed successfully!\n");
    return 0;
}
```
