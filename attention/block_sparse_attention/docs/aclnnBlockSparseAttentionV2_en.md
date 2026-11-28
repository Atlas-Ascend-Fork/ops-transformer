# aclnnBlockSparseAttentionV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:14:23.836Z -->

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

- **Description**: BlockSparseAttentionV2 performs sparse attention computation. It supports flexible block-level sparse patterns: blockSparseMaskOptional specifies the KV blocks selected for each Q block, enabling efficient sparse attention computation.

  Compared with BlockSparseAttention, this API adds the qDequantScaleOptional, kDequantScaleOptional, and vDequantScaleOptional parameters.

- **Formula**: Sparse block size: $blockShapeX \times blockShapeY$, blockSparseMaskOptional specifies the sparse pattern

  $$
  attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T + atten\_mask) \cdot value_{sparse}
  $$

  Here, `scale` is the scaling factor controlled by `scaleValue`. `atten_mask` is the attention mask corresponding to `attenMaskOptional`, which currently must be `nullptr`.

  BlockSparseAttentionV2 supports interpreting the data layout of the query, key, and value inputs from multiple dimension arrangements, which can be passed in through qInputLayout and kvInputLayout.

  - B: Indicates the batch size of the input samples (Batch).
  - T: Indicates the length of B and S concatenated along the axis and tightly packed (Total tokens).
  - S: Indicates the sequence length of the input samples (Seq-Length).
  - H: Indicates the size of the hidden layer (Head-Size).
  - N: Number of heads (Head-Num).
  - D: Minimum unit size of the hidden layer, satisfying D = H/N (Head-Dim).

  Currently supported layouts:

  - qInputLayout: "TND" "BNSD" "BSND"
  - kvInputLayout: "TND" "BNSD" "BSND"

<!-- npu="950" id7 -->
- **FP8 feature description (supported only on <term>Ascend 950PR&950DT products</term>)**:

  This operator newly supports inputs of the FP8 data type to improve computing efficiency and reduce memory usage. When FP8 inputs are used, the corresponding quantization scaling factors must be provided for dequantization.

<!-- end id7 -->
- **Quantization Scaling Factor**:

  When the query, key, and value inputs use the FLOAT8_E4M3FN data type, the following quantization scaling factor parameters are required:

  qDequantScale (query quantization scaling factor)
  - Data type: FLOAT32
  - shape: (Batch, HeadNum, CeilDiv(maxQSeqLength, blockShapeX), 1)
  - Purpose: Perform Dequantization on query during QK Matrix Multiplication.

  kDequantScale (key Quantization Scaling factor)
  - Data type: FLOAT32
  - shape: (Batch, KVHeadNum, CeilDiv(maxKVSeqLength, blockShapeY), 1)
  - Purpose: Perform Dequantization on key during QK Matrix Multiplication.

  vDequantScale (value Quantization Scaling factor)
  - Data type: FLOAT32
  - shape: (Batch, KVHeadNum, CeilDiv(maxKVSeqLength, blockShapeY), 1)
  - Purpose: Perform Dequantization on value during PV Matrix Multiplication.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnBlockSparseAttentionV2GetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call "aclnnBlockSparseAttentionV2" to perform the computation.

```c++
aclnnStatus aclnnBlockSparseAttentionV2GetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *blockSparseMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *blockShapeOptional,
  const aclIntArray *actualSeqLengthsOptional,
  const aclIntArray *actualSeqLengthsKvOptional,
  const aclTensor   *blockTableOptional,
  const aclTensor   *qDequantScale,
  const aclTensor   *kDequantScale,
  const aclTensor   *vDequantScale,
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
  aclTensor         *attentionOut,
  aclTensor         *softmaxLseOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnBlockSparseAttentionV2(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnBlockSparseAttentionV2GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px">
  <colgroup>
  <col style="width: 269px">
  <col style="width: 132px">
  <col style="width: 270px">
  <col style="width: 350px">
  <col style="width: 185px">
  <col style="width: 119px">
  <col style="width: 146px">
  <col style="width: 145px">
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
      <td>Query in the formula.</td>
      <td>Supported shapes are as follows:
        <ul><li>TND: [totalQTokens, headNum, headDim].</li>
        <li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li>
        <li>BSND: [batch, maxQSeqLength, headNum, headDim].</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>key (aclTensor*)</td>
      <td>Input</td>
      <td>Key in the formula.</td>
      <td>The supported shapes are as follows:
        <ul>
          <li>TND: [totalKTokens, numKeyValueHeads, headDim].</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li>
          <li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim].</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>value (aclTensor*)</td>
      <td>Input</td>
      <td>The value in the formula.</td>
      <td>        The supported shapes are as follows:
        <ul>
          <li>TND: [totalVTokens, numKeyValueHeads, headDim].</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li>
          <li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim].</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockSparseMaskOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the actual sparse pattern.</td>
      <td>        Optional Input (required in the current version)
        <ul>
          <li>The shape is [batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)].</li>
          <li>Indicates which blocks participate in the computation (value 1) and which blocks do not (value 0) after block partitioning.</li>
          <li>If nullptr is passed in, block sparse computation is disabled; that is, the attention scores between all tokens are computed.</li>
        </ul>
      </td>
      <td>INT8</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>attenMaskOptional(aclTensor*)</td>
      <td>Input</td>
      <td>The atten_mask in the formula.</td>
      <td>The atten_mask takes effect together with the sparse pattern. Currently not supported; pass in nullptr.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
    <tr>
      <td>blockShapeOptional(aclIntArray*)</td>
      <td>Input</td>
      <td>Sparse block shape array.</td>
      <td>        Used together with blockSparseMaskOptional:
        <ul>
          <li>When blockSparseMaskOptional is configured: if this input is configured, the operator obtains the sparse block size from it; if this input is not configured, the operator defaults the sparse block size to [128,128].</li>
          <li>When blockSparseMaskOptional is not configured: the operator ignores this input regardless of its configuration.</li>
        </ul>
        When this input is configured: it must contain two elements [blockShapeX, blockShapeY]
        <ul>
          <li>blockShapeX: Block size along the Q direction. The value must be greater than 0.</li>
          <li>blockShapeY: Block size along the KV direction. The value must be greater than 0; it must be a multiple of 16 on <term>Ascend 950PR&950DT products</term>, and a multiple of 128 on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
      <td>actualSeqLengthsOptional (aclIntArray*)</td>
      <td>Input</td>
      <td>Describes the query sequence length for each Batch.</td>
      <td>        Optional Input, used for variable-length sequence scenarios:
        <ul>
          <li>When qInputLayout is "TND": this input must be configured.</li>
          <li>When qInputLayout is "BNSD" or "BSND": if this input is configured, the operator processes data based on the actual sequence length specified by this input; if this input is not configured (pass in nullptr), the operator processes data based on S in the query shape.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKvOptional(aclIntArray*)</td>
      <td>Input</td>
      <td>Describes the key/value sequence length corresponding to each Batch.</td>
      <td>        Optional Input, used for variable-length sequence scenarios:
        <ul>
          <li>When kvInputLayout is "TND": this input must be configured.</li>
          <li>When kvInputLayout is "BNSD" or "BSND": if this input is configured, the operator processes data based on the actual sequence length specified by this input; if this input is not configured (pass in nullptr), the operator processes data based on S in the key/value shape.</li>
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
      <td>The block table is used for PagedAttention.</td>
      <td>Currently not supported; must pass in nullptr.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>qDequantScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Dequantization scaling factor of query.</td>
      <td>        Optional Input:
        <ul>
          <li>When the data types of query, key, and value are FLOAT8_E4M3FN, this parameter must be passed in.</li>
          <li>When the data types of query, key, and value are FLOAT16 or BFLOAT16, this parameter must be set to nullptr.</li>
        </ul>
        When this input is configured:
        <ul>
          <li>The shape is [batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), 1].</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>kDequantScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Dequantization scaling factor of key.</td>
      <td>        Optional input:
        <ul>
          <li>If the data types of query, key, and value are FLOAT8_E4M3FN, this parameter must be passed in.</li>
          <li>If the data types of query, key, and value are FLOAT16 or BFLOAT16, nullptr should be passed in.</li>
        </ul>
        If this input is configured:
        <ul>
          <li>The shape is [batch, kvHeadNum, ceilDiv(maxKvSeqLength, blockShapeY), 1].</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>vDequantScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Dequantization scaling factor of value.</td>
      <td>        Optional Input:
        <ul>
          <li>When the data types of query, key, and value are FLOAT8_E4M3FN, this parameter must be passed in.</li>
          <li>When the data types of query, key, and value are FLOAT16 or BFLOAT16, nullptr should be passed in for this parameter.</li>
        </ul>
        When this input is configured:
        <ul>
          <li>The shape is [batch, kvHeadNum, ceilDiv(maxKvSeqLength, blockShapeY), 1].</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>qInputLayout (char*)</td>
      <td>Input</td>
      <td>Indicates the data layout of the input query.</td>
      <td>The value ranges are "TND"/"BNSD"/"BSND". For details, see Constraints.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kvInputLayout (char*)</td>
      <td>Input</td>
      <td>Indicates the data layout of the input key and value.</td>
      <td>Value range: "TND"/"BNSD"/"BSND". For details about the constraints, see Constraints.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKeyValueHeads (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of key/value heads.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType (int64_t)</td>
      <td>Input</td>
      <td>Indicates the mask type in the attention computation.</td>
      <td>        Currently, only 0 can be passed in.
        <ul>
          <li>0: indicates the use case where no mask is added</li>
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
      <td>Indicates the scale in the formula, that is, the scaling factor.</td>
      <td>Generally set to D^-0.5.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise (int64_t)</td>
      <td>Input</td>
      <td>Precision level used for the Softmax calculation.</td>
      <td>        Controls the data types used for calculation in the online softmax and rescale phases. Currently, only 0, 1, or 4 can be passed in, where <term>Ascend 950PR&950DT products</term> supports only 4, and <term>Atlas A2 products</term> and <term>Atlas A3 products</term> support only 0 or 1
        <ul>
          <li>0: Indicates that both online softmax and rescale use the fp32 data type, which is suitable for use cases requiring high calculation precision.</li>
          <li>1: Can be configured only when the input query, key, and value are all of the fp16 data type. Indicates that both online softmax and rescale use the fp16 data type. This setting delivers better performance but lower precision, and numeric overflow may occur during calculation. Users need to determine whether to use it based on the value range.</li>
          <li>4: Indicates mixed-precision calculation, which strikes a balance between performance and precision. Online softmax uses the fp16/bf16 data type (the same as the data types of query, key, and value), and rescale uses the fp32 data type. Numeric overflow may occur in the online softmax phase.</li>
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
      <td>Used for PagedAttention use cases. Currently, PagedAttention is not supported, so only 0 can be passed in.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of tokens that the sliding window needs to include before the current token in the sliding window attention use case.</td>
      <td>Used for the sliding window attention use case. Currently, sliding window attention is not supported. Only 2147483647 can be passed in.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens (int64_t)</td>
      <td>Input</td>
      <td>In the sliding window attention use case, indicates the number of tokens that the sliding window needs to include backward.</td>
      <td>Used for the sliding window attention use case. Sliding window attention is currently not supported; only 2147483647 can be passed in.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxLseFlag (int64_t)</td>
      <td>Input</td>
      <td>Flag indicating whether to enable the softmaxLse output.</td>
      <td>        Currently, only 0 or 1 can be passed in.
        <ul>
          <li>0: Indicates that softmaxLse is not output.</li>
          <li>1: Indicates that softmaxLse is output, which may cause performance loss compared with not outputting softmaxLse.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut (aclTensor*)</td>
      <td>Output</td>
      <td>attentionOut in the formula.</td>
      <td>The data type and shape are the same as those of query.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3/4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxLseOptional(aclTensor*)</td>
      <td>Output</td>
      <td>Intermediate log-sum-exp result of the Softmax computation.</td>
      <td>        The supported shapes vary with the shape of query:
        <ul>
          <li>If query is "TND": [totalQTokens, headNum, 1].</li>
          <li>If query is "BNSD": [batch, headNum, maxQSeqLength, 1].</li>
          <li>If query is "BSND": [batch, maxQSeqLength, headNum, 1].</li>
        </ul>
      </td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3/4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, including the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

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
      <td>The input query, key, and value are passed in as null pointers.</td>
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
      <td>innerPrecise is invalid (it must be 0, 1, or 4).</td>
    </tr>
  </tbody>
  </table>

## aclnnBlockSparseAttentionV2

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnBlockSparseAttentionV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnBlockSparseAttentionV2 defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- qInputLayout currently supports only "TND", "BNSD", and "BSND".
- kvInputLayout currently supports only "TND", "BNSD", and "BSND".
- Currently, the InputLayout of query, key, and value must be consistent.
- The data types of the input query, key, and value must be consistent, supporting FLOAT16, BFLOAT16, and FLOAT8_E4M3FN (FLOAT8_E4M3FN is supported only on <term>Ascend 950PR&950DT products</term>).
- The D axis of query, key, and value currently only supports being configured as 64 or 128.
- If blockShapeOptional is passed in, it must contain two elements [blockShapeX, blockShapeY], and the values must be greater than 0. blockShapeY must be a multiple of 16 on <term>Ascend 950PR&950DT products</term>, and a multiple of 128 on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>.
- blockSparseMaskOptional currently must be passed in, and its shape must be [batch, headNum, ceilDiv(maxQS, blockShapeX), ceilDiv(maxKVS, blockShapeY)].
- attentionMaskOptional currently only supports passing nullptr.
- actualSeqLengthsOptional is mandatory when qInputLayout is "TND"; actualSeqLengthsKvOptional is mandatory when kvInputLayout is "TND".
- actualSeqLengthsOptional and actualSeqLengthsKvOptional currently must be configured together or not configured together. Configuring only one of them will be intercepted by the operator.
- blockTableOptional currently only supports passing nullptr, which indicates that the PagedAttention feature is not enabled.
- innerPrecise supports only the value 4, indicating mixed-precision computation to achieve a trade-off between performance and accuracy.
- softmaxLseFlag supports only the values 0 or 1, indicating that the softmaxLse output is disabled/enabled, respectively.
- qSeqlen and kvSeqlen do not need to be divisible by blockShape. Non-aligned scenarios are supported, and the actual number of blocks is calculated by rounding up.
- Let the headNum of the input query be N1 and the headNum of the input key and value be N2. Then N1 >= N2 && N1 % N2 == 0.
- maskType currently only supports input 0, which indicates that no mask is applied.
- blockSize currently only supports input 0, which indicates that paged cache is not supported.
- preTokens and nextTokens currently only support passing in 2147483647, which indicates that all tokens before and after the current token participate in the attention computation, i.e., sliding window attention is not supported.
- **FP8-related constraints (new)**:

  <!-- npu="950" id8 -->
  - Supported only on <term>Ascend 950PR&950DT products</term>.
  <!-- end id8 -->
  - If the data type of any of query, key, and value is FLOAT8_E4M3FN, query, key, and value must all be of the FLOAT8_E4M3FN data type.
  - When FP8 inputs are used, the corresponding quantization scaling factor inputs qDequantScale, kDequantScale, and vDequantScale must be provided.
  - The data type of the quantization scaling factor must be FLOAT32.
  - The shape of qDequantScale must be (Batch, HeadNum, CeilDiv(maxQSeqLength, blockShapeX), 1).
  - The shapes of kDequantScale and vDequantScale must be the same, which is (Batch, KVHeadNum, CeilDiv(maxKVSeqLength, blockShapeY), 1).
  - If the data type of any of query, key, or value is not FLOAT8_E4M3FN, qDequantScale, kDequantScale, and vDequantScale must be passed in as nullptr.
  - blockShapeOptional must be passed in.
  - The quantization block sizes of q and kv must be the same as the two element sizes of blockShapeOptional, respectively.

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
#include "aclnnop/aclnn_block_sparse_attention.h"
#include "aclnnop/aclnn_block_sparse_attention_v2.h"

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
    // Boilerplate for resource initialization
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

    // Call aclrtMalloc to allocate device memory.
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
              return ret);

    // Call aclrtMemcpy to copy data from the host side to the device memory.
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

    // Call the aclCreateTensor API to create aclTensor.
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
    // 1. (Boilerplate) Initialize device/stream.
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

    // Calculate dimensions in TND format.
    int64_t totalQTokens = batch * qSeqlen;
    int64_t totalKvTokens = batch * kvSeqlen;
    int32_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;  // Number of Q blocks in the X dimension
    int32_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY;  // Number of KV blocks in the Y dimension

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

    // 3. Create the Query tensor (TND format: [totalQTokens, numHeads, headDim]).
    std::vector<int64_t> queryShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> queryHostData(GetShapeSize(queryShape), 1.0f);
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 4. Create the Key/Value tensor (TND format: [totalKvTokens, numKvHeads, headDim]).
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

    // 5. Create the blockSparseMask tensor ([batch, numHeads, qBlockNum, kvBlockNum]).
    std::vector<int64_t> blockSparseMaskShape = {batch, numHeads, qBlockNum, kvBlockNum};
    std::vector<int8_t> blockSparseMaskHostData(GetShapeSize(blockSparseMaskShape), 1);
    ret = CreateAclTensor(blockSparseMaskHostData, blockSparseMaskShape, &blockSparseMaskDeviceAddr, aclDataType::ACL_INT8, &blockSparseMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create block sparse mask tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 6. Create the output tensor.
    std::vector<int64_t> attentionOutShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> attentionOutHostData(GetShapeSize(attentionOutShape), 0.0f);
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create attentionOut tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 7. Create the blockShape array.
    std::vector<int64_t> blockShapeData = {blockShapeX, blockShapeY};
    blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_RET(blockShape != nullptr, LOG_PRINT("Failed to create blockShape array\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return -1);

    // 8. Create actualSeqLengths and actualSeqLengthsKv (required parameters).
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

    // 9. Prepare the string parameters (ensure the buffer size is sufficient and includes the null terminator).
    const char* qLayoutStr = "TND";
    const char* kvLayoutStr = "TND";
    char qLayoutBuffer[16] = {0};
    char kvLayoutBuffer[16] = {0};
    strncpy(qLayoutBuffer, qLayoutStr, sizeof(qLayoutBuffer) - 1);
    strncpy(kvLayoutBuffer, kvLayoutStr, sizeof(kvLayoutBuffer) - 1);

    // 10. Calculate scaleValue.
    float scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));

    // 11. Call the first-phase API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnBlockSparseAttentionV2GetWorkspaceSize(
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
        attentionOutTensor,       // attentionOut
        nullptr,                  // softmaxLseOptional
        &workspaceSize,           // workspaceSize (out)
        &executor);               // executor (out)

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionV2GetWorkspaceSize failed. ERROR: %d\n", ret);
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

    // 12. Allocate the workspace.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                              actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                              valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                              workspaceAddr, deviceId, &stream);
                  return ret);
    }

    // 13. Call the second-phase API.
    ret = aclnnBlockSparseAttentionV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionV2 failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 14. Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 15. Obtain the output value and copy the result from the device to the host.
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
