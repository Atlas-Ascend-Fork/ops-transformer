# aclnnKvRmsNormRopeCacheV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:54:56.438Z -->

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

- Description: Fuses the RMSNorm normalization computation in the MLA (Multi-head Latent Attention) structure (corresponding to $rms\_size$) with the RoPE (Rotary Position Embedding) rotary position encoding (corresponding to $rope\_size$), and the ScatterUpdate operation for updating the KVCache. This API supports two use cases and is backward compatible with aclnnKvRmsNormRopeCache.

- Supported use cases:

    | Scenario Type | Source of kv Components | Description |
    | :------: | :------: | :------ |
    |V1|rms_size=Dv=512<br>rope_size=Dk=64<br>vOptional=None|kv axis fusion mode: The trailing axis of the input tensor kv is split into a left half for rms_norm computation and a right half for rope computation. The computation results are then scattered into the two caches, respectively.<li>Closely related to the DeepSeekV3 network structure. Only the use case with N=1 is supported.</li><li>Dv, the data required for rms_norm computation, and Dk, the data required for rope computation, are split from D of the input kv. Dk and Dv must satisfy Dk+Dv=Dkv.</li>|
    |V2|Dv=128<br>rms_size=Dk=Dkv=192<br>rope_size=64<br>vOptional has a shape of [Bkv, Nkv, Skv, Dv]|kv axis separation mode: rms_norm computation is performed on the input tensor kv. Then rope computation is performed on the first 64 dimensions of the trailing axis, and the results overwrite the corresponding elements. The final result is scattered into kCacheRef. The input tensor vOptional undergoes intermediate processing, and the final result is scattered into ckvCacheRef.<li>N=1/2/4/8 is supported.</li><li>In this use case, the trailing axes of k and v are separated. kv stores only the trailing axis of the k component, and vOptional stores the trailing axis of the v component.</li>|

    - <term>Ascend 950PR&950DT products</term>: Only the V1 use case is supported.
    - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: Support the V1 and V2 use cases.

- Formula:

  The input tensor kv has a shape of $[Bkv, N, Skv, Dkv]$, and the tensor vOptional has a shape of $[Bkv, N, Skv, Dv]$.

  (1) RmsNorm:

  $$
  x=kv[...,:rms\_size]
  $$

  $$
  \operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  y = \operatorname{RmsNorm}(x)
  $$

  (2) interleaveRope:

  $$
  x=\begin{cases}
  kv[...,Dv:], \quad vOptional = None\\
  {\operatorname{RmsNorm}(\mathbf{x})}[...,:rope\_size],  \quad vOptional != None
  \end{cases}
  $$

  $$
  x1=x[...,::2]
  $$

  $$
  x2=x[...,1::2]
  $$

  $$
  x\_part1=torch.cat((x1,x2),dim=-1)
  $$

  $$
  x\_part2=torch.cat((-x2,x1),dim=-1)
  $$

  $$
  y\_rope=x\_part1*cos+x\_part2*sin
  $$

  $$
  rope\_out=\begin{cases}y\_rope, \quad vOptional = None \\
  concat(y\_rope, {\operatorname{RmsNorm}(\mathbf{x})}[...,rope\_size:]),  \quad vOptional != None
  \end{cases}
  $$

  (3) Quantization computation:

  x indicates the raw data to be written to kCacheRef and ckvCacheRef, serving as the input of the quantization process.

  $$
  x = x * scale,\ if\ scale\ !=\ None
  $$

  $$
  x = x + offset,\ if\ offset\ !=\ None
  $$

  $$
  y = \begin{cases}x, \quad scale == None \space and \space offset == None \\ round(x).clamp(-128,127), \quad others
  \end{cases}
  $$

  (4) Scatter write-out:

  The input tensor index corresponds to the index mapping table of each element in the input KV cache. It takes the indices $b \in Bkv$ and $s \in Skv$ of specific elements in x, where $n$ is the attention head index,

  $$
  scatter\_idx = index(b, s) \\
  $$

   Quant denotes the aforementioned quantization computation process, which updates parameters k\_cache and ckv\_cache in place:

  $$
  k\_cache[scatter\_idx, ...] = Quant[x = rope\_out, scale = k\_scale, offset = k\_offset]\left(b, n, s\right)
  $$

  $$
  ckv\_cache[scatter\_idx, ...] = \begin{cases} Quant(x = \operatorname{RmsNorm}(x), scale = v\_scale, offset = v\_offset)[b, n, s], \quad vOptional = None \\
  Quant[x = vOptional, scale = v\_scale, offset = v\_offset]\left(b, n, s\right),  \quad vOptional != None
  \end{cases}
  $$

  (5) Raw result write-out:

  When $is\_output\_kv=True$ exists and is valid:

  $$
  k\_rope = rope\_out
  $$

  $$
  c\_kv = \begin{cases}\operatorname{RmsNorm}(x), \quad vOptional = None \\
  vOptional,  \quad vOptional != None
  \end{cases}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnKvRmsNormRopeCacheV2GetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size according to the procedure, and then call the "aclnnKvRmsNormRopeCacheV2" API to perform the computation.

```Cpp
aclnnStatus aclnnKvRmsNormRopeCacheV2GetWorkspaceSize(
  const aclTensor* kv,
  const aclTensor* gamma,
  const aclTensor* cos,
  const aclTensor* sin,
  const aclTensor* index,
  aclTensor*       kCacheRef,
  aclTensor*       ckvCacheRef,
  const aclTensor* kRopeScaleOptional,
  const aclTensor* cKvScaleOptional,
  const aclTensor* kRopeOffsetOptional,
  const aclTensor* cKvOffsetOptional,
  const aclTensor* vOptional,
  double           epsilon,
  char*            cacheModeOptional,
  bool             isOutputKv,
  const aclTensor* kRopeOut,
  const aclTensor* cKvOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnKvRmsNormRopeCacheV2(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnKvRmsNormRopeCacheV2GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 403px">
  <col style="width: 403px">
  <col style="width: 275px">
  <col style="width: 118px">
  <col style="width: 138px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>kv</td>
      <td>Input</td>
      <td>Input data used in the formula to split out Dv for the rms_norm computation and Dk for the RoPE computation.</td>
      <td><ul><li>Only the 4D shape [Bkv,N,Skv,D] is supported.</li><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Input data used for the rms_norm computation in the formula.</td>
      <td><ul><li>Same data type as kv; shape is 1D [Dv,] for V1 and [Dk,] (192) for V2.</li><li>Empty tensor is not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>Input</td>
      <td>Input data used for RoPE computation in the formula. Applies cosine transformation to the input tensor Dk.</td>
      <td><ul><li>Same as the data type of kv, shape is 4D [Bkv,N,Skv,Dk] or [Bkv,N,1,Dk].</li><li>Empty tensor is not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>Optional input</td>
      <td>Input data used for RoPE computation in the formula. Performs the sine transformation on the input tensor Dk.</td>
      <td><ul><li>Data type is the same as that of kv. Shape is 4D [Bkv,N,Skv,Dk] or [Bkv,N,1,Dk], consistent with the shape of cos.</li><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Input</td>
      <td>Specifies the index for writing to the cache. If the value of index is -1, the update is skipped.</td>
      <td><ul><li>If cacheModeOptional is Norm, the shape is 2D [Bkv,Skv].</li><li>If cacheModeOptional is PA_BNSD or PA_NZ, the shape is 1D [Bkv * Skv].</li><li>If cacheModeOptional is PA_BLK_BNSD or PA_BLK_NZ, the shape is 1D [Bkv\*ceil_div(Skv,BlockSize)].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kCacheRef</td>
      <td>Input/Output</td>
      <td>Cache applied for in advance. Input and output reuse the same address.</td>
      <td><ul><li>In non-quantized use cases, the data type is the same as that of input kv.</li><li>In quantized use cases, the data type is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.</li><li>When cacheModeOptional is a PA use case (cacheModeOptional is PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ), the shape is 4D [BlockNum,BlockSize,N,Dk].</li><li>When cacheModeOptional is a Norm use case, the shape is 4D [Bcache,N,Scache,Dk].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8E5M2, FLOAT8E4M3FN</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ckvCacheRef</td>
      <td>Input/Output</td>
      <td>Cache applied for in advance. Input and output reuse the same address.</td>
      <td><ul><li>In non-quantization use cases, the data type is the same as that of input kv.</li><li>In quantization use cases, the data type is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.</li><li>When cacheModeOptional is a PA use case (cacheModeOptional is PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ), the shape is 4D [BlockNum,BlockSize,N,Dv].</li><li>When cacheModeOptional is a Norm use case, the shape is 4D [Bcache,N,Scache,Dv].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8E5M2, FLOAT8E4M3FN</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kRopeScaleOptional</td>
      <td>Input</td>
      <td>This input parameter is required when the data type of kCacheRef is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.</td>
      <td><ul><li>The shape is 2D [N,Dk], 1D [Dk,], or 1D [1,].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ckvScaleOptional</td>
      <td>Input</td>
      <td>This input parameter is required when the data type of ckvCacheRef is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.</td>
      <td><ul><li>The shape is 2D [N,Dv]; or 1D [Dv,]; or 1D [1,].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kRopeOffsetOptional</td>
      <td>Input</td>
      <td>Required when the kCacheRef data type is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN, the corresponding kRopeScaleOptional input exists, and the quantization use case is asymmetric quantization.</td>
      <td><ul><li>The shape is 2D [N,Dk], 1D [Dk,], or 1D [1,].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cKvOffsetOptional</td>
      <td>Input</td>
      <td>This parameter is required when the data type of ckvCacheRef is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN, the corresponding ckvScaleOptional input exists, and the quantization use case is asymmetric quantization.</td>
      <td><ul><li>The shape is 2D [N,Dv], 1D [Dv,], or 1D [1,].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOptional</td>
      <td>Input</td>
      <td>Used only in the KV separated use case (V2) as the input source of the Dv component for immediate scatter.<br>The first three dimensions of the shape must be the same as those of kv, and the data type must be the same as that of kv.</td>
      <td><ul><li>The shape is 4D [Bkv,N,Skv,Dv].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Attribute</td>
      <td>Prevents division by zero in rms_norm computation.</td>
      <td>The recommended value is 1e-5.</td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cacheModeOptional</td>
      <td>Attribute</td>
      <td>Flag for selecting the cache format.</td>
      <td>Types include Norm, PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, and PA_BLK_NZ. Norm is recommended.</td>
      <td>char*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isOutputKv</td>
      <td>Attribute</td>
      <td>Output control flag for kRopeOut and cKvOut.</td>
      <td>If isOutputKv is true, kRopeOut and cKvOut are output. It is recommended to set it to false; on Atlas A3 products, non-null tensor descriptors for kRopeOut and cKvOut are still required when it is false.</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kRopeOut</td>
      <td>Output</td>
      <td>Controlled by isOutputKv, corresponding to `rope_out` in the interleaveRope formula.<br>Required when isOutputKv is true. The data type is the same as that of the input kv.</td>
      <td>The shape is 4D [Bkv,N,Skv,Dk].</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cKvOut</td>
      <td>Output</td>
      <td>Controlled by isOutputKv, corresponding to `y` in the rmsNorm formula.<br>Required when isOutputKv is true. The data type is the same as that of the input kv.</td>
      <td>The shape is 4D [Bkv,N,Skv,Dv].</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
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

  <table style="undefined;table-layout: fixed; width: 1260px"><colgroup>
  <col style="width: 325px">
  <col style="width: 126px">
  <col style="width: 809px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input or output tensor is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The input and output data types are not within the supported range.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="4">561002</td>
      <td>Shape validation of parameters kv, gamma, cos, sin, index, kCacheRef, ckvCacheRef, kRopeScaleOptional, ckvScaleOptional, kRopeOffsetOptional, and cKvOffsetOptional fails.</td>
    </tr>
    <tr>
      <td>Dtype validation of parameters kv, gamma, cos, sin, index, kCacheRef, ckvCacheRef, kRopeScaleOptional, ckvScaleOptional, kRopeOffsetOptional, and cKvOffsetOptional fails.</td>
    </tr>
    <tr>
      <td>In the PA use case (cacheModeOptional is PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ), the BlockSize dimension value of the cache fails validation.</td>
    </tr>
    <tr>
      <td>In the NZ use case (cacheModeOptional is PA_NZ or PA_BLK_NZ), the dimension values of Dk and Dv fail validation.</td>
    </tr>
  </tbody>
  </table>

## aclnnKvRmsNormRopeCacheV2

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
      <td>Workspace size allocated on the device. The value is obtained by the first-phase API aclnnKvRmsNormRopeCacheV2GetWorkspaceSize.</td>
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

  - This operator defaults to a deterministic implementation.
  - Input shape constraints:

      - kv is a 4D tensor with shape [Bkv,N,Skv,D]. Bkv is the batch size of the input kv. Skv is the sequence length of the input kv. Its size depends on the user input use case and has no explicit limit.
      - N is the head number of the input kv. The V1 use case is strongly related to the DeepSeekV3 network structure and supports only N=1. The V2 use case supports N=1/2/4/8.
      - D is the head dim of the input kv. According to the rope rule, Dk must be an even number. If cacheModeOptional is an NZ use case (cacheModeOptional is PA_NZ or PA_BLK_NZ), Dk and Dv must be 32B-aligned. This rule applies to all use cases and computation types.
      - If cacheModeOptional is a PA use case (cacheModeOptional is PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ), block_size must be 32B-aligned.
      - For the preceding 32B alignment cases, the alignment value depends on the data type of the cache. Take block_size as an example: if the data type of the cache is int8, block_size%32=0 is required; if the data type of the cache is float16, block_size%16=0 is required; if the dtypes of kCacheRef and ckvCacheRef are inconsistent, block_size must satisfy both block_size%32=0 and block_size%16=0.
      - block_num is the number of memory blocks written to the cache. Its size depends on the user input use case and has no explicit limit.
      - The rotary position encoding (RoPE) parameters (cos, sin) must satisfy the following shape constraints:
        - The shape can be 4-dimensional [Bkv,N,Skv,Dk] or [Bkv,N,1,Dk].
        - The shapes of cos and sin must be the same.
      - The quantization parameters (kRopeScaleOptional, kRopeOffsetOptional, ckvScaleOptional, cKvOffsetOptional) must satisfy the following shape constraints:
        - The number of dimensions and the N-axis size (if present) must be the same for all quantization parameters.
        - The valid shape constraints of quantization parameters vary by use case:

          | Scenario Type | Quantization Parameter Shape |
          | :------: | :------ |
          |V1|<li>The supported shapes of kRopeScaleOptional and kRopeOffsetOptional are [1, Dk], [Dk,], and [1,].</li><li>The supported shapes of ckvScaleOptional and cKvOffsetOptional are [1, Dv], [Dv,], and [1,].</li>|
          |V2|<li>The supported shapes of kRopeScaleOptional and kRopeOffsetOptional are [N, Dk].</li><li>The supported shapes of ckvScaleOptional and cKvOffsetOptional are [N, Dv].</li>|

          - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: For the V1 use case, [1,] is not supported. For the V2 use case, broadcasting of quantization parameter items is not supported.

      - Input tensors do not support empty tensors.
      - The inputs do not support invalid values, including but not limited to ±inf and nan.

  - Supported data types of cache:

    - Non-quantization mode: The type of cache must be the same as that of kv.
      - <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, and <term>Atlas A2 products</term>: BFLOAT16 and FLOAT16 are supported.
      - <term>Kirin X90/Kirin 9030 processors</term>: Only FLOAT16 is supported.
    - Quantization mode:
      - <term>Ascend 950PR&950DT products</term>: INT8, HIFLOAT8, FLOAT8E5M2, and FLOAT8E4M3FN are supported.
      - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Kirin X90/Kirin 9030 processor products</term>: only INT8 is supported.

  - Parameters:

    - The data types of kv, gamma, cos, sin, and vOptional in the input parameters must be exactly the same.
    - kCacheRef and ckvCacheRef are <b>in-place update parameters</b>. Their data types depend on the corresponding input components and the corresponding scale and offset. See details below:

      | cache_type | offset==None | offset!=None |
      | :-------------------------: | :-----------: | :----------------------------------------------: |
      | scale==None | Same as kv | Invalid input, intercepted |
      | scale!=None | INT8, HIFLOAT8, FLOAT8E5M2, FLOAT8E4M3FN | INT8, HIFLOAT8, FLOAT8E5M2, FLOAT8E4M3FN |

      - In non-quantization mode, the quantization parameters (kRopeScaleOptional, kRopeOffsetOptional, ckvScaleOptional, cKvOffsetOptional) must be set to None, and the dtype of kCacheRef and ckvCacheRef must be the same as that of kv.
      - In quantization mode, the dtype of kCacheRef and ckvCacheRef must be the data type supported by the corresponding product.

    - The mapping between the input components and the quantization factor scale and quantization offset scale is as follows:

      | kv component | Component input source | Corresponding scale input | Corresponding offset input | Corresponding output cache |
      | :------: | :------: | :------: | :------: | :------: |
      | k component | In the V1 use case, it corresponds to kv[..., Dv:].<br>In the V2 use case, it corresponds to kv[..., :]. | kRopeScaleOptional | kRopeOffsetOptional | kCacheRef |
      | v component | In the V1 use case, it corresponds to kv[..., :Dv].<br>In the V2 use case, it corresponds to vOptional[..., :]. |ckvScaleOptional | cKvOffsetOptional | ckvCacheRef |

      - kCacheRef: The quantization coefficients are kRopeScaleOptional and kRopeOffsetOptional.
      - ckvCacheRef: the corresponding quantization coefficients are ckvScaleOptional and cKvOffsetOptional.

    - In the output parameters, the data types of k_rope and c_kv must be the same as that of kv.

  - Quantization mode constraints:

    <table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
      <col style="width: 400px">
      <col style="width: 160px">
      <col style="width: 448px">
      </colgroup>
      <thead>
        <tr>
          <th>Quantization mode description<br>(scale and offset inputs are in an "AND" relationship)</th>
          <th>Supported cache modes</th>
          <th>Availability</th>
        </tr></thead>
      <tbody>
      <tr>
          <td rowspan="5">No quantization mode<li>scale input: kRopeScaleOptional==None && ckvScaleOptional==None </li><li>offset input: kRopeOffsetOptional==None && cKvOffsetOptional==None</li></td>
          <td rowspan="1">Norm</td>
          <td rowspan="5">Supported for both V1 and V2.</td>
      </tr>
      <tr>
          <td>PA/PA_BNSD</td>
      </tr>
      <tr>
          <td>PA_NZ</td>
      </tr>
      <tr>
          <td>PA_BLK_BNSD</td>
      </tr>
      <tr>
          <td>PA_BLK_NZ</td>
      </tr>
      <tr>
          <td rowspan="5">Static quantization mode<li>scale input: At least one of kRopeScaleOptional and ckvScaleOptional must be non-null.</li><li>offset input: Valid only when the corresponding scale is non-null. If the corresponding offset is null, <b>static symmetric quantization</b> is used; if the corresponding offset is non-null, <b>static asymmetric quantization</b> is used.</li></td>
          <td rowspan="1">Norm</td>
          <td rowspan="5"><li>Support for <b>static symmetric quantization</b> and <b>static asymmetric quantization</b> differs.</li><li>K and V can independently select different quantization modes.</li></td>
      </tr>
      <tr>
          <td>PA/PA_BNSD</td>
      </tr>
      <tr>
          <td>PA_NZ</td>
      </tr>
      <tr>
          <td>PA_BLK_BNSD</td>
      </tr>
      <tr>
          <td>PA_BLK_NZ</td>
      </tr>
      </tbody>
    </table>

    - Static quantization mode support details:
      - Ascend 950PR&950DT products: Only the V1 use case is supported, with <b>static symmetric quantization</b> and <b>static asymmetric quantization</b> supported.
      - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
        - V1 use case: For cachemodes other than `Norm`, only <b>static symmetric quantization</b> is supported. Even if a valid offset is passed in, it is not processed by the operator, and <b>static symmetric quantization</b> still applies. For the `Norm` cachemode, no static quantization is supported.
        - V2 use case: For all cachemodes, <b>static symmetric quantization</b> and <b>static asymmetric quantization</b> are supported.

  - Constraints related to cache and index:

    | cachemode| kCacheRef Shape | ckvCacheRef Shape | index Shape | Description |
    | :------: | :------: | :------: | :------: | :------ |
    |Norm|[Bkv, N, Scache, Dk]|[Bkv, N, Scache, Dv]|[Bkv, Skv]|KV-Cache update mode. index indicates the offset of each batch.<br>The value range of index is [-1,Scache). Values can be duplicated across different Bkv.<br>$Scache \ge Skv$|
    |PA/PA_BNSD|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × Skv]|PagedAttention mode. index indicates the offset of each token.<br>The value range of index is [-1,block_num * block_size). Values must not be duplicated.<br>$block\_size>1,\\ block\_num \ge Floor(Skv / block\_size) * Bkv$|
    |PA_NZ|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × Skv]|PagedAttention mode where the cache data format is FRACTAL_NZ. index indicates the offset of each token.<br>The value range of index is [-1,block_num * block_size). Values must not be duplicated.<br>$block\_size>1,\\ block\_num \ge Floor(Skv / block\_size) * Bkv$|
    |PA_BLK_BNSD|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × ceil(Skv / block_size)]|Special PagedAttention mode. index indicates the start offset of each block (not one-to-one mapped to tokens).<br>The value range of index is [-1,block_num * block_size). value/block_size must not be duplicated.<br>$block\_size>1,\\ block\_num \ge Floor(Skv / block\_size) * Bkv$|
    |PA_BLK_NZ|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × ceil(Skv / block_size)]|Special PagedAttention mode where the cache data format is FRACTAL_NZ. index indicates the start offset of each block.<br>The value range of index is [-1,block_num * block_size). value/block_size must not be duplicated.<br>$block\_size>1,\\ block\_num \ge Floor(Skv / block\_size) * Bkv$|

    - Scache is the sequence length of the input cache. Its size is determined by the user input use case, with no explicit limit.
    - When cacheModeOptional is Norm, the shape is 2D [Bkv, Skv], and the value range of index must be [-1, Scache). Values can be duplicated across different Bkv.

    - When cacheModeOptional is PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ, the data layout in the cache is as follows:
      - In non-quantization mode: kCacheRef is [block_num, Dk//16, block_size, 1, 16]; ckvCacheRef is [block_num, Dv//16, block_size, 1, 16].
      - In static quantization mode: kCacheRef is [block_num, Dk//32, block_size, 1, 32]; ckvCacheRef is [block_num, Dv//32, block_size, 1, 32].
    - When cacheModeOptional is PA_BNSD or PA_NZ, the shape is 1-dimensional [Bkv *Skv]. The values of index must be in the range [-1, block_num* block_size) and cannot be duplicated.
    - When cacheModeOptional is PA_BLK_BNSD or PA_BLK_NZ, the shape is 1-dimensional [Bkv *ceil_div(Skv,block_size)]. The values of index must be in the range [-1, block_num* block_size). The values of value/block_size cannot be duplicated.

  - isOutputKv constraints:
    - Outputs the intermediate processing results of a specific use case, enabling the two outputs kRopeOut and cKvOut.
    - Valid when cacheModeOptional is PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ.
    - When cacheModeOptional is Norm, valid only in the V2 use case when the quantization mode is enabled.

  - Input combination constraints:

    - Items in this constraint apply only to <b>Atlas A3 products</b> and <b>Atlas A2 products</b>.
    - Under all cacheModeOptional modes, `non-broadcast mode` is always supported: that is, the `rotary position encoding (RoPE) parameters` and `quantization parameters` whose three dimensions `[B, N, S]` strictly match those of `kv`.
    - The valid input shape modes supported under each cacheModeOptional are as follows:

      <table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
        <col style="width: 160px">
        <col style="width: 280px">
        <col style="width: 280px">
        <col style="width: 320px">
        </colgroup>
        <thead>
          <tr>
            <th>cache mode</th>
            <th>kv Shape</th>
            <th>Rotary position encoding (RoPE) parameter<br>Shape</th>
            <th>Other parameters</th>
          </tr>
          </thead>
        <tbody>
        <tr>
            <td>Norm</td>
            <td rowspan="6"><li>V1 use case: [Bkv, 1, Skv, Dv+Dk] (kv)</li><li>V2 use case: <ul><li>[Bkv, N, Skv, Dk] (kv)</li><li>[Bkv, N, Skv, Dv] (vOptional)</li></ul></li></td>
            <td rowspan="2"><li>No broadcast on the S axis is supported: Srope=Skv.</li><li>Broadcast on the S axis is supported: Srope=1.</li></td>
            <td>In <b>no quantization mode</b>: isOutputKv, kRopeOut, and cKvOut are invalid. <li>V1 use case: <ul><li>Only <b>no quantization mode</b> is supported. All quantization parameter items are invalid.</li></ul></li><li>V2 use case: <ul><li><b>No quantization mode</b> and <b>static quantization mode</b> are supported.</li><li>The quantization parameter items for k and v must be 2D [N, D].</li></ul></li></td>
        </tr>
        <tr>
            <td>PA</td>
            <td>In <b>no quantization mode</b>, both broadcast and no broadcast on the S axis are supported. <li>In <b>static quantization mode</b>, only no broadcast on the S axis is supported. Otherwise, the input is invalid.</li><li>Quantization parameter items: <ul><li>V1 use case: Supports a shape of 2D [1,D] or 1D [D]. Only <b>static symmetric quantization</b> is supported.</li><li>V2 use case: Supports a shape of 2D [N,D]. Both <b>static symmetric quantization</b> and <b>static asymmetric quantization</b> are supported.</li></ul></li></td>
        </tr>
        <tr>
            <td>PA_BNSD</td>
            <td rowspan="4"><li>Broadcasting along the S axis is not supported: Srope=Skv must be satisfied.</li></td>
            <td rowspan="4"><li>Non-quantization mode is supported.</li><li>Quantization parameters:<ul><li>V1 use case: Supports the 2D [1,D] or 1D [D] shape. Only <b>static symmetric quantization</b> is supported.</li><li>V2 use case: Supports the 2D [N,D] shape. Both <b>static symmetric quantization</b> and <b>static asymmetric quantization</b> are supported.</li></ul></li></td>
        </tr>
        <tr>
            <td>PA_NZ</td>
        </tr>
        <tr>
            <td>PA_BLK_BNSD</td>
        </tr>
        <tr>
            <td>PA_BLK_NZ</td>
        </tr>
        </tbody>
      </table>

    - <b>Input combinations not covered in the preceding table will cause undefined behavior of the operator</b>.

  - vOptional:
    - This parameter is available only for the aclnnKvRmsNormRopeCacheV2 API. The aclnnKvRmsNormRopeCache API does not support this parameter!
    - This parameter is available only for <b>Atlas A3 products</b> and <b>Atlas A2 products</b>.
      - This parameter is mandatory only in the <b>KV separation use case (V2)</b>. In other types, it is ignored as an invalid parameter.
      - If vOptional exists, its type must be the same as kv, and its `[B, N, S]` dimensions must also be the same as kv.

    - Ascend 950PR&950DT products: This parameter is not checked, but the actual function does not support it and does not process it.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kv_rms_norm_rope_cache_v2.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclFloat16> resultData(size, aclFloatToFloat16(0.0f));
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  auto batchKv = shape[0];
  auto batchSeq = shape[2];
  // Print by [B, S]
  uint64_t bIdx = 0;
  uint64_t sIdx = 0;
  for (uint64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%lu] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    // Fast diagonal traverse
    bIdx = (bIdx >= batchKv) ? batchKv : (bIdx + 1);
    sIdx = (sIdx >= batchSeq) ? batchSeq : (sIdx + 1);
    i += bIdx * batchKv * batchSeq + sIdx * batchSeq;
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate: resource initialization.
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
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. Boilerplate: device/stream initialization. For details, see the acl API manual.
  // Fill in deviceId based on your actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output according to the API definition.
  // This example demonstrates the V1 use case (kv merged mode, with nullptr passed to vOptional). It can run on all products that support this operator.
  // The V2 use case (kv separated mode) is supported only on Atlas A2 products and Atlas A3 products. A non-null vOptional must be passed. In this case:
  // N: 1/2/4/8; kv shape: [Bkv, N, Skv, 192]; gamma shape: [192]; vOptional shape: [Bkv, N, Skv, 128];
  // kCacheRef shape: [Bkv, N, Scache, 192]; ckvCacheRef shape: [Bkv, N, Scache, 128];
  // kRopeOut shape: [Bkv, N, Skv, 192]; cKvOut shape: [Bkv, N, Skv, 128].
  uint64_t totalBatch = 32;
  uint64_t hDimV = 512;    // Tail axis length Dv of the rms_norm computation in the V1 use case
  uint64_t hDimRope = 64;  // Tail axis length Dk of the Rope computation
  uint64_t hDimKv = hDimV + hDimRope;  // In the V1 use case, the tail axis length of kv must satisfy Dkv = Dv + Dk
  std::vector<int64_t> kvShape = {totalBatch, 1, 1, hDimKv};
  std::vector<int64_t> gammaShape = {hDimV,};
  std::vector<int64_t> cosShape = {totalBatch, 1, 1, hDimRope};
  std::vector<int64_t> sinShape = {totalBatch, 1, 1, hDimRope};
  std::vector<int64_t> indexShape = {totalBatch, 1};  // In Norm mode, the shape of index is 2D [Bkv, Skv].
  std::vector<int64_t> kpeCacheShape = {totalBatch, 1, 1, hDimRope};
  std::vector<int64_t> ckvCacheShape = {totalBatch, 1, 1, hDimV};
  std::vector<int64_t> kRopeShape = {totalBatch, 1, 1, hDimRope};
  std::vector<int64_t> cKvShape = {totalBatch, 1, 1, hDimV};

  uint64_t totalEle = totalBatch * 1 * 1;  // In this example, N=1 and Skv=1, so totalEle = Bkv * N * Skv.
  std::vector<int16_t> kvHostData(totalEle * hDimKv, 0);
  std::vector<int16_t> gammaHostData(hDimV, 0);
  std::vector<int16_t> cosHostData(totalEle * hDimRope, 0);
  std::vector<int16_t> sinHostData(totalEle * hDimRope, 0);
  std::vector<int64_t> indexHostData(totalEle, 0);  // The value range of index is [-1, Scache). A value of -1 indicates that the update is skipped.
  std::vector<int16_t> kpeCacheHostData(totalEle * hDimRope, 0);
  std::vector<int16_t> ckvCacheHostData(totalEle * hDimV, 0);
  std::vector<int16_t> kRopeHostData(totalEle * hDimRope, 0);
  std::vector<int16_t> cKvHostData(totalEle * hDimV, 0);

  void* kvDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* cosDeviceAddr = nullptr;
  void* sinDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* kpeCacheDeviceAddr = nullptr;
  void* ckvCacheDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* cKvDeviceAddr = nullptr;

  aclTensor* kv = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* cos = nullptr;
  aclTensor* sin = nullptr;
  aclTensor* index = nullptr;
  aclTensor* kpeCache = nullptr;
  aclTensor* ckvCache = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* cKv = nullptr;

  double epsilon = 1e-5;
  char cacheMode[] = "Norm";
  bool isOutputKv = false;

  ret = CreateAclTensor(kvHostData, kvShape, &kvDeviceAddr, aclDataType::ACL_FLOAT16, &kv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kpeCacheHostData, kpeCacheShape, &kpeCacheDeviceAddr, aclDataType::ACL_FLOAT16, &kpeCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(ckvCacheHostData, ckvCacheShape, &ckvCacheDeviceAddr, aclDataType::ACL_FLOAT16, &ckvCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cKvHostData, cKvShape, &cKvDeviceAddr, aclDataType::ACL_FLOAT16, &cKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnKvRmsNormRopeCacheV2. Pass nullptr to vOptional to indicate the V1 use case.
  ret = aclnnKvRmsNormRopeCacheV2GetWorkspaceSize(kv,gamma,cos,sin,index,
                                                kpeCache,ckvCache,nullptr,nullptr,nullptr,nullptr,nullptr,epsilon,cacheMode,isOutputKv,kRope,cKv,&workspaceSize,&executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvRmsNormRopeCacheV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnKvRmsNormRopeCacheV2.
  ret = aclnnKvRmsNormRopeCacheV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvRmsNormRopeCacheV2 failed. ERROR: %d\n", ret); return ret);

  // 4. Boilerplate. Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values. Copy the result from device memory to host memory. Modify the code according to the definition of the specific API as required.
  PrintOutResult(kpeCacheShape, &kpeCacheDeviceAddr);
  PrintOutResult(ckvCacheShape, &ckvCacheDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code according to the definition of the specific API as required.
  aclDestroyTensor(kv);
  aclDestroyTensor(gamma);
  aclDestroyTensor(cos);
  aclDestroyTensor(sin);
  aclDestroyTensor(index);
  aclDestroyTensor(kpeCache);
  aclDestroyTensor(ckvCache);
  aclDestroyTensor(kRope);
  aclDestroyTensor(cKv);

  // 7. Release device resources.
  aclrtFree(kvDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(cosDeviceAddr);
  aclrtFree(sinDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(kpeCacheDeviceAddr);
  aclrtFree(ckvCacheDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(cKvDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
