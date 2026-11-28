# aclnnBlockSparseAttention

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:11:28.084Z -->

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

- **Description**: BlockSparseAttention performs sparse attention computation, supporting flexible block-level sparse patterns. It uses blockSparseMaskOptional to specify the KV blocks selected for each Q block, enabling efficient sparse attention computation.

- **Formula**: Sparse block size: $blockShapeX \times blockShapeY$, blockSparseMaskOptional specifies the sparse pattern

  $$
  attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T + atten\_mask) \cdot value_{sparse}
  $$

  Here, `scale` is the scaling factor controlled by `scaleValue`. `atten_mask` is the attention mask corresponding to `attenMaskOptional`, which currently must be `nullptr`.

  The data arrangement format of the query, key, and value inputs for BlockSparseAttention supports interpretation from multiple dimension arrangements, which can be passed in through qInputLayout and kvInputLayout.

  - B: Indicates the batch size of the input samples (Batch).
  - T: Indicates the length of B and S concatenated along the axis and tightly packed (Total tokens).
  - S: Indicates the sequence length of the input samples (Seq-Length).
  - H: Indicates the size of the hidden layer (Head-Size).
  - N: Number of heads (Head-Num).
  - D: Minimum unit size of the hidden layer, satisfying D = H/N (Head-Dim).

  Currently supported layouts:

  - qInputLayout: "TND" "BNSD" "BSND"
  - kvInputLayout: "TND" "BNSD" "BSND"

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnBlockSparseAttentionGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the computation process, and then call the "aclnnBlockSparseAttention" API to perform the computation.

```c++
aclnnStatus aclnnBlockSparseAttentionGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *blockSparseMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *blockShapeOptional,
  const aclIntArray *actualSeqLengthsOptional,
  const aclIntArray *actualSeqLengthsKvOptional,
  const aclTensor   *blockTableOptional,
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
aclnnStatus aclnnBlockSparseAttention(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnBlockSparseAttentionGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px">
  <colgroup>
  <col style="width: 269px">
  <col style="width: 132px">
  <col style="width: 318px">
  <col style="width: 236px">
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
      <td>query</td>
      <td>Input</td>
      <td>The query in the formula.</td>
      <td>Supported shapes:
        <ul><li>TND: [totalQTokens, headNum, headDim].</li>
        <li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li>
        <li>BSND: [batch, maxQSeqLength, headNum, headDim].</li></ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>The key in the formula.</td>
      <td>The supported shapes are:
        <ul>
          <li>TND: [totalKTokens, numKeyValueHeads, headDim].</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li>
          <li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim].</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>value in the formula.</td>
      <td>        The supported shapes are:
        <ul>
          <li>TND: [totalVTokens, numKeyValueHeads, headDim].</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li>
          <li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim].</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockSparseMaskOptional</td>
      <td>Input</td>
      <td>Indicates the actual sparse pattern.</td>
      <td>        Optional input (required in the current version)
        <ul>
          <li>The shape is [batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)].</li>
          <li>Indicates which blocks need to participate in the computation (1) and which blocks do not (0) after division by block.</li>
          <li>If nullptr is passed in, block sparse computation is disabled, and the attention scores between all tokens are computed.</li>
        </ul>
      </td>
      <td>INT8</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Input</td>
      <td>atten_mask in the formula.</td>
      <td>The atten_mask takes effect together with the sparse pattern. Currently not supported; must be set to nullptr.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
    <tr>
      <td>blockShapeOptional</td>
      <td>Input</td>
      <td>Array of sparse block shapes.</td>
      <td>        Used together with blockSparseMaskOptional:
        <ul>
          <li>When blockSparseMaskOptional is configured: if this input is configured, the operator obtains the sparse block size from it; if this input is not configured, the operator defaults the sparse block size to [128,128].</li>
          <li>When blockSparseMaskOptional is not configured: the operator ignores this input regardless of its configuration.</li>
        </ul>
        If configured, this input must contain two elements: [blockShapeX, blockShapeY].
        <ul>
          <li>blockShapeX: Block size along the Q direction. The value must be greater than 0.</li>
          <li>blockShapeY: Block size along the KV direction. The value must be greater than 0. It must be a multiple of 16 on <term>Ascend 950PR&950DT products</term>, and a multiple of 128 on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
      <td>actualSeqLengthsOptional</td>
      <td>Input</td>
      <td>Describes the query sequence length for each batch.</td>
      <td>        Optional input for variable-length sequences:
        <ul>
          <li>When qInputLayout is "TND": this input must be configured.</li>
          <li>When qInputLayout is "BNSD" or "BSND": if this input is configured, the operator processes sequences based on the actual sequence lengths specified by this input; if this input is not configured (nullptr is passed), the operator processes sequences based on S in the query shape.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKvOptional</td>
      <td>Input</td>
      <td>Describes the key/value sequence length of each batch.</td>
      <td>        Optional input for variable-length sequences:
        <ul>
          <li>When kvInputLayout is "TND": this input must be configured.</li>
          <li>When kvInputLayout is "BNSD" or "BSND": if this input is configured, the operator processes sequences based on the actual sequence lengths specified by this input; if this input is not configured (nullptr is passed), the operator processes sequences based on S in the key/value shape.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>Input</td>
      <td>Block table used for PagedAttention.</td>
      <td>Currently not supported. Pass in nullptr.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>qInputLayout</td>
      <td>Input</td>
      <td>Data layout format of the input query.</td>
      <td>Currently, only "TND", "BNSD", and "BSND" are supported. qInputLayout must be the same as kvInputLayout.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kvInputLayout</td>
      <td>Input</td>
      <td>Represents the data layout format of the input key and value.</td>
      <td>Currently, only "TND", "BNSD", and "BSND" are supported. qInputLayout must be the same as kvInputLayout.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKeyValueHeads</td>
      <td>Input</td>
      <td>Represents the number of key/value heads.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>Input</td>
      <td>Specifies the mask type used in attention computation.</td>
      <td>        Currently, only 0 is supported.
        <ul>
          <li>0: no mask is applied</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>Scale in the formula, representing the scaling factor.</td>
      <td>Generally set to D^-0.5.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise</td>
      <td>Input</td>
      <td>Precision level used for Softmax computation.</td>
      <td>        Controls the data types used for operations in the online softmax and rescale stages. Currently, only 0, 1, or 4 can be passed. <term>Ascend 950PR&950DT products</term> supports only 4. <term>Atlas A2 products</term> and <term>Atlas A3 products</term> support only 0 or 1.
        <ul>
          <li>0: both online softmax and rescale use the fp32 data type. Suitable for use cases that require high computation precision.</li>
          <li>1: can be configured only when the input query, key, and value are all of the fp16 data type. Both online softmax and rescale use the fp16 data type, which delivers better performance but lower precision. Numerical overflow may occur during computation. Determine whether to use it based on the value range.</li>
          <li>4: mixed-precision computation, which balances performance and precision. Online softmax uses the fp16/bf16 data type (the same as the data types of query, key, and value), and rescale uses the fp32 data type. Numerical overflow may occur in the online softmax stage.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize</td>
      <td>Input</td>
      <td>Block size of PagedAttention.</td>
      <td>Used in PagedAttention use cases. PagedAttention is currently not supported, so only 0 can be passed.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>Input</td>
      <td>Number of tokens that the sliding window needs to include forward in the sliding window attention use case.</td>
      <td>Used for the sliding window attention use case. Currently, sliding window attention is not supported. Only 2147483647 can be passed.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>Input</td>
      <td>Number of backward tokens included in the sliding window in the sliding window attention use case.</td>
      <td>Used for the sliding window attention use case. Currently, sliding window attention is not supported. Only 2147483647 can be passed.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxLseFlag</td>
      <td>Input</td>
      <td>Flag indicating whether the softmaxLse output is enabled.</td>
      <td>        Only 0 or 1 is supported.
        <ul>
          <li>0: softmaxLse is not output.</li>
          <li>1: softmaxLse is output. A performance loss may occur compared with not outputting softmaxLse.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>Output</td>
      <td>attentionOut in the formula.</td>
      <td>Data type and shape are the same as those of query.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3/4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxLseOptional</td>
      <td>Output</td>
      <td>Intermediate log-sum-exp result of the Softmax computation.</td>
      <td>        The supported shapes vary with the query shape:
        <ul>
          <li>query is "TND": [totalQTokens, headNum, 1].</li>
          <li>query is "BNSD": [batch, headNum, maxQSeqLength, 1].</li>
          <li>query is "BSND": [batch, maxQSeqLength, headNum, 1].</li>
        </ul>
      </td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3/4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
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
      <td>The input query, key, or value is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of query, key, and value are out of the supported range.</td>
    </tr>
    <tr>
      <td>qInputLayout or kvInputLayout is invalid.</td>
    </tr>
    <tr>
      <td>blockShape is invalid (the number of elements is not equal to 2, or the value is less than or equal to 0).</td>
    </tr>
    <tr>
      <td>innerPrecise is invalid (must be 0, 1, or 4).</td>
    </tr>
  </tbody>
  </table>

## aclnnBlockSparseAttention

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
      <td>Address of the workspace memory allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnBlockSparseAttentionGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnBlockSparseAttention defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- qInputLayout supports only "TND", "BNSD", and "BSND".
- kvInputLayout supports only "TND", "BNSD", and "BSND".
- Currently, the InputLayout of query, key, and value must be consistent.
- The data types of the input query, key, and value must be consistent, supporting FLOAT16 and BFLOAT16.
- The D axis of query, key, and value currently only supports being configured as 64 or 128.
- If blockShapeOptional is passed in, it must contain two elements [blockShapeX, blockShapeY], both greater than 0. blockShapeY must be a multiple of 16 on <term>Ascend 950PR&950DT products</term>, and a multiple of 128 on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>.
- blockSparseMaskOptional currently must be passed in, and its shape must be [batch, headNum, ceilDiv(maxQS, blockShapeX), ceilDiv(maxKVS, blockShapeY)].
- attentionMaskOptional currently only supports passing nullptr.
- actualSeqLengthsOptional is mandatory when qInputLayout is "TND"; actualSeqLengthsKvOptional is mandatory when kvInputLayout is "TND".
- actualSeqLengthsOptional and actualSeqLengthsKvOptional currently must be configured together or not configured together. Configuring only one of them will be intercepted by the operator.
- blockTableOptional currently only supports passing nullptr, which indicates that the PagedAttention feature is not enabled.

<!-- npu="950,A3,910b" id7 -->
- innerPrecise must be 0, 1, or 4. <term>Ascend 950PR&950DT products</term> supports only 4, and <term>Atlas A2 products</term> and <term>Atlas A3 products</term> support only 0 or 1.
<!-- end id7 -->
- softmaxLseFlag supports only 0 or 1, indicating disabling/enabling the softmaxLse output respectively.
- qSeqlen and kvSeqlen do not need to be divisible by blockShape. Non-aligned scenarios are supported, and the actual number of blocks is calculated by rounding up.
- Let the headNum of the input query be N1 and the headNum of the input key and value be N2. Then N1 >= N2 && N1 % N2 == 0.
- maskType currently only supports input 0, which indicates that no mask is applied.
- blockSize currently only supports input 0, which indicates that paged cache is not supported.
- preTokens and nextTokens currently only support passing in 2147483647, which indicates that all tokens before and after the current token participate in the attention computation, i.e., sliding window attention is not supported.

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
    // Boilerplate: initialize resources.
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

    // Check whether the hostData size matches.
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

    // Call aclrtMemcpy to copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              return ret);

    // Compute the strides of the contiguous tensor.
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
                  void *blockSparseMaskDeviceAddr, void *attentionOutDeviceAddr, void *workspaceAddr,
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
    // 1. (Boilerplate) Initialize the device/stream.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Set the parameters.
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
    int32_t blockShapeY = 128;

    // Calculate dimensions for the TND format.
    int64_t totalQTokens = batch * qSeqlen;
    int64_t totalKvTokens = batch * kvSeqlen;
    int32_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;  // Number of Q blocks in the X dimension
    int32_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY;  // Number of KV blocks in the Y dimension

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *blockSparseMaskTensor = nullptr;
    aclTensor *attentionOutTensor = nullptr;
    aclIntArray *actualSeqLengths = nullptr;
    aclIntArray *actualSeqLengthsKv = nullptr;
    aclIntArray *blockShape = nullptr;

    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *blockSparseMaskDeviceAddr = nullptr;
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

    // 9. Prepare string parameters (ensure the buffer is large enough to include the null terminator).
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

    ret = aclnnBlockSparseAttentionGetWorkspaceSize(
        queryTensor,           // query
        keyTensor,             // key
        valueTensor,           // value
        blockSparseMaskTensor, // blockSparseMask
        nullptr,               // attenMaskOptional
        blockShape,            // blockShape
        actualSeqLengths,      // actualSeqLengthsOptional
        actualSeqLengthsKv,    // actualSeqLengthsKvOptional
        nullptr,               // blockTableOptional
        qLayoutBuffer,         // qInputLayout
        kvLayoutBuffer,        // kvInputLayout
        numKvHeads,            // numKeyValueHeads
        0,                     // maskType
        scaleValue,            // scaleValue
        innerPrecise,          // innerPrecise
        0,                     // blockSize
        2147483647,            // preTokens
        2147483647,            // nextTokens
        0,                     // softmaxLseFlag
        attentionOutTensor,    // attentionOut
        nullptr,               // softmaxLseOptional
        &workspaceSize,        // workspaceSize (out)
        &executor);            // executor (out)

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
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
                      FreeResource(
                      queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor, actualSeqLengths,
                      actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr, valueDeviceAddr,
                      blockSparseMaskDeviceAddr, attentionOutDeviceAddr, workspaceAddr, deviceId, &stream);
                  return ret);
    }

    // 12. Call the second-phase API.
    ret = aclnnBlockSparseAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttention failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 13. Wait synchronously for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           workspaceAddr, deviceId, &stream);
              return ret);

    // 14. Obtain the output value and copy the result from the device memory to the host.
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

    // 15. Print some of the results.
    uint64_t printNum = 10;
    LOG_PRINT("attentionOut results (first %lu elements):\n", printNum);
    for (uint64_t i = 0; i < printNum && i < resultData.size(); i++) {
        LOG_PRINT("  index %lu: %f\n", i, static_cast<float>(resultData[i]));
    }

    // 16. Release resources.
    FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor, actualSeqLengths,
                 actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr, valueDeviceAddr,
                 blockSparseMaskDeviceAddr, attentionOutDeviceAddr, workspaceAddr, deviceId, &stream);
    LOG_PRINT("Test completed successfully!\n");
    return 0;
}
```
