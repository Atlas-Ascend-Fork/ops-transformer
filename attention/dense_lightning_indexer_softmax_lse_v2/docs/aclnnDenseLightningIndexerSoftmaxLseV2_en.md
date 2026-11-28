# aclnnDenseLightningIndexerSoftmaxLseV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:41:37.846Z -->

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

- Description: The DenseLightningIndexerSoftmaxLseV2 operator is a branch operator used to compute the Softmax input for the DenseLightningIndexerGradKlLoss operator. Compared with aclnnDenseLightningIndexerSoftmaxLse, it adds support for Compressed Attention and supports split-core load balancing through the metadata preceding operator.

  The main computation process is as follows:
  1. Computes the attention score of query_index and key_index.
  2. Performs a weighted sum of attention score based on weight.
  3. Performs Softmax computation on the weighted result and outputs the log-sum-exp (LSE) result.

- Formula:

  $$
  \text{res}=\text{AttentionMask}\left(\text{ReduceSum}\left(W\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right)\right)
  $$

  $$
  \text{lse}=\text{ReduceMax}\left(\text{res}\right)+\text{log}\left(\text{ReduceSum}\left(\text{exp}\left(\text{res}-\text{ReduceMax}\left(\text{res}\right)\right)\right)\right)
  $$

  The lse is passed as the output to the DenseLightningIndexerGradKlLoss operator as input for Softmax computation.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the computation process, and then call the "aclnnDenseLightningIndexerSoftmaxLseV2" API to perform the computation.

```c++
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize(
    const aclTensor  *queryIndex,
    const aclTensor  *keyIndex,
    const aclTensor  *weight,
    const aclTensor  *cuSeqLensQOptional,
    const aclTensor  *cuSeqLensKOptional,
    const aclTensor  *seqUsedQOptional,
    const aclTensor  *seqUsedKOptional,
    const aclTensor  *cmpResidualKOptional,
    const aclTensor  *metadataOptional,
    const char       *layoutQ,
    const char       *layoutK,
    int64_t           maskMode,
    int64_t           cmpRatio,
    const aclTensor  *softmaxLseOut,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor);
```

```c++
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1550px">
    <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 300px">
            <col style="width: 400px">
            <col style="width: 212px">
            <col style="width: 100px">
            <col style="width: 190px">
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
    </tr></thead>
    <tbody>
     <tr>
      <td>queryIndex (aclTensor*)</td>
      <td>Input</td>
      <td>Input queryIndex of the lightningIndexer structure.</td>
      <td><ul><li>B: supports generalization and must be consistent with the B of keyIndex.</li><li>N1: 1-128, and must be divisible by N2.</li><li>D: 128.</li><li>T1: accumulation of S1 across multiple batches.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(B,S1,N1,D);(T1,N1,D)</td>
      <td>√</td>
     </tr>
     <tr>
      <td>keyIndex (aclTensor*)</td>
      <td>Input</td>
      <td>Input keyIndex of the lightningIndexer structure.</td>
      <td><ul><li>B: support generalization and remain consistent with B of queryIndex.</li><li>S2: support generalization.</li><li>N2: 1.</li><li>D: 128, and remain consistent with D of queryIndex.</li><li>T2: accumulation of S2 across multiple batches.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(B,S2,N2,D);(T2,N2,D)</td>
      <td>√</td>
     </tr>
     <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>Weight tensor.</td>
      <td><ul><li>B: support generalization and remain consistent with B of queryIndex.</li><li>S1: support generalization and remain consistent with S1 of queryIndex.</li><li>N1: remain consistent with N1 of queryIndex.</li><li>T1: accumulation of S1 across multiple batches.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(B,S1,N1);(T1,N1)</td>
      <td>√</td>
     </tr>
     <tr>
      <td>cuSeqLensQOptional (aclTensor*)</td>
      <td>Optional input</td>
      <td>Accumulated sum of the valid token counts of q in the current batch and preceding batches.</td>
      <td>For TND, pass B+1 cumulative token counts starting with 0. For BSND, this parameter may be omitted (nullptr).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B+1,)</td>
      <td>√</td>
     </tr>
     <tr>
      <td>cuSeqLensKOptional (aclTensor*)</td>
      <td>Optional input</td>
      <td>Accumulation of the number of valid tokens of k in the current batch and preceding batches.</td>
      <td>For TND, pass B+1 cumulative token counts starting with 0. For BSND, this parameter may be omitted (nullptr).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B+1,)</td>
      <td>√</td>
     </tr>
     <tr>
      <td>seqUsedQOptional (aclTensor*)</td>
      <td>Optional input</td>
      <td>Actual used length of q in different batches.</td>
      <td><ul><li>Supports empty Tensor. When this input is present, each batch participates in the computation based on the actual value of seqused_q.</li><li>The value of seqused_q must not be greater than the actual seqlen_q of each batch.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B,)</td>
      <td>√</td>
     </tr>
     <tr>
      <td>seqUsedKOptional (aclTensor*)</td>
      <td>Optional input</td>
      <td>Actual used length of k in different batches.</td>
      <td><ul><li>Supports empty Tensor. When this input exists, each batch participates in computation based on the actual value of seqused_k.</li><li>The value of seqused_k is not greater than the actual seqlen_k of each batch.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B,)</td>
      <td>√</td>
     </tr>
     <tr>
      <td>cmpResidualKOptional (aclTensor*)</td>
      <td>Optional input</td>
      <td>Indicates the residual of the sequence length of k related to cmpRatio.</td>
      <td><ul><li>Supports empty Tensor.</li><li>Must be passed in when maskMode=3 and cmpRatio>1.</li><li>Each value of cmp_residual_k must be less than cmpRatio.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B,)</td>
      <td>√</td>
     </tr>
     <tr>
      <td>metadataOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Per-core load balancing information output by the preceding AICPU operator.</td>
      <td><ul><li>Must be passed in. Output by the aclnnDenseLightningIndexerSoftmaxLseV2Metadata operator to reduce host array access in the tiling phase.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D. Pass the output tensor of <a href="../../dense_lightning_indexer_softmax_lse_v2_metadata/docs/aclnnDenseLightningIndexerSoftmaxLseV2Metadata_en.md#Metadata">aclnnDenseLightningIndexerSoftmaxLseV2Metadata</a> directly.</td>
      <td>√</td>
     </tr>
     <tr>
      <td>layoutQ (char*)</td>
      <td>Input</td>
      <td>Indicates the layout format of the query.</td>
      <td>Supports BSND and TND. The default value is BSND if a null pointer is passed in. layoutQ and layoutK must be consistent.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
      <td>layoutK (char*)</td>
      <td>Input</td>
      <td>Indicates the layout format of the key side.</td>
      <td>Supports BSND and TND. Defaults to BSND when a null pointer is passed in. layoutK must be consistent with layoutQ.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
      <td>maskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the mask mode.</td>
      <td><ul><li>0: No mask.</li><li>3: mask in rightDownCausal mode, corresponding to the lower-triangular use case with the right vertex as the division point.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Indicates the compression ratio of the key.</td>
      <td>Value range: [1, 128], indicating no compression.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
      <td>softmaxLseOut(aclTensor*)</td>
      <td>Output</td>
      <td>LSE (log-sum-exp) values used in the softmax computation.</td>
      <td><ul><li>B: support generalization, and must remain consistent with the B of queryIndex.</li><li>N2: number of heads of key, currently fixed to 1.</li><li>S1: support generalization, and must remain consistent with the S1 of queryIndex.</li><li>T1: accumulation of S1 across multiple batches.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(B,N2,S1);(N2,T1)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>executor (aclOpExecutor**)</td>
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

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed;width: 1155px">
        <colgroup>
            <col style="width: 319px">
            <col style="width: 144px">
            <col style="width: 671px">
        </colgroup>
        <thead>
            <tr>
                <th>Return Value</th>
                <th>Error Code</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>queryIndex, keyIndex, weight, or softmaxLseOut is nullptr.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The input data type is not a supported type.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_INNER_TILING_ERROR</td>
                <td>561002</td>
                <td>Tiling fails if layoutQ and layoutK differ or metadataOptional is nullptr.</td>
            </tr>
        </tbody>
    </table>

## aclnnDenseLightningIndexerSoftmaxLseV2

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 184px">
    <col style="width: 134px">
    <col style="width: 833px">
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
      <td>Workspace memory address allocated on the Device side.</td>
     </tr>
     <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize.</td>
     </tr>
     <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the computation flow of the operator.</td>
     </tr>
     <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
     </tr>
    </tbody>
    </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - Deterministic computation: aclnnDenseLightningIndexerSoftmaxLseV2 defaults to a deterministic implementation.

  - General Constraints
    - Handling of empty input parameters:
        - If queryIndex is an empty tensor, the API returns directly.

    <table style="undefined;table-layout: fixed; width: 901px"><colgroup>
    <col style="width: 168px">
    <col style="width: 565px">
    <col style="width: 168px">
    </colgroup>
    <thead>
     <tr>
      <th>maskMode</th>
      <th>Meaning</th>
      <th>Remarks</th>
     </tr>
    </thead>
    <tbody>
     <tr>
      <td>0</td>
      <td>No mask, meaning no mask operation is performed.</td>
      <td>Supported</td>
     </tr>
     <tr>
      <td>3</td>
      <td>Mask in rightDownCausal mode, corresponding to the lower-triangular use case divided by the top-right vertex.</td>
      <td>Supported</td>
     </tr>
    </tbody>
    </table>

  - Specification constraints

    <table style="undefined;table-layout: fixed; width: 909px"><colgroup>
    <col style="width: 125px">
    <col style="width: 182px">
    <col style="width: 602px">
    </colgroup>
    <thead>
    <tr>
      <th>Specification Item</th>
      <th>Value</th>
      <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td>B</td>
      <td>1-256</td>
      <td>-</td>
    </tr>
    <tr>
      <td>S1, S2</td>
      <td>0-128K</td>
      <td>S1 and S2 support different lengths.</td>
    </tr>
    <tr>
      <td>N1</td>
      <td>1-128</td>
      <td>Must be divisible by N2.</td>
    </tr>
    <tr>
      <td>N2</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>D</td>
      <td>128</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>BSND/TND</td>
      <td>layoutQ and layoutK must be consistent.</td>
    </tr>
    <tr>
      <td>cmpRatio</td>
      <td>1-128</td>
      <td>1 indicates no compression.</td>
    </tr>
    <tr>
      <td>maskMode</td>
      <td>0, 3</td>
      <td>0=No mask, 3=rightDownCausal.</td>
    </tr>
    </tbody>
    </table>

  - Special constraints
    - In BSND use cases, information about batch_size and max_seqlen_q (indicated by the tensor shape) must be passed in.
    - In TND use cases, cuSeqLensQ and cuSeqLensK must be passed in.
    - When maskMode=3 and cmpRatio>1, cmpResidualK must be passed in, and each value of cmpResidualK must be less than cmpRatio.
    - Each value of seqUsedQ must not be greater than the actual seqlen_q of each batch.
    - Each value of seqUsedK must not be greater than the actual seqlen_k of each batch.
    - metadataOptional is the output of the preceding operator aclnnDenseLightningIndexerSoftmaxLseV2Metadata and must be passed in to execute the related load balancing policy.

## Example

The sample code for the API call is as follows, for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse_v2.h"
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse_v2_metadata.h"

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
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateContext(context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetCurrentContext(*context);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // BSND layout: q[B,S1,N1,D] k[B,S2,N2,D] w[B,S1,N1] lse[B,N2,S1]
  int64_t B = 2;
  int64_t S1 = 128;
  int64_t S2 = 256;
  int64_t N1 = 8;
  int64_t N2 = 1;
  int64_t D = 128;

  std::vector<int64_t> qIndexShape = {B, S1, N1, D};
  std::vector<int64_t> kIndexShape = {B, S2, N2, D};
  std::vector<int64_t> weightShape = {B, S1, N1};
  std::vector<int64_t> softmaxLseShape = {B, N2, S1};
  // In this BSND example, B * S1 = 256, so metadata requires 2184 INT32 elements.
  const int64_t metadataLength = 136 + 8 * B * S1;
  std::vector<int64_t> metadataShape = {metadataLength};

  void* qIndexDeviceAddr = nullptr;
  void* kIndexDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* softmaxLseDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;

  aclTensor* qIndex = nullptr;
  aclTensor* kIndex = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* softmaxLse = nullptr;
  aclTensor* metadata = nullptr;

  std::vector<aclFloat16> qIndexHostData(B * S1 * N1 * D, aclFloatToFloat16(0.2));
  std::vector<aclFloat16> kIndexHostData(B * S2 * 1 * D, aclFloatToFloat16(0.1));
  std::vector<float> weightHostData(B * S1 * N1, 0.005f);
  std::vector<float> softmaxLseHostData(B * N2 * S1, 0.0f);
  std::vector<int32_t> metadataHostData(metadataLength, 0);

  ret = CreateAclTensor(qIndexHostData, qIndexShape, &qIndexDeviceAddr, aclDataType::ACL_FLOAT16, &qIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kIndexHostData, kIndexShape, &kIndexDeviceAddr, aclDataType::ACL_FLOAT16, &kIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr,
      aclDataType::ACL_FLOAT, &softmaxLse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // BSND layout, maskMode=0 (defaultMask), cmpRatio=1
  constexpr const char layoutQStr[] = "BSND";
  constexpr const char layoutKStr[] = "BSND";
  constexpr size_t layoutQLen = sizeof(layoutQStr);
  constexpr size_t layoutKLen = sizeof(layoutKStr);
  char layoutQ[layoutQLen];
  char layoutK[layoutKLen];
  errno_t memcpyRet = memcpy_s(layoutQ, sizeof(layoutQ), layoutQStr, layoutQLen);
  if (memcpyRet != 0) {
    LOG_PRINT("memcpy_s layoutQ failed. ERROR: %d\n", memcpyRet);
    return -1;
  }
  memcpyRet = memcpy_s(layoutK, sizeof(layoutK), layoutKStr, layoutKLen);
  if (memcpyRet != 0) {
    LOG_PRINT("memcpy_s layoutK failed. ERROR: %d\n", memcpyRet);
    return -1;
  }
  int64_t maskMode = 0;
  int64_t cmpRatio = 1;

  // 1. Call the preceding metadata operator to generate core-splitting load balancing information.
  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;
  ret = aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize(
      nullptr, nullptr, nullptr, nullptr, nullptr, B, S1, S2, N1, N2, D, layoutQ, layoutK, maskMode, cmpRatio,
      metadata, &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnDenseLightningIndexerSoftmaxLseV2Metadata(metadataWorkspaceAddr, metadataWorkspaceSize,
                                                        metadataExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseV2Metadata failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 2. Call the main operator.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize(
            qIndex, kIndex, weight,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            metadata,
            layoutQ, layoutK, maskMode, cmpRatio,
            softmaxLse,
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnDenseLightningIndexerSoftmaxLseV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseV2 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  PrintOutResult(softmaxLseShape, &softmaxLseDeviceAddr);
  LOG_PRINT("pass\n");

  aclDestroyTensor(qIndex);
  aclDestroyTensor(kIndex);
  aclDestroyTensor(weight);
  aclDestroyTensor(softmaxLse);
  aclDestroyTensor(metadata);

  aclrtFree(qIndexDeviceAddr);
  aclrtFree(kIndexDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(softmaxLseDeviceAddr);
  aclrtFree(metadataDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  if (metadataWorkspaceSize > 0) {
    aclrtFree(metadataWorkspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
