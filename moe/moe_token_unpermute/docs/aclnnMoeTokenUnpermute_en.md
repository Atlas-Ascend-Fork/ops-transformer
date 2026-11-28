# aclnnMoeTokenUnpermute

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:33:39.998Z -->

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

- Description: Retrieves the input data stored in permutedTokens based on the indices stored in sortedIndices. If probs data exists, permutedTokens is multiplied by probs. Finally, cumulative summation is performed and the computation result is output.
- Formula:

    - When probs is not None, the formula is as follows:

      $$
      T[k] = T[S[k]]
      $$

      $$
      T[k] = T[k] * P[i][j]
      $$

      $$
      O[i] = \sum_{k=i*topK}^{(i+1)*topK - 1 } T[k]
      $$

      Where $i \in {0,1,...,tokens-1}$; $j \in {0,1,...,topK-1}$; $k \in {0,1,...,tokens*topK-1}$; T denotes permutedTokens; S denotes sortedIndices; P denotes probs; O denotes out; topK denotes topK_num; tokens denotes tokens_num.

    - When probs is None, topK_num = 1, and the formula is as follows:

      $$
      T[i] = T[S[i]]
      $$

      $$
      O[i] = T[i]
      $$

      Where $i \in {0,1,...,tokens-1}$; T denotes permutedTokens; S denotes sortedIndices; O denotes out; and tokens denotes tokens_num.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMoeTokenUnpermuteGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the "aclnnMoeTokenUnpermute" API to perform the computation.

```c++
aclnnStatus aclnnMoeTokenUnpermuteGetWorkspaceSize(
    const aclTensor   *permutedTokens,
    const aclTensor   *sortedIndices,
    const aclTensor   *probsOptional,
    bool               paddedMode,
    const aclIntArray *restoreShapeOptional,
    aclTensor         *out,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnMoeTokenUnpermute(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMoeTokenUnpermuteGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 550px">
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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>permutedTokens (aclTensor)</td>
      <td>Input</td>
      <td>Input tokens, corresponding to T in the formula.</td>
      <td>Shape: (tokens_num * topK_num, hidden_size), where tokens_num indicates the number of input tokens, topK_num indicates the number of experts processing each token, and hidden_size indicates the length of the vector representation of each token.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sortedIndices (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the positions of the data to be computed in permutedTokens.</td>
      <td><ul><li>The shape is (tokens_num * topK_num).</li><li>The value range is [0, tokens_num * topK_num - 1], with no duplicate indices.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>probsOptional (aclTensor)</td>
      <td>Input</td>
      <td>P in the formula.</td>
      <td><ul><li>If probs is specified, topK_num equals the second dimension of probs. If probs is not specified, topK_num=1.</li><li>The shape is (tokens_num, topK_num).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>paddedMode (bool)</td>
      <td>Input</td>
      <td>Specifies whether to enable paddedMode.</td>
      <td><ul><li>When paddedMode is true, restoreShapeOptional takes effect; otherwise, it is not operated on.</li><li>Currently, only false is supported.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional (aclIntArray)</td>
      <td>Input</td>
      <td>Shape of the output result when paddedMode is true.</td>
      <td><ul><li>When paddedMode is true, restoreShapeOptional takes effect, and the shape of out is represented as restoreShapeOptional.</li><li>Currently, only nullptr is supported.</li></ul></td>
      <td>aclIntArray*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor)</td>
      <td>Output</td>
      <td>Output result.</td>
      <td><ul><li>When paddedMode=false, the shape is (tokens_num, hidden_size); when paddedMode=true, the shape is the same as that of restoreShapeOptional.</li><li>Same data type as permutedTokens.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Returns the workspace size that the user needs to allocate on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td> ACLNN_ERR_PARAM_NULLPTR </td>
      <td> 161001 </td>
      <td>The mandatory input, mandatory output, or mandatory attribute is a null pointer.</td>
      </tr>
      <tr>
      <td> ACLNN_ERR_PARAM_INVALID </td>
      <td> 161002 </td>
      <td>The data type and data format of inputs and outputs are not within the supported range.</td>
      </tr>
      <tr>
      <td rowspan="2"> ACLNN_ERR_INNER_TILING_ERROR </td>
      <td rowspan="2"> 561002 </td>
      <td>The shape information of multiple input tensors does not match.</td>
      </tr>
      <tr>
      <td>The shape information between input attributes and input tensors does not match.</td>
      </tr>
  </tbody>
  </table>

## aclnnMoeTokenUnpermute

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1030px"> <colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>Workspace size allocated on the Device side, obtained by the first-phase API <code>aclnnMoeTokenUnpermuteGetWorkspaceSize</code>.</td>
  </tr>
  <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation process.</td>
  </tr>
  <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream where tasks are executed.</td>
  </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeTokenUnpermute defaults to a deterministic implementation.

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - `topK_num <= 512`.
  - If a value in `probs` is zero, the operator skips the corresponding token in `permutedTokens`.
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>:
  When this API is called, the framework internally calls [aclnnMoeFinalizeRoutingV2](../../moe_finalize_routing_v2/docs/aclnnMoeFinalizeRoutingV2_en.md). If a parameter error message is displayed, see the following parameter mapping:
  - The permutedTokens input is equivalent to the expandedX input of the aclnnMoeFinalizeRoutingV2 API.
  - The sortedIndices input is equivalent to the expandedRowIdx input of the aclnnMoeFinalizeRoutingV2 API.
  - The probsOptional input is equivalent to the scalesOptional input of the aclnnMoeFinalizeRoutingV2 API.
  - The paddedMode input is equivalent to the dropPadMode input of the aclnnMoeFinalizeRoutingV2 API.
  - The out output is equivalent to the out output of the aclnnMoeFinalizeRoutingV2 API.
<!-- end id8 -->
<!-- npu="310p" id9 -->
- |<term>Atlas inference products</term>:
  - The supported data types for permutedTokens and probsOptional are FLOAT16 and FLOAT32.
  - topK_num <= 512.
  - hidden_size is a multiple of 128 and less than 10240.

<!-- end id9 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute.h"
#include <iostream>
#include <vector>
#include <cstdio>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // Boilerplate: resource initialization.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // Call aclrtMemcpy to copy host-side data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct the input and output. Customize the construction based on the API.

  std::vector<float> permutedTokensData = {1, 2, 3, 4};
  std::vector<int64_t> permutedTokensShape = {2, 2};
  void *permutedTokensAddr = nullptr;
  aclTensor *permutedTokens = nullptr;

  ret = CreateAclTensor(permutedTokensData, permutedTokensShape,
                        &permutedTokensAddr, aclDataType::ACL_FLOAT,
                        &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int> sortedIndicesData = {0,1};
  std::vector<int64_t> sortedIndicesShape = {2};
  void *sortedIndicesAddr = nullptr;
  aclTensor *sortedIndices = nullptr;

  ret =
      CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> probsOptionalData = {1, 1};
  std::vector<int64_t> probsOptionalShape = {1, 2};
  void *probsOptionalAddr = nullptr;
  aclTensor *probsOptional = nullptr;

  ret =
      CreateAclTensor(probsOptionalData, probsOptionalShape, &probsOptionalAddr,
                      aclDataType::ACL_FLOAT, &probsOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> outData = {0, 0};
  std::vector<int64_t> outShape = {1, 2};
  void *outAddr = nullptr;
  aclTensor *out = nullptr;

  ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT,
                        &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the aclnnMoeTokenUnpermute first-phase API.
  ret = aclnnMoeTokenUnpermuteGetWorkspaceSize(permutedTokens, sortedIndices,
                                               probsOptional, false, nullptr,
                                               out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteGetWorkspaceSize failed. ERROR: %d\n",
                ret);
      return ret);

  // Allocate device memory using the workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the aclnnMoeTokenUnpermute second-phase API.
  ret = aclnnMoeTokenUnpermute(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermute failed. ERROR: %d\n", ret);
            return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value. Copy the result from the device side to the host side. Modify the code based on the actual API definition.
  PrintOutResult(outShape, &outAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the actual API definition.
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(probsOptional);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(permutedTokensAddr);
  aclrtFree(sortedIndicesAddr);
  aclrtFree(probsOptionalAddr);
  aclrtFree(outAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
