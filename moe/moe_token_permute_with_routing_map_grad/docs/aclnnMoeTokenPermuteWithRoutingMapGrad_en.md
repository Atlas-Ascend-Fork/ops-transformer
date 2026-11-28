# aclnnMoeTokenPermuteWithRoutingMapGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:35:22.451Z -->

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

- **Description**: Backward propagation of aclnnMoeTokenPermuteWithRoutingMap.
- **Formula**:

    The following formulas describe nonempty inputs. `tokensGradOut` is computed independently of whether the probability gradient is provided. First, initialize `tokensGradOut` to zero:

    $$
    tokensGradOut= zeros((tokensNum, hidden_size), dtype=permutedTokenOutputGrad.dtype, device=permutedTokenOutputGrad.device)
    $$

    - When `dropAndPad` is false, `topK_num = permutedTokenOutputGrad.size(0) / tokensNum`. For each element of `sortedIndices` with `sortedIndices[i] >= 0`:

    $$
    tokensGradOut[i / topK\_num] += permutedTokenOutputGrad[sortedIndices[i]]
    $$

    Here, `i / topK_num` uses integer division.

    - When `dropAndPad` is true, the values of `sortedIndices` are token indices. For each element:

    $$
    tokensGradOut[sortedIndices[i]] += permutedTokenOutputGrad[i]
    $$

    - When both `permutedProbsOutputGradOptional` and `probsGradOutOptional` are non-null, compute the probability gradient:

    $$
    probsGradOutOptional = zeros(tokensNum, numExperts)
    $$

    - When dropAndPad is true, `capacity = permutedTokenOutputGrad.size(0) / numExperts`:

    $$
    probsGradOutOptional [sortedIndices[i], i/capacity] = permutedProbsOutputGradOptional[i]
    $$

    - When dropAndPad is false:

    $$
    probsGradOutOptional = transpose(maskedscatter(transpose(probsGradOutOptional), transpose(routingMapOptional), permutedProbsOutputGradOptional))
    $$
    Here, transpose exchanges the token and expert dimensions, and maskedscatter fills values in expert-major order.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation flow, and then call the "aclnnMoeTokenPermuteWithRoutingMapGrad" API to perform the computation.

```c++
aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize(
    const aclTensor *permutedTokenOutputGrad,
    const aclTensor *permutedProbsOutputGradOptional,
    const aclTensor *sortedIndices,
    const aclTensor *routingMapOptional,
    int64_t          numExperts,
    int64_t          tokensNum,
    bool             dropAndPad,
    aclTensor       *tokensGradOut,
    aclTensor       *probsGradOutOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGrad(
    void                *workspace,
    uint64_t             workspaceSize,
    aclOpExecutor       *executor,
    const aclrtStream    stream)
```

## aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 187px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 387px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 187px">
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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
  </tr></thead>
  <tbody>
  <tr>
      <td>permutedTokenOutputGrad</td>
      <td>Input</td>
      <td>Gradient of the forward output permutedTokens.</td>
      <td>shape supports 2D dimensions; empty tensor is not supported. topK_num indicates the maximum quantity of experts that can be selected for each token, and capacity indicates the quantity of tokens selected by each expert.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>      • Non-dropAndPad mode: (tokensNum * topK_num, hidden_size)<br>
      • dropAndPad mode: (numExperts * capacity, hidden_size)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>permutedProbsOutputGradOptional</td>
      <td>Optional input</td>
      <td>Gradient of the forward output permutedProbs.</td>
      <td>      •  If not passed, probsGradOutOptional does not need to be computed;<br>
      •  The shape is a 1D dimension, where topK_num indicates the upper limit of the quantity of experts that can be selected by each token, and capacity indicates the quantity of tokens selected by each expert.<br>
      •  The data type must be the same as that of permutedTokenOutputGrad, or FLOAT is supported when permutedTokenOutputGrad is BFLOAT16.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>      •  Non-dropAndPad mode: (tokensNum * topK_num)<br>
      •  dropAndPad mode: (numExperts * capacity)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>sortedIndices</td>
      <td>Input</td>
      <td>Sorted index values.</td>
      <td>Non-dropAndPad mode: the index value range is [0, tokensNum * topK_num - 1] or -1, where -1 indicates an invalid slot that does not participate in the computation, and topK_num indicates the upper limit of the quantity of experts selected by each token; dropAndPad mode: indices identify tokens and range from 0 to tokensNum - 1, and capacity indicates the quantity of tokens selected by each expert.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>      • non-dropAndPad mode: (tokensNum * topK_num,)<br>
      • dropAndPad mode: (numExperts * capacity)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>routingMapOptional</td>
      <td>Optional input</td>
      <td>Represents the mapping from token to expert.</td>
      <td>The shape must be a 2D tensor. In non-dropAndPad mode, each row must contain no more than topK true or 1 values.</td>
      <td>INT8, bool (0 and 1 for INT8, true and false for bool)</td>
      <td>ND</td>
      <td>(tokensNum, numExperts)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>numExperts</td>
      <td>Input</td>
      <td>Indicates the number of experts participating in the computation.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
      <tr>
      <td>tokensNum</td>
      <td>Input</td>
      <td>Indicates the number of tokens participating in the computation.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>dropAndPad</td>
      <td>Input</td>
      <td>-</td>
      <td>true indicates that dropAndPad is enabled, and false indicates that dropAndPad is disabled.</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>tokensGradOut</td>
      <td>Output</td>
      <td>Gradient of input permutedTokens.</td>
      <td>Must be a tensor with 2 dimensions.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(tokensNum, hidden_size)</td>
      <td>×</td>
  </tr>
  <tr>
      <td>probsGradOutOptional</td>
      <td>Optional output</td>
      <td>Gradient of the forward operator's input probs.</td>
      <td>The shape supports 2D dimensions.</td>
      <td>Same as permutedProbsOutputGradOptional</td>
      <td>ND</td>
      <td>(tokensNum, numExperts)</td>
      <td>×</td>
  </tr>
  <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be side-applied on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
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

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
      <col style="width: 267px">
      <col style="width: 124px">
      <col style="width: 775px">
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
          <td>The required input, required output, or required attribute passed in is a null pointer.</td>
          </tr>
          <tr>
          <td> ACLNN_ERR_PARAM_INVALID </td>
          <td> 161002 </td>
          <td>              1.The data types and data formats of inputs and outputs are not within the supported range.<br>
              2.The shapes of inputs and outputs are not within the supported range.
          </td>
          </tr>
      </tbody>
  </table>

## aclnnMoeTokenPermuteWithRoutingMapGrad

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
  </colgroup>
      <thead>
          <tr>
          <th>Parameter</th>
          <th>Input/Output</th>
          <th>Description</th>
          </tr>
      </thead>
      <tbody>
          <tr>
          <td>workspace</td>
          <td>Input</td>
          <td>Address of the workspace memory side-applied on the Device.</td>
          </tr>
          <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>Size of the workspace side-applied on the Device, obtained by the first-phase API aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize.</td>
          </tr>
          <tr>
          <td>executor</td>
          <td>Input</td><td> Op executor that contains the operator computation flow.</td>
          </tr>
          <tr>
          <td>stream</td>
          <td>Input</td>
          <td> Stream for task execution.</td>
          </tr>
      </tbody>
  </table>

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeTokenPermuteWithRoutingMapGrad defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtSetSysParamOpt.

- For non-dropAndPad use cases, topK_num <= 512

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "aclnnop/aclnn_moe_token_permute_with_routing_map_grad.h"
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <iomanip>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
bool WriteFile(const std::string &filePath, int64_t size, std::vector<T>& hostData)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        LOG_PRINT("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, reinterpret_cast<char*>(hostData.data()), size * sizeof(T));
    (void)close(fd);
    if (writeSize != size * sizeof(T)) {
        LOG_PRINT("Write file Failed.");
        return false;
    }

    return true;
}
void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate: resource initialization
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
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

int main()
{

    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
    // Set deviceId according to your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;

    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API.

    int64_t num_token = 4096;
    int64_t hidden_size = 7168;
    int64_t num_expert = 256;
    int64_t num_capacity = 16;
    std::vector<float> permuted_output_grad_Data(num_expert * num_capacity * hidden_size, 0);
    std::vector<int64_t> permuted_output_grad_Shape = {num_expert * num_capacity, hidden_size};
    void* permuted_output_grad_Addr = nullptr;
    aclTensor* permuted_output_grad = nullptr;

    ret = CreateAclTensor(permuted_output_grad_Data, permuted_output_grad_Shape, &permuted_output_grad_Addr,
                          aclDataType::ACL_FLOAT, &permuted_output_grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> permutedProbsOutputGradOptional(num_expert * num_capacity, 0.1);
    std::vector<int64_t> permutedProbsOutputGradOptionalShape = {num_expert * num_capacity};
    void* permutedProbsOutputGrad_Addr = nullptr;
    aclTensor* ppermutedProbsOutputGrad = nullptr;
    ret = CreateAclTensor(permutedProbsOutputGradOptional, permutedProbsOutputGradOptionalShape,
                          &permutedProbsOutputGrad_Addr, aclDataType::ACL_FLOAT, &ppermutedProbsOutputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int> sortedIndicesData(num_expert * num_capacity, 0);
    std::vector<int64_t> sortedIndicesShape = {num_expert * num_capacity};
    void* sortedIndicesAddr = nullptr;
    aclTensor* sortedIndices = nullptr;
    ret = CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr, aclDataType::ACL_INT32,
                          &sortedIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<char> routingMapOptionalData(num_token * num_expert, 1);
    std::vector<int64_t> routingMapOptionalShape = {num_token , num_expert};
    void* routingMapOptionalAddr = nullptr;
    aclTensor* proutingMapOptional = nullptr;

    ret = CreateAclTensor(routingMapOptionalData, routingMapOptionalShape, &routingMapOptionalAddr,
                          aclDataType::ACL_INT8, &proutingMapOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outData(num_token * hidden_size, 0.0f);
    std::vector<int64_t> outShape = {num_token, hidden_size};
    // std::vector<int64_t> outShape = {num_token};
    void* outAddr = nullptr;
    aclTensor* out = nullptr;

    ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outData2(num_token * num_expert, 0.0f);
    std::vector<int64_t> outShape2 = {num_token, num_expert};
    void* outAddr2 = nullptr;
    aclTensor* out2 = nullptr;

    ret = CreateAclTensor(outData2, outShape2, &outAddr2, aclDataType::ACL_FLOAT, &out2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnMoeTokenPermuteWithRoutingMapGrad.
    ret = aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize(permuted_output_grad, ppermutedProbsOutputGrad, sortedIndices, proutingMapOptional, num_expert, num_token, true,
                                                                 out, out2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnMoeTokenPermuteWithRoutingMapGrad.
    ret = aclnnMoeTokenPermuteWithRoutingMapGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGrad failed. ERROR: %d\n", ret);
              return ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device side to the host side. Modify it based on the specific API definition.
    PrintOutResult(outShape, &outAddr);
    PrintOutResult(outShape2, &outAddr2);

    // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
    aclDestroyTensor(permuted_output_grad);
    aclDestroyTensor(sortedIndices);
    aclDestroyTensor(out);

    // 7. Release device resources.
    aclrtFree(permuted_output_grad_Addr);
    aclrtFree(sortedIndicesAddr);
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
