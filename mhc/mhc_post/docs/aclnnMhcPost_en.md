# aclnnMhcPost

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:45:05.795Z -->

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

- Description: MhcPost performs Post Mapping on the output $h_{t}^{out}$ of the previous layer in the mHC architecture based on a series of computations, performs Res Mapping on the layer input $x_l$ of the previous layer, and then performs residual connection on the two to obtain the layer input $x_{l+1}$ of the next layer.

- Formula:

  When hRes is provided:

  $$
  x_{l+1} = (H_{l}^{res})^{T} \times x_l + h_{l}^{out} \otimes H_{t}^{post}
  $$

  When hRes is not provided (supported only on Ascend 950PR&950DT products):

  $$
  x_{l+1} = x_l + h_{l}^{out} \otimes H_{t}^{post}
  $$

## Function Prototype

The operator execution APIs are [two-phase APIs](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnMhcPostGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnMhcPost" API to execute the computation.

```c++
aclnnStatus aclnnMhcPostGetWorkspaceSize(
    const aclTensor  *x,
    const aclTensor  *hRes,
    const aclTensor  *hOut,
    const aclTensor  *hPost,
    aclTensor        *out,
    uint64_t         *workspaceSize,
    aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnMhcPost(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnMhcPostGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1470px"><colgroup>
  <col style="width: 145px">
  <col style="width: 120px">
  <col style="width: 441px">
  <col style="width: 158px">
  <col style="width: 186px">
  <col style="width: 120px">
  <col style="width: 155px">
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
      <td>x</td>
      <td>Input</td>
      <td>Tensor to be computed, representing the mHC layer input data in the network.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[B,S,N,D], [T,N,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hRes</td>
      <td>Input (optional)</td>
      <td>hRes transformation matrix of mHC, which is a doubly stochastic matrix after the Sinkhorn transformation. Passing nullptr degrades it to a direct residual connection (supported only on Ascend 950PR&950DT products).</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,N,N], [T,N,N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hOut</td>
      <td>Input</td>
      <td>Output of the Attention/MLP layer.</td>
      <td>Same data type as x.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[B,S,D], [T,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hPost</td>
      <td>Input</td>
      <td>hPost transformation matrix of mHC.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,N], [T,N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output data of the mHC layer in the network, serving as the layer input of the next layer.</td>
      <td>The data type is the same as that of x.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[B,S,N,D], [T,N,D]</td>
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

  <table style="undefined;table-layout: fixed;width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
    </colgroup>
      <thead>
          <th>Return Value</th>
          <th>Error Code</th>
          <th>Description</th>
      </thead>
      <tbody>
          <tr>
              <td>ACLNN_ERR_PARAM_NULLPTR</td>
              <td>161001</td>
              <td>x, hOut, hPost, or out is a null pointer.</td>
          </tr>
          <tr>
              <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
              <td rowspan="3">161002</td>
              <td>The data type of x, hRes (when non-empty), hOut, hPost, or out is not within the supported range.</td>
          </tr>
            <tr>
              <td>The shape dimensions of x, hRes (when non-empty), hOut, hPost, or out are not within the supported range.</td>
          </tr>
          <tr>
              <td>The data type or shape of x, hRes (when non-empty), hOut, hPost, or out does not match.</td>
          </tr>
      </tbody>
  </table>

## aclnnMhcPost

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 969px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnMhcPostGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation process.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing the task.</td>
      </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- aclnnMhcPost defaults to a deterministic implementation.
- After parameter validation succeeds, if any of x, hOut, or hPost is an empty tensor, or if hRes is non-null and is an empty tensor, the two-phase API calls return success, workspaceSize is 0, no computation task is generated, and out is not written. The caller must skip consuming the output.

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: hRes supports passing in nullptr. In this case, it degrades to a direct residual connection.
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: hRes is a required parameter and does not support passing in nullptr.

  | Specification Item | Value              | Specification                          |
  | :------- | :----------------- | :------------------------------------- |
  | n        | 4                  | Fixed to 4.                            |
  | d        | Greater than 0 and less than 100000  | Multiple of 128.                       |

<!-- end id8 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_post.h"
#include "securec.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
    // Fixed writing method, resource initialization.
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to request device side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    // Call aclrtMemcpy to copy host side data to device side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    // Calculate the strides of continuous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor interface to create aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

} // namespace

int main() {
    // 1. (Fixed writing method)  device/stream initialization. Refer to the API's external interface list.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
    // Example: BSND format (B, S, n, D)
    std::vector<int64_t> xShape = {1, 1024, 4, 5120};   // BSND
    std::vector<int64_t> hResShape = {1, 1024, 4, 4};   // BSNN
    std::vector<int64_t> hOutShape = {1, 1024, 5120};   // BSD
    std::vector<int64_t> hPostShape = {1, 1024, 4};     // BSN
    std::vector<int64_t> outShape = {1, 1024, 4, 5120};  // BSND

    void *xDeviceAddr = nullptr;
    void *hResDeviceAddr = nullptr;
    void *hOutDeviceAddr = nullptr;
    void *hPostDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;

    aclTensor *xTensor = nullptr;
    aclTensor *hResTensor = nullptr;
    aclTensor *hOutTensor = nullptr;
    aclTensor *hPostTensor = nullptr;
    aclTensor *outTensor = nullptr;

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t hResShapeSize = GetShapeSize(hResShape);
    int64_t hOutShapeSize = GetShapeSize(hOutShape);
    int64_t hPostShapeSize = GetShapeSize(hPostShape);
    int64_t outShapeSize = GetShapeSize(outShape);

    std::vector<aclFloat16> xHostData(xShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> hResHostData(hResShapeSize, 1.0f);
    std::vector<aclFloat16> hOutHostData(hOutShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> hPostHostData(hPostShapeSize, 1.0f);
    std::vector<aclFloat16> outHostData(outShapeSize, aclFloatToFloat16(0.0f));

    // Create x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &xTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hRes aclTensor.
    ret = CreateAclTensor(hResHostData, hResShape, &hResDeviceAddr, aclDataType::ACL_FLOAT, &hResTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hOut aclTensor.
    ret = CreateAclTensor(hOutHostData, hOutShape, &hOutDeviceAddr, aclDataType::ACL_FLOAT16, &hOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hPost aclTensor.
    ret = CreateAclTensor(hPostHostData, hPostShape, &hPostDeviceAddr, aclDataType::ACL_FLOAT, &hPostTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // 3. Call CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first interface.
    ret = aclnnMhcPostGetWorkspaceSize(
        xTensor, hResTensor, hOutTensor, hPostTensor, outTensor,
        &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPostGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    // Allocate device memory based on the workspaceSize calculated from the first interface paragraph.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }
    // Call the second interface.
    ret = aclnnMhcPost(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPost failed. ERROR: %d\n", ret);
        return ret;
    }

    // 4. (Fixed writing method) Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    // 5. Retrieve the output value, copy the result from the device side memory to the host side.
    auto size = GetShapeSize(outShape);
    std::vector<aclFloat16> resultData(size, aclFloatToFloat16(0.0f));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    // 6. Release resources.
    aclDestroyTensor(xTensor);
    aclDestroyTensor(hResTensor);
    aclDestroyTensor(hOutTensor);
    aclDestroyTensor(hPostTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(xDeviceAddr);
    aclrtFree(hResDeviceAddr);
    aclrtFree(hOutDeviceAddr);
    aclrtFree(hPostDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
