# aclnnInplacePartialRotaryMul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:44:05.048Z -->

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

- Description: Performs the Inplace computation of single-path rotary position encoding, directly modifying the input tensor xRef without generating a new output tensor.
- Formula:

    (1) interleave mode (rotary_mode = 1):

    $$
    x1 = x[..., ::2].view(-1, 1)
    $$

    $$
    x2 = x[..., 1::2].view(-1, 1)
    $$

    $$
    x\_rotate = torch.cat((-x2, x1), dim=-1).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    $$

    $$
    x = x * cos + x\_rotate * sin
    $$

- partialSlice parameter description: Supports partial rotary position encoding on the last dimension of the input tensor. The `partialSlice` parameter specifies the [start, end) range. Rotary position encoding is performed only on the data in this range.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnInplacePartialRotaryMulGetWorkspaceSize" API to obtain input parameters and calculate the required workspace size according to the procedure, and then call the "aclnnInplacePartialRotaryMul" API to perform the computation.

```c++
aclnnStatus aclnnInplacePartialRotaryMulGetWorkspaceSize(
  const aclTensor   *xRef,
  const aclTensor   *cos,
  const aclTensor   *sin,
  int64_t           rotary_mode,
  const aclIntArray *partialSlice,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```c++
aclnnStatus aclnnInplacePartialRotaryMul(
  void          *workspace,
  uint64_t      workspaceSize,
  aclOpExecutor *executor,
  aclrtStream   stream)
```

## aclnnInplacePartialRotaryMulGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1461px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 332px">
  <col style="width: 169px">
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
      <th>Data type</th>
      <th>Data format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>xRef</td>
      <td>Input</td>
      <td>Tensor to which rotary position encoding is applied, corresponding to x in the formula. Inplace mode: xRef also serves as the output and stores the result.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>Input</td>
      <td>Position encoding tensor, corresponding to cos in the formula.</td>
      <td>Same data type as xRef, or FLOAT32.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>Input</td>
      <td>Position encoding tensor, which is sin in the formula.</td>
      <td>Same data type as xRef, or FLOAT32</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>rotary_mode</td>
      <td>Input</td>
      <td>Rotary mode.</td>
      <td>0 indicates half mode, 1 indicates interleave mode. Currently, only interleave mode is supported.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>partialSlice</td>
      <td>Input</td>
      <td>Slice range [start, end) for partial rotation, applied to the last dimension.</td>
      <td>If not passed, rotary encoding is applied to the entire D dimension by default. If start equals end, no rotary encoding is performed.</td>
      <td>INT64 array</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be requested on the Device side.</td>
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

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table>
  <tr>
  <td align="center" style="width:169px;">Return Value</td>
  <td align="center" style="width:125px;">Error Code</td>
  <td align="center" style="width:855px;">Description</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">The passed-in xRef, cos, or sin is a null pointer.</td>
  </tr>
  <tr>
  <td rowspan="4" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="4" align="left">161002</td>
  <td align="left">The data type of the passed-in xRef, cos, or sin is not in the supported range, or the data types of cos and sin are inconsistent, or the precision combination does not meet the requirement.</td>
  </tr>
  <tr><td align="left">The passed-in rotary_mode is not 1 (only the interleave mode is supported).</td></tr>
  <tr><td align="left">The passed-in partialSlice length is not 2, or the value range is invalid.</td></tr>
  <tr><td align="left">The shapes of the passed-in xRef, cos, and sin do not meet the constraints (the number of dimensions is not 4, or the shapes of cos and sin are inconsistent, or the size of the last dimension of xRef exceeds 1024, or the last dimension of xRef is not a multiple of 2, or the length of the partialSlice slice is not a multiple of 2).</td></tr>
  </table>

## aclnnInplacePartialRotaryMul

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 849px">
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
      <td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnInplacePartialRotaryMulGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor. It contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnInplacePartialRotaryMul defaults to a deterministic implementation.
- Non-contiguous tensors are not supported.
- Only the interleave mode (rotary_mode = 1) is supported.
- Inplace execution: the input xRef and the output share the same tensor, and the computation result is written back to the input xRef directly.
- The shape of the input tensor xRef is in BSND layout. The shape constraints of each parameter are as follows:
  - The size of the last dimension (D) of the input tensor xRef does not exceed 1024.
  - In interleave mode (rotary_mode = 1), the size of the last dimension (D) of xRef must be a multiple of 2. The partialSlice slice length (that is, partialSlice[1] - partialSlice[0]) must be a multiple of 2.
  - The size of the last dimension of the input tensors cos and sin must be the same and must equal the partialSlice slice length (that is, partialSlice[1] - partialSlice[0]).
  - The shape of cos/sin must satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with xRef. The following constraints apply:

    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT products</term>: Currently, only BSND, B1ND, B11D, and 111D layouts are supported for the shape of cos/sin.
    <!-- end id7 -->
    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: Currently, only BS1D and B11D layouts are supported for the shape of cos/sin, which requires the B axis to be the same.
    <!-- end id8 -->

  - partialSlice constraints: sliceStart ≥ 0, sliceEnd ≥ 0, sliceEnd ≤ the size of the last dimension (D) of xRef, and sliceLength = sliceEnd - sliceStart >= 0. When sliceEnd is equal to sliceStart, no rotary position encoding is performed and the function returns directly.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_inplace_partial_rotary_mul.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
    // 1. Boilerplate. Initialize the device and stream. For details, see the acl API manual.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input based on the API definition.
    std::vector<int64_t> xShape = {96, 1, 1, 512};
    std::vector<int64_t> cosShape = {96, 1, 1, 64};
    std::vector<int64_t> sinShape = {96, 1, 1, 64};
    int64_t rotary_mode = 1;
    std::vector<int64_t> partialSlice = {448, 512};

    void* xDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;

    // Create host data (example data)
    std::vector<float> xHostData = { /* fill with your data */ };
    std::vector<float> cosHostData = { /* fill with your data */ };
    std::vector<float> sinHostData = { /* fill with your data */ };

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray* partialSliceArray = aclCreateIntArray(partialSlice.data(), partialSlice.size());

    // 3. Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnInplacePartialRotaryMul.
    ret = aclnnInplacePartialRotaryMulGetWorkspaceSize(x, cos, sin, rotary_mode, partialSliceArray,
                                                        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnInplacePartialRotaryMul.
    ret = aclnnInplacePartialRotaryMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InplacePartialRotaryMul failed. ERROR: %d\n", ret); return ret);

    // 4. Boilerplate. Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Cleanup
    aclDestroyIntArray(partialSliceArray);
    aclDestroyTensor(x);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclrtFree(xDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
