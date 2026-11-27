# aclnnGroupedMatmulAdd

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:13:27.017Z -->

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

- Description: Splits x and weight along K, multiplies each pair of groups, and adds each result to the corresponding group of yRef in place.

  - For group $i$, the matrices have shapes $[M,k_i]$ and $[k_i,N]$ after applying transposeX. The group size $k_i$ may vary. Each result has shape $[M,N]$.
- Formula:

  $$
  yRef_i^{out}=x_i\times weight_i + yRef_i^{in}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call "aclnnGroupedMatmulAddGetWorkspaceSize" to obtain the input parameters and calculate the required workspace size based on the computation process. Then, call "aclnnGroupedMatmulAdd" to perform the computation.

```c++
aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *groupList,
    aclTensor       *yRef,
    bool             transposeX,
    bool             transposeWeight,
    int64_t          groupType,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnGroupedMatmulAdd(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream
)
```

## aclnnGroupedMatmulAddGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 300px">
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
        <td>x</td>
        <td>Input</td>
        <td>Input x in the formula.</td>
        <td>The shape of x is [K, M] when transposeX is true and [M, K] when it is false.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Weight in the formula.</td>
        <td>Shape [K, N].</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
    <tr>
      <td>groupList</td>
      <td>Input</td>
      <td>Indicates the matmul size distribution along the grouping axis of the inputs and outputs. Device-side aclTensor type.</td>
      <td>Only the cumulative sum (cumsum) mode is supported.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
        <td>yRef</td>
        <td>Input/Output</td>
        <td>Input/output tensor for in-place accumulation.</td>
        <td>Shape [E, M, N], where E is the number of groups. Atlas A2/A3 also support the 2D layout [E × M, N]. FLOAT16 and BFLOAT16 are supported only on Atlas A2/A3 and must match the data type of x.</td>
        <td>FLOAT32, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2, 3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>transposeX</td>
        <td>attribute</td>
        <td>Indicates whether matrix x is transposed.</td>
        <td>Supports true and false.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeWeight</td>
        <td>Attribute</td>
        <td>Indicates whether the weight matrix is transposed.</td>
        <td>Only false is supported.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupType</td>
        <td>Attribute</td>
        <td>Indicates the grouping type.</td>
        <td>Only 2 (K axis grouping) is supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device.</td>
        <td></td>
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

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:

    - A 2-dimensional shape is not supported for yRef.

  <!-- end id7 -->

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:
    <table style="undefined;table-layout: fixed; width: 1150px"> <colgroup>
    <col style="width: 280px">
    <col style="width: 100px">
    <col style="width: 900px">
      </colgroup><thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>x, weight, groupList, or yRef is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="5">161002</td>
        <td>The data type or data format of x, weight, groupList, or yRef is not within the supported range.</td>
      </tr>
      <tr>
        <td>The data type of x is inconsistent with that of weight.</td>
      </tr>
      <tr>
        <td>The shapes of x, weight, or yRef do not meet the matmul constraint requirements.</td>
      </tr>
      <tr>
        <td>transposeWeight is true or groupType is not 2.</td>
      </tr>
      <tr>
        <td>yRef is non-contiguous, or a 16-bit yRef violates the view offset or N alignment requirements.</td>
      </tr>
    </tbody>
    </table>

## aclnnGroupedMatmulAdd

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1150px"> <colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 900px">
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
        <td>Workspace size allocated on the Device, obtained by calling the first-phase API <code>aclnnGroupedMatmulAddGetWorkspaceSize</code>.</td>
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
    </tbody></table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- aclnnGroupedMatmulAdd defaults to a deterministic implementation.
- After 32-byte alignment, the size of each dimension of each tensor group in x and weight must be less than the maximum INT32 value, 2147483647.
- On Atlas A2/A3, the last dimensions of x and weight must be less than 65536. For x, this is M when transposeX is true and K when it is false. For weight, it is N.
- Supported input types:
  - x is FLOAT16, weight is FLOAT16, and y is FLOAT32.
  - x is BFLOAT16, weight is BFLOAT16, and y is FLOAT32.
  - On Atlas A2/A3, x, weight and y may all be FLOAT16 or all BFLOAT16.
- On Ascend 950PR/950DT, transposeX=false may require workspace to rearrange x. Use the size returned by the first-phase API.
- yRef must be contiguous. For FLOAT16 or BFLOAT16 yRef, its view offset must be a non-negative multiple of 16 elements and N must be a multiple of 16. The matrix product and yRef are added in FLOAT32 before conversion to the data type of yRef.
- On Atlas A2/A3, when transposeX is false or yRef is 16-bit, M, K, N and the group count must fit in a non-negative int32 value. The output element count (M × N × group count) must fit in int64.
- groupList must be a non-negative, monotonically non-decreasing sequence (cumulative sum), and the last value must not be greater than the K dimension of x. Example with K=256 and E=4 (the group sizes are 64, 0, 128, and 64 in sequence): `[64, 64, 192, 256]`
- The first dimension of groupList supports a maximum of 1024, i.e., a maximum of 1024 groups.

## Example

The sample code is as follows, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).
<!-- npu="A3,910b" id8 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:

    ```c++
    #include <iostream>
    #include <vector>
    #include "acl/acl.h"
    #include "aclnnop/aclnn_grouped_matmul_add.h"

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
      // Boilerplate code for resource initialization.
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
    }

    template <typename T>
    int CreateAclTensor_New(const std::vector<int64_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate Device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // Call aclrtMemcpy to copy Host data to Device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensors.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
    }

    template <typename T>
    int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate Device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // Call aclrtMemcpy to copy Host data to Device memory.
      std::vector<T> hostData(GetShapeSize(shape), 0);
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensors.
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
      // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
      // Fill in deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // Handle the check as required.
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct inputs and outputs as required by the API.
      std::vector<int64_t> xShape = {64, 32};
      std::vector<int64_t> weightShape= {64, 32};
      std::vector<int64_t> yShape = {64, 32};
      std::vector<int64_t> groupListShape = {2};
      std::vector<int64_t> groupListData = {32, 64};
      void* xDeviceAddr;
      void* weightDeviceAddr;
      void* yDeviceAddr;

      void* groupListDeviceAddr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* groupedList = nullptr;
      aclTensor* y = nullptr;
      aclTensor* yRef = nullptr;

      bool transpose_x = true;
      bool transpose_weight = false;
      int group_type = 2;

      // Create the x aclTensor
      ret = CreateAclTensor<uint16_t>(xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the weight aclTensor
      ret = CreateAclTensor<uint16_t>(weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the y aclTensor
      ret = CreateAclTensor<float>(yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the group_list aclTensor
      ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      yRef = y;

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // 3. Call the CANN operator library API
      // Call the first-phase API of aclnnGroupedMatmulAdd
      ret = aclnnGroupedMatmulAddGetWorkspaceSize(x, weight, groupedList, yRef, transpose_x, transpose_weight, group_type, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory using the workspaceSize computed by the first-phase API
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API aclnnGroupedMatmulAdd.
      ret = aclnnGroupedMatmulAdd(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulAdd failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronously wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the value of the output and copy the result from Device memory to Host memory. Modify the code based on the specific API definition.
      auto size = GetShapeSize(yShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr,
                            size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t j = 0; j < size; j++) {
          LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
      }


      // 6. Release the aclTensor and aclScalar. Modify the code based on the specific API definition.
      aclDestroyTensor(x);
      aclDestroyTensor(weight);
      aclDestroyTensor(y);

      // 7. Release device resources. Modify the code based on the specific API definition.
      aclrtFree(xDeviceAddr);
      aclrtFree(weightDeviceAddr);
      aclrtFree(yDeviceAddr);

      if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
    }
    ```
<!-- end id8 -->

<!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:

    ```c++
    #include <iostream>
    #include <vector>
    #include "acl/acl.h"
    #include "aclnnop/aclnn_grouped_matmul_add.h"

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
    int CreateAclTensor_New(const std::vector<int64_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate Device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // Call aclrtMemcpy to copy data from the Host to Device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
    }

    template <typename T>
    int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate memory on the Device side.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // Call aclrtMemcpy to copy Host-side data to the Device-side memory.
      std::vector<T> hostData(GetShapeSize(shape), 0);
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
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
      // 1. (boilerplate) device/stream initialization. For details, see the acl API manual.
      // Fill in deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // Handle the check as required.
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. The construction must be custom based on the API.
      std::vector<int64_t> xShape = {64, 32};
      std::vector<int64_t> weightShape= {64, 32};
      std::vector<int64_t> yShape = {2, 32, 32};
      std::vector<int64_t> groupListShape = {2};
      std::vector<int64_t> groupListData = {32, 64};
      void* xDeviceAddr;
      void* weightDeviceAddr;
      void* yDeviceAddr;

      void* groupListDeviceAddr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* groupedList = nullptr;
      aclTensor* y = nullptr;
      aclTensor* yRef = nullptr;

      bool transpose_x = true;
      bool transpose_weight = false;
      int group_type = 2;

      // Create the x aclTensor.
      ret = CreateAclTensor<uint16_t>(xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the weight aclTensor.
      ret = CreateAclTensor<uint16_t>(weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the y aclTensor.
      ret = CreateAclTensor<float>(yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the group_list aclTensor.
      ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      yRef = y;

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // 3. Call the CANN operator library API.
      // Call the first-phase API of aclnnGroupedMatmulAdd.
      ret = aclnnGroupedMatmulAddGetWorkspaceSize(x, weight, groupedList, yRef, transpose_x, transpose_weight, group_type, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnGroupedMatmulAdd.
      ret = aclnnGroupedMatmulAdd(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulAdd failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Synchronize and wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device side to the host side. Modify it based on the specific API definition.
      auto size = GetShapeSize(yShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr,
                            size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t j = 0; j < size; j++) {
          LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
      }


      // 6. Release the aclTensor and aclScalar. Modify it based on the specific API definition.
      aclDestroyTensor(x);
      aclDestroyTensor(weight);
      aclDestroyTensor(y);

      // 7. Release device resources. Modify it based on the definition of the specific API.
      aclrtFree(xDeviceAddr);
      aclrtFree(weightDeviceAddr);
      aclrtFree(yDeviceAddr);

      if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
    }
    ```
<!-- end id9 -->
