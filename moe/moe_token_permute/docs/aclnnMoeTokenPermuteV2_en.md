# aclnnMoeTokenPermuteV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:26:11.401Z -->

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

- **Description**: MoE permute computation, which broadcasts and sorts tokens based on the indices.
- **Extended Capabilities**: Compared with `aclnnMoeTokenPermute`, this API adds `quantMode` and `expandedScaleOut`; MXFP8 and MXFP4 quantization output is supported on Ascend 950PR&950DT products.
- **Platform Behavior**:
  - Ascend 950PR&950DT products: `aclnnMoeTokenPermuteV2` supports `quantMode=-1/2/3/9`.
  - Non-Ascend 950: Quantization parameters are silently ignored, and the non-quantized compatibility path is used.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMoeTokenPermuteV2GetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnMoeTokenPermuteV2" API to execute the computation.

```cpp
aclnnStatus aclnnMoeTokenPermuteV2GetWorkspaceSize(
    const aclTensor  *tokens,
    const aclTensor  *indices,
    int64_t           numOutTokens,
    bool              paddedMode,
    int64_t           quantMode,
    const aclTensor  *permuteTokensOut,
    const aclTensor  *sortedIndicesOut,
    const aclTensor  *expandedScaleOut,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeTokenPermuteV2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnMoeTokenPermuteV2GetWorkspaceSize

| Name | Input/Output | Description | Usage Instructions | Data Type | Data Format | Dimension |
| --- | --- | --- | --- | --- | --- | --- |
| `tokens` | Input | Input token features. | Empty tensors are supported; the dimension must be 2, with the first dimension being `num_tokens`. In quantization mode, Ascend 950PR&950DT products support only FLOAT16 and BFLOAT16. | Non-quantized: FLOAT16, BFLOAT16, FLOAT32, INT8; quantized: FLOAT16, BFLOAT16 | ND | 2 |
| `indices` | Input | Input indices. | Empty tensors are supported; the shape is 1D or 2D. When `paddedMode=false`, the tensor represents the topK expert indices corresponding to each input token. When `aclnnMoeTokenPermuteV2` is called on Ascend 950PR&950DT products, elements represent expert IDs, and the value range is `[0, 10240)`. | INT32, INT64 | ND | 1 or 2 |
| `numOutTokens` | Input | Number of valid output tokens. | `0` indicates that no tokens are deleted; a value greater than 0 retains the first `numOutTokens` tokens after sorting; a value less than 0 is processed as a negative slice index. | - | - | - |
| `paddedMode` | Input | Whether to use padding mode. | `true` is currently not supported; it is recommended to set the value to `false`. | - | - | - |
| `quantMode` | Input | Quantization mode. | `-1` indicates non-quantization; `2` indicates MXFP8 E5M2; `3` indicates MXFP8 E4M3FN; `9` indicates MXFP4 E2M1. Quantization parameters are silently ignored on platforms other than Ascend 950PR&950DT products. | - | - | - |
| `permuteTokensOut` | Output | Tokens extended and sorted based on `indices`. | The data type is the same as `tokens` in non-quantized mode; when `quantMode=2/3/9`, the data types are FLOAT8_E5M2, FLOAT8_E4M3FN, and FLOAT4_E2M1, respectively. | FLOAT16, BFLOAT16, FLOAT32, INT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1 | ND | ≥2 |
| `sortedIndicesOut` | Output | Mapping between `permuteTokensOut` and `tokens`. | 1D tensor with the shape `[indices.numel()]`. | INT32 | ND | 1 |
| `expandedScaleOut` | Output | Per-token block scale after quantization. | A valid output must be provided in quantization use cases; in non-quantization use cases, an empty tensor with the shape `[0]` and dtype FLOAT32 can be passed. | FLOAT8_E8M0, FLOAT32 | ND | 1, 2, or 3 |
| `workspaceSize` | Output | Returns the workspace size to be applied on the Device side. | - | - | - | - |
| `executor` | Output | Returns the op executor, which contains the operator computation flow. | - | - | - | - |

## Output Shape

First, calculate:

```text
flatten_size = indices.numel()

if numOutTokens > 0:
    M = min(numOutTokens, flatten_size)
else:
    M = max(numOutTokens + flatten_size, 0)

H = tokens.shape[1]
```

The output shapes of each mode are as follows:

| quantMode | permuteTokensOut shape | sortedIndicesOut shape | expandedScaleOut shape |
| --- | --- | --- | --- |
| `-1` | `[M, H]` | `[flatten_size]` | `[0]` |
| `2` | `[M, H]` | `[flatten_size]` | `[M, AlignUp(CeilDiv(H, 32), 2)]` |
| `3` | `[M, H]` | `[flatten_size]` | `[M, AlignUp(CeilDiv(H, 32), 2)]` |
| `9` | `[M, H]` | `[flatten_size]` | `[M, CeilDiv(H, 64), 2]` |

Where:

```text
CeilDiv(a, b) = (a + b - 1) // b
AlignUp(a, b) = ((a + b - 1) // b) * b
```

## Return Value

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

Common error cases:

| Return Value | Error Code | Description |
| --- | --- | --- |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | Input or output tensor is a null pointer. |
| ACLNN_ERR_PARAM_INVALID | 161002 | Input or output data type is out of the supported range, or input or output shape fails API validation; on Ascend 950PR&950DT products, `quantMode` is not in `-1/2/3/9` or `paddedMode=true`. |
| ACLNN_ERR_INNER_TILING_ERROR | 561002 | Operator tiling fails internally, for example, when tiling rejects `paddedMode=true` on the compatibility path for platforms other than Ascend 950PR&950DT products. |

## aclnnMoeTokenPermuteV2

| Name | Input/Output | Description |
| --- | --- | --- |
| `workspace` | Input | Device-side allocated workspace memory address. |
| `workspaceSize` | Input | Size of the workspace requested on the Device, obtained by calling the first-phase API `aclnnMoeTokenPermuteV2GetWorkspaceSize`. |
| `executor` | Input | OP executor, which contains the computation flow of the operator. |
| `stream` | Input | Stream used to execute the task. |

## Constraints

- The number of elements in `indices` must be less than `16777215`. Each element must be greater than or equal to `0` and less than `16777215`.
- When `aclnnMoeTokenPermuteV2` is called on Ascend 950PR&950DT products, `indices` represents the expert ID. The value range is `[0, 10240)`. The maximum value is `10239`, and `10240` is not supported.
- `paddedMode=true` is not supported.
- In quantization mode on Ascend 950PR&950DT products, `tokens` currently supports FLOAT16 and BFLOAT16.
- When `quantMode=9`, the hidden dimension `H` of `tokens` must be an even number, and the logical hidden dimension of `permuteTokensOut` is `H`. Two FLOAT4 elements are packed for storage; packing does not change the logical shape.
- On platforms other than Ascend 950PR&950DT products, `quantMode` is not validated or used; `2/3/9` are processed through the non-quantization compatible path.

## Example

The following example shows a non-quantized call. For a quantized call, adjust the data types and shapes of `permuteTokensOut` and `expandedScaleOut` according to `quantMode`.

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_v2.h"
#include <iostream>
#include <vector>

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
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    *deviceAddr = nullptr;
    if (size > 0) {
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> indicesShape = {3, 2};
    std::vector<int64_t> expandedXOutShape = {6, 4};
    std::vector<int64_t> sortedIndicesOutShape = {6};
    std::vector<int64_t> expandedScaleOutShape = {0};

    void* xDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* sortedIndicesOutDeviceAddr = nullptr;
    void* expandedScaleOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* expandedXOut = nullptr;
    aclTensor* sortedIndicesOut = nullptr;
    aclTensor* expandedScaleOut = nullptr;

    std::vector<float> xHostData = {
        0.1, 0.1, 0.1, 0.1,
        0.2, 0.2, 0.2, 0.2,
        0.3, 0.3, 0.3, 0.3
    };
    std::vector<int32_t> indicesHostData = {1, 2, 0, 1, 0, 2};
    std::vector<float> expandedXOutHostData(24, 0);
    std::vector<int32_t> sortedIndicesOutHostData(6, 0);
    std::vector<float> expandedScaleOutHostData;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr,
                          ACL_FLOAT, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesOutHostData, sortedIndicesOutShape, &sortedIndicesOutDeviceAddr,
                          ACL_INT32, &sortedIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedScaleOutHostData, expandedScaleOutShape, &expandedScaleOutDeviceAddr,
                          ACL_FLOAT, &expandedScaleOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t numOutTokens = 0;
    bool paddedMode = false;
    int64_t quantMode = -1;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMoeTokenPermuteV2GetWorkspaceSize(
        x, indices, numOutTokens, paddedMode, quantMode, expandedXOut, sortedIndicesOut, expandedScaleOut,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMoeTokenPermuteV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnMoeTokenPermuteV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteV2 failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<float> expandedXData(24, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(float), expandedXOutDeviceAddr,
                      expandedXData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy expandedXOut failed. ERROR: %d\n", ret); return ret);
    std::vector<int32_t> sortedIndicesData(6, 0);
    ret = aclrtMemcpy(sortedIndicesData.data(), sortedIndicesData.size() * sizeof(int32_t),
                      sortedIndicesOutDeviceAddr, sortedIndicesData.size() * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy sortedIndicesOut failed. ERROR: %d\n", ret); return ret);

    aclDestroyTensor(x);
    aclDestroyTensor(indices);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(sortedIndicesOut);
    aclDestroyTensor(expandedScaleOut);
    aclrtFree(xDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(sortedIndicesOutDeviceAddr);
    if (expandedScaleOutDeviceAddr != nullptr) {
        aclrtFree(expandedScaleOutDeviceAddr);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ACL_SUCCESS;
}
```
