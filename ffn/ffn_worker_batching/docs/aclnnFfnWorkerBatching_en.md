# aclnnFfnWorkerBatching

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T11:18:13.555Z -->

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

- Description: Token regrouping operator on the FFN worker side for use cases where Attention and FFN are deployed separately. Attention sends tokens to the pre-allocated data area of the corresponding FFN worker based on expert routing. This API scans the scheduling information in the data area, aggregates and regroups tokens by expert dimension, and outputs a contiguous token data block for each expert.

  **This operator is not recommended for standalone use. Use it with other operators such as FfnWorkerScheduler to form a complete workflow.**

    1. Receives data sent from the Attention side. The data is stored in the memory layout of the ScheduleContext structure. For details, see [Example](#example). The structure contains the CommonArea, ControlArea, AttentionArea, and FfnArea fields. This API reads token_info_buf and token_data_buf from FfnArea to obtain the token data and description information to be rearranged, and obtains routing information such as layer_id, session_id, micro_batch_id, and expert_ids.

    2. Sorts the expert IDs of all tokens in `expert_ids_in` (masked tokens are initialized to a large value) to generate gather indices.

    3. Extracts the hidden states and dynamic scale of tokens from `token_data` based on the gather indices in parallel on multiple cores, and obtains the corresponding session_id, micro_batch_id, and token_id through table lookup.

    4. Scans the sorted expert ID sequence on a single core to find jump points and generates `groupList` (start and end offsets of the tokens processed by each expert).

    where $Y = A \times BS \times (K+1)$, $A$ is the number of Attention workers, $BS$ is the micro batch size, and $K+1$ is topK plus the number of shared experts.

- Formula:

  $$
  \text{gather\_idx}, \_ = \text{sort}(\text{expert\_ids\_in})
  $$

  $$
  y[i] = \text{token\_data}[\text{gather\_idx}[i]]
  $$

  $$
  \text{groupList}[e] = [\text{expert\_id}_e, \text{expert\_token\_num}_e]
  $$

  $$
  \text{actualTokenNum} = \sum_{e} \text{expert\_token\_num}_e
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnFfnWorkerBatchingGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call "aclnnFfnWorkerBatching" to perform the computation.

```Cpp
aclnnStatus aclnnFfnWorkerBatchingGetWorkspaceSize(
    const aclTensor   *scheduleContext,
    int64_t            expertNum,
    const aclIntArray *maxOutShape,
    int64_t            tokenDtype,
    int64_t            needSchedule,
    int64_t            layerNum,
    const aclTensor   *y,
    const aclTensor   *groupList,
    const aclTensor   *sessionIds,
    const aclTensor   *microBatchIds,
    const aclTensor   *tokenIds,
    const aclTensor   *expertOffsets,
    const aclTensor   *dynamicScale,
    const aclTensor   *actualTokenNum,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnFfnWorkerBatching(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream)
```

## aclnnFfnWorkerBatchingGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1540px"><colgroup>
  <col style="width: 210px">
  <col style="width: 135px">
  <col style="width: 360px">
  <col style="width: 280px">
  <col style="width: 130px">
  <col style="width: 110px">
  <col style="width: 170px">
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
      <td>scheduleContext</td>
      <td>Input</td>
      <td>Scheduling context received on the FFN side, containing CommonArea, ControlArea, AttentionArea, and FfnArea. The operator reads token_info_buf and token_data_buf from FfnArea to obtain the token data to be rearranged and the description information.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1D, shape fixed to (1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertNum</td>
      <td>Input</td>
      <td>Total number of experts on the current device, equal to the number of experts on the device per layer × layerNum. Used to derive the output size of groupList.</td>
      <td>The value range is (0, 8192].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxOutShape</td>
      <td>Input</td>
      <td>Upper limit of the output shape, in the format of {A, BS, topK+1, H}. Used to derive the shape upper limit Y = A × BS × (topK+1) of the y output and the H value.</td>
      <td>The array length must be 4. The value range of A is (0, 1024], BS must be greater than 0, the value range of topK+1 is (0, 64], and H must be greater than 0.</td>
      <td>LIST_INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tokenDtype</td>
      <td>Input</td>
      <td>Data type of input tokens. 0: FP16; 1: BF16; 2: INT8 dynamic quantization (INT8 data and FP32 dynamic scale are arranged contiguously). dynamic_scale must be output when the value is 2.</td>
      <td>0, 1, or 2</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>needSchedule</td>
      <td>Input</td>
      <td>Scheduling mode. 0 indicates batching only, without data scanning; 1 indicates data scanning before batching.</td>
      <td>0 or 1</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layerNum</td>
      <td>Input</td>
      <td>Layer number. Each layer has an independent expert index.</td>
      <td>Value range: [0, expertNum].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Rearranged token hidden states, contiguously stored after being sorted by expert ID.</td>
      <td>-</td>
      <td>FP16, BF16, INT8</td>
      <td>ND</td>
      <td>2 dimensions, (Y, H), where Y = A × BS × (topK+1)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupList</td>
      <td>Output</td>
      <td>Token range processed by each expert. Each row is in the format [expert_id, expert_token_num]. Unused experts are filled with [0, 0].</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>2-dimensional, (expertNum, 2)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sessionIds</td>
      <td>Output</td>
      <td>Attention session ID corresponding to each output token.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-dimensional, (Y)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>microBatchIds</td>
      <td>Output</td>
      <td>Micro batch ID of each output token.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, (Y)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>tokenIds</td>
      <td>Output</td>
      <td>Position index of each output token in the original input.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-dimension, (Y)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertOffsets</td>
      <td>Output</td>
      <td>Offset of each output token within its expert group.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1 dimension, (Y)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dynamicScale</td>
      <td>Output</td>
      <td>Dynamic quantization scale value. Valid only when tokenDtype=2. Empty tensor when tokenDtype is 0 or 1.</td>
      <td>-</td>
      <td>FP32</td>
      <td>ND</td>
      <td>1-D, (Y)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>actualTokenNum</td>
      <td>Output</td>
      <td>Sum of valid token counts of all experts.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1-D, (1)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>Return Value</th>
            <th>Error Codes</th>
            <th>Description</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>Input is a null pointer.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The input data type is out of the supported range.</td>
            </tr>
        </tbody>
    </table>

## aclnnFfnWorkerBatching

- **Parameters**

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
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnFfnWorkerBatchingGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Parameter A (number of Attention workers): ≤ 1024.
- micro_batch_num in the scheduling context (number of micro batches): ≤ 64.
- Parameter K+1 (topK plus the number of shared experts): ≤ 64.
- Parameters BS (micro batch size) and Y support generalization with no hard upper limit (limited by memory).
- Parameter H (hidden size) supports generalization.
- Deterministic computation:
  - aclnnFfnWorkerBatching defaults to a deterministic implementation.

## Example

The following is the sample code for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

In the example, `selected_expert_num` is topK plus the number of shared experts (9 here), and `needSchedule=0` means batching only.

```Cpp
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn_worker_batching.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> stride(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        stride[i] = shape[i + 1] * stride[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int CreateAclTensorNoData(const std::vector<int64_t> &shape, void **deviceAddr, aclDataType dataType,
                          aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(int8_t);
    if (dataType == ACL_INT32) {
        size = GetShapeSize(shape) * sizeof(int32_t);
    }
    if (dataType == ACL_INT64) {
        size = GetShapeSize(shape) * sizeof(int64_t);
    }
    if (dataType == ACL_FLOAT) {
        size = GetShapeSize(shape) * sizeof(float);
    }
    if (dataType == ACL_FLOAT16) {
        size = GetShapeSize(shape) * sizeof(int16_t);
    }
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> stride(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        stride[i] = shape[i + 1] * stride[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

constexpr uint64_t kBufAlignSize = 512;

inline uint64_t AlignUp(uint64_t num, uint64_t align)
{
    return ((num + align - 1) / align) * align;
}

#pragma pack(push, 1)
struct FfnDataDesc {
    volatile int32_t flag;
    volatile int32_t layer_id;
    volatile int32_t expert_ids[0];
};

struct ScheduleContext {
    struct CommonArea {
        uint32_t session_num; // Number of attention nodes
        uint32_t micro_batch_num;
        uint32_t micro_batch_size;
        uint32_t selected_expert_num; // topK + 1
        uint32_t expert_num;          // experts per layer
        uint32_t attn_to_ffn_token_size;
        uint32_t ffn_to_attn_token_size;
        int32_t schedule_mode; // 0: Ffn only 1: Attention only
        int8_t reserve0[96];
    };
    struct ControlArea {
        int32_t run_flag; // 0 : exited  1 : running
        int8_t reserve2[124];
    };
    struct FfnArea {
        uint64_t token_info_buf; // Points to device memory.
        uint64_t token_info_buf_size;
        uint64_t token_data_buf; // Points to device memory.
        uint64_t token_data_buf_size;
        uint64_t polling_index;
        int8_t reserve3[88];
        uint64_t layer_ids_buf;
        uint64_t layer_ids_buf_size;
        uint64_t session_ids_buf;
        uint64_t session_ids_buf_size;
        uint64_t micro_batch_ids_buf;
        uint64_t micro_batch_ids_buf_size;
        uint64_t expert_ids_buf;
        uint64_t expert_ids_buf_size;
        uint32_t out_num;
        int8_t reserve4[60];
    };
    struct AttentionArea {
        uint64_t token_info_buf;
        uint64_t token_info_buf_size;
        uint64_t token_data_buf;
        uint64_t token_data_buf_size;
        uint32_t micro_batch_id;
        int8_t reserve5[92];
    };
    CommonArea common;
    ControlArea control;
    AttentionArea attention;
    FfnArea ffn;
    int8_t reserve6[384]; // Padding to 1KB.
};
static_assert(sizeof(ScheduleContext) == 1024, "ScheduleContext size must be 1KB");
#pragma pack(pop)

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs. Custom construction is required based on the API.
    ScheduleContext scheduleContext = {};
    scheduleContext.common.session_num = 1;
    scheduleContext.common.micro_batch_num = 1;
    scheduleContext.common.micro_batch_size = 8;
    scheduleContext.common.selected_expert_num = 9;
    scheduleContext.common.expert_num = 8;
    scheduleContext.common.attn_to_ffn_token_size = 512;
    scheduleContext.common.ffn_to_attn_token_size = 512;
    scheduleContext.common.schedule_mode = 0; // Ffn only
    scheduleContext.control.run_flag = 1;     // running
    scheduleContext.ffn.polling_index = 0;

    // Attribute parameters
    int64_t expertNum = 8;                                  // Number of experts per layer × layer_num
    int64_t A = scheduleContext.common.session_num;         // Number of Attention workers
    int64_t BS = scheduleContext.common.micro_batch_size;   // micro batch size
    int64_t KPlus1 = scheduleContext.common.selected_expert_num;
    int64_t H = 4096;                                       // hidden size
    int64_t Y = A * BS * KPlus1;
    std::vector<int64_t> maxOutShapeValue = {A, BS, KPlus1, H};
    int64_t tokenDtype = 0; // FP16
    int64_t needSchedule = 0;
    int64_t layerNum = 1;

    // Initialize the Ffn token_info_buf.
    uint64_t flagAndLayerIdSize = sizeof(FfnDataDesc);
    uint64_t tokenInfoSize = (sizeof(int32_t) * static_cast<uint64_t>(KPlus1) * BS + flagAndLayerIdSize) *
                             static_cast<uint64_t>(scheduleContext.common.micro_batch_num) *
                             static_cast<uint64_t>(scheduleContext.common.session_num);
    scheduleContext.ffn.token_info_buf_size = tokenInfoSize;
    void *tokenInfoBuf = nullptr;
    ret = aclrtMalloc(&tokenInfoBuf, tokenInfoSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc token info buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.ffn.token_info_buf = reinterpret_cast<uint64_t>(tokenInfoBuf);

    // Populate token_info_buf: flag=1, layer_id, and expert_ids.
    std::vector<int32_t> hostTokenInfo(tokenInfoSize / sizeof(int32_t), 0);
    int32_t *pInt = hostTokenInfo.data();
    for (uint32_t s = 0; s < scheduleContext.common.session_num; ++s) {
        for (uint32_t m = 0; m < scheduleContext.common.micro_batch_num; ++m) {
            *pInt++ = 1; // flag
            *pInt++ = 0; // layer_id
            for (uint32_t i = 0; i < BS * KPlus1; ++i) {
                *pInt++ = static_cast<int32_t>(i % scheduleContext.common.expert_num); // expert_id
            }
        }
    }
    ret = aclrtMemcpy(tokenInfoBuf, tokenInfoSize, hostTokenInfo.data(), tokenInfoSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy token info buf failed. ERROR: %d\n", ret); return ret);

    // Initialize the Ffn token_data_buf.
    uint64_t tokenDataSize = static_cast<uint64_t>(Y) * H * sizeof(int16_t);
    scheduleContext.ffn.token_data_buf_size = tokenDataSize;
    void *tokenDataBuf = nullptr;
    ret = aclrtMalloc(&tokenDataBuf, tokenDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc token data buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.ffn.token_data_buf = reinterpret_cast<uint64_t>(tokenDataBuf);
    std::vector<int16_t> hostTokenData(Y * H, 1);
    ret = aclrtMemcpy(tokenDataBuf, tokenDataSize, hostTokenData.data(), tokenDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy token data buf failed. ERROR: %d\n", ret); return ret);

    // Create the scheduleContext aclTensor.
    std::vector<int64_t> scheduleContextShape = {1024};
    void *scheduleContextDeviceAddr = nullptr;
    aclTensor *scheduleContextRef = nullptr;
    // ==== Flat id buffers in FfnArea ====
    // These four fields are second-level pointers: the context stores device addresses, which the kernel dereferences on its own. The framework cannot see the inner buffers,
    // leaving them empty still passes tiling validation, and the problem only surfaces as errcode:(95) MTE out-of-bounds DDR access when the kernel accesses memory.
    // The NORM path (needSchedule=0) reads them in both generations of implementations. They must be provided.
    std::vector<int32_t> hostExpertIds(Y);
    for (int64_t i = 0; i < Y; ++i) {
        hostExpertIds[i] = static_cast<int32_t>(i % expertNum);
    }
    void *expertIdsBuf = nullptr;
    ret = aclrtMalloc(&expertIdsBuf, hostExpertIds.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc expert ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(expertIdsBuf, hostExpertIds.size() * sizeof(int32_t), hostExpertIds.data(),
                      hostExpertIds.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy expert ids buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.ffn.expert_ids_buf = reinterpret_cast<uint64_t>(expertIdsBuf);
    scheduleContext.ffn.expert_ids_buf_size = hostExpertIds.size() * sizeof(int32_t);

    std::vector<int32_t> hostSessionIds(A, 0);
    std::vector<int32_t> hostMicroBatchIds(A, 0);
    std::vector<int32_t> hostLayerIds(A, 0);
    const uint64_t idsBufSize = static_cast<uint64_t>(A) * sizeof(int32_t);
    void *sessionIdsBuf = nullptr;
    void *microBatchIdsBuf = nullptr;
    void *layerIdsBuf = nullptr;
    ret = aclrtMalloc(&sessionIdsBuf, idsBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc session ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&microBatchIdsBuf, idsBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc micro batch ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&layerIdsBuf, idsBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc layer ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(sessionIdsBuf, idsBufSize, hostSessionIds.data(), idsBufSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy session ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(microBatchIdsBuf, idsBufSize, hostMicroBatchIds.data(), idsBufSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy micro batch ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(layerIdsBuf, idsBufSize, hostLayerIds.data(), idsBufSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy layer ids buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.ffn.session_ids_buf = reinterpret_cast<uint64_t>(sessionIdsBuf);
    scheduleContext.ffn.session_ids_buf_size = idsBufSize;
    scheduleContext.ffn.micro_batch_ids_buf = reinterpret_cast<uint64_t>(microBatchIdsBuf);
    scheduleContext.ffn.micro_batch_ids_buf_size = idsBufSize;
    scheduleContext.ffn.layer_ids_buf = reinterpret_cast<uint64_t>(layerIdsBuf);
    scheduleContext.ffn.layer_ids_buf_size = idsBufSize;

    std::vector<int8_t> hostCtxData(1024, 0);
    std::memcpy(hostCtxData.data(), &scheduleContext, sizeof(ScheduleContext));
    ret = CreateAclTensor(hostCtxData, scheduleContextShape, &scheduleContextDeviceAddr, aclDataType::ACL_INT8,
                          &scheduleContextRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the output aclTensor.
    std::vector<int64_t> yShape = {Y, H};
    void *yDeviceAddr = nullptr;
    aclTensor *yRef = nullptr;
    ret = CreateAclTensorNoData(yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &yRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> groupListShape = {expertNum, 2};
    void *groupListDeviceAddr = nullptr;
    aclTensor *groupListRef = nullptr;
    ret = CreateAclTensorNoData(groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupListRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> oneDimShape = {Y};
    void *sessionIdsAddr = nullptr;
    aclTensor *sessionIdsRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &sessionIdsAddr, aclDataType::ACL_INT32, &sessionIdsRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *microBatchIdsAddr = nullptr;
    aclTensor *microBatchIdsRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &microBatchIdsAddr, aclDataType::ACL_INT32, &microBatchIdsRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *tokenIdsAddr = nullptr;
    aclTensor *tokenIdsRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &tokenIdsAddr, aclDataType::ACL_INT32, &tokenIdsRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *expertOffsetsAddr = nullptr;
    aclTensor *expertOffsetsRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &expertOffsetsAddr, aclDataType::ACL_INT32, &expertOffsetsRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *dynamicScaleAddr = nullptr;
    aclTensor *dynamicScaleRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &dynamicScaleAddr, aclDataType::ACL_FLOAT, &dynamicScaleRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> actualTokenNumShape = {1};
    void *actualTokenNumAddr = nullptr;
    aclTensor *actualTokenNumRef = nullptr;
    ret = CreateAclTensorNoData(actualTokenNumShape, &actualTokenNumAddr, aclDataType::ACL_INT64, &actualTokenNumRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the maxOutShape aclIntArray.
    aclIntArray *maxOutShapeArray = aclCreateIntArray(maxOutShapeValue.data(), maxOutShapeValue.size());

    // 3. Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnFfnWorkerBatchingGetWorkspaceSize(scheduleContextRef, expertNum, maxOutShapeArray, tokenDtype,
                                                 needSchedule, layerNum, yRef, groupListRef, sessionIdsRef,
                                                 microBatchIdsRef, tokenIdsRef, expertOffsetsRef, dynamicScaleRef,
                                                 actualTokenNumRef, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFfnWorkerBatchingGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnFfnWorkerBatching(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFfnWorkerBatching failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output result, and copy the result on the device memory to the host.
    int64_t actualTokenNum = 0;
    ret = aclrtMemcpy(&actualTokenNum, sizeof(int64_t), actualTokenNumAddr, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy actual_token_num failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("actual_token_num = %ld.\n", actualTokenNum);

    std::vector<int64_t> groupListHost(expertNum * 2, 0);
    ret = aclrtMemcpy(groupListHost.data(), expertNum * 2 * sizeof(int64_t), groupListDeviceAddr,
                      expertNum * 2 * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy group_list failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expertNum; i++) {
        LOG_PRINT("group_list[%ld] = [expert_id=%ld, token_num=%ld]\n", i, groupListHost[i * 2],
                  groupListHost[i * 2 + 1]);
    }

    // 6. Release the aclTensor and aclIntArray.
    aclDestroyTensor(scheduleContextRef);
    aclDestroyTensor(yRef);
    aclDestroyTensor(groupListRef);
    aclDestroyTensor(sessionIdsRef);
    aclDestroyTensor(microBatchIdsRef);
    aclDestroyTensor(tokenIdsRef);
    aclDestroyTensor(expertOffsetsRef);
    aclDestroyTensor(dynamicScaleRef);
    aclDestroyTensor(actualTokenNumRef);
    aclDestroyIntArray(maxOutShapeArray);

    // 7. Release the device resources.
    aclrtFree(scheduleContextDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(sessionIdsAddr);
    aclrtFree(microBatchIdsAddr);
    aclrtFree(tokenIdsAddr);
    aclrtFree(expertOffsetsAddr);
    aclrtFree(dynamicScaleAddr);
    aclrtFree(actualTokenNumAddr);
    aclrtFree(tokenInfoBuf);
    aclrtFree(tokenDataBuf);
    aclrtFree(expertIdsBuf);
    aclrtFree(sessionIdsBuf);
    aclrtFree(microBatchIdsBuf);
    aclrtFree(layerIdsBuf);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    Finalize(deviceId, stream);
    return 0;
}
```
