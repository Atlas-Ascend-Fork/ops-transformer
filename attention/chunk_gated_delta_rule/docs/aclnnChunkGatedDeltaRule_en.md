# aclnnChunkGatedDeltaRule

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:24:11.018Z -->

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

- Description: Performs chunked Gated Delta Rule computation.

- Formula:

  Gated Delta Rule (GDR) is an operator used in recurrent neural networks and also in a linear attention mechanism. At each time step $t$, GDR computes the current attention output $o_t$ and the new hidden state $S_t$ based on the current inputs $q_t$, $k_t$, and $v_t$, the previous hidden state $S_{t-1}$, the decay factor $\alpha_t$, and the update strength $\beta_t$. The computation formulas are as follows:
  $$
  S_t := S_{t-1}(\alpha_t(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1}k_t)k_t^T
  $$
  $$
  o_t := S_t (q_t \cdot scale)
  $$

  where $S_{t-1},S_t \in \mathbb{R}^{D_v \times D_k}$, $q_t, k_t \in \mathbb{R}^{D_k}$, $v_t \in \mathbb{R}^{D_v}$, $\alpha_t \in \mathbb{R}$, $\beta_t \in \mathbb{R}$, $o_t \in \mathbb{R}^{D_v}$.

  Chunked Gated Delta Rule is the chunk-based implementation of GDR ([reference paper](https://arxiv.org/abs/2412.06464)). It splits the input sequence into chunks, achieving a certain degree of parallelism. In long-context use cases, its computational efficiency is higher than that of Recurrent Gated Delta Rule, making it suitable for the prefill stage. Given an input sequence of length L, this operator computes the output of each step $o_t, t \in \{1, .., L\}$ and the final state matrix $S_L$.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnChunkGatedDeltaRuleGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnChunkGatedDeltaRule" API to perform the computation.

```cpp
aclnnStatus aclnnChunkGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *beta,
    const aclTensor *initialState,
    const aclTensor *actualSeqLengths,
    const aclTensor *gOptional,
    float           scaleValue,
    const aclTensor *out,
    const aclTensor *finalState,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnChunkGatedDeltaRule(
    void          *workspace,
    uint64_t      workspaceSize,
    aclOpExecutor *executor,
    aclrtStream   stream)
```

## aclnnChunkGatedDeltaRuleGetWorkspaceSize

- Parameters

  <table style="undefined; table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 165px">
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
      <td>q in the formula.</td>
      <td>Does not support empty tensors.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nk, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>k in the formula.</td>
      <td>Does not support empty tensors.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nk, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>v in the formula.</td>
      <td>Does not support empty tensors.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv, Dv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>β in the formula.</td>
      <td>Does not support empty tensors.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>initialState</td>
      <td>Input</td>
      <td>Initial state matrix S_0.</td>
      <td>      <ul>
          <li>Does not support empty tensors.</li>
          <li>Supports non-contiguity on axis 0 and axis 1.</li>
      </ul>
      </td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(B, Nv, Dv, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqLengths</td>
      <td>Input</td>
      <td>Valid sequence length of each batch.</td>
      <td>Does not support empty tensors.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gOptional</td>
      <td>Input</td>
      <td>Exponent of the decay coefficient, where α=e^g in the formula.</td>
      <td>Optional input.<ul><li>Does not support empty Tensor.</li><li>If nullptr is passed in, it indicates an all-zero tensor.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(T, Nv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>Scale factor of query, corresponding to scale in the formula.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>o in the formula.</td>
      <td>-</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv, Dv)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>finalState</td>
      <td>Output</td>
      <td>Final state matrix S_L.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(B, Nv, Dv, Dk)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied by the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  In the formula, $B$ indicates the batch size, $L_i$ indicates the length of the i-th sequence, and $T=\sum_i^B L_i$ indicates the cumulative sequence length. $N_k$ indicates the number of key heads, $N_v$ indicates the number of value heads, $D_k$ indicates the hidden layer dimension of key, and $D_v$ indicates the hidden layer dimension of value.

- Return Value

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1050px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 670px">
  </colgroup>
  <thead>
    <tr>
      <th>return value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>Null pointers exist in query, key, value, beta, initialState, actualSeqLengths, out, or finalState.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types and formats of the input query, key, value, beta, initialState, actualSeqLengths, out, and finalState are not within the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The shapes of the input query, key, value, beta, initialState, actualSeqLengths, out, and finalState do not match.</td>
    </tr>
  </tbody>
  </table>

## aclnnChunkGatedDeltaRule

- Parameters

  <table style="undefined;table-layout: fixed; width: 1050px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 670px">
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
      <td>workspace (void*)</td>
      <td>Input</td>
      <td>Workspace memory address applied by the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Input</td>
      <td>Workspace size applied by the Device, obtained by the first-phase API aclnnChunkGatedDeltaRuleGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor*)</td>
      <td>Input</td>
      <td>Executor of the operator, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream (aclrtStream)</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- Return Value:
  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnChunkGatedDeltaRule defaults to a deterministic implementation.
- Dimension Constraints:
  - $0 \lt Nv \le 64, 0 \lt Nk \le 64$ and $Nv \bmod Nk = 0$
  - $0 \lt Dv \le 128$, $0 \lt Dk \le 128$
  - $B \gt 0$, $T \gt 0$
- Due to algorithm characteristics, users must ensure the following numerical constraints; otherwise, the computation results may overflow:
  - $-1 \le query[i][j][k] \le 1$, and query must be normalized
  - $-1 \le key[i][j][k] \le 1$, and key must be normalized
  - $-1 \le g[i][j] \le 0$
  - $0 < beta[i][j] < 1$
- The float32 data type is supported for initialState and finalState only on Ascend 950PR&950DT products.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_gated_delta_rule.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr, const char* name)
{
    auto size = GetShapeSize(shape);
    std::vector<uint16_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        if (i >= 5) { // print the first five data
            break;
        }
        float result = 0.0f;
        uint32_t val = static_cast<uint32_t>(resultData[i]) << 16U;
        std::memcpy(&result, &val, sizeof(result));
        LOG_PRINT("%s result[%ld] is: %f\n", name, i, result);
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // AscendCL initialization
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    // 1. Initialize device/context/stream. For details, see the list of AscendCL external APIs.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs. Custom construction is required based on the API.
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *gamaDeviceAddr = nullptr;
    void *betaDeviceAddr = nullptr;
    void *initStateDeviceAddr = nullptr;
    void *actSeqLenDeviceAddr = nullptr;
    void *attnOutDeviceAddr = nullptr;
    void *finalStateDeviceAddr = nullptr;

    aclTensor *query = nullptr;
    aclTensor *key = nullptr;
    aclTensor *value = nullptr;
    aclTensor *gama = nullptr;
    aclTensor *beta = nullptr;
    aclTensor *initState = nullptr;
    aclTensor *actSeqLen = nullptr;
    aclTensor *attnOut = nullptr;
    aclTensor *finalState = nullptr;

    // Custom inputs and attributes
    int32_t batchSize = 2;
    int32_t seqLength = 32;
    int32_t headKNum = 4;
    int32_t headVNum = 8;
    int32_t dimV = 32;
    int32_t dimK = 32;


    std::vector<int64_t> stateShape = {batchSize, headVNum, dimV, dimK};
    std::vector<int64_t> qkShape = {batchSize * seqLength, headKNum, dimK};
    std::vector<int64_t> vShape = {batchSize * seqLength, headVNum, dimV};
    std::vector<int64_t> gamaShape = {batchSize * seqLength, headVNum};
    std::vector<int64_t> actSeqLenShape = {batchSize};
    std::vector<int16_t> initStateHostData(GetShapeSize(stateShape));
    std::vector<int16_t> queryHostData(GetShapeSize(qkShape));
    std::vector<int16_t> keyHostData(GetShapeSize(qkShape));
    std::vector<int16_t> valueHostData(GetShapeSize(vShape));
    std::vector<float> gamaHostData(GetShapeSize(gamaShape));
    std::vector<int16_t> betaHostData(GetShapeSize(gamaShape));
    std::vector<int32_t> actSeqLenHostData(batchSize, seqLength);
    int16_t bfloatHalf = 16128;   // The binary form of 16128 in int16_t corresponds to 0.5 in bfloat16.
    int16_t bfloatQuarter = 16000; // The binary form of 16000 in int16_t corresponds to 0.25 in bfloat16.
    for (int i = 0; i < initStateHostData.size(); i++) {
        initStateHostData[i] = bfloatQuarter;
    }
    for (int i = 0; i < queryHostData.size(); i++) {
        queryHostData[i] = bfloatQuarter;
    }
    for (int i = 0; i < keyHostData.size(); i++) {
        keyHostData[i] = bfloatQuarter;
    }
    for (int i = 0; i < valueHostData.size(); i++) {
        valueHostData[i] = bfloatHalf;
    }
    for (int i = 0; i < betaHostData.size(); i++) {
        betaHostData[i] = bfloatHalf;
    }
    for (int i = 0; i < gamaHostData.size(); i++) {
        gamaHostData[i] = -0.5f;
    }

    std::vector<int16_t> attnOutHostData(valueHostData);
    std::vector<int16_t> finalStateHostData(GetShapeSize(stateShape));

    ret = CreateAclTensor(initStateHostData, stateShape, &initStateDeviceAddr, aclDataType::ACL_BF16, &initState);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryHostData, qkShape, &queryDeviceAddr, aclDataType::ACL_BF16, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, qkShape, &keyDeviceAddr, aclDataType::ACL_BF16, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueHostData, vShape, &valueDeviceAddr, aclDataType::ACL_BF16, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gamaHostData, gamaShape, &gamaDeviceAddr, aclDataType::ACL_FLOAT, &gama);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, gamaShape, &betaDeviceAddr, aclDataType::ACL_BF16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actSeqLenHostData, actSeqLenShape, &actSeqLenDeviceAddr, aclDataType::ACL_INT32, &actSeqLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnOutHostData, vShape, &attnOutDeviceAddr, aclDataType::ACL_BF16, &attnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(finalStateHostData, stateShape, &finalStateDeviceAddr, aclDataType::ACL_BF16, &finalState);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    float scale = 1.0;
    aclOpExecutor *executor;
    // Call the aclnnChunkGatedDeltaRuleGetWorkspaceSize API (first-stage interface)
    ret = aclnnChunkGatedDeltaRuleGetWorkspaceSize(query, key, value, beta, initState, actSeqLen, gama,
                                                       scale, attnOut, finalState, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChunkGatedDeltaRuleGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnChunkGatedDeltaRule.
    ret = aclnnChunkGatedDeltaRule(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChunkGatedDeltaRule failed. ERROR: %d\n", ret); return ret);

    // 4. Wait synchronously for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code based on the specific API definition.
    PrintOutResult(vShape, &attnOutDeviceAddr, "out");
    PrintOutResult(stateShape, &finalStateDeviceAddr, "finalState");

    // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(gama);
    aclDestroyTensor(beta);
    aclDestroyTensor(initState);
    aclDestroyTensor(actSeqLen);
    aclDestroyTensor(attnOut);
    aclDestroyTensor(finalState);

    // 7. Release device resources.
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(gamaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(initStateDeviceAddr);
    aclrtFree(actSeqLenDeviceAddr);
    aclrtFree(attnOutDeviceAddr);
    aclrtFree(finalStateDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
