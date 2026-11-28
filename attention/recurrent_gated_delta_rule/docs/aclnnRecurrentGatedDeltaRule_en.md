# aclnnRecurrentGatedDeltaRule

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:53:07.109Z -->

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

- Description: Completes the Recurrent Gated Delta Rule computation with a variable step size.

- Formula:

  Recurrent Gated Delta Rule (RGDR) is an operator applied in recurrent neural networks and is also used in a type of linear attention mechanism.
  At each time step $t$, the network computes the current attention output $o_t$ and the new hidden state $S_t$ based on the current inputs $q_t$, $k_t$, $v_t$ and the previous hidden state $S_{t-1}$.
  In this process, the gating unit determines how much new information is stored into the hidden state and how much old information needs to be forgotten.

  $$
  S_t := S_{t-1}(\alpha_t Diag(\alpha_{kt})(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t Diag(\alpha_{kt})S_{t-1} + \beta_t (v_t - \alpha_t Diag(\alpha_{kt})S_{t-1}k_t)k_t^T
  $$

  $$
  o := \frac{S_t q_t}{\sqrt{d_k}}
  $$

  where $S_{t-1},S_t \in R^{d_v \times d_k}$, $q_t, k_t \in R^{d_k}$, $v_t \in R^{d_v}$, $\alpha_t \in R$, $\alpha_{kt} \in R^{d_k}$, $\beta_t \in R$, $o \in R^{d_v}$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnRecurrentGatedDeltaRuleGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnRecurrentGatedDeltaRule" API to perform the computation.

```cpp
aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *beta,
    aclTensor       *stateRef,
    const aclTensor *actualSeqLengths,
    const aclTensor *ssmStateIndices,
    const aclTensor *g,
    const aclTensor *gk,
    const aclTensor *numAcceptedTokens,
    float           scaleValue,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnRecurrentGatedDeltaRule(
    void          *workspace,
    uint64_t      workspaceSize,
    aclOpExecutor *executor,
    aclrtStream   stream)
```

## aclnnRecurrentGatedDeltaRuleGetWorkspaceSize

- **Parameters**

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
      <td>Does not support null Tensor.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nk, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>k in the formula.</td>
      <td>Does not support null tensors.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nk, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>v in the formula.</td>
      <td>Does not support null tensors.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv, Dv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>β in the formula.</td>
      <td>Does not support null tensors.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>stateRef</td>
      <td>Input&Output</td>
      <td>State matrix, S in the formula.</td>
      <td>Does not support null Tensor.</td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(BlockNum, Nv, Dv, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqLengths</td>
      <td>Input</td>
      <td>Valid sequence length of each batch.</td>
      <td>Does not support null tensors.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ssmStateIndices</td>
      <td>Input</td>
      <td>Mapping indices from input sequences to the state matrix.</td>
      <td><ul><li>Does not support null Tensor.</li><li>state[ssmStateIndices[i]] indicates the state matrix of the i-th token.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>(T,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>g</td>
      <td>Input</td>
      <td>Decay coefficient, α=e^g in the formula.</td>
      <td><ul><li>Does not support null Tensor.</li><li>If nullptr is passed in, it indicates a tensor of all 0s.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(T, Nv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gk</td>
      <td>Input</td>
      <td>Decay coefficient, αk=e^gk in the formula.</td>
      <td><ul><li>Does not support null tensors.</li><li>If nullptr is passed in, it indicates an all-0 tensor.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(T, Nv, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numAcceptedTokens</td>
      <td>Optional input</td>
      <td>Number of tokens accepted by each sequence. When provided, the initial state for sequence i is selected by ssmStateIndices[seqStart + numAcceptedTokens[i] - 1], where seqStart is the sum of the lengths of preceding sequences. This parameter does not change the number of tokens processed in the sequence.</td>
      <td><ul><li>Optional input; pass nullptr to omit it.</li><li>When provided, an empty tensor is not supported.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>Scale factor of query, corresponding to 1/sqrt(d_k) in the formula.</td>
      <td>-</td>
      <td>-</td>
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
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  Where $B$ indicates the batch size, let $L_i$ indicate the length of the i-th sequence, then $T=\sum_i^B L_i$ indicates the cumulative sequence length. $N_k$ indicates the number of key heads, $N_v$ indicates the number of value heads, $D_k$ indicates the dimension of the key vector, and $D_v$ indicates the dimension of the value vector.

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1050px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 670px">
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
      <td>query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of the input tensor is out of the supported range.</td>
    </tr>
    <tr>
      <td>The data format of the input tensor is out of the supported range.</td>
    </tr>
    <tr>
      <td>The shape of the input tensor is out of the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnRecurrentGatedDeltaRule

- **Parameters**
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
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnRecurrentGatedDeltaRuleGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnRecurrentGatedDeltaRule defaults to a deterministic implementation.
- The input shape sizes must meet the following constraints: $0 < L_i \le 8$, $0 < N_k \le 256$, $N_k \le N_v \le 256$, $N_v$ % $N_k == 0$, $0 < D_k \le 512$, $0 < D_v \le 512$, $0 < T$, $0 < B$, $T \le BlockNum$.
- stateRef supports only non-contiguous tensors on axis 0 and axis 1.
- The following constraints must be ensured by the user because the operator cannot obtain specific values in the tensor and does not validate them:
  - $ssmStateIndices[i] < BlockNum$
  - $0 < actualSeqLengths[i] \le 8$, and the cumulative sum of $actualSeqLengths[i]$ equals $T$
  - When numAcceptedTokens is provided, $1 \le numAcceptedTokens[i] \le actualSeqLengths[i]$.
  - $-1 \le query[i][j][k] \le 1$
  - $-1 \le key[i][j][k] \le 1$
  - $g[i][j] < 0$
  - $gk[i][j][k] < 0$
  - $0 < beta[i][j] < 1$

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_recurrent_gated_delta_rule.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr, const char *name)
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
    // Initialize AscendCL.
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
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Call aclCreateTensor to create an aclTensor.
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

    // 2. Construct the inputs and outputs. Custom construction is required based on the API.
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *gDeviceAddr = nullptr;
    void *gkDeviceAddr = nullptr;
    void *betaDeviceAddr = nullptr;
    void *stateRefDeviceAddr = nullptr;
    void *actSeqLenDeviceAddr = nullptr;
    void *ssmStaIdDeviceAddr = nullptr;
    void *numAccTokDeviceAddr = nullptr;
    void *attnOutDeviceAddr = nullptr;

    aclTensor *query = nullptr;
    aclTensor *key = nullptr;
    aclTensor *value = nullptr;
    aclTensor *g = nullptr;
    aclTensor *gk = nullptr;
    aclTensor *beta = nullptr;
    aclTensor *stateRef = nullptr;
    aclTensor *actSeqLen = nullptr;
    aclTensor *ssmStaId = nullptr;
    aclTensor *numAccTok = nullptr;
    aclTensor *attnOut = nullptr;

    // Custom inputs and attributes.
    int32_t batchSize = 2;
    int32_t mtp = 2;
    int32_t headKNum = 4;
    int32_t headVNum = 8;
    int32_t dimV = 32;
    int32_t dimK = 32;


    std::vector<int64_t> stateShape = {batchSize * mtp, headVNum, dimV, dimK};
    std::vector<int64_t> qkShape = {batchSize * mtp, headKNum, dimK};
    std::vector<int64_t> vShape = {batchSize * mtp, headVNum, dimV};
    std::vector<int64_t> gShape = {batchSize * mtp, headVNum};
    std::vector<int64_t> gkShape = {batchSize * mtp, headVNum, dimK};
    std::vector<int64_t> actSeqLenShape = {batchSize};
    std::vector<int64_t> ssmStaIdShape = {batchSize * mtp};
    std::vector<float> stateRefHostData(GetShapeSize(stateShape));
    std::vector<float> queryHostData(GetShapeSize(qkShape));
    std::vector<float> keyHostData(GetShapeSize(qkShape));
    std::vector<float> valueHostData(GetShapeSize(vShape));
    std::vector<float> gHostData(GetShapeSize(gShape));
    std::vector<float> gkHostData(GetShapeSize(gkShape));
    std::vector<float> betaHostData(GetShapeSize(gShape));
    std::vector<int32_t> actSeqLenHostData(batchSize, mtp);
    std::vector<int32_t> ssmStaIdHostData(batchSize * mtp);
    std::vector<int32_t> numAccTokHostData(batchSize, 1);
    for (int i = 0; i < stateRefHostData.size(); i++) {
        stateRefHostData[i] = 0.5;
    }
    for (int i = 0; i < queryHostData.size(); i++) {
        queryHostData[i] = 0.5;
    }
    for (int i = 0; i < keyHostData.size(); i++) {
        keyHostData[i] = 0.5;
    }
    for (int i = 0; i < valueHostData.size(); i++) {
        valueHostData[i] = 0.5;
    }
    for (int i = 0; i < gHostData.size(); i++) {
        gHostData[i] = -0.5;
    }
    for (int i = 0; i < gkHostData.size(); i++) {
        gkHostData[i] = -0.5;
    }
    for (int i = 0; i < betaHostData.size(); i++) {
        betaHostData[i] = 0.5;
    }
    for (int i = 0; i < ssmStaIdHostData.size(); i++) {
        ssmStaIdHostData[i] = i;
    }

    std::vector<float> attnOutHostData(valueHostData);

    ret = CreateAclTensor(stateRefHostData, stateShape, &stateRefDeviceAddr, aclDataType::ACL_BF16, &stateRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryHostData, qkShape, &queryDeviceAddr, aclDataType::ACL_BF16, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, qkShape, &keyDeviceAddr, aclDataType::ACL_BF16, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueHostData, vShape, &valueDeviceAddr, aclDataType::ACL_BF16, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gHostData, gShape, &gDeviceAddr, aclDataType::ACL_FLOAT, &g);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gkHostData, gkShape, &gkDeviceAddr, aclDataType::ACL_FLOAT, &gk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, gShape, &betaDeviceAddr, aclDataType::ACL_BF16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actSeqLenHostData, actSeqLenShape, &actSeqLenDeviceAddr, aclDataType::ACL_INT32, &actSeqLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(ssmStaIdHostData, ssmStaIdShape, &ssmStaIdDeviceAddr, aclDataType::ACL_INT32, &ssmStaId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numAccTokHostData, actSeqLenShape, &numAccTokDeviceAddr, aclDataType::ACL_INT32, &numAccTok);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnOutHostData, vShape, &attnOutDeviceAddr, aclDataType::ACL_BF16, &attnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    float scale = 1.0;
    aclOpExecutor *executor;
    // Call the first-phase API aclnnRecurrentGatedDeltaRuleGetWorkspaceSize.
    ret = aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(query, key, value, beta, stateRef, actSeqLen, ssmStaId, g,
                                                       gk, numAccTok, scale, attnOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRecurrentGatedDeltaRuleGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API aclnnRecurrentGatedDeltaRule.
    ret = aclnnRecurrentGatedDeltaRule(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRecurrentGatedDeltaRule failed. ERROR: %d\n", ret); return ret);

    // 4. Synchronize and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value, and copy the result on the device side to the host side. The code needs to be modified based on the specific API definition.
    PrintOutResult(stateShape, &stateRefDeviceAddr, "finalState");
    PrintOutResult(vShape, &attnOutDeviceAddr, "out");

    // 6. Release aclTensor and aclScalar. The code needs to be modified based on the specific API definition.
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(g);
    aclDestroyTensor(gk);
    aclDestroyTensor(beta);
    aclDestroyTensor(stateRef);
    aclDestroyTensor(actSeqLen);
    aclDestroyTensor(ssmStaId);
    aclDestroyTensor(numAccTok);
    aclDestroyTensor(attnOut);

    // 7. Release device resources.
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(gDeviceAddr);
    aclrtFree(gkDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(stateRefDeviceAddr);
    aclrtFree(actSeqLenDeviceAddr);
    aclrtFree(ssmStaIdDeviceAddr);
    aclrtFree(numAccTokDeviceAddr);
    aclrtFree(attnOutDeviceAddr);
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
