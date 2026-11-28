# aclnnMoeDistributeDispatchTeardown

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:26:20.327Z -->

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

- Description:

    - Receives data sent by the AllToAllV communication in the EP (Expert Parallelism) domain of the MOE layer. Data sending is completed by aclnnMoeDistributeDispatchSetup. This API confirms the communication status and organizes the data.

    - Note that this API must be used together with aclnnMoeDistributeDispatchSetup, aclnnMoeDistributeCombineSetup, and aclnnMoeDistributeCombineTeardown.

## Function Prototype

Each operator uses a two-phase API. Call "aclnnMoeDistributeDispatchTeardownGetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call "aclnnMoeDistributeDispatchTeardown" to execute the computation.

```cpp
aclnnStatus aclnnMoeDistributeDispatchTeardownGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* y,
    const aclTensor* expertIds,
    const aclTensor* commCmdInfo,
    const char* groupEp,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    int64_t expertShardType,
    int64_t sharedExpertNum,
    int64_t sharedExpertRankNum,
    int64_t quantMode,
    int64_t globalBs,
    int64_t expertTokenNumsType,
    int64_t commType,
    char* commAlg,
    aclTensor* expandXOut,
    aclTensor* dynamicScalesOut,
    aclTensor* assistInfoForCombineOut,
    aclTensor* expertTokenNumsOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnMoeDistributeDispatchTeardown(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeDispatchTeardownGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1550px"> <colgroup>
    <col style="width: 110px">
    <col style="width: 120px">
    <col style="width: 305px">
    <col style="width: 330px">
    <col style="width: 210px">
    <col style="width: 100px">
    <col style="width: 180px">
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
    <th>Dimensions(shape)</th>
    <th>Non-contiguous Tensor</th>
    </tr>
    </thead>
    <tbody>

    <tr>
    <td>x</td>
    <td>Input</td>
    <td>Indicates the token data sent by the current device.</td>
    <td>Must be a 2D tensor.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>(Bs, H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>y</td>
    <td>Input</td>
    <td>Output of aclnnMoeDistributeDispatchSetup. Indicates the communication data to be sent by the current device. The communication data is the input token data reordered by the algorithm. If quantization is required, quantize the input tokens first, and then reorder the data.</td>
    <td>Must be a 2D Tensor.</td>
    <td>FLOAT16, BFLOAT16, INT8, HIFP8, FP8E5M2, FP8E4M3</td>
    <td>ND</td>
    <td>(BS * (K + sharedExpertNum), tokenMsgSize)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>TopK expert indexes of each token.</td>
    <td>Must be a 2D Tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(Bs, K)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>commCmdInfo</td>
    <td>Input</td>
    <td>Output of aclnnMoeDistributeDispatchSetup, communication cmd information.</td>
    <td>Must be a 1D Tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(BS * (K + sharedExpertNum) + epWorldSize * localExpertNum) * 16</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>EP communication domain name (expert parallelism communication domain).</td>
    <td>String length range: [1, 128).</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>Input</td>
    <td>Size of the EP communication domain.</td>
    <td>Value range: [2, 384].</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>Input</td>
    <td>ID of the current device in the EP domain.</td>
    <td>Value range [0, epWorldSize). The epRankId of each device in the same EP communication domain must be unique.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>Input</td>
    <td>MoE expert count.</td>
    <td>Value range (0, 512]. Must satisfy moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>Input</td>
    <td>Distribution type of shared expert devices.</td>
    <td>Currently, only 0 is supported, indicating that shared expert devices are placed before MoE expert devices.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>Input</td>
    <td>Indicates the shared expert count.</td>
    <td>Value range in the current version: [0, 4]</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>Input</td>
    <td>Indicates the quantity of shared expert devices.</td>
    <td>Value range: [0, epWorldSize / 2]</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>quantMode</td>
    <td>Input</td>
    <td>Indicates the quantization mode.</td>
    <td>Value range: [0, 4]. 0 indicates non-quantization, 1 indicates static quantization, 2 indicates Pertoken dynamic quantization, 3 indicates Pergroup dynamic quantization, and 4 indicates MX quantization. Currently, only 0 and 4 are supported.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBs</td>
    <td>Input</td>
    <td>Global batch size across the EP domain.</td>
    <td><ul><li>When all ranks have the same Bs, globalBs = Bs * epWorldSize or 0.</li><li>When Bs differs across ranks, globalBs = maxBs * epWorldSize (maxBs is the maximum Bs of a single device).</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNumsType</td>
    <td>Input</td>
    <td>Indicates the semantic type of the values in the output expertTokenNums.</td>
    <td><ul><li>0 indicates that the output in expertTokenNums is the prefix sum of the number of tokens processed by each expert; 1 indicates that the output in expertTokenNums is the number of tokens processed by each expert.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commType</td>
    <td>Input</td>
    <td>Indicates the communication scheme selection.</td>
    <td>Value range [0, 2]. 0 indicates the AICPU-SDMA scheme, 1 indicates the CCU scheme, and 2 indicates the URMA scheme. Only 2 is supported in the current version.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>Input</td>
    <td>Indicates the communication-affinity memory layout algorithm.</td>
    <td>Reserved field. Not supported in the current version. Pass a null pointer or an empty string.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandXOut</td>
    <td>Output</td>
    <td>Token features expanded based on expertIds.</td>
    <td>Must be a 2D tensor.</td>
    <td>FLOAT16, BFLOAT16, INT8, FP8E5M2, FP8E4M3, HIFP8</td>
    <td>ND</td>
    <td>(A, H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dynamicScalesOut</td>
    <td>Output</td>
    <td>Dynamic quantization parameters of each token.</td>
    <td>Must be a 1D tensor. Actual valid values are output only when quantMode is 2, 3, or 4.</td>
    <td>FLOAT32, FP8E8M0</td>
    <td>ND</td>
    <td>(A, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombineOut</td>
    <td>Output</td>
    <td>Number of tokens sent to the same expert, corresponding to assistInfoForCombine in the Combine series of operators.</td>
    <td>Must be a 1D Tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(A * 128, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertTokenNumsOut</td>
    <td>Output</td>
    <td>Indicates the number of tokens received by each expert.</td>
    <td>Must be a 1D tensor.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>(localExpertNum, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the size of the workspace to be allocated on the Device side.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the op executor, which contains the calculation process of the operator.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT products</term>:
        - The value range of groupEp string length is [1, 128).
        - The value range of epWorldSize is [2, 384]. Currently, only 2 and 8 are supported.
        - The value range of epRankId is [0, epWorldSize). epRankId must be unique for each device in the same EP communication domain.
        - The value range of moeExpertNum is (0, 512].
        - expertShardType supports only 0 currently, indicating that shared expert devices are placed before MoE expert devices.
        - sharedExpertNum: the current value range is [0, 4].
        - sharedExpertRankNum: value range [0, epWorldSize / 2].
        - globalBs: if the number of Bs is the same on each rank, globalBs = Bs \* epWorldSize or globalBs = 0; if the number of Bs is different on each rank, globalBs = maxBs \* epWorldSize, where maxBs indicates the maximum value of Bs on a single device.
        - expertTokenNumsType supports only 1 currently.
        - commType supports only 2 currently.
        - commAlg is a reserved field and is not supported in the current version. Pass NULL.

    <!-- end id7 -->
    <!-- npu="A3" id8 -->
    - <term>Atlas A3 products</term>  :
        - groupEp: string length range [1, 128).
        - epWorldSize: value range [2, 384].
        - The value range of epRankId is [0, epWorldSize). epRankId must be unique for each device in the same EP communication domain.
        - The value range of moeExpertNum is (0, 512].
        - expertShardType currently supports only 0, indicating that shared expert devices are placed before MoE expert devices.
        - The value range of sharedExpertNum is currently [0, 4].
        - The value range of sharedExpertRankNum is [0, epWorldSize / 2].
        - globalBs: if the Bs value is the same on each rank, globalBs = Bs \* epWorldSize or globalBs = 0; if the Bs value differs across ranks, globalBs = maxBs \* epWorldSize, where maxBs indicates the maximum Bs value on a single device.
        - commType currently supports only 0.
        - commAlg is not supported in the current version. Pass a null pointer.

    <!-- end id8 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 282px">
  <col style="width: 120px">
  <col style="width: 747px">
  </colgroup>
  <thead>
  <tr>
  <th>Return Values</th>
  <th>Error Code</th>
  <th>Description</th>
  </tr>
  </thead>
  <tbody>
  <tr>
  <td>ACLNN_ERR_PARAM_NULLPTR</td>
  <td>161001</td>
  <td>The required input or output tensor is a null pointer.</td>
  </tr>
  <tr>
  <td>ACLNN_ERR_PARAM_INVALID</td>
  <td>161002</td>
  <td>The data type of an input or output tensor is not in the supported range.</td>
  </tr>
  <tr>
  <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
  <td rowspan="2">561002</td>
  <td>The input and output shapes are not in the supported range.</td>
  </tr>
  <tr>
      <td>The parameter value is not in the supported range.</td>
  </tr>
  </tbody>
  </table>

## aclnnMoeDistributeDispatchTeardown

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 770px">
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
        <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnMoeDistributeCombineV2GetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor of the op. It contains the operator computation flow.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
    </tr>
    </tbody>
    </table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

1. Deterministic computation:
     - aclnnMoeDistributeDispatchTeardown defaults to a deterministic implementation.

2. The aclnnMoeDistributeDispatchTeardown API must be used together with the aclnnMoeDistributeDispatchSetup, aclnnMoeDistributeCombineSetup, and aclnnMoeDistributeCombineTeardown APIs.

3. The values of the `groupEp`, `epWorldSize`, `moeExpertNum`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBs`, `commQuantMode`, `commType`, and `commAlg` parameters used during the API call must be consistent across all devices; the values of the `groupEp`, `epWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBs`, `commQuantMode`, `commType`, and `commAlg` parameters must also be consistent across different layers in the network and consistent with the corresponding parameters of `aclnnMoeDistributeDispatchTeardown`, `aclnnMoeDistributeCombineSetup`, and `aclnnMoeDistributeCombineTeardown`.

4. <term>Atlas A3 products</term>: A single device contains dual DIEs (referred to as "die" or "bare die"), so "the current device" in the Parameters section refers to a single DIE.

5. Description of the shape format in Parameters:
    - A: Indicates the maximum token quantity that the current device can receive. The value range is as follows:

      - For MoE experts, when globalBs is 0, A = `BS` \* `epWorldSize` \* min(`localExpertNum`, `K`) must be met; when `globalBs` is not 0, A = `globalBs` \* min(`localExpertNum`, `K`) must be met.
      - For shared experts, when `globalBs` is 0, A = `BS` \* `epWorldSize` \* `sharedExpertNum` / `sharedExpertRankNum` must be met; when globalBs is not 0, A = `globalBs` \* `sharedExpertNum` / `sharedExpertRankNum` must be met.
    - H: Indicates the hidden size. The value range is [1024, 8192]. Currently, only 4096 and 7168 are supported.
    - BS: Indicates the batch sequence size, that is, the quantity of tokens finally output by the current device. The value range is 0 < `BS` ≤ 512. Currently, only 8, 16, and 256 are supported.
    - K: Indicates that topK experts are selected. The value range is 0 < `K` ≤ 16, and 0 < `K` ≤ `moeExpertNum` must also be met. Currently, only 6 and 8 are supported.
    - localExpertNum: indicates the number of experts on the current card.

      - For a shared expert card, localExpertNum = 1.
      - For MoE expert devices, localExpertNum = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`). Currently, only 32 is supported for moeExpertNum.
    - tokenMsgSize: Indicates the dimension information of each token during data communication.
      - In non-quantization use cases, tokenMsgSize = Align256(H).
      - In quantization use cases, tokenMsgSize = Align512(Align32(H) + 4 ), where AlignN(x) = ((x + N - 1) / N) \* N.

    - Shared experts are not supported in the current version. sharedExpertNum and sharedExpertRankNum currently support only 0.

6. HCCL_BUFFSIZE:

    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT products</term>:
      Before calling this API, check whether the value of the `HCCL_BUFFSIZE` environment variable is appropriate. This environment variable specifies the memory size occupied by a single communication domain, in MB. The default value is 200 MB if it is not configured. The value must be >= 2 and satisfy >= 4 \* (`localExpertNum` \* `maxBs` \* `epWorldSize` \* Align512(Align32(2 \* H) + 44) + (`K` + `sharedExpertNum`) \* `maxBs` \* Align512(2 \* `H`)), where `localExpertNum` indicates the number of local experts on devices that use MoE experts, Align512(x) = ((x + 512 - 1) / 512) \* 512, and Align32(x) = ((x + 32 - 1) / 32) \* 32.

    <!-- end id9 -->
    <!-- npu="A3" id10 -->
    - <term>Atlas A3 products</term>  :
      Before calling this API, check whether the value of the `HCCL_BUFFSIZE` environment variable is appropriate. This environment variable specifies the memory size occupied by a single communication domain, in MB. The default value is 200 MB if it is not configured. The value must be >= 2 and satisfy >= 2 \* (`localExpertNum` \* `maxBs` \* `epWorldSize` \* Align512(Align32(2 \* H) + 44) + (`K` + `sharedExpertNum`) \* `maxBs` \* Align512(2 \* `H`)), where `localExpertNum` indicates the number of local experts on devices that use MoE experts, Align512(x) = ((x + 512 - 1) / 512) \* 512, and Align32(x) = ((x + 32 - 1) / 32) \* 32.

    <!-- end id10 -->

7. Communication domain usage constraints:
    - In a model, the aclnnMoeDistributeDispatchSetup, aclnnMoeDistributeDispatchTeardown, aclnnMoeDistributeCombineSetup, and aclnnMoeDistributeCombineTeardown APIs support only the same EP communication domain. No other operators are allowed in this communication domain.

8. Communication mode constraints:

    <!-- npu="950" id11 -->
    - <term>Ascend 950PR&950DT products</term>: Supports URMA communication only.

    <!-- end id11 -->

## Example

<!-- npu="950" id12 -->
- <term>Ascend 950PR&950DT products</term>: aclnnMoeDistributeDispatchTeardown must be called after aclnnMoeDistributeDispatchSetup. For the example, see [aclnnMoeDistributeDispatchSetup](../../moe_distribute_dispatch_setup/docs/aclnnMoeDistributeDispatchSetup_en.md).

<!-- end id12 -->
