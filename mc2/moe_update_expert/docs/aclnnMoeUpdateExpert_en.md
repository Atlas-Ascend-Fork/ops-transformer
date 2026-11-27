# aclnnMoeUpdateExpert

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:35:31.747Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
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

This API supports load balancing and expert pruning. The mapped expert table and mask can be passed to the MoE layer for data distribution and processing.

- Load balancing: To address load imbalance scenarios, this operator maps the topK Logical Expert IDs of each token to physical card IDs. The calculation method is as follows:

  For the i-th value in flattened expert_ids, the token index is i // K. F is the number of columns in eplb_table:

    ```python
    # expert_ids and eplb_table are flattened in row-major order.
    expert_id = expert_ids[i]
    table_offset = expert_id * F
    count = eplb_table[table_offset]
    token_id = i // K
    if count == 1:
        place_idx = 1
    elif balance_mode == 0:
        mode_value = (world_size + count - 1) // count
        place_idx = local_rank_id // mode_value + 1
    else:
        place_idx = token_id % count + 1
    new_expert_id = eplb_table[table_offset + place_idx]
    ```

- Expert pruning: supports pruning the topK experts sent by a token based on a threshold. The calculation method is as follows:

  Broadcast active_mask with shape (BS,) to active_mask_tensor with shape (BS, K), where experts corresponding to False in BS are directly pruned. For elements where active_mask_tensor is True, those meeting the condition will also be pruned.

    ```python
    active_mask_tensor = broadcast(active_mask, (BS, K))
    for i in range(BS):
        expert_scales_vec[:] = sum(expert_scales[i, :] * pruning_threshold[:])
        balanced_active_mask[i, :] = (expert_scales[i, :] < expert_scales_vec[:]) && active_mask_tensor[i, :]
    ```

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMoeUpdateExpertGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the computation process of the operator, and then call the "aclnnMoeUpdateExpert" API to execute the computation.

```cpp
aclnnStatus aclnnMoeUpdateExpertGetWorkspaceSize(
    const aclTensor* expertIds,
    const aclTensor* eplbTable,
    const aclTensor* expertScalesOptional,
    const aclTensor* pruningThresholdOptional,
    const aclTensor* activeMaskOptional,
    int64_t          localRankId,
    int64_t          worldSize,
    int64_t          balanceMode,
    aclTensor*       balancedExpertIds,
    aclTensor*       balancedActiveMask,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeUpdateExpert(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnMoeUpdateExpertGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1390px"> <colgroup>
    <col style="width: 150px">
    <col style="width: 160px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 180px">
    <col style="width: 180px">
    <col style="width: 190px">
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
    <td>expertIds(aclTensor*)</td>
    <td>Input</td>
    <td>topK expert indices for each token.</td>
    <td>A 2D tensor with shape (BS, K).</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>eplbTable(aclTensor*)</td>
    <td>Input</td>
    <td>Mapping table from logic experts to physical experts (the caller must ensure the values are correct).</td>
    <td><ul><li>A total of world_size * place_per_rank expert instances.</li><li>The first column of each row indicates the number of deployed instances of the corresponding Logic Expert, count (Value Range [1, min(world_size, F - 1)]), and the subsequent [1, count] columns indicate instance IDs (Value Range [0, world_size*place_per_rank), without duplicates).</li><li>Must be a 2D tensor with shape (moeExpertNum, F).</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>expertScalesOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Scale weights of the topK experts for each token.</td>
    <td><ul><li>Must be sorted in descending order within each token.</li><li>Valid data or a null pointer is acceptable. If valid data is passed, pruningThresholdOptional must also be passed with valid data.</li><li>Must be a 2D tensor with shape (BS, K).</li></ul></td>
    <td>FLOAT16, BFLOAT16, FLOAT</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>pruningThresholdOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Minimum threshold for expert scale weights (a token is pruned if its corresponding expert scale is less than the threshold).</td>
    <td><ul><li>Valid data or a null pointer. If valid data is passed, valid data must also be passed for expertScalesOptional.</li><li>A 1D or 2D tensor with shape (K,) or (1, K).</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>1/2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>activeMaskOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Indicates whether tokens participate in communication.</td>
    <td><ul><li>Can be valid data or a null pointer. If valid data is passed, valid data must also be passed for both expertScalesOptional and pruningThresholdOptional.</li><li>true indicates participation in communication, and true must precede false (for example, {true, false, true} is invalid).</li><li>If a null pointer is passed, all tokens participate in communication by default.</li><li>Must be a 1D tensor with shape (BS,).</li></ul></td>
    <td>BOOL</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>localRankId (int64_t)</td>
    <td>Input</td>
    <td>ID of this device.</td>
    <td><ul><li>When balanceMode=0, the value range is [0, worldSize).</li><li>localRankId of each device in the same communicator is unique.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>worldSize (int64_t)</td>
    <td>Input</td>
    <td>Size of the communicator.</td>
    <td>Value range [2, 768] when balanceMode=0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>balanceMode (int64_t)</td>
    <td>Input</td>
    <td>Balancing rule. Defaults to 0.</td>
    <td><ul><li>0: dispatch by rank.</li><li>1: dispatch by token.</li><li>Value range [0, 1].</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>balancedExpertIds (aclTensor*)</td>
    <td>Output</td>
    <td>Instance IDs of the physical experts where the topK experts of each token reside after mapping.</td>
    <td><ul><li>Must be a 2D tensor with the shape (BS, K).</li><li>The data type and data format must be the same as those of expertIds.</li></ul></td>
    <td>INT32, INT64</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>balancedActiveMask (aclTensor*)</td>
    <td>Output</td>
    <td>Balanced activeMask after pruning. Valid only when valid data is passed to expertScalesOptional and pruningThresholdOptional.</td>
    <td>2D tensor with shape (BS, K).</td>
    <td>BOOL</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>workspaceSize (uint64_t*)</td>
    <td>Output</td>
    <td>Returns the workspace size to be allocated on the Device.</td>
    <td>-</td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor(aclOpExecutor**)</td>
    <td>Output</td>
    <td>Returns the op executor, which contains the computation process of the operator.</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
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
    <td>The required input or output Tensor is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type of the input or output is out of the supported range.</td>
    </tr>
    <tr>
    <td class="merged-cell" rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td class="merged-cell" rowspan="2">561002</td>
    <td>1. The shape of the input or output is out of the supported range;</td>
    </tr>
    <tr>
    <td>2. The parameter value is out of the supported range.</td>
    </tr>
    </tbody>
    </table>

## aclnnMoeUpdateExpert

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
    <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Workspace size allocated on the Device side, obtained by the first-phase API <code>aclnnMoeUpdateExpertGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Executor of the op, which contains the operator computation process.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Stream for executing the task.</td>
    </tr>
    </tbody>
    </table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

1. Deterministic computation:
     - aclnnMoeUpdateExpert defaults to a deterministic implementation.

2. **Interface Compatibility and Call Sequence**:
    This API must be used together with `aclnnMoeDistributeDispatchV2` and `aclnnMoeDistributeCombineV2`/`aclnnMoeDistributeCombineAddRmsNorm`. **The call sequence is fixed as follows**:
    `aclnnMoeUpdateExpert` → `aclnnMoeDistributeDispatchV2` → `aclnnMoeDistributeCombineV2`/`aclnnMoeDistributeCombineAddRmsNorm`;

    Or use it together with `aclnnMoeDistributeDispatchV3` and `aclnnMoeDistributeCombineV3`/`aclnnMoeDistributeCombineAddRmsNormV2`. **The call sequence is fixed as follows**:
    `aclnnMoeUpdateExpert` → `aclnnMoeDistributeDispatchV3` → `aclnnMoeDistributeCombineV3`/`aclnnMoeDistributeCombineAddRmsNormV2`; For details, see [Example](#example).

3. **Parameter Consistency Requirements**:
   The `worldSize` and `moeExpertNum` parameter values used during the call must be kept consistent across all cards and across different layers of the network, and must be consistent with the corresponding parameters of `aclnnMoeDistributeDispatchV2` and `aclnnMoeDistributeCombineV2`/`aclnnMoeDistributeCombineAddRmsNorm`.

4. **Hardware-Related Definitions**:
   <term>Atlas A3 products</term>: Each device contains dual DIEs ("die" or "bare die"). Therefore, "this device" in the parameter description refers to **a single DIE**.

5. **Parameter shape format constraints**:
   - **BS**: Number of tokens finally output by this device. Value range: ( 0 < BS ≤ 512 ).
   - **K**: Number of selected topK experts. Value range: ( 0 < K ≤ 16 ), and ( 0 < K ≤ moeExpertNum ) must be satisfied.
   - **moeExpertNum**: Number of MoE experts. Value range: (0, 1024].
   - **F**: The number of columns in the mapping table `eplbTable`. Value range: [2, worldSize + 1]. The first column is the deployed instance count, count, which must satisfy 1 <= count <= min(worldSize, F - 1). Columns [1, count] contain the corresponding distinct instance IDs; the remaining columns do not participate in instance selection. F need not equal worldSize + 1 when count is smaller than worldSize.
   - **Limit on the total number of instances**: The total number of MoE expert instances deployed on all devices is at most 1024, that is, ( place_per_rank * world_size ≤ 1024 ) (`place_per_rank` indicates the number of instances deployed on a single device).
   - **Per-Card Instance Count Consistency**: The number of expert instances deployed on each card must be identical.

## Example

The following example calls the MoeUpdateExpert, MoeDistributeDispatchV2, and MoeDistributeCombineAddRmsNorm operators on <term>Atlas A3 products</term> and <term>Ascend 950PR&950DT products</term>.

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_update_expert.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_v2.h"
    #include "aclnnop/aclnn_moe_distribute_combine_v2.h"

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

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        uint32_t tpRankId;
        HcclComm hcclEpComm;
        HcclComm hcclTpComm;
        aclrtStream eplbStream;
        aclrtStream dispatchStream;
        aclrtStream combineStream;
        aclrtContext context;
    };

    constexpr uint32_t EP_WORLD_SIZE = 2;
    constexpr uint32_t TP_WORLD_SIZE = 1;
    constexpr uint32_t DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

    inline int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
        aclDataType dataType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(
            shape.data(), shape.size(), dataType, strides.data(), 0,
            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    int LaunchOneProcessUpdateExpertAndDispatchAndCombine(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
        char hcomTpName[128] = {0};

        int64_t BS = 8;
        int64_t H = 7168;
        int64_t K = 2;
        int64_t F = 2;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 0;
        int64_t sharedExpertRankNum = 0;
        int64_t moeExpertNum = 2;
        int64_t quantMode = 0;
        int64_t globalBS = BS * EP_WORLD_SIZE;
        int64_t balanceMode = 0;
        int64_t expertTokenNumsType = 1;
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupList_type = 1;
        int64_t localExpertNum;
        int64_t A;
        if (args.epRankId < sharedExpertRankNum) {
            // Shared expert device
            localExpertNum = 1;
            A = globalBS / sharedExpertRankNum;
        } else {
            // Moe expert device
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBS * (localExpertNum < K ? localExpertNum : K);
        }

        /* Construct device-side input and output variables based on the current use case */
        // Declare device-side input and output variables.
        void *xDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *eplbTableDeviceAddr = nullptr;
        void *scalesDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;
        void *expandXDeviceAddr = nullptr;
        void *dynamicScalesDeviceAddr = nullptr;
        void *expandIdxDeviceAddr = nullptr;
        void *expertTokenNumsDeviceAddr = nullptr;
        void *epRecvCountsDeviceAddr = nullptr;
        void *tpRecvCountsDeviceAddr = nullptr;
        void *expandScalesDeviceAddr = nullptr;
        void *residualXDeviceAddr = nullptr;
        void *sharedExpertXDeviceAddr = nullptr;
        void *gammaDeviceAddr = nullptr;
        void *yOutDeviceAddr = nullptr;
        void *rstdOutDeviceAddr = nullptr;
        void *xOutDeviceAddr = nullptr;
        void *balancedExpertIdsDeviceAddr = nullptr;
        void *balancedActiveMaskDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *eplbTable = nullptr;
        aclTensor *scales = nullptr;
        aclTensor *expertScales = nullptr;
        aclTensor *expandX = nullptr;
        aclTensor *dynamicScales = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *expertTokenNums = nullptr;
        aclTensor *epRecvCounts = nullptr;
        aclTensor *tpRecvCounts = nullptr;
        aclTensor *expandScales = nullptr;
        aclTensor *residualX = nullptr;
        aclTensor *sharedExpertX = nullptr;
        aclTensor *gamma = nullptr;
        aclTensor *yOut = nullptr;
        aclTensor *rstdOut = nullptr;
        aclTensor *xOut = nullptr;
        aclTensor *balancedExpertIds = nullptr;
        aclTensor *balancedActiveMask = nullptr;

        // Define the dimensions of each variable for the current use case.
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> eplbTableShape{moeExpertNum, F};
        std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? moeExpertNum + 1 : moeExpertNum, H};
        std::vector<int64_t> expertScalesShape{BS, K};
        std::vector<int64_t> expandXShape{TP_WORLD_SIZE * A, H};
        std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE * A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE * localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE * localExpertNum};
        std::vector<int64_t> expandScalesShape{A};
        std::vector<int64_t> residualXShape{BS, 1, H};
        std::vector<int64_t> sharedExpertXShape{BS, 1, H};
        std::vector<int64_t> gammaShape{H, };
        std::vector<int64_t> yOutShape{BS, 1, H};
        std::vector<int64_t> rstdOutShape{BS, 1, 1};
        std::vector<int64_t> xOutShape{BS, 1, H};
        std::vector<int64_t> balancedExpertIdsShape{BS, K};
        std::vector<int64_t> balancedActiveMaskShape{BS, K};

        int64_t xShapeSize = GetShapeSize(xShape);
        int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
        int64_t scalesShapeSize = GetShapeSize(scalesShape);
        int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);
        int64_t expandXShapeSize = GetShapeSize(expandXShape);
        int64_t dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
        int64_t expandIdxShapeSize = GetShapeSize(expandIdxShape);
        int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
        int64_t epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
        int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
        int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);
        int64_t residualXShapeSize = GetShapeSize(residualXShape);
        int64_t sharedExpertXShapeSize = GetShapeSize(sharedExpertXShape);
        int64_t gammaShapeSize = GetShapeSize(gammaShape);
        int64_t yOutShapeSize = GetShapeSize(yOutShape);
        int64_t rstdOutShapeSize = GetShapeSize(rstdOutShape);
        int64_t xOutShapeSize = GetShapeSize(xOutShape);
        int64_t balancedExpertIdsShapeSize = GetShapeSize(balancedExpertIdsShape);
        int64_t balancedActiveMaskShapeSize = GetShapeSize(balancedActiveMaskShape);

        // Construct host-side variables.
        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            // Each token is sent to MoE experts {0, 1, ... k - 1}.
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }
        // Construct eplb_table data: 8 MoE experts, each with 1 instance, and one MoE expert instance deployed on each device. For example, the first two numbers 1, 0 indicate that 1 instance of the first MoE expert is deployed on place0.
        std::vector<int32_t> eplbTableHostData = {1, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7};

        std::vector<float> scalesHostData(scalesShapeSize, 0.1);
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0.1);
        std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
        std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
        std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);
        std::vector<int16_t> residualXHostData(residualXShapeSize, 1);
        std::vector<int16_t> sharedExpertXHostData(sharedExpertXShapeSize, 1);
        std::vector<int16_t> gammaHostData(gammaShapeSize, 1);
        std::vector<int16_t> yOutHostData(yOutShapeSize, 0);
        std::vector<float> rstdOutHostData(rstdOutShapeSize, 0);
        std::vector<int16_t> xOutHostData(xOutShapeSize, 0);
        std::vector<int32_t> balancedExpertIdsHostData(balancedExpertIdsShapeSize, 0);
        std::vector<int32_t> balancedActiveMaskHostData(balancedActiveMaskShapeSize, 0);

        // Construct device-side variables.
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        ret = CreateAclTensor(eplbTableHostData, eplbTableShape, &eplbTableDeviceAddr, aclDataType::ACL_INT32, &eplbTable);
        ret = CreateAclTensor(balancedExpertIdsHostData, balancedExpertIdsShape, &balancedExpertIdsDeviceAddr, aclDataType::ACL_INT32, &balancedExpertIds);
        ret = CreateAclTensor(balancedActiveMaskHostData, balancedActiveMaskShape, &balancedActiveMaskDeviceAddr, aclDataType::ACL_BOOL, &balancedActiveMask);
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32, &epRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(residualXHostData, residualXShape, &residualXDeviceAddr, aclDataType::ACL_BF16, &residualX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(sharedExpertXHostData, sharedExpertXShape, &sharedExpertXDeviceAddr, aclDataType::ACL_BF16, &sharedExpertX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_BF16, &gamma);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_BF16, &yOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT, &rstdOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_BF16, &xOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        /* Declare variables required for operator execution. */
        uint64_t eplbworkspaceSize = 0;
        aclOpExecutor *eplbexecutor = nullptr;
        void *eplbWorkspaceAddr = nullptr;

        uint64_t dispatchWorkspaceSize = 0;
        aclOpExecutor *dispatchExecutor = nullptr;
        void *dispatchWorkspaceAddr = nullptr;

        uint64_t combineWorkspaceSize = 0;
        aclOpExecutor *combineExecutor = nullptr;
        void *combineWorkspaceAddr = nullptr;

        /**************************************** Call eplb ********************************************/
        ret = aclnnMoeUpdateExpertGetWorkspaceSize(expertIds, eplbTable, nullptr, nullptr, nullptr, args.epRankId,
            EP_WORLD_SIZE, balanceMode, balancedExpertIds, balancedActiveMask, &eplbworkspaceSize, &eplbexecutor);

        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnMoeUpdateExpertGetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (eplbworkspaceSize > 0) {
            ret = aclrtMalloc(&eplbWorkspaceAddr, eplbworkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMoeUpdateExpert(eplbWorkspaceAddr, eplbworkspaceSize, eplbexecutor, args.eplbStream);
        ret = aclrtSynchronizeStreamWithTimeout(args.eplbStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeUpdateExpert failed. ret = %d \n", ret);  \
            return ret);

        /**************************************** Call dispatch ********************************************/

        ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(x, balancedExpertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
                args.tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBS,
                expertTokenNumsType, nullptr, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMoeDistributeDispatchV2(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2 failed. ret = %d \n", ret);  \
            return ret);

        /**************************************** Call combine ********************************************/

        // Call the first-phase API of the combine operator.
        ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(expandX, expertIds, expandIdx, epRecvCounts, expertScales,
            tpRecvCounts, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE, args.tpRankId,
            expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode,
            groupList_type, nullptr, x, &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2GetWorkspaceSize failed. ret = %d \n", ret); return ret
        );
        // Allocate device memory based on the workspaceSize calculated by the first-phase API of the combine operator.
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d \n", ret);
                    return ret);
        }

        // Call the second-phase API of the combine operator.
        ret = aclnnMoeDistributeCombineV2(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2 failed. ret = %d \n", ret);
                return ret);

        LOG_PRINT("[INFO] device_%d aclnnMoeUpdateExpert, aclnnMoeDistributeDispatchV2 and aclnnMoeDistributeCombineV2    \
                    execute successfully.\n", args.rankId);

        // Release device resources.
        if (dispatchWorkspaceSize > 0) {
            aclrtFree(dispatchWorkspaceAddr);
        }
        if (combineWorkspaceSize > 0) {
            aclrtFree(combineWorkspaceAddr);
        }
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (expertIds != nullptr) {
            aclDestroyTensor(expertIds);
        }
        if (eplbTable != nullptr) {
            aclDestroyTensor(eplbTable);
        }
        if (scales != nullptr) {
            aclDestroyTensor(scales);
        }
        if (expertScales != nullptr) {
            aclDestroyTensor(expertScales);
        }
        if (expandX != nullptr) {
            aclDestroyTensor(expandX);
        }
        if (dynamicScales != nullptr) {
            aclDestroyTensor(dynamicScales);
        }
        if (expandIdx != nullptr) {
            aclDestroyTensor(expandIdx);
        }
        if (expertTokenNums != nullptr) {
            aclDestroyTensor(expertTokenNums);
        }
        if (epRecvCounts != nullptr) {
            aclDestroyTensor(epRecvCounts);
        }
        if (tpRecvCounts != nullptr) {
            aclDestroyTensor(tpRecvCounts);
        }
        if (expandScales != nullptr) {
            aclDestroyTensor(expandScales);
        }
        if (residualX != nullptr) {
            aclDestroyTensor(residualX);
        }
        if (sharedExpertX != nullptr) {
            aclDestroyTensor(sharedExpertX);
        }
        if (gamma != nullptr) {
            aclDestroyTensor(gamma);
        }
        if (yOut != nullptr) {
            aclDestroyTensor(yOut);
        }
        if (rstdOut != nullptr) {
            aclDestroyTensor(rstdOut);
        }
        if (xOut != nullptr) {
            aclDestroyTensor(xOut);
        }
        if (balancedExpertIds != nullptr) {
            aclDestroyTensor(balancedExpertIds);
        }
        if (balancedActiveMask != nullptr) {
            aclDestroyTensor(balancedActiveMask);
        }
        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (expertIdsDeviceAddr != nullptr) {
            aclrtFree(expertIdsDeviceAddr);
        }
        if (eplbTableDeviceAddr != nullptr) {
            aclrtFree(eplbTableDeviceAddr);
        }
        if (scalesDeviceAddr != nullptr) {
            aclrtFree(scalesDeviceAddr);
        }
        if (expertScalesDeviceAddr != nullptr) {
            aclrtFree(expertScalesDeviceAddr);
        }
        if (expandXDeviceAddr != nullptr) {
            aclrtFree(expandXDeviceAddr);
        }
        if (dynamicScalesDeviceAddr != nullptr) {
            aclrtFree(dynamicScalesDeviceAddr);
        }
        if (expandIdxDeviceAddr != nullptr) {
            aclrtFree(expandIdxDeviceAddr);
        }
        if (expertTokenNumsDeviceAddr != nullptr) {
            aclrtFree(expertTokenNumsDeviceAddr);
        }
        if (epRecvCountsDeviceAddr != nullptr) {
            aclrtFree(epRecvCountsDeviceAddr);
        }
        if (expandScalesDeviceAddr != nullptr) {
            aclrtFree(expandScalesDeviceAddr);
        }
        if (tpRecvCountsDeviceAddr != nullptr) {
            aclrtFree(tpRecvCountsDeviceAddr);
        }
        if (residualXDeviceAddr != nullptr) {
            aclrtFree(residualXDeviceAddr);
        }
        if (sharedExpertXDeviceAddr != nullptr) {
            aclrtFree(sharedExpertXDeviceAddr);
        }
        if (gammaDeviceAddr != nullptr) {
            aclrtFree(gammaDeviceAddr);
        }
        if (yOutDeviceAddr != nullptr) {
            aclrtFree(yOutDeviceAddr);
        }
        if (rstdOutDeviceAddr != nullptr) {
            aclrtFree(rstdOutDeviceAddr);
        }
        if (xOutDeviceAddr != nullptr) {
            aclrtFree(xOutDeviceAddr);
        }
        if (balancedExpertIdsDeviceAddr != nullptr) {
            aclrtFree(balancedExpertIdsDeviceAddr);
        }
        if (balancedActiveMaskDeviceAddr != nullptr) {
            aclrtFree(balancedActiveMaskDeviceAddr);
        }
        HcclCommDestroy(args.hcclEpComm);
        HcclCommDestroy(args.hcclTpComm);
        aclrtDestroyStream(args.eplbStream);
        aclrtDestroyStream(args.dispatchStream);
        aclrtDestroyStream(args.combineStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

        aclrtStream eplbStream[DEV_NUM];
        aclrtStream dispatchStream[DEV_NUM];
        aclrtStream combineStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&eplbStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        }

        int32_t devicesEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
                devicesEp[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp[tpId], commsEp[tpId]);
            CHECK_RET(ret == ACL_SUCCESS,
                        LOG_PRINT("[ERROR] HcclCommInitAll ep %d failed, ret %d\n", tpId, ret); return ret);
        }

        int32_t devicesTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
                devicesTp[epId][tpId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            ret = HcclCommInitAll(TP_WORLD_SIZE, devicesTp[epId], commsTp[epId]);
            CHECK_RET(ret == ACL_SUCCESS,
                        LOG_PRINT("[ERROR] HcclCommInitAll tp %d failed, ret %d\n", epId, ret); return ret);
        }

        Args args[DEV_NUM];
        // Each thread calls each device to execute the operator.
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId / TP_WORLD_SIZE;
            uint32_t tpRankId = rankId % TP_WORLD_SIZE;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].tpRankId = tpRankId;
            args[rankId].hcclEpComm = commsEp[tpRankId][epRankId];
            args[rankId].hcclTpComm = commsTp[epRankId][tpRankId];
            args[rankId].eplbStream = eplbStream[rankId];
            args[rankId].dispatchStream = dispatchStream[rankId];
            args[rankId].combineStream = combineStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new (std::nothrow) std::thread(&LaunchOneProcessUpdateExpertAndDispatchAndCombine, std::ref(args[rankId])));
        }

        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");

        return 0;
    }
    ```
