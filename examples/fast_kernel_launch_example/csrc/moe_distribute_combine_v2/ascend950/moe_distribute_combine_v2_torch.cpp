/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */
/*!
 * \file moe_distribute_combine_v2_torch.cpp
 * \brief A5 (ascend950) torch binding for the <<<>>> direct-launch MoeDistributeCombineV2 example.
 */

#include <ATen/ATen.h>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <torch/all.h>

#include "acl/acl_prof.h"
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "moe_distribute_combine_v2_entry.h"
#include "mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2_tiling.h"
#include "moe_distribute_combine_v2_torch.h"
#include "mc2_cc_tiling_shim.h"
#include "moe_distribute_combine_v2_torch_validate.h"
#include "../../common/inc/kernel/mc2_profiling.h"

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {

/*
 * A5 example 本地补充：镜像 mc2 host 侧 tiling 符号（mc2_tiling_utils.h /
 * matmul_formulaic_tiling.h 的 AicpuComType / A5_CCU_ENGINE）。
 * 不直接 include 原头：其依赖 graph/matmul 等 ops 框架头，与 example 自建构建不兼容
 * （mc2_hcom_topo_info.h 缺失 + formulaic_tiling_datatype.h 枚举冲突）。
 */
namespace mc2tiling {
enum class AicpuComType {
    HCCL_CMD_ALLTOALLV = 8, // A5-MTE 路径（对齐 A3 tiling SetHcommCfg 的 OP_TYPE_ALL_TO_ALL）
    HCCL_CMD_HALFALLTOALLV = 20,
};
constexpr uint8_t AIV_ENGINE = 3; // MTE 引擎（对齐 A3 tiling 的 AIV_ENGINE）
constexpr uint8_t A5_CCU_ENGINE = 6;
} // namespace mc2tiling

namespace MoeDistributeCombineV2 {

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("MoeDistributeCombineV2(Tensor expand_x, Tensor expert_ids, Tensor assist_info_for_combine, "
          "Tensor ep_send_counts, Tensor expert_scales, Tensor mc2_context, str group_ep, int ep_world_size, "
          "int ep_rank_id, int moe_expert_num, int total_winsize_ep, *, "
          "Tensor? x_active_mask=None, Tensor? shared_expert_x=None, Tensor? ori_x=None, "
          "Tensor? const_expert_alpha_1=None, Tensor? const_expert_alpha_2=None, Tensor? const_expert_v=None, "
          "Tensor? performance_info=None, "
          "int expert_shard_type=0, int shared_expert_num=0, int shared_expert_rank_num=0, "
          "int global_bs=0, int comm_quant_mode=0, "
          "str comm_alg=\"\", int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0) "
          "-> Tensor");
}

constexpr uint32_t DIM_ONE = 1UL;
constexpr uint32_t DIM_TWO = 2UL;
constexpr uint32_t TILINGKEY_XTYPE = 10;
constexpr uint32_t WORKSPACESIZE = 16 * 1024 * 1024;
constexpr uint32_t AI_VECTOR_CORE = 2;
constexpr uint32_t BUFFER_SINGLE = 1;
constexpr uint32_t BUFFER_DOUBLE = 2;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint32_t DTYPE_SIZE_HALF = 2;
constexpr uint32_t ALIGNED_LEN = 256U;
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint64_t A5_WIN_SIZE_MULTIPLY = 2UL;
constexpr uint8_t INT8_COMM_QUANT = 2;

static void calculate_tilingkey_a5(int32_t& tilingKey, at::ScalarType xType, int64_t commQuantMode)
{
    tilingKey += static_cast<uint64_t>(commQuantMode); // 个位: commQuantMode
    if (xType == at::kBFloat16) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_XTYPE); // 十位: bf16
    }
    return;
}

static void calculate_buffernum(MoeDistributeCombineV2Info& tilingData, const at::Tensor& expand_x,
                                int64_t comm_quant_mode)
{
    uint32_t axisH = tilingData.h;
    uint32_t axisBS = tilingData.bs;
    uint32_t axisK = tilingData.k;
    uint32_t zeroExpertNum = tilingData.zeroExpertNum;
    uint32_t copyExpertNum = tilingData.copyExpertNum;
    uint32_t constExpertNum = tilingData.constExpertNum;
    bool isInputExpertMaskFlag = tilingData.isExpertMask;
    bool isInputTokenMaskFlag = tilingData.isTokenMask;
    bool enableSpecialExpert = (constExpertNum + zeroExpertNum + copyExpertNum > 0U);
    uint32_t maxSizeTokenBuf = (axisH * expand_x.element_size() + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t hExpandXTypeSize = axisH * expand_x.element_size();
    uint32_t activeMaskAlignSize = axisBS * ((axisK * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
    uint32_t hExpandXAlign32Size = (hExpandXTypeSize + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t hFloatSize = axisH * static_cast<uint32_t>(sizeof(float));
    uint32_t hFloatAlign32Size = (hFloatSize + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size;
    uint32_t flagRcvCount = axisK + tilingData.sharedExpertNum;
    uint32_t hFloatAlign256Size = (hFloatSize + ALIGNED_LEN - 1) / ALIGNED_LEN * ALIGNED_LEN;
    uint32_t bsKNum = axisBS * axisK;
    uint32_t bsKFloatAlign = (bsKNum * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t mulBufSize = hFloatAlign256Size > bsKFloatAlign ? hFloatAlign256Size : bsKFloatAlign;

    if (isInputExpertMaskFlag || enableSpecialExpert) {
        uint32_t activeMaskAlignHalfSize = activeMaskAlignSize * sizeof(DTYPE_SIZE_HALF);
        maxSizeTokenBuf = (activeMaskAlignSize > hExpandXAlign32Size ? activeMaskAlignSize : hExpandXAlign32Size);
        maxSizeRowTmpFloatBuf =
            (activeMaskAlignHalfSize > hFloatAlign32Size ? activeMaskAlignHalfSize : hFloatAlign32Size);
    }

    // LocalWindowCopy的ub使用总量
    uint32_t totalBufferSize = 0;
    totalBufferSize = maxSizeTokenBuf + maxSizeRowTmpFloatBuf + mulBufSize + hFloatAlign32Size +
                      hExpandXAlign32Size * BUFFER_DOUBLE + flagRcvCount * STATE_OFFSET * BUFFER_DOUBLE + UB_ALIGN;
    if (isInputTokenMaskFlag) {
        uint32_t axisBsAlignSize = (axisBS * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        totalBufferSize += axisBsAlignSize + axisBsAlignSize * sizeof(DTYPE_SIZE_HALF) * BUFFER_DOUBLE;
    }
    if (isInputExpertMaskFlag) {
        totalBufferSize += (axisBS * sizeof(DTYPE_SIZE_HALF) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN +
                           (axisBS * sizeof(int32_t) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN +
                           (axisBS * axisK * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    }
    if (enableSpecialExpert && !isInputExpertMaskFlag) {
        totalBufferSize += (axisBS * sizeof(DTYPE_SIZE_HALF) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    }
    if (comm_quant_mode == INT8_COMM_QUANT) {
        // int8 通信量化额外 scale 缓冲（对齐 A3 example 的 calculate_buffernum 逻辑）
        uint32_t scaleNum =
            (hExpandXAlign32Size / expand_x.element_size()) / static_cast<uint32_t>(UB_ALIGN / sizeof(float));
        uint32_t scaleNumAlignSize = (scaleNum * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        totalBufferSize += scaleNumAlignSize;
    }
    tilingData.bufferNum = totalBufferSize > tilingData.totalUbSize ? BUFFER_SINGLE : BUFFER_DOUBLE;
}

static void calculate_tilingdata_a5(MoeDistributeCombineV2TilingData& tilingData, c10::string_view group_ep,
                                    int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num,
                                    int64_t total_winsize_ep, int64_t expert_shard_type, int64_t shared_expert_num,
                                    int64_t shared_expert_rank_num, int64_t global_bs, int64_t bs, int64_t h, int64_t k,
                                    int64_t a, int64_t zero_expert_num, int64_t copy_expert_num,
                                    int64_t const_expert_num, int64_t comm_quant_mode, const at::Tensor& expand_x,
                                    const c10::optional<at::Tensor>& x_active_mask,
                                    const c10::optional<at::Tensor>& shared_expert_x,
                                    const c10::optional<at::Tensor>& performance_info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSizePlatFrom;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatFrom);

    // mc2通信tiling：对齐arch35 tiling（A5仅支持HalfAllToAllV，A5_CCU_ENGINE）
    // 经本地 shim 调用 AscendC::Mc2CcTilingConfig（libtiling_api.a 为旧 ABI，example 侧新 ABI，无法直链）
    uint32_t opType = static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_HALFALLTOALLV);
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    void* mc2CcTilingCfg = A5Mc2CcTilingCreate(std::string(group_ep).c_str(), opType, algConfigStr.c_str());
    TORCH_CHECK(mc2CcTilingCfg != nullptr, "A5Mc2CcTilingCreate failed, mc2CcTilingCfg is nullptr");
    A5Mc2CcTilingSetCommEngine(mc2CcTilingCfg, mc2tiling::A5_CCU_ENGINE);
    A5Mc2CcTilingGetInitTiling(mc2CcTilingCfg, &tilingData.mc2InitTiling);
    A5Mc2CcTilingGetCcTiling(mc2CcTilingCfg, &tilingData.mc2CcTiling1);
    A5Mc2CcTilingDestroy(mc2CcTilingCfg);

    auto& info = tilingData.moeDistributeCombineV2Info;
    info.epWorldSize = ep_world_size;
    info.tpWorldSize = 1; // A5不支持TP通信
    info.epRankId = ep_rank_id;
    info.tpRankId = 0; // A5不支持TP通信
    info.expertShardType = expert_shard_type;
    info.sharedExpertNum = shared_expert_num;
    info.sharedExpertRankNum = shared_expert_rank_num;
    info.moeExpertNum = moe_expert_num;
    info.moeExpertPerRankNum = moe_expert_num / (ep_world_size - shared_expert_rank_num);
    info.zeroExpertNum = zero_expert_num;
    info.copyExpertNum = copy_expert_num;
    info.constExpertNum = const_expert_num;
    info.globalBs = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    info.bs = bs;
    info.k = k;
    info.h = h;
    info.a = a;
    info.aivNum = ascendcPlatform->GetCoreNumAiv();
    info.isTokenMask = (x_active_mask.has_value() && x_active_mask->dim() == DIM_ONE);
    info.isExpertMask = (x_active_mask.has_value() && x_active_mask->dim() == DIM_TWO);
    info.hasSharedExpertX = shared_expert_x.has_value();
    info.hasElasticInfo = false;                       // 样例不使能弹性缩容
    info.isPerformance = performance_info.has_value(); // whether performance or not
    info.isMc2Context =
        false; // 对齐 A5-MTE：内核经 GetHcclContext（SetHcclContext<0> 指向传入上下文）读 HcclCombinOpParam
    info.hasExpertScales = true; // expertScales为必选输入
    info.reserved1 = false;
    info.totalUbSize = ubSizePlatFrom;
    // 对齐 mc2 arch35（moe_distribute_combine_tiling_arch35.cpp:771）：
    // totalWinSizeEp = GetMaxWindowSize()*2（GetMaxWindowSize = HCCL_BUFFSIZE(MB)*1024^2），
    // 不减状态区/rank 偏移，与 mc2 框架完全一致
    {
        constexpr uint64_t MAX_HCCL_BUFFSIZE_MB = 1024UL; // 环境变量上界，防无符号回绕
        uint64_t defaultWindowSize = 200;
        const char* envValue = std::getenv("HCCL_BUFFSIZE");
        if (envValue != nullptr) {
            long long parsed = std::atoll(envValue);
            // 负数/超大值不采用（parsed<=0 已排除 atoll 失败返回 0 与负数）
            if (parsed > 0 && static_cast<uint64_t>(parsed) <= MAX_HCCL_BUFFSIZE_MB) {
                defaultWindowSize = static_cast<uint64_t>(parsed);
            }
        }
        info.totalWinSizeEp = defaultWindowSize * 1024UL * 1024UL * 2UL;
    }
    info.armAvgFactor = 1.0f; // AddRmsNorm参数，样例未使能（HasAddRmsNorm=false）
    info.epsilon = 1e-5f;     // AddRmsNorm参数，样例未使能（HasAddRmsNorm=false）
    calculate_buffernum(info, expand_x, comm_quant_mode);
}

void MoeDistributeCombineV2_api(
    aclrtStream stream, int64_t comm_quant_mode, const at::Tensor& expand_x, const at::Tensor& expert_ids,
    const at::Tensor& assist_info_for_combine, const at::Tensor& ep_send_counts, const at::Tensor& expert_scales,
    const at::Tensor& new_workspace, const at::Tensor& tiling_buffer, const at::Tensor& mc2_context,
    const c10::optional<at::Tensor>& x_active_mask, const c10::optional<at::Tensor>& shared_expert_x,
    const c10::optional<at::Tensor>& ori_x, const c10::optional<at::Tensor>& const_expert_alpha_1,
    const c10::optional<at::Tensor>& const_expert_alpha_2, const c10::optional<at::Tensor>& const_expert_v,
    const c10::optional<at::Tensor>& performance_info, at::Tensor& x_out,
    const MoeDistributeCombineV2TilingData& tilingData)
{
    auto expandX_ptr = get_first_tensor_address<at::Tensor>(expand_x.scalar_type(), expand_x, false);
    auto expertIds_ptr = get_first_tensor_address<at::Tensor>(expert_ids.scalar_type(), expert_ids, false);
    auto expandIdx_ptr =
        get_first_tensor_address<at::Tensor>(assist_info_for_combine.scalar_type(), assist_info_for_combine, false);
    auto epSendCount_ptr = get_first_tensor_address<at::Tensor>(ep_send_counts.scalar_type(), ep_send_counts, false);
    auto expertScales_ptr = get_first_tensor_address<at::Tensor>(expert_scales.scalar_type(), expert_scales, false);
    auto workspace_ptr = get_first_tensor_address<at::Tensor>(new_workspace.scalar_type(), new_workspace, false);
    auto tilingGM_ptr = get_first_tensor_address<at::Tensor>(tiling_buffer.scalar_type(), tiling_buffer, false);
    auto mc2Context_ptr = get_first_tensor_address<at::Tensor>(mc2_context.scalar_type(), mc2_context, false);

    void* xActiveMask_ptr = nullptr;
    if (x_active_mask.has_value()) {
        xActiveMask_ptr =
            get_first_tensor_address<c10::optional<at::Tensor>>(x_active_mask->scalar_type(), x_active_mask, false);
    }

    void* sharedExpertX_ptr = nullptr;
    if (shared_expert_x.has_value()) {
        sharedExpertX_ptr =
            get_first_tensor_address<c10::optional<at::Tensor>>(shared_expert_x->scalar_type(), shared_expert_x, false);
    }

    void* oriX_ptr = nullptr;
    if (ori_x.has_value()) {
        oriX_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(ori_x->scalar_type(), ori_x, false);
    }

    void* constExpertAlpha1_ptr = nullptr;
    if (const_expert_alpha_1.has_value()) {
        constExpertAlpha1_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(const_expert_alpha_1->scalar_type(),
                                                                                    const_expert_alpha_1, false);
    }

    void* constExpertAlpha2_ptr = nullptr;
    if (const_expert_alpha_2.has_value()) {
        constExpertAlpha2_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(const_expert_alpha_2->scalar_type(),
                                                                                    const_expert_alpha_2, false);
    }

    void* constExpertV_ptr = nullptr;
    if (const_expert_v.has_value()) {
        constExpertV_ptr =
            get_first_tensor_address<c10::optional<at::Tensor>>(const_expert_v->scalar_type(), const_expert_v, false);
    }

    void* performanceInfo_ptr = nullptr;
    if (performance_info.has_value()) {
        performanceInfo_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(performance_info->scalar_type(),
                                                                                  performance_info, false);
    }
    auto XOut_ptr = get_first_tensor_address<at::Tensor>(x_out.scalar_type(), x_out, false);

    int32_t tilingKey = 200;
    calculate_tilingkey_a5(tilingKey, expand_x.scalar_type(), comm_quant_mode);

    aclprofTensorInfo tensorInfo;
    INIT_ACL_PROF_TENSOR_INFO("moe_distribute_combine_v2", "MoeDistributeCombineV2",
                              tilingData.moeDistributeCombineV2Info.aivNum, AI_VECTOR_CORE, tensorInfo, stream,
                              INPUT(expand_x), INPUT(expert_ids), INPUT(assist_info_for_combine), INPUT(ep_send_counts),
                              INPUT(expert_scales), OUTPUT(x_out));
    aclprofEventAttributes attrs = {1, sizeof(aclprofEventAttributes::message), 0, &tensorInfo};
    aclprofRangePushEx(&attrs);

    // tiling结构体H2D到device buffer，kernel内通过tilingGM以GET_TILING_DATA_WITH_STRUCT读取
    // tiling结构体H2D到device buffer（pinned host 内存，兼容 torch.compile 图捕获）
    static void* pTilingHost = nullptr;
    if (pTilingHost == nullptr) {
        auto mr = aclrtMallocHost(&pTilingHost, sizeof(MoeDistributeCombineV2TilingData));
        TORCH_CHECK(mr == ACL_SUCCESS, "aclrtMallocHost tiling failed, ret: ", static_cast<int32_t>(mr));
    }
    memcpy(pTilingHost, &tilingData, sizeof(MoeDistributeCombineV2TilingData));
    auto ret = aclrtMemcpyAsync(tilingGM_ptr, sizeof(MoeDistributeCombineV2TilingData), pTilingHost,
                                sizeof(MoeDistributeCombineV2TilingData), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemcpyAsync tilingData failed, ret: ", static_cast<int32_t>(ret));

    // 说明：residualX/gamma/activationScale/weightScale/groupList/expandScales/tpSendCounts/elasticInfo
    // 在A5 v2内核路径中未使用（apt.cpp中固定nullptr或不传入），此处传nullptr
    moe_distribute_combine_v2_entry(
        tilingKey, tilingData.moeDistributeCombineV2Info.aivNum, (void*)stream, (GM_ADDR)expandX_ptr,
        (GM_ADDR)expertIds_ptr, (GM_ADDR)expandIdx_ptr, (GM_ADDR)epSendCount_ptr, (GM_ADDR)expertScales_ptr, nullptr,
        (GM_ADDR)xActiveMask_ptr, nullptr, nullptr, nullptr, nullptr, (GM_ADDR)sharedExpertX_ptr, nullptr,
        (GM_ADDR)oriX_ptr, (GM_ADDR)constExpertAlpha1_ptr, (GM_ADDR)constExpertAlpha2_ptr, (GM_ADDR)constExpertV_ptr,
        (GM_ADDR)performanceInfo_ptr, (GM_ADDR)XOut_ptr, (GM_ADDR)workspace_ptr, (GM_ADDR)tilingGM_ptr,
        (GM_ADDR)mc2Context_ptr);

    aclprofRangePop();
}

at::Tensor npu_moe_distribute_combine_v2(
    const at::Tensor& expand_x, const at::Tensor& expert_ids, const at::Tensor& assist_info_for_combine,
    const at::Tensor& ep_send_counts, const at::Tensor& expert_scales, const at::Tensor& mc2_context,
    c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num,
    int64_t total_winsize_ep, const c10::optional<at::Tensor>& x_active_mask,
    const c10::optional<at::Tensor>& shared_expert_x, const c10::optional<at::Tensor>& ori_x,
    const c10::optional<at::Tensor>& const_expert_alpha_1, const c10::optional<at::Tensor>& const_expert_alpha_2,
    const c10::optional<at::Tensor>& const_expert_v, const c10::optional<at::Tensor>& performance_info,
    int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs,
    int64_t comm_quant_mode, c10::string_view comm_alg, int64_t zero_expert_num, int64_t copy_expert_num,
    int64_t const_expert_num)
{
    // A5样例支持非量化(0)与int8通信量化(2)
    TORCH_CHECK(comm_quant_mode == 0 || comm_quant_mode == 2,
                "A5 example only supports comm_quant_mode=0 (non-quant) or 2 (int8 comm quant), but got ",
                comm_quant_mode);

    MoeDistributeCombineV2ValidateParams validate_params;
    validate_params.ep_world_size = ep_world_size;
    validate_params.ep_rank_id = ep_rank_id;
    validate_params.moe_expert_num = moe_expert_num;
    validate_params.expert_shard_type = expert_shard_type;
    validate_params.shared_expert_num = shared_expert_num;
    validate_params.shared_expert_rank_num = shared_expert_rank_num;
    validate_params.global_bs = global_bs;
    validate_params.comm_quant_mode = comm_quant_mode;
    validate_params.zero_expert_num = zero_expert_num;
    validate_params.copy_expert_num = copy_expert_num;
    validate_params.const_expert_num = const_expert_num;

    TORCH_CHECK(
        ValidateMoeDistributeCombineV2Input(expand_x, expert_ids, assist_info_for_combine, ep_send_counts,
                                            expert_scales, x_active_mask, shared_expert_x, ori_x, const_expert_alpha_1,
                                            const_expert_alpha_2, const_expert_v, performance_info, validate_params),
        "Input validation failed for npu_moe_distribute_combine_v2");

    auto expand_x_size = expand_x.sizes();
    auto expert_ids_size = expert_ids.sizes();

    int64_t bs = expert_ids_size[0];
    int64_t h = expand_x_size[1];
    int64_t k = expert_ids_size[1];

    bool shared_front = (expert_shard_type == 0);
    int64_t local_moe_expert_num = 1;
    int64_t global_bs_real = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    int64_t a = 0;
    if (shared_front) {
        if (ep_rank_id < shared_expert_rank_num) {
            local_moe_expert_num = 1;
            TORCH_CHECK(ep_world_size > 0, "ep_world_size must be > 0, but got ", ep_world_size);
            TORCH_CHECK(shared_expert_num > 0, "shared_expert_num must be > 0, but got ", shared_expert_num);
            int64_t max_bs = global_bs_real / ep_world_size;
            int64_t rank_num_per_shared_expert = shared_expert_rank_num / shared_expert_num;
            TORCH_CHECK(rank_num_per_shared_expert > 0, "rank_num_per_shared_expert must be > 0, but got ",
                        rank_num_per_shared_expert);
            int64_t max_shared_group_num =
                (ep_world_size + rank_num_per_shared_expert - 1) / rank_num_per_shared_expert;
            a = max_bs * max_shared_group_num;
        } else {
            TORCH_CHECK(ep_world_size - shared_expert_rank_num > 0,
                        "ep_world_size - shared_expert_rank_num must be > 0, but got ep_world_size=", ep_world_size,
                        ", shared_expert_rank_num=", shared_expert_rank_num);
            local_moe_expert_num = moe_expert_num / (ep_world_size - shared_expert_rank_num);
            a = global_bs_real * std::min(local_moe_expert_num, k);
        }
    }
    TORCH_CHECK((expand_x.dim() == DIM_TWO) && (expert_ids.dim() == DIM_TWO), "The x and expert_ids should be 2D");
    TORCH_CHECK((expand_x.scalar_type() == at::kBFloat16) || (expand_x.scalar_type() == at::kHalf),
                "dtype of expand_x should be BFloat16 or Float16, but got " +
                    std::string(c10::toString(expand_x.scalar_type())));
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "dtype of expert_ids should be Int, but got " + std::string(c10::toString(expert_ids.scalar_type())));

    at::Tensor output;
    // 对齐原 mc2/torch_npu 持久 workspace：缓存复用，避免 per-call 新建导致
    // profiling memory_record 出现瞬时双块（两个 16MB workspace 并存）
    static at::Tensor cachedWorkspace;
    if (!cachedWorkspace.defined()) {
        cachedWorkspace = at::empty({WORKSPACESIZE / 4}, expert_ids.options().dtype(at::kInt));
    }
    at::Tensor new_workspace = cachedWorkspace;
    at::Tensor tiling_buffer =
        at::empty({(int64_t)sizeof(MoeDistributeCombineV2TilingData)}, expand_x.options().dtype(at::kByte));
    output = at::empty({bs, h}, expert_ids.options().dtype(expand_x.scalar_type()));

    MoeDistributeCombineV2TilingData tilingData;
    calculate_tilingdata_a5(tilingData, group_ep, ep_world_size, ep_rank_id, moe_expert_num, total_winsize_ep,
                            expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, bs, h, k, a,
                            zero_expert_num, copy_expert_num, const_expert_num, comm_quant_mode, expand_x,
                            x_active_mask, shared_expert_x, performance_info);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() mutable -> int {
        // A3 式：mc2_context 设备地址直接传入内核
        MoeDistributeCombineV2_api(stream, comm_quant_mode, expand_x, expert_ids, assist_info_for_combine,
                                   ep_send_counts, expert_scales, new_workspace, tiling_buffer, mc2_context,
                                   x_active_mask, shared_expert_x, ori_x, const_expert_alpha_1, const_expert_alpha_2,
                                   const_expert_v, performance_info, output, tilingData);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("moeDistributeCombineV2", acl_call);

    return output;
}

at::Tensor npu_moe_distribute_combine_v2_meta(
    const at::Tensor& expand_x, const at::Tensor& expert_ids, const at::Tensor& assist_info_for_combine,
    const at::Tensor& ep_send_counts, const at::Tensor& expert_scales, const at::Tensor& mc2_context,
    c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num,
    int64_t total_winsize_ep, const c10::optional<at::Tensor>& x_active_mask,
    const c10::optional<at::Tensor>& shared_expert_x, const c10::optional<at::Tensor>& ori_x,
    const c10::optional<at::Tensor>& const_expert_alpha_1, const c10::optional<at::Tensor>& const_expert_alpha_2,
    const c10::optional<at::Tensor>& const_expert_v, const c10::optional<at::Tensor>& performance_info,
    int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs,
    int64_t comm_quant_mode, c10::string_view comm_alg, int64_t zero_expert_num, int64_t copy_expert_num,
    int64_t const_expert_num)
{
    auto expert_ids_size = expert_ids.sizes();
    auto expand_x_size = expand_x.sizes();

    int64_t bs = expert_ids_size[0];
    int64_t h = expand_x_size[1];

    at::Tensor output = at::empty({bs, h}, expert_ids.options().dtype(expand_x.scalar_type()));
    return output;
}

TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("MoeDistributeCombineV2", TORCH_FN(npu_moe_distribute_combine_v2));
}

TORCH_LIBRARY_IMPL(ascend_ops, Meta, m)
{
    m.impl("MoeDistributeCombineV2", &npu_moe_distribute_combine_v2_meta);
}

} // namespace MoeDistributeCombineV2
} // namespace ascend_ops
