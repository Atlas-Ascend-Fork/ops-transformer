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
 * \file moe_distribute_dispatch_v2_torch.cpp
 * \brief A5 (ascend950) torch binding for the <<<>>> direct-launch MoeDistributeDispatchV2 example.
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
#include "moe_distribute_dispatch_v2_entry.h"
#include "mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_dispatch_v2_torch.h"
#include "mc2_cc_tiling_shim.h"
#include "moe_distribute_dispatch_v2_torch_validate.h"
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

namespace MoeDistributeDispatchV2 {

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def(
        "MoeDistributeDispatchV2(Tensor x, Tensor expert_ids, Tensor mc2_context, str group_ep, int ep_world_size, "
        "int ep_rank_id, int moe_expert_num, int total_winsize_ep, *, Tensor? scales=None, Tensor? x_active_mask=None, "
        "Tensor? expert_scales=None, Tensor? performance_info=None, "
        "int expert_shard_type=0, int shared_expert_num=0, "
        "int shared_expert_rank_num=0, int quant_mode=0, int global_bs=0, int expert_token_nums_type=0, "
        "str comm_alg=\"\", int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0 "
        ") "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

constexpr uint32_t DIM_ONE = 1UL;
constexpr uint32_t DIM_TWO = 2UL;
constexpr uint32_t TILINGKEY_XTYPE = 10;
constexpr uint32_t TILINGKEY_SCALES = 100;
constexpr uint32_t WORKSPACESIZE = 16 * 1024 * 1024;
constexpr uint32_t MIX_AIV = 5;
constexpr uint64_t A5_WIN_SIZE_MULTIPLY = 2UL;

static void calculate_tilingkey_a5(int32_t& tilingKey, at::ScalarType xType, int64_t quantMode, const bool isScales)
{
    tilingKey += static_cast<uint64_t>(quantMode); // 个位: quantMode
    if (xType == at::kBFloat16) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_XTYPE); // 十位: bf16
    }
    if (isScales) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_SCALES); // 百位: smoothScale 存在
    }
    return;
}

void moe_distribute_dispatch_v2_api(
    aclrtStream stream, int64_t quant_mode, bool is_scales, const at::Tensor& x, const at::Tensor& expert_ids,
    const at::Tensor& new_workspace, const at::Tensor& tiling_buffer, const at::Tensor& mc2_context,
    c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num,
    const c10::optional<at::Tensor>& scales, const c10::optional<at::Tensor>& x_active_mask,
    const c10::optional<at::Tensor>& expert_scales, const c10::optional<at::Tensor>& performance_info,
    at::Tensor& expand_x, at::Tensor& dynamic_scales, at::Tensor& assist_info_forcombine, at::Tensor& expert_token_nums,
    at::Tensor& ep_recv_counts, at::Tensor& expand_scales, const MoeDistributeDispatchV2TilingData& tilingData)
{
    auto x_ptr = get_first_tensor_address<at::Tensor>(x.scalar_type(), x, false);
    auto expertIds_ptr = get_first_tensor_address<at::Tensor>(expert_ids.scalar_type(), expert_ids, false);
    auto workspace_ptr = get_first_tensor_address<at::Tensor>(new_workspace.scalar_type(), new_workspace, false);
    auto tilingGM_ptr = get_first_tensor_address<at::Tensor>(tiling_buffer.scalar_type(), tiling_buffer, false);
    auto mc2Context_ptr = get_first_tensor_address<at::Tensor>(mc2_context.scalar_type(), mc2_context, false);
    void* scales_ptr = nullptr;
    if (scales.has_value()) {
        scales_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(scales->scalar_type(), scales, false);
    }
    void* xActiveMask_ptr = nullptr;
    if (x_active_mask.has_value()) {
        xActiveMask_ptr =
            get_first_tensor_address<c10::optional<at::Tensor>>(x_active_mask->scalar_type(), x_active_mask, false);
    }
    void* expertScales_ptr = nullptr;
    if (expert_scales.has_value()) {
        expertScales_ptr =
            get_first_tensor_address<c10::optional<at::Tensor>>(expert_scales->scalar_type(), expert_scales, false);
    }
    void* performanceInfo_ptr = nullptr;
    if (performance_info.has_value()) {
        performanceInfo_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(performance_info->scalar_type(),
                                                                                  performance_info, false);
    }
    auto expandXOut_ptr = get_first_tensor_address<at::Tensor>(expand_x.scalar_type(), expand_x, false);
    auto dynamicScalesOut_ptr =
        get_first_tensor_address<at::Tensor>(dynamic_scales.scalar_type(), dynamic_scales, false);
    auto assistInfoOut_ptr =
        get_first_tensor_address<at::Tensor>(assist_info_forcombine.scalar_type(), assist_info_forcombine, false);
    auto expertTokenNumsOut_ptr =
        get_first_tensor_address<at::Tensor>(expert_token_nums.scalar_type(), expert_token_nums, false);
    auto epSendCountsOut_ptr =
        get_first_tensor_address<at::Tensor>(ep_recv_counts.scalar_type(), ep_recv_counts, false);
    auto expandScalesOut_ptr = get_first_tensor_address<at::Tensor>(expand_scales.scalar_type(), expand_scales, false);

    int32_t tilingKey = 20000;
    calculate_tilingkey_a5(tilingKey, x.scalar_type(), quant_mode, is_scales);

    // 对齐 mc2 host tiling SetPlatformInfo：launch blockDim = CalcTschBlockDim(aivNum, 0, aivNum)
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aivNum = tilingData.moeDistributeDispatchV2Info.aivNum;
    uint32_t numBlocks = ascendcPlatform->CalcTschBlockDim(aivNum, 0, aivNum);

    aclprofTensorInfo tensorInfo;
    INIT_ACL_PROF_TENSOR_INFO(
        "moe_distribute_dispatch_v2", "MoeDistributeDispatchV2", tilingData.moeDistributeDispatchV2Info.aivNum, MIX_AIV,
        tensorInfo, stream, INPUT(x), INPUT(expert_ids), OUTPUT(expand_x), OUTPUT(dynamic_scales),
        OUTPUT(assist_info_forcombine), OUTPUT(expert_token_nums), OUTPUT(ep_recv_counts), OUTPUT(expand_scales));
    aclprofEventAttributes attrs = {1, sizeof(aclprofEventAttributes::message), 0, &tensorInfo};
    aclprofRangePushEx(&attrs);

    // tiling结构体H2D到device buffer（pinned host 内存，兼容 torch.compile 图捕获），
    // kernel内通过tilingGM以GET_TILING_DATA_WITH_STRUCT读取
    static void* pTilingHost = nullptr;
    if (pTilingHost == nullptr) {
        auto mr = aclrtMallocHost(&pTilingHost, sizeof(MoeDistributeDispatchV2TilingData));
        TORCH_CHECK(mr == ACL_SUCCESS, "aclrtMallocHost tiling failed, ret: ", static_cast<int32_t>(mr));
    }
    memcpy(pTilingHost, &tilingData, sizeof(MoeDistributeDispatchV2TilingData));
    auto ret = aclrtMemcpyAsync(tilingGM_ptr, sizeof(MoeDistributeDispatchV2TilingData), pTilingHost,
                                sizeof(MoeDistributeDispatchV2TilingData), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemcpyAsync tilingData failed, ret: ", static_cast<int32_t>(ret));

    moe_distribute_dispatch_v2_entry(
        tilingKey, numBlocks, (void*)stream, (GM_ADDR)x_ptr, (GM_ADDR)expertIds_ptr, (GM_ADDR)scales_ptr,
        (GM_ADDR)xActiveMask_ptr, (GM_ADDR)expertScales_ptr, nullptr, (GM_ADDR)performanceInfo_ptr,
        (GM_ADDR)expandXOut_ptr, (GM_ADDR)dynamicScalesOut_ptr, (GM_ADDR)assistInfoOut_ptr,
        (GM_ADDR)expertTokenNumsOut_ptr, (GM_ADDR)epSendCountsOut_ptr, nullptr, (GM_ADDR)expandScalesOut_ptr,
        (GM_ADDR)workspace_ptr, (GM_ADDR)tilingGM_ptr, (GM_ADDR)mc2Context_ptr);

    aclprofRangePop();
}

static void calculate_tilingdata_a5(
    MoeDistributeDispatchV2TilingData& tilingData, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
    int64_t moe_expert_num, int64_t total_winsize_ep, int64_t expert_shard_type, int64_t shared_expert_num,
    int64_t shared_expert_rank_num, int64_t global_bs, int64_t bs, int64_t h, int64_t k, int64_t a, int64_t quant_mode,
    int64_t zero_expert_num, int64_t copy_expert_num, int64_t const_expert_num, int64_t expert_token_nums_type,
    const c10::optional<at::Tensor>& scales, const c10::optional<at::Tensor>& x_active_mask,
    const c10::optional<at::Tensor>& expert_scales, const c10::optional<at::Tensor>& performance_info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSizePlatFrom;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatFrom);

    // mc2通信tiling：对齐 A5-MTE 的 SetHcommCfg（A3 tiling：OP_TYPE_ALL_TO_ALL + AIV_ENGINE）
    // 经本地 shim 调用 AscendC::Mc2CcTilingConfig（libtiling_api.a 为旧 ABI，example 侧新 ABI，无法直链）
    uint32_t opType = static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLTOALLV);
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    void* mc2CcTilingCfg = A5Mc2CcTilingCreate(std::string(group_ep).c_str(), opType, algConfigStr.c_str());
    TORCH_CHECK(mc2CcTilingCfg != nullptr, "A5Mc2CcTilingCreate failed, mc2CcTilingCfg is nullptr");
    A5Mc2CcTilingSetCommEngine(mc2CcTilingCfg, mc2tiling::AIV_ENGINE);
    A5Mc2CcTilingGetInitTiling(mc2CcTilingCfg, &tilingData.mc2InitTiling);
    A5Mc2CcTilingGetCcTiling(mc2CcTilingCfg, &tilingData.mc2CcTiling1);
    A5Mc2CcTilingDestroy(mc2CcTilingCfg);

    // 初始化TilingData结构体的成员变量
    auto& info = tilingData.moeDistributeDispatchV2Info;
    info.epWorldSize = ep_world_size;                                                  // epWorldSize
    info.tpWorldSize = 1;                                                              // A5不支持TP通信
    info.epRankId = ep_rank_id;                                                        // epRankId
    info.tpRankId = 0;                                                                 // A5不支持TP通信
    info.expertShardType = expert_shard_type;                                          // expert type
    info.sharedExpertNum = shared_expert_num;                                          // shared expert number
    info.sharedExpertRankNum = shared_expert_rank_num;                                 // shared expert rank number
    info.moeExpertNum = moe_expert_num;                                                // moe expert number
    info.quantMode = quant_mode;                                                       // quant mode
    info.globalBs = (global_bs == 0) ? (bs * ep_world_size) : global_bs;               // globalBs = gBS * worldSize
    info.bs = bs;                                                                      // bs
    info.k = k;                                                                        // k
    info.h = h;                                                                        // h
    info.a = a;                                                                        // a
    info.aivNum = ascendcPlatform->GetCoreNumAiv();                                    // aivNum
    info.isTokenMask = (x_active_mask.has_value() && x_active_mask->dim() == DIM_ONE); // input active mask 1dims or not
    info.isExpertMask =
        (x_active_mask.has_value() && x_active_mask->dim() == DIM_TWO); // input active mask 2dims or not
    info.hasElasticInfo = false;                                        // 样例不使能弹性缩容
    info.isPerformance = performance_info.has_value();                  // whether performance or not
    info.isQuant = (quant_mode != 0);                                   // whether quant or not
    info.isMc2Context =
        false; // 对齐 A5-MTE：内核经 GetHcclContext（SetHcclContext<0> 指向传入上下文）读 HcclCombinOpParam
    info.hasExpertScales =
        expert_scales.has_value(); // 对齐 mc2 infershape：由 expert_scales 输入决定是否输出 expand_scales
    info.reserved1 = false;
    info.totalUbSize = ubSizePlatFrom;
    // 对齐 mc2 arch35（moe_distribute_dispatch_v2_tiling_arch35.cpp:907）：
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
    info.expertTokenNumsType = expert_token_nums_type; // expert token nums type, support 0: cumsum mode, 1: count mode
    info.zeroComputeExpertNum =
        zero_expert_num + copy_expert_num + const_expert_num; // sum of zero, copy and const expert nums
    info.maxSizeForUbBuffer = 0;
    info.cumsumTmpMinSize = 0;
    if (scales.has_value()) {
        TORCH_CHECK(scales.value().dim() == 2, "scales should be 2D, but got dim=", scales.value().dim());

        info.scalesRow = scales.value().size(0);
        info.scalesCol = scales.value().size(1);
        info.scalesTypeSize = at::elementSize(scales.value().scalar_type());
        info.scalesCount = info.scalesRow * info.scalesCol;
    } else {
        info.scalesRow = 0;
        info.scalesCol = 0;
        info.scalesTypeSize = 0;
        info.scalesCount = 0;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_moe_distribute_dispatch_v2(
    const at::Tensor& x, const at::Tensor& expert_ids, const at::Tensor& mc2_context, c10::string_view group_ep,
    int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, int64_t total_winsize_ep,
    const c10::optional<at::Tensor>& scales, const c10::optional<at::Tensor>& x_active_mask,
    const c10::optional<at::Tensor>& expert_scales, const c10::optional<at::Tensor>& performance_info,
    int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t quant_mode,
    int64_t global_bs, int64_t expert_token_nums_type, c10::string_view comm_alg, int64_t zero_expert_num,
    int64_t copy_expert_num, int64_t const_expert_num)
{
    // A5样例支持非量化(0)与perToken动态量化(2, int8输出)
    TORCH_CHECK(quant_mode == 0 || quant_mode == 2,
                "A5 example only supports quant_mode=0 (non-quant) or 2 (perToken int8), but got ", quant_mode);
    // smoothScale(scales) 仅 perToken 量化场景有意义：非量化(quant_mode=0)下无 smoothScale 概念，
    // 若放行会生成 tilingKey 20100/20110，而 switch 无对应 case，输出保持未初始化内存。
    TORCH_CHECK(!scales.has_value() || quant_mode == 2,
                "A5 example scales (smoothScale) is only supported with quant_mode=2 (perToken), but got quant_mode=",
                quant_mode);

    MoeDistributeDispatchV2ValidateParams validate_params;
    validate_params.ep_world_size = ep_world_size;
    validate_params.ep_rank_id = ep_rank_id;
    validate_params.moe_expert_num = moe_expert_num;
    validate_params.shared_expert_num = shared_expert_num;
    validate_params.shared_expert_rank_num = shared_expert_rank_num;
    validate_params.quant_mode = quant_mode;
    validate_params.global_bs = global_bs;
    validate_params.expert_token_nums_type = expert_token_nums_type;
    validate_params.comm_alg = std::string(comm_alg);
    validate_params.zero_expert_num = zero_expert_num;
    validate_params.copy_expert_num = copy_expert_num;
    validate_params.const_expert_num = const_expert_num;

    ValidateMoeDistributeDispatchV2Input(x, expert_ids, scales, x_active_mask, expert_scales, performance_info,
                                         validate_params);

    int64_t bs = validate_params.bs;
    int64_t h = validate_params.h;
    int64_t k = validate_params.k;

    bool shared_front = (expert_shard_type == 0);
    int64_t local_moe_expert_num = 1;
    int64_t global_bs_real = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    int64_t a = 0;
    int64_t ep_recv_cnt_num = 0;
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
    ep_recv_cnt_num = ep_world_size * local_moe_expert_num;

    auto output_dtype = (!scales.has_value() && quant_mode == 0) ? x.scalar_type() : at::kChar;
    at::Tensor expand_x;
    at::Tensor dynamic_scales;

    expand_x = at::empty({a, h}, x.options().dtype(output_dtype));
    dynamic_scales = at::empty({a}, x.options().dtype(at::kFloat));

    at::Tensor expert_token_nums = at::empty({local_moe_expert_num}, x.options().dtype(at::kLong));
    at::Tensor ep_recv_counts = at::empty({ep_recv_cnt_num}, x.options().dtype(at::kInt));
    at::Tensor assist_info_forcombine{nullptr};
    // 对齐原 mc2/torch_npu 持久 workspace：缓存复用，避免 per-call 新建导致
    // profiling memory_record 出现瞬时双块（两个 16MB workspace 并存）
    static at::Tensor cachedWorkspace;
    if (!cachedWorkspace.defined()) {
        cachedWorkspace = at::empty({WORKSPACESIZE / 4}, x.options().dtype(at::kInt));
    }
    at::Tensor new_workspace = cachedWorkspace;
    at::Tensor tiling_buffer =
        at::empty({(int64_t)sizeof(MoeDistributeDispatchV2TilingData)}, x.options().dtype(at::kByte));

    at::Tensor expand_scales = at::empty({a}, x.options().dtype(at::kFloat));

    // 仅支持 MTE 通信；scales 存在即启用 smoothScale（perToken 量化场景）
    bool is_scales = scales.has_value();

    assist_info_forcombine = at::empty({std::max(bs * k, a * 128)}, x.options().dtype(at::kInt));

    MoeDistributeDispatchV2TilingData tilingData;
    calculate_tilingdata_a5(tilingData, group_ep, ep_world_size, ep_rank_id, moe_expert_num, total_winsize_ep,
                            expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, bs, h, k, a,
                            quant_mode, zero_expert_num, copy_expert_num, const_expert_num, expert_token_nums_type,
                            scales, x_active_mask, expert_scales, performance_info);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() mutable -> int {
        // 对齐 mc2 框架：把 mc2_context 设备地址传入内核，由 SetHcclContext<0> 写入 g_hcclContextReserved
        moe_distribute_dispatch_v2_api(
            stream, quant_mode, is_scales, x, expert_ids, new_workspace, tiling_buffer, mc2_context, group_ep,
            ep_world_size, ep_rank_id, moe_expert_num, scales, x_active_mask, expert_scales, performance_info, expand_x,
            dynamic_scales, assist_info_forcombine, expert_token_nums, ep_recv_counts, expand_scales, tilingData);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("moeDistributeDispatchV2", acl_call);

    return std::tie(expand_x, dynamic_scales, assist_info_forcombine, expert_token_nums, ep_recv_counts, expand_scales);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_moe_distribute_dispatch_v2_meta(
    const at::Tensor& x, const at::Tensor& expert_ids, const at::Tensor& mc2_context, c10::string_view group_ep,
    int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, int64_t total_winsize_ep,
    const c10::optional<at::Tensor>& scales, const c10::optional<at::Tensor>& x_active_mask,
    const c10::optional<at::Tensor>& expert_scales, const c10::optional<at::Tensor>& performance_info,
    int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t quant_mode,
    int64_t global_bs, int64_t expert_token_nums_type, c10::string_view comm_alg, int64_t zero_expert_num,
    int64_t copy_expert_num, int64_t const_expert_num)
{
    auto x_size = x.sizes();
    auto expert_ids_size = expert_ids.sizes();

    int64_t bs = x_size[0];
    int64_t h = x_size[1];
    int64_t k = expert_ids_size[1];

    bool shared_front = (expert_shard_type == 0);
    int64_t local_moe_expert_num = 1;
    int64_t global_bs_real = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    int64_t a = 0;
    int64_t ep_recv_cnt_num = 0;
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
    ep_recv_cnt_num = ep_world_size * local_moe_expert_num;

    auto output_dtype = (!scales.has_value() && quant_mode == 0) ? x.scalar_type() : at::kChar;
    at::Tensor expand_x;
    at::Tensor dynamic_scales;

    expand_x = at::empty({a, h}, x.options().dtype(output_dtype));
    dynamic_scales = at::empty({a}, x.options().dtype(at::kFloat));

    at::Tensor expert_token_nums = at::empty({local_moe_expert_num}, x.options().dtype(at::kLong));
    at::Tensor ep_recv_counts = at::empty({ep_recv_cnt_num}, x.options().dtype(at::kInt));
    at::Tensor assist_info_forcombine{nullptr};
    at::Tensor expand_scales = at::empty({a}, x.options().dtype(at::kFloat));

    assist_info_forcombine = at::empty({std::max(bs * k, a * 128)}, x.options().dtype(at::kInt));
    return std::tie(expand_x, dynamic_scales, assist_info_forcombine, expert_token_nums, ep_recv_counts, expand_scales);
}

// Register Ascend implementations for MoeDistributeDispatchV2
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("MoeDistributeDispatchV2", TORCH_FN(npu_moe_distribute_dispatch_v2));
}

TORCH_LIBRARY_IMPL(ascend_ops, Meta, m)
{
    m.impl("MoeDistributeDispatchV2", &npu_moe_distribute_dispatch_v2_meta);
}

} // namespace MoeDistributeDispatchV2
} // namespace ascend_ops
