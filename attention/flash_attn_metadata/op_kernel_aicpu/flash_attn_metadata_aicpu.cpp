/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_metadata_aicpu.cpp
 * \brief
 */

#include "flash_attn_metadata_aicpu.h"
#include <cmath>
#include <cstdio>
#include <numeric>
#include <algorithm>
#include "log.h"
#include "status.h"
#include "flash_attn_grad_metadata_split.h"
#include "../../common/op_kernel/aicpu_common.h"
#include "../../flash_attn/op_host/arch35/flash_attn_kernel_config_arch35.h"
#include "../../flash_attn/op_host/arch92/fa_adjust_sinner_souter_arch92.h"

constexpr uint32_t FA_KERNEL_STATUS_OK = 0;
constexpr uint32_t FA_KERNEL_STATUS_PARAM_INVALID = 1;

using namespace optiling;

namespace aicpu {
namespace {
// FAG uses cu_seqlens exclusively for physical packed offsets, and seqused
// (when present) for the effective attention rectangle within each batch.
bool CheckFagSequence(const std::vector<int64_t>& cu, const std::vector<int64_t>& used, int64_t batchSize,
                      int64_t maxSeqlen, bool isTnd, const char* name)
{
    if (!isTnd) {
        if (!cu.empty()) {
            KERNEL_LOG_ERROR("FAG %s cu_seqlens is supported only for TND layout", name);
            return false;
        }
        if (used.empty()) {
            return true;
        }
        // Optional per the interface doc (seqused or max_seqlen, one of the
        // two): when this side provides seqused, max_seqlen may be omitted.
        // Skip the upper-bound check in that case; the missing-side
        // requirement (max as the default fill source) is enforced in
        // ParamsCheck.
        if (batchSize <= 0 || used.size() != static_cast<size_t>(batchSize)) {
            KERNEL_LOG_ERROR("FAG %s seqused requires B entries", name);
            return false;
        }
        for (size_t i = 0; i < used.size(); ++i) {
            if (used[i] < 0 || (maxSeqlen >= 0 && used[i] > maxSeqlen)) {
                KERNEL_LOG_ERROR("FAG %s seqused at batch %zu must be in [0, physical S]", name, i);
                return false;
            }
        }
        return true;
    }
    if (batchSize <= 0 || maxSeqlen < 0 || cu.size() != static_cast<size_t>(batchSize) + 1 || cu[0] != 0) {
        KERNEL_LOG_ERROR("FAG %s requires B+1 cumulative lengths starting at zero and non-negative max_seqlen", name);
        return false;
    }
    if (!used.empty() && used.size() != static_cast<size_t>(batchSize)) {
        KERNEL_LOG_ERROR("FAG %s seqused must have exactly B entries", name);
        return false;
    }
    for (size_t i = 0; i < static_cast<size_t>(batchSize); ++i) {
        if (cu[i + 1] < cu[i] || cu[i + 1] - cu[i] > maxSeqlen) {
            KERNEL_LOG_ERROR("FAG %s physical length at batch %zu must be in [0, max_seqlen]", name, i);
            return false;
        }
        if (!used.empty() && (used[i] < 0 || used[i] > cu[i + 1] - cu[i])) {
            KERNEL_LOG_ERROR("FAG %s seqused at batch %zu must be in [0, physical length]", name, i);
            return false;
        }
    }
    return true;
}
} // namespace

uint32_t FlashAttnMetadataCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_LOG_INFO("FlashAttnMetadata Compute enter.");
    bool success = Prepare(ctx);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Prepare data failed!");

    load_balance::SectionStreamKResult splitRes{};
    success = BalanceSchedule(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Schedule load balance failed!");

    success = GenMetadata(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Generate balance result failed!");

    // ===== FAG metadata start =====
    // Re-read seqused tensors right before FAG via direct int32 access.
    // When an AicpuTaskSpace cache hit causes the reused task descriptor to
    // carry a stale (undefined) dtype for an optional-input slot, the
    // GetTensorDataAsInt64 switch falls through and returns a zero-filled
    // vector. The zeros then masquerade as a legitimate all-zero seqused input
    // in CheckActualQuerySeq/CheckActualKvSeq, polluting actualSeqlenQ_/Kv_.
    // Direct pointer reads bypass the dtype dispatch and always see the real
    // device data, so we use them to detect and recover from this condition.
    // Only triggers when the early-read vector is all-zero AND the direct
    // read yields non-zero values — legitimate all-zero seqused inputs are
    // never touched.
    if (fagIsGradEnabled_) {
        auto isAllZero = [](const std::vector<int64_t>& v) {
            if (v.empty()) {
                return false;
            }
            for (size_t i = 0; i < v.size(); i++) {
                if (v[i] != 0) {
                    return false;
                }
            }
            return true;
        };
        auto directReadInt32 = [](Tensor* t) -> std::vector<int64_t> {
            // NOTE: This assumes INT32 data (enforced by host-side
            // FlashAttnMetadataCheck::CheckSeqLens in flash_attn_metadata_check.h).
            // If the dtype restriction is relaxed, this reader must be updated.
            if (t == nullptr || t->GetData() == nullptr) {
                return {};
            }
            auto sh = t->GetTensorShape();
            const int64_t n = (sh != nullptr && sh->GetDims() > 0) ? sh->GetDimSize(0) : 0;
            if (n <= 0 || n > 65536) {
                return {};
            }
            const int32_t* p = static_cast<const int32_t*>(t->GetData());
            std::vector<int64_t> v;
            v.reserve(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; i++) {
                v.push_back(p[i]);
            }
            return v;
        };
        std::vector<int64_t> sqDirect = directReadInt32(sequsedQ_);
        if (!sqDirect.empty() && isAllZero(actualSeqlenQ_) && !isAllZero(sqDirect)) {
            actualSeqlenQ_ = sqDirect;
            isActualSeqlenQAccum_ = false;
        }
        std::vector<int64_t> skvDirect = directReadInt32(sequsedKv_);
        if (!skvDirect.empty() && isAllZero(actualSeqlenKv_) && !isAllZero(skvDirect)) {
            actualSeqlenKv_ = skvDirect;
            isActualSeqlenKvAccum_ = false;
        }
    }

    // FAG (Flash Attn Grad) metadata: only executed when is_grad_enabled=true.
    if (fagIsGradEnabled_) {
        InitFagParams();
        DoFagSparse();
        GenFagMetadata(splitRes.sectionNum);
        KERNEL_CHECK_FALSE(GenFagDeterMetadata(), FA_KERNEL_STATUS_PARAM_INVALID,
                           "Generate FAG deterministic metadata failed!");
        KERNEL_CHECK_FALSE(GenFagTndMetadata(), FA_KERNEL_STATUS_PARAM_INVALID, "Generate FAG TND metadata failed!");
    }
    // ===== FAG metadata end =====

    KERNEL_LOG_INFO("FlashAttnMetadata Compute success.");
    return FA_KERNEL_STATUS_OK;
}

bool FlashAttnMetadataCpuKernel::Prepare(CpuKernelContext& ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    KERNEL_CHECK_FALSE((metadata_ != nullptr && metadata_->GetData() != nullptr), false, "metadata is empty");

    bool requiredAttrs =
        GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
        GetAttrValue(ctx, "head_dim", headDim_) && GetAttrValue(ctx, "soc_version", socVersion_) &&
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) && GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    KERNEL_CHECK_FALSE(requiredAttrs, false, "Missing Required attrs missing!");

    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "win_left", winLeft_);
    GetAttrValueOpt(ctx, "win_right", winRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "layout_out", layoutOut_);
    GetAttrValueOpt(ctx, "head_dim_v", headDimV_);
    // head_dim_v 未指定(-1)时, v 的 head_dim 等于 head_dim
    if (headDimV_ == -1) {
        headDimV_ = headDim_;
    }
    GetAttrValueOpt(ctx, "is_grad_enabled", fagIsGradEnabled_);
    fagDeterMaxQ_ = maxSeqlenQ_;
    fagDeterMaxKv_ = maxSeqlenKv_;
    auto dropoutPAttr = ctx.GetAttr("dropout_p");
    dropoutP_ = dropoutPAttr == nullptr ? 0.0f : dropoutPAttr->GetFloat();

    KERNEL_CHECK_FALSE(ParamsCheck(), false, "Params check failed");
    return ParamsInit();
}

bool FlashAttnMetadataCpuKernel::GenFagTndMetadata()
{
    namespace fd = optiling::fag_deter;
    namespace ft = fag_pto_tnd;
    auto* metadata = static_cast<uint32_t*>(metadata_->GetData());
    const uint64_t capacity = metadata_->GetDataSize() / sizeof(int32_t);
    KERNEL_CHECK_FALSE(capacity > ft::HEAD_MAGIC, false, "FAG TND metadata header is truncated");
    std::fill(metadata + ft::HEAD_OFFSET, metadata + ft::HEAD_MAGIC + 1, uint32_t{0});
    if (layoutQ_ != "TND" || layoutKv_ != "TND") {
        return true;
    }
    const uint64_t baseEnd = static_cast<uint64_t>(metadata[HEAD_FAG_START_OFFSET_INDEX]) + FAG_METADATA_SIZE;
    // Reserve the entire WJF plan even when its admission is false. Publishing
    // either descriptor never changes the other extension's address or contents.
    const uint64_t wjfOffset = (baseEnd + 15) / 16 * 16;
    const uint64_t offset = (wjfOffset + fd::METADATA_I32_WORDS + 15) / 16 * 16;
    const int64_t required = ft::WordCount(batchSize_);
    KERNEL_CHECK_FALSE(batchSize_ > 0 && batchSize_ <= 65536 && required > 0 && offset <= UINT32_MAX &&
                           offset <= capacity && static_cast<uint64_t>(required) <= capacity - offset,
                       false, "FAG metadata output is too small for independent TND extension");
    auto plan = ft::Build(actualSeqlenQ_, actualSeqlenKv_, numHeadsKv_, fagG_, headDim_, headDimV_, aicCoreNum_,
                          maskMode_, winLeft_, winRight_, fagBlockOuter_);
    KERNEL_CHECK_FALSE(plan.status == 0, false, "FAG TND schedule input is outside the supported range");
    const bool finalBn2s2Tnd = fagSplitAxis_ == FAG_SPLIT_AXIS_BN2S2 && fagLayoutType_ == FAG_INPUT_FORMAT_TND &&
                               !fagFallbackToGs_ && fagG_ == 1 &&
                               (!fagIsSeqExistZero_ || (maskMode_ == 0 && !fagIsSparse_));
    plan.normalSwizzle = fag_pto_normal_swizzle::Build(actualSeqlenQ_, actualSeqlenKv_, numHeadsQ_, headDim_,
                                                       aicCoreNum_, maskMode_, winLeft_, winRight_, fagIsSparse_,
                                                       finalBn2s2Tnd, fagBlockOuter_, fagBlockStarts_, fagBlockEnds_);
    KERNEL_CHECK_FALSE(ft::Serialize(plan, aicCoreNum_, metadata + offset, capacity - offset), false,
                       "FAG TND schedule serialization failed");
    metadata[ft::HEAD_OFFSET] = static_cast<uint32_t>(offset);
    metadata[ft::HEAD_SIZE] = static_cast<uint32_t>(required);
    metadata[ft::HEAD_VERSION] = ft::VERSION;
    metadata[ft::HEAD_MAGIC] = ft::MAGIC;
    return true;
}

bool FlashAttnMetadataCpuKernel::GenFagDeterMetadata()
{
    namespace fd = optiling::fag_deter;
    // The extension is optional for legacy FA/FD callers, including forward
    // callers that request the original FAG metadata for a later backward.
    auto* metadata = static_cast<uint32_t*>(metadata_->GetData());
    const uint64_t capacity = metadata_->GetDataSize() / sizeof(int32_t);
    if (capacity <= fd::METADATA_HEAD_MAGIC_INDEX) {
        KERNEL_LOG_ERROR("FAG metadata output lacks capacity for the original header");
        return false;
    }
    std::fill(metadata + fd::METADATA_HEAD_OFFSET_INDEX, metadata + fd::METADATA_HEAD_MAGIC_INDEX + 1, 0);
    const uint64_t oldEnd = static_cast<uint64_t>(metadata[HEAD_FAG_START_OFFSET_INDEX]) + FAG_METADATA_SIZE;
    if (oldEnd > capacity) {
        KERNEL_LOG_ERROR("FAG metadata output lacks capacity for the original FAG region");
        return false;
    }
    // The candidate plan is independent of the later FAG execution mode.
    // Keep legacy/non-TND outputs unchanged and use only the supported 950
    // resource contract. No runtime-state array crosses the AICPU boundary.
    if (socVersion_.find("Ascend950") != 0 || layoutQ_ != "TND" || layoutKv_ != "TND" || layoutOut_ != "TND") {
        return true;
    }
    fd::PlanInputs inputs;
    auto& schedule = inputs.schedule;
    auto& route = schedule.route;
    route.batch = batchSize_;
    route.maxQ = fagDeterMaxQ_;
    route.maxKv = fagDeterMaxKv_;
    route.headsQ = numHeadsQ_;
    route.headsKv = numHeadsKv_;
    route.dimQ = headDim_;
    route.dimV = headDimV_;
    route.dtype = fd::DType::F16_ANY;
    route.maskMode = maskMode_ == 4 && winLeft_ == -1 && winRight_ == -1 ? 0 : maskMode_;
    route.allLayoutsTnd = true;
    route.deterministic = true;
    auto present = [](Tensor* tensor) {
        // AICPU may retain a shape placeholder for an absent optional input.
        // Match GetTensorDataAsInt64: only a nonempty input with data is present.
        return tensor != nullptr && tensor->GetData() != nullptr && tensor->NumElements() > 0;
    };
    route.hasSequsedQ = present(sequsedQ_);
    route.hasSequsedKv = present(sequsedKv_);
    auto validCu = [&](Tensor* tensor) {
        return tensor != nullptr && tensor->GetData() != nullptr && tensor->GetTensorShape() != nullptr &&
               tensor->GetTensorShape()->GetDims() == 1 && tensor->NumElements() == route.batch + 1 &&
               tensor->GetDataSize() >= static_cast<uint64_t>(route.batch + 1) * sizeof(int32_t);
    };
    if (route.batch <= 0 || !validCu(cuSeqlensQ_) || !validCu(cuSeqlensKv_)) {
        KERNEL_LOG_ERROR("Invalid FAG metadata cumulative sequence descriptors");
        return false;
    }
    const auto* cuQ = static_cast<const int32_t*>(cuSeqlensQ_->GetData());
    const auto* cuKv = static_cast<const int32_t*>(cuSeqlensKv_->GetData());
    auto readMaximum = [&](const int32_t* cu, int64_t declaredMax, int64_t& actualMax) -> bool {
        if (cu[0] != 0) {
            return false;
        }
        actualMax = 0;
        for (int64_t i = 0; i < route.batch; ++i) {
            if (cu[i + 1] < cu[i]) {
                return false;
            }
            actualMax = std::max(actualMax, static_cast<int64_t>(cu[i + 1]) - cu[i]);
        }
        return declaredMax < 0 || actualMax <= declaredMax;
    };
    int64_t actualMaxQ = 0;
    int64_t actualMaxKv = 0;
    if (!readMaximum(cuQ, route.maxQ, actualMaxQ) || !readMaximum(cuKv, route.maxKv, actualMaxKv)) {
        KERNEL_LOG_ERROR("Invalid FAG metadata cumulative sequence values or declared maximum");
        return false;
    }
    route.totalQ = cuQ[route.batch];
    route.totalKv = cuKv[route.batch];
    // Legacy metadata accepts max_seqlen upper bounds, larger batches and
    // windows. Only the deterministic extension requires exact physical maxima
    // and the scheduler's bounded domain; do not reject valid forward callers.
    if (fd::Admit(route) != fd::Admission::ELIGIBLE || actualMaxQ != route.maxQ || actualMaxKv != route.maxKv ||
        route.batch > 2048 || winLeft_ < -1 || winRight_ < -1 || winLeft_ > INT32_MAX || winRight_ > INT32_MAX) {
        return true;
    }
    const uint64_t offset = (oldEnd + 15) / 16 * 16;
    if (offset > UINT32_MAX || offset > capacity || fd::METADATA_I32_WORDS > capacity - offset) {
        return true; // An original-sized output remains valid without the optional extension.
    }
    schedule.aicNum = aicCoreNum_;
    schedule.aivNum = aivCoreNum_;
    schedule.windowLeft = winLeft_;
    schedule.windowRight = winRight_;
    schedule.l2Bytes = fd::PLAN_L2_ESTIMATE_BYTES;
    inputs.ubBytes = fd::ASCEND950_UB_BYTES;
    fd::MetadataWords output;
    const auto result = fd::BuildMetadataPlans(inputs, cuQ, route.batch + 1, cuKv, route.batch + 1, output);
    if (result.error != fd::PlanError::OK) {
        KERNEL_LOG_ERROR("FAG deterministic metadata planning failed: plan=%d schedule=%d",
                         static_cast<int>(result.error), static_cast<int>(result.scheduleError));
        return false;
    }
    std::memcpy(metadata + offset, output.data(), fd::METADATA_BYTES);
    // Publish the extension directory only after the entire payload is ready.
    metadata[fd::METADATA_HEAD_OFFSET_INDEX] = static_cast<uint32_t>(offset);
    metadata[fd::METADATA_HEAD_MAGIC_INDEX] = fd::METADATA_HEAD_MAGIC;
    return true;
}

bool FlashAttnMetadataCpuKernel::ParamsInit()
{
    InitBaseInfo();
    InitLoadBalanceParams();
    InitDeviceInfo();
    return true;
}

bool FlashAttnMetadataCpuKernel::ParamsCheck()
{
    KERNEL_CHECK_FALSE(std::isfinite(dropoutP_) && dropoutP_ >= 0.0f && dropoutP_ < 1.0f, false,
                       "dropout_p must be finite and in [0, 1)");
    KERNEL_CHECK_FALSE(dropoutP_ == 0.0f || (layoutKv_ != "PA_BBND" && layoutKv_ != "PA_BNBD" && layoutKv_ != "PA_NZ"),
                       false, "dropout does not support paged KV cache (PA_BBND/PA_BNBD/PA_NZ)");
    KERNEL_CHECK_FALSE(CheckActualQuerySeq(), false, "Check query sequence failed");
    KERNEL_CHECK_FALSE(CheckActualKvSeq(), false, "Check kv sequence failed");
    // Non-TND layouts: metadata has no q/k/v tensors, so a side without
    // seqused can only derive its physical length (default fill in
    // InitFagActualSeqlen and the split grid) from max_seqlen. When that side
    // provides seqused, max_seqlen is optional per the interface doc.
    // TND layouts derive max_seqlen from cu_seqlens (see CheckActual*Seq).
    if (fagIsGradEnabled_ && (layoutQ_ != "TND" || layoutKv_ != "TND")) {
        KERNEL_CHECK_FALSE(batchSize_ > 0, false, "FAG metadata requires B > 0");
    }
    return true;
}

bool FlashAttnMetadataCpuKernel::CheckActualQuerySeq()
{
    isActualSeqlenQAccum_ = false;
    actualSeqlenQ_.clear();
    std::vector<int64_t> cuSeqlensQ{};
    std::vector<int64_t> sequsedQ{};

    cuSeqlensQ = GetTensorDataAsInt64(cuSeqlensQ_);
    sequsedQ = GetTensorDataAsInt64(sequsedQ_);
    // Optional per the interface doc: under TND layout, max_seqlen_q may be
    // omitted (<0) and is derived from cu_seqlens_q. Use the packed token
    // total (cu.back()) so host (q.shape T-dim fallback) and AICPU share the
    // same derived value in grid/split formulas.
    if (fagIsGradEnabled_ && layoutQ_ == "TND" && maxSeqlenQ_ < 0 && !cuSeqlensQ.empty()) {
        maxSeqlenQ_ = static_cast<int32_t>(cuSeqlensQ.back());
    }
    if (fagIsGradEnabled_ &&
        !CheckFagSequence(cuSeqlensQ, sequsedQ, batchSize_, maxSeqlenQ_, layoutQ_ == "TND", "query")) {
        return false;
    }
    fagPhysicalTokensQ_ = cuSeqlensQ.empty() ? 0 : cuSeqlensQ.back();
    fagHasSeqused_ = !sequsedQ.empty();

    for (size_t i = 0; i < sequsedQ.size(); ++i) {
        if (sequsedQ[i] < 0) {
            KERNEL_LOG_ERROR("The elements of sequsedQ must be non-negative, but %zuth element is %ld", i, sequsedQ[i]);
            return false;
        }
    }

    if (!cuSeqlensQ.empty()) {
        if (cuSeqlensQ[0] != 0) {
            KERNEL_LOG_ERROR("The first element of cuSeqlensQ must be 0, but got %ld", cuSeqlensQ[0]);
            return false;
        }
    }

    for (size_t i = 1; i < cuSeqlensQ.size(); ++i) {
        if (cuSeqlensQ[i] < cuSeqlensQ[i - 1]) {
            KERNEL_LOG_ERROR(
                "The %zuth element of cuSeqlensQ must not be less than the %zuth element, but got %ld and %ld", i,
                i - 1, cuSeqlensQ[i], cuSeqlensQ[i - 1]);
            return false;
        }
    }

    if (!sequsedQ.empty()) {
        isActualSeqlenQAccum_ = false;
        actualSeqlenQ_ = sequsedQ;
    } else if (!cuSeqlensQ.empty()) {
        isActualSeqlenQAccum_ = true;
        actualSeqlenQ_.assign(cuSeqlensQ.begin() + 1, cuSeqlensQ.end());
    }

    return true;
}

bool FlashAttnMetadataCpuKernel::CheckActualKvSeq()
{
    isActualSeqlenKvAccum_ = false;
    actualSeqlenKv_.clear();
    std::vector<int64_t> cuSeqlensKv{};
    std::vector<int64_t> sequsedKv{};

    cuSeqlensKv = GetTensorDataAsInt64(cuSeqlensKv_);
    sequsedKv = GetTensorDataAsInt64(sequsedKv_);
    // Optional per the interface doc: under TND layout, max_seqlen_kv may be
    // omitted (<0) and is derived from cu_seqlens_kv (packed token total).
    if (fagIsGradEnabled_ && layoutKv_ == "TND" && maxSeqlenKv_ < 0 && !cuSeqlensKv.empty()) {
        maxSeqlenKv_ = static_cast<int32_t>(cuSeqlensKv.back());
    }
    if (fagIsGradEnabled_ &&
        !CheckFagSequence(cuSeqlensKv, sequsedKv, batchSize_, maxSeqlenKv_, layoutKv_ == "TND", "kv")) {
        return false;
    }
    fagPhysicalTokensKv_ = cuSeqlensKv.empty() ? 0 : cuSeqlensKv.back();
    fagHasSeqused_ = fagHasSeqused_ || !sequsedKv.empty();

    for (size_t i = 0; i < sequsedKv.size(); ++i) {
        if (sequsedKv[i] < 0) {
            KERNEL_LOG_ERROR("The elements of sequsedKv must be non-negative, but %zuth element is %ld", i,
                             sequsedKv[i]);
            return false;
        }
    }

    if (!cuSeqlensKv.empty()) {
        if (cuSeqlensKv[0] != 0) {
            KERNEL_LOG_ERROR("The first element of cuSeqlensKv must be 0, but got %ld", cuSeqlensKv[0]);
            return false;
        }
    }

    for (size_t i = 1; i < cuSeqlensKv.size(); ++i) {
        if (cuSeqlensKv[i] < cuSeqlensKv[i - 1]) {
            KERNEL_LOG_ERROR(
                "The %zuth element of cuSeqlensKv must not be less than the %zuth element, but got %ld and %ld", i,
                i - 1, cuSeqlensKv[i], cuSeqlensKv[i - 1]);
            return false;
        }
    }

    if (!sequsedKv.empty()) {
        isActualSeqlenKvAccum_ = false;
        actualSeqlenKv_ = sequsedKv;
    } else if (!cuSeqlensKv.empty()) {
        isActualSeqlenKvAccum_ = true;
        actualSeqlenKv_.assign(cuSeqlensKv.begin() + 1, cuSeqlensKv.end());
    }

    return true;
}

void FlashAttnMetadataCpuKernel::InitDeviceInfo()
{
    // Keep aicCoreNum_/aivCoreNum_ unchanged: they define metadata strides and the FAG layout.
    // 核数已在 InitLoadBalanceParams 里与基本块/模板一起算出, 这里直接使用。
    uint32_t maxUsedAivCores = maxUsedAicCores_ * (aivCoreNum_ / aicCoreNum_);
    deviceInfo.aicCoreMaxNum = maxUsedAicCores_;
    deviceInfo.aivCoreMaxNum = maxUsedAivCores;
    deviceInfo.aicCoreMinNum = maxUsedAicCores_;
    deviceInfo.aivCoreMinNum = maxUsedAivCores;
}

int64_t FlashAttnMetadataCpuKernel::CostFunc(uint32_t basicM, uint32_t basicS2)
{
    const uint32_t mAlignCoef = 16U;
    const uint32_t s2AlignCoef = 64U;
    const float mCoef = -0.037f;
    const float s2Coef = 0.11f;
    const float constant = 3.45f;

    float alignBasicM = static_cast<float>(load_balance::CeilDiv(basicM, mAlignCoef));
    float alignBasicS2 = static_cast<float>(load_balance::CeilDiv(basicS2, s2AlignCoef));

    return static_cast<int64_t>(mCoef * alignBasicM + s2Coef * alignBasicS2 + constant);
}

void FlashAttnMetadataCpuKernel::InitLoadBalanceParams()
{
    uint32_t cvRatio = static_cast<uint32_t>(aivCoreNum_ / aicCoreNum_);
    uint32_t gSize = static_cast<uint32_t>(numHeadsQ_ / numHeadsKv_);
    bool dnUnmergedRouted = false;
    if (socVersion_.find("Ascend960") != std::string::npos) {
        uint32_t templateId = 0;
        optiling::flash_attn::fa_tiling_util::GetConfigInfo(static_cast<uint32_t>(headDimV_), maskMode_, maxSeqlenQ_,
                                                            templateId, mBaseSize_, s2BaseSize_);
        maxUsedAicCores_ = optiling::flash_attn::fa_tiling_util::GetMaxUsedAicCores(
            aicCoreNum_, baseInfo.batchSize, numHeadsKv_, gSize, maxSeqlenQ_, maxSeqlenKv_, mBaseSize_ * cvRatio,
            s2BaseSize_);
        dnUnmergedRouted = templateId == FA_Template_DN_Unmerged;
        param.costFunc = nullptr;
    } else {
        // 与 host tiling 同一个入口: 模板 -> 基本块 -> 核数, 一次算出。
        optiling::flash_attn::fa_tiling_util::FaKernelParamArch35 kernelParam{
            .aicNum = aicCoreNum_,
            .cvRatio = cvRatio,
            .batchSize = static_cast<uint32_t>(batchSize_),
            .kvHeads = static_cast<uint32_t>(numHeadsKv_),
            .gSize = gSize,
            .qkHeadDim = static_cast<uint32_t>(headDim_),
            .vHeadDim = static_cast<uint32_t>(headDimV_),
            .maskMode = maskMode_,
            .maxSeqlenQ = maxSeqlenQ_,
            .maxSeqlenKv = maxSeqlenKv_,
            .winLeft = baseInfo.preToken,
            .winRight = baseInfo.nextToken,
            .hasDrop = dropoutP_ > 0.0f,
        };
        optiling::flash_attn::fa_tiling_util::FaKernelConfigArch35 kernelConfig{};
        optiling::flash_attn::fa_tiling_util::CalcFaKernelConfigArch35::CalcConfig(kernelParam, kernelConfig);

        mBaseSize_ = kernelConfig.sOuter;
        s2BaseSize_ = kernelConfig.sInner;
        maxUsedAicCores_ = kernelConfig.usedCoreNum;
        dnUnmergedRouted = kernelConfig.templateId == FA_Template_DN_Unmerged;
        param.costFunc = FlashAttnMetadataCpuKernel::CostFunc;
        if (!actualSeqlenKv_.empty() || !actualSeqlenQ_.empty()) {
            param.isConsiderHbm = true;
            param.hbmCostFunc = [](uint32_t a, uint32_t b) -> int64_t {
                return FlashAttnMetadataCpuKernel::CostFunc(a, b) / 4; // 4: 经验值
            };
        }
    }
    mBaseSize_ *= (aivCoreNum_ / aicCoreNum_);
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    // Reserve one quarter of Ascend350's 64 MiB L2, matching the existing 950 budget ratio.
    const uint32_t l2BudgetMiB = socVersion_.find("Ascend350") != std::string::npos ? 48U : 96U;
    param.l2Byte = l2BudgetMiB * 1024U * 1024U;
    param.fdTolerance = 10; // 10: tolerance block
    param.fdLeastBlock = 3; // 3: least block
    param.fdOn = true;
    param.kernelSplitMode =
        dnUnmergedRouted ? load_balance::KernelSplitMode::BN1_S1_S2 : load_balance::KernelSplitMode::BN2_S1G_S2;
    KERNEL_LOG_DEBUG("InitLoadBalanceParams: mBaseSize=%u, s2BaseSize=%u, kernelSplitMode=%u", param.mBaseSize,
                     param.s2BaseSize, static_cast<uint32_t>(param.kernelSplitMode));
}

void FlashAttnMetadataCpuKernel::InitBaseInfo()
{
    baseInfo.batchSize = (actualSeqlenQ_.empty()) ? batchSize_ : actualSeqlenQ_.size();
    baseInfo.querySeqSize = maxSeqlenQ_;
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.kvSeqSize = maxSeqlenKv_;
    baseInfo.kvHeadNum = numHeadsKv_;
    baseInfo.headDimQk = headDim_;
    baseInfo.headDimV = headDimV_;
    load_balance::SparseMode maskMode = load_balance::SparseMode::BUTT;
    if (maskMode_ != 0) {
        maskMode = static_cast<load_balance::SparseMode>(maskMode_);
    }
    baseInfo.attenMaskFlag = (maskMode != load_balance::SparseMode::BUTT);
    baseInfo.sparseMode = static_cast<uint32_t>(maskMode);
    baseInfo.preToken = (winLeft_ == -1) ? static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) : winLeft_;
    baseInfo.nextToken = (winRight_ == -1) ? static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) : winRight_;
    baseInfo.layoutQuery = load_balance::ConvertToLayout(layoutQ_);
    baseInfo.layoutKv = load_balance::ConvertToLayout(layoutKv_);
    baseInfo.queryType = load_balance::DataType::FP16;
    baseInfo.kvType = load_balance::DataType::FP16;
    baseInfo.isCumulativeKvSeq = isActualSeqlenKvAccum_;
    baseInfo.actualKvSeqSize = actualSeqlenKv_;
    baseInfo.isCumulativeQuerySeq = isActualSeqlenQAccum_;
    baseInfo.actualQuerySeqSize = actualSeqlenQ_;
    KERNEL_LOG_DEBUG("InitBaseInfo: batchSize=%u, querySeqSize=%u, kvSeqSize=%u, queryHeadNum=%u, kvHeadNum=%u",
                     baseInfo.batchSize, baseInfo.querySeqSize, baseInfo.kvSeqSize, baseInfo.queryHeadNum,
                     baseInfo.kvHeadNum);
    KERNEL_LOG_DEBUG("InitBaseInfo: headDimQk=%u, headDimV=%u, attenMaskFlag=%d, sparseMode=%u, preToken=%ld, "
                     "nextToken=%ld",
                     baseInfo.headDimQk, baseInfo.headDimV, static_cast<int32_t>(baseInfo.attenMaskFlag),
                     baseInfo.sparseMode, baseInfo.preToken, baseInfo.nextToken);
    KERNEL_LOG_DEBUG("InitBaseInfo: layoutQuery=%u, layoutKv=%u, isCumulativeQuerySeq=%d, isCumulativeKvSeq=%d",
                     static_cast<uint32_t>(baseInfo.layoutQuery), static_cast<uint32_t>(baseInfo.layoutKv),
                     static_cast<int32_t>(baseInfo.isCumulativeQuerySeq),
                     static_cast<int32_t>(baseInfo.isCumulativeKvSeq));
}

bool FlashAttnMetadataCpuKernel::BalanceSchedule(load_balance::SectionStreamKResult& splitRes)
{
    bool success =
        load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == SECTION_STREAM_K_SUCCESS;
    KERNEL_LOG_DEBUG("BalanceSchedule: sectionNum=%u", splitRes.sectionNum);
    return success;
}

bool FlashAttnMetadataCpuKernel::GenMetadata(load_balance::SectionStreamKResult& splitRes)
{
    const auto shape = metadata_->GetTensorShape();
    const int64_t words = shape != nullptr && shape->GetDims() == 1 ? shape->GetDimSize(0) : 0;
    const int64_t required = METADATA_STRIDE +
                             static_cast<int64_t>(splitRes.sectionNum) *
                                 (static_cast<int64_t>(aicCoreNum_) + static_cast<int64_t>(aivCoreNum_)) *
                                 METADATA_STRIDE +
                             (fagIsGradEnabled_ ? FAG_METADATA_SIZE : 0);
    KERNEL_CHECK_FALSE(words >= required, false, "metadata storage is too short for the generated sections");
    detail::FaMetadata faMetadata(aicCoreNum_, aivCoreNum_, splitRes.sectionNum, metadata_->GetData());
    faMetadata.Clear(); // set to all 0

    SetMetadataHead(splitRes, faMetadata);
    SetMetadataFa(splitRes, faMetadata);
    SetMetadataFd(splitRes, faMetadata);
    return true;
}

bool FlashAttnMetadataCpuKernel::CheckNeedInitOutput(uint32_t& uniformQLen) const
{
    // For padded layouts, the kernel must compare the uniform used Q length with physical S.
    const bool checkUniformQ = layoutQ_ != "TND" && !actualSeqlenQ_.empty();
    uniformQLen = 0;
    // seqused may leave physical query rows without a scheduled task.
    const auto cuSeqlensQ = GetTensorDataAsInt64(cuSeqlensQ_);
    for (uint32_t batchIdx = 0; batchIdx < baseInfo.batchSize; ++batchIdx) {
        const int64_t qLen = baseInfo.GetQuerySeqSize(batchIdx);
        const int64_t kvLen = baseInfo.GetKvSeqSize(batchIdx);
        if (checkUniformQ) {
            if (batchIdx == 0) {
                uniformQLen = static_cast<uint32_t>(qLen);
            } else if (qLen != uniformQLen) {
                return true;
            }
        } else {
            int64_t physicalQLen = maxSeqlenQ_;
            if (layoutQ_ == "TND") {
                if (cuSeqlensQ.size() <= static_cast<size_t>(batchIdx) + 1) {
                    return true; // Physical coverage cannot be established.
                }
                physicalQLen = cuSeqlensQ[batchIdx + 1] - cuSeqlensQ[batchIdx];
            }
            if (physicalQLen < 0 || qLen < physicalQLen) {
                return true;
            }
        }
        if (qLen == 0) {
            continue; // Physical Q coverage is checked separately; no KV is needed for zero effective Q.
        }
        if (kvLen == 0) {
            return true;
        }
        if (maskMode_ == static_cast<int32_t>(load_balance::SparseMode::RIGHT_DOWN_CAUSAL) && qLen > kvLen) {
            return true;
        }
        if (maskMode_ == static_cast<int32_t>(load_balance::SparseMode::BAND) && winRight_ != -1 &&
            qLen - kvLen > winRight_) {
            return true;
        }
    }
    return false;
}

void FlashAttnMetadataCpuKernel::SetMetadataHead(const load_balance::SectionStreamKResult& splitRes,
                                                 optiling::detail::FaMetadata& faMetadata)
{
    faMetadata.SetHeadMetadata(HEAD_SECTION_NUM_INDEX, splitRes.sectionNum);
    uint32_t uniformQLen = 0;
    uint32_t skipInitOutput = 0;
    if (!CheckNeedInitOutput(uniformQLen)) {
        // Mode 2 requires a physical-Q check in the kernel; older kernels conservatively initialize.
        skipInitOutput = (layoutQ_ != "TND" && !actualSeqlenQ_.empty()) ? 2U : 1U;
    }
    faMetadata.SetHeadMetadata(HEAD_SKIP_INIT_OUTPUT_INDEX, skipInitOutput);
    faMetadata.SetHeadMetadata(HEAD_UNIFORM_Q_LEN_INDEX, skipInitOutput == 2U ? uniformQLen : 0U);
    faMetadata.SetHeadMetadata(HEAD_M_BASE_SIZE_INDEX, mBaseSize_);
    faMetadata.SetHeadMetadata(HEAD_S2_BASE_SIZE_INDEX, s2BaseSize_);
    if (std::any_of(splitRes.sectionFdResult.begin(), splitRes.sectionFdResult.end(),
                    [](load_balance::SectionStreamKFdResult result) { return result.usedVecNum > 0U; })) {
        faMetadata.SetHeadMetadata(HEAD_IS_FD_INDEX, 1U);
        KERNEL_LOG_INFO("FlashDecode is enabled.");
    }
    faMetadata.SetHeadMetadata(HEAD_AIC_NUM_INDEX, aicCoreNum_);
    faMetadata.SetHeadMetadata(HEAD_AIV_NUM_INDEX, aivCoreNum_);
    faMetadata.SetHeadMetadata(HEAD_OUTPUT_LAYOUT_INDEX, static_cast<FA_METADATA_T>(param.kernelSplitMode));
}

void FlashAttnMetadataCpuKernel::SetMetadataFa(const load_balance::SectionStreamKResult& splitRes,
                                               optiling::detail::FaMetadata& faMetadata)
{
    load_balance::SectionStreamKFaResult dummyHead{static_cast<uint32_t>(aicCoreNum_)}; // all zeror dummy head
    for (uint32_t secIdx = 0; secIdx < splitRes.sectionNum; ++secIdx) {
        auto& faRes = splitRes.sectionFaResult[secIdx];
        for (uint32_t aicIdx = 0; aicIdx < faRes.usedCoreNum; ++aicIdx) {
            auto& prevFaRes = (secIdx == 0U) ? dummyHead : splitRes.sectionFaResult[secIdx - 1U];
            auto prevLastCore = (secIdx == 0U) ? 0U : prevFaRes.usedCoreNum - 1U;
            FA_METADATA_T bnStart = (aicIdx == 0) ? prevFaRes.bNEnd[prevLastCore] : faRes.bNEnd[aicIdx - 1U];
            FA_METADATA_T mStart = (aicIdx == 0) ? prevFaRes.mEnd[prevLastCore] : faRes.mEnd[aicIdx - 1U];
            FA_METADATA_T s2Start = (aicIdx == 0) ? prevFaRes.s2End[prevLastCore] : faRes.s2End[aicIdx - 1U];

            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN_START_INDEX, bnStart);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_START_INDEX, mStart);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_START_INDEX, s2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN_END_INDEX, faRes.bNEnd[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_END_INDEX, faRes.mEnd[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_END_INDEX, faRes.s2End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                     faRes.firstFdDataWorkspaceIdx[aicIdx]);
        }
    }
}

void FlashAttnMetadataCpuKernel::SetMetadataFd(const load_balance::SectionStreamKResult& splitRes,
                                               optiling::detail::FaMetadata& faMetadata)
{
    for (uint32_t secIdx = 0; secIdx < splitRes.sectionNum; ++secIdx) {
        auto& fdRes = splitRes.sectionFdResult[secIdx];
        for (uint32_t aivIdx = 0; aivIdx < fdRes.usedVecNum; ++aivIdx) {
            uint32_t t = fdRes.taskIdx[aivIdx];
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_BN_IDX_INDEX, fdRes.bNIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_IDX_INDEX, fdRes.mIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_IDX_INDEX, fdRes.workspaceIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_NUM_INDEX, fdRes.s2SplitNum[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_START_INDEX, fdRes.mStart[aivIdx]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_NUM_INDEX, fdRes.mLen[aivIdx]);
        }
    }
}

namespace {
static const char* kernelType = "FlashAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, FlashAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu

// Deterministic FAG planning for the existing metadata producer.
namespace optiling::fag_deter {
namespace asc {
#ifdef _MSC_VER
#pragma warning(push)
// Retain ASC's float load-balancing weights and original narrowing in its L2
// sets on CPU-checking MSVC builds. Do not promote weights to double.
#pragma warning(disable : 4244 4267)
#endif
namespace {
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// Migrated from ASC; maintain the documented local corrections alongside this source.
// Source: attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_varlen_regbase.cpp
// SHA256: 4fd887cae30ee387310c2e323209de0edf9c6bce1e6e99009e82f633e9c3fd25
// Source: attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp
// SHA256: cb140d6d29ee900ff1bbc4a1ebaa1217c265698a1f3a2c05843d782a470dab5d
// Source: attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_common_regbase.cpp
// SHA256: 1b53a86de6273de021585b37b53a1e4c81bf9851061360566ba7386c7f6f1873

// Preserve this planner's enum values, signed division and token convention.
struct TndMathPolicy {
    using Params = FuzzyBaseInfoParamsRegbase;
    using SparseMode = optiling::fag_deter::asc::SparseMode;
    using DeterSparseType = optiling::fag_deter::asc::DeterSparseType;
    static constexpr size_t PREFIX_CAPACITY = DETER_PREFIX_ARRAY_SIZE;
    static int64_t Ceil(int64_t value, int64_t divisor)
    {
        return CeilDivideBy(value, divisor);
    }
    static void ActualTokens(Params& p, int64_t batch, int64_t& q, int64_t& kv)
    {
        CalcleActualToken(p, batch, q, kv);
    }
    static std::vector<int64_t> Slice(const std::vector<int64_t>& source, int64_t step)
    {
        return SliceVector(source, step);
    }
};
using TndMath = aicpu::fag_tnd_math::Functions<TndMathPolicy>;
using TNDLineDeterPack = aicpu::fag_tnd_math::TNDLineDeterPack;

inline void ApplyTNDLineDeterParam(FuzzyBaseInfoParamsRegbase& p, const TNDLineDeterPack& pack)
{
    p.splitAxis = SplitAxisEnum::BN2GS1S2;
    p.tndLineDeter = 1;
    p.deterMaxRound = pack.rLine;
    TndMath::StoreTNDPrefix(pack.prefixSolo, p.deterPrefixStep, pack.rLine, p.deterPrefix0);
    TndMath::StoreTNDPrefix(pack.prefixSingle, p.deterPrefixStep, pack.prefixSingle.back(), p.deterPrefix1);

    std::vector<int64_t> deterPrefix = {0};
    std::vector<int64_t> deterPrefixAlign = {0};
    for (int64_t bIdx = 0; bIdx < p.b; bIdx++) {
        deterPrefix.push_back(deterPrefix.back() + p.actualSeqQlen[bIdx] * p.actualSeqKvlen[bIdx]);
        deterPrefixAlign.push_back(
            deterPrefixAlign.back() +
            p.actualSeqQlen[bIdx] * AlignTo(p.actualSeqKvlen[bIdx], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));
    }
    deterPrefix = SliceVector(deterPrefix, p.deterPrefixStep);
    deterPrefixAlign = SliceVector(deterPrefixAlign, p.deterPrefixStep);
    std::copy(deterPrefix.begin(), deterPrefix.end(), p.deterPrefix);
    std::copy(deterPrefixAlign.begin(), deterPrefixAlign.end(), p.deterPrefixAlign);
}

inline int64_t PositiveCeilDiv(int64_t dividend, int64_t divisor)
{
    return dividend / divisor + static_cast<int64_t>(dividend % divisor != 0);
}

inline bool IsRoundGrowthWithinLimit(int64_t candidateRound, int64_t legacyRound, int64_t maxGrowthPercent,
                                     int64_t percentBase)
{
    return candidateRound > 0 && legacyRound > 0 &&
           candidateRound * percentBase <= legacyRound * (percentBase + maxGrowthPercent);
}

inline bool IsTndDeterSwizzleScheduleSafe(const FuzzyBaseInfoParamsRegbase& params)
{
    if (params.aicNum == 0 || params.n1 <= 0) {
        return false;
    }

    const int64_t cubeBaseM = static_cast<int64_t>(params.s1Inner) * params.s1CvRatio;
    const int64_t cubeBaseN = static_cast<int64_t>(params.s2Inner) * params.s2CvRatio;
    if (cubeBaseM <= 0 || cubeBaseN <= 0 || params.actualSeqQlen.size() != params.actualSeqKvlen.size()) {
        return false;
    }

    // Mode 2 assigns consecutive linear IDs to S2 blocks inside each head and rotates S1 by S2 + round.
    // One head has at most min(k, n) active cores in a round. The same number of S1 blocks is required to
    // keep their dQ coordinates unique. The Python recommendation m >= n is the k >= n special case.
    for (size_t bIdx = 0; bIdx < params.actualSeqQlen.size(); ++bIdx) {
        const int64_t actualSeqQLen = params.actualSeqQlen[bIdx];
        const int64_t actualSeqKvLen = params.actualSeqKvlen[bIdx];
        // BuildInitialSchedule validates nonnegative lengths. Empty batches
        // have no coordinates and retain their original physical cu offsets.
        if (actualSeqQLen == 0 || actualSeqKvLen == 0) {
            continue;
        }
        const int64_t m = PositiveCeilDiv(actualSeqQLen, cubeBaseM);
        const int64_t n = PositiveCeilDiv(actualSeqKvLen, cubeBaseN);
        if (m < std::min(static_cast<int64_t>(params.aicNum), n)) {
            return false;
        }
    }
    return true;
}

inline bool IsTndDeterSwizzlePerformanceEnough(const FuzzyBaseInfoParamsRegbase& params)
{
    if (params.aicNum == 0 || params.n2 <= 0 || params.g <= 0 ||
        params.actualSeqQlen.size() != params.actualSeqKvlen.size()) {
        return false;
    }

    const int64_t aicNum = static_cast<int64_t>(params.aicNum);
    const int64_t cubeBaseM = static_cast<int64_t>(params.s1Inner) * params.s1CvRatio;
    const int64_t cubeBaseN = static_cast<int64_t>(params.s2Inner) * params.s2CvRatio;
    if (cubeBaseM <= 0 || cubeBaseN <= 0) {
        return false;
    }

    int64_t validSlotNum = 0;
    int64_t totalSlotNum = 0;
    int64_t maxS1Outer = 0;
    int64_t maxS2Outer = 0;
    for (size_t bIdx = 0; bIdx < params.actualSeqQlen.size(); ++bIdx) {
        const int64_t actualSeqQLen = params.actualSeqQlen[bIdx];
        const int64_t actualSeqKvLen = params.actualSeqKvlen[bIdx];
        // BuildInitialSchedule validates nonnegative lengths. Empty batches
        // have no coordinates and retain their original physical cu offsets.
        if (actualSeqQLen == 0 || actualSeqKvLen == 0) {
            continue;
        }
        const int64_t m = PositiveCeilDiv(actualSeqQLen, cubeBaseM);
        const int64_t n = PositiveCeilDiv(actualSeqKvLen, cubeBaseN);
        const int64_t columnNum = n * params.n2 * params.g;
        const int64_t batchRound = PositiveCeilDiv(columnNum, aicNum) * m;
        validSlotNum += m * columnNum;
        totalSlotNum += batchRound * aicNum;
        maxS1Outer = std::max(maxS1Outer, m);
        maxS2Outer = std::max(maxS2Outer, n);
    }
    const bool slotUtilizationEnough =
        totalSlotNum > 0 &&
        validSlotNum * TND_DETER_SWIZZLE_PERCENT_BASE >= totalSlotNum * TND_DETER_SWIZZLE_MIN_SLOT_UTILIZATION_PERCENT;
    const int64_t candidateMaxRound = totalSlotNum / aicNum;
    // Keep the legacy bound identical to CalcleTNDDenseDeterParam.
    const int64_t legacyMaxRound = std::max({PositiveCeilDiv(validSlotNum, aicNum), maxS1Outer * params.g, maxS2Outer});
    const bool roundGrowthWithinLimit = IsRoundGrowthWithinLimit(
        candidateMaxRound, legacyMaxRound, TND_DETER_SWIZZLE_MAX_ROUND_GROWTH_PERCENT, TND_DETER_SWIZZLE_PERCENT_BASE);
    return slotUtilizationEnough && roundGrowthWithinLimit;
}

inline bool IsTndDeterSwizzleSupported(const FuzzyBaseInfoParamsRegbase& params)
{
    if (!IsTndDeterSwizzleScheduleSafe(params)) {
        return false;
    }
    return IsTndDeterSwizzlePerformanceEnough(params);
}

inline void ConfigureTndDeterBn2S2Swizzle(FuzzyBaseInfoParamsRegbase& params, const TndBaseInfo& tndBaseInfo)
{
    // Mode 2 keeps every (B, N2, S2) column on one core, so dK/dV never need a cross-core tail merge.
    std::fill(std::begin(params.deterPrefix2), std::end(params.deterPrefix2), static_cast<int64_t>(-1));

    // Different columns still atomically accumulate dQ. Synchronize consecutive rounds to keep the accumulation
    // order deterministic, as in the BN2GS1S2 deterministic swizzle path.
    std::fill(std::begin(params.startNeedSyncRound), std::end(params.startNeedSyncRound), static_cast<uint64_t>(0));
    std::fill(std::begin(params.endNeedSyncRound), std::end(params.endNeedSyncRound), static_cast<uint64_t>(0));
    params.startNeedSyncRound[0] = 1;
    params.endNeedSyncRound[0] = tndBaseInfo.tndS2BlockPrefixSum[params.b];
}

inline void GetOffset(FuzzyBaseInfoParamsRegbase& fBaseParams, int64_t& currentDqOffset, int64_t& currentDkDvOffset,
                      int64_t blockIdx)
{
    int64_t boIdx = 0;
    int64_t bDimTail = 0;
    int64_t n2oIdx = 0;
    int64_t n2DimTail = 0;
    int64_t goIdx = 0;
    int64_t gDimTail = 0;
    int64_t s2oIdx = 0;
    int64_t s1oIdx = 0;

    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    int64_t s2Offset = 0;
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        int64_t resbaseIdx = blockIdx;
        for (int64_t bIdx = 0; bIdx < fBaseParams.b; bIdx++) {
            int64_t actualS1Len = fBaseParams.actualSeqQlen[bIdx];
            int64_t actualS2Len = fBaseParams.actualSeqKvlen[bIdx];
            int64_t s1OuterTmp = (actualS1Len + fBaseParams.s1Inner * S1CV_RATIO_DEFAULT - 1) /
                                 (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT);
            int64_t s2OuterTmp = (actualS2Len + fBaseParams.s2Inner - 1) / fBaseParams.s2Inner;
            int64_t totalBaseIdx = fBaseParams.n2 * fBaseParams.g * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                boIdx = bIdx;
                bDimTail = resbaseIdx;
                n2oIdx = bDimTail / (fBaseParams.g * s1OuterTmp * s2OuterTmp);
                n2DimTail = bDimTail % (fBaseParams.g * s1OuterTmp * s2OuterTmp);
                goIdx = n2DimTail / (s1OuterTmp * s2OuterTmp);
                gDimTail = n2DimTail % (s1OuterTmp * s2OuterTmp);
                s2oIdx = gDimTail / s1OuterTmp;
                s1oIdx = gDimTail % s1OuterTmp;
                break;
            } else {
                resbaseIdx -= totalBaseIdx;
            }
        }
        // caculate dq offset
        for (int64_t bIdx = 0; bIdx < boIdx; bIdx++) {
            bOffset += fBaseParams.actualSeqQlen[bIdx] * (fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
        }
        s1Offset = s1oIdx * fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
        n2Offset = n2oIdx * fBaseParams.g * fBaseParams.d;
        gOffset = goIdx * fBaseParams.d;
        currentDqOffset = bOffset + n2Offset + gOffset + s1Offset;
        // caculate dk dv offset
        bOffset = 0;
        for (int64_t bIdx = 0; bIdx < boIdx; bIdx++) {
            bOffset += fBaseParams.actualSeqKvlen[bIdx] * (fBaseParams.n2 * fBaseParams.d);
        }
        s2Offset = s2oIdx * fBaseParams.s2Inner * fBaseParams.n2 * fBaseParams.d;
        n2Offset = n2oIdx * fBaseParams.d;
        currentDkDvOffset = bOffset + n2Offset + s2Offset;
    } else {
        boIdx = blockIdx / (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
        bDimTail = blockIdx % (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
        n2oIdx = bDimTail / (fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
        n2DimTail = bDimTail % (fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
        goIdx = n2DimTail / (fBaseParams.s1Outer * fBaseParams.s2Outer);
        gDimTail = n2DimTail % (fBaseParams.s1Outer * fBaseParams.s2Outer);
        s2oIdx = gDimTail / fBaseParams.s1Outer;
        s1oIdx = gDimTail % fBaseParams.s1Outer;
        // caculate dq offset
        if (fBaseParams.layoutType == INPUT_FORMAT_BN2GS2D) {
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.d);
            n2Offset = n2oIdx * (fBaseParams.g * fBaseParams.s1 * fBaseParams.d);
            gOffset = goIdx * (fBaseParams.s1 * fBaseParams.d);
            s1Offset = s1oIdx * fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * fBaseParams.d;
        } else if (fBaseParams.layoutType == INPUT_FORMAT_S2BN2GD) {
            s1Offset = s1oIdx * fBaseParams.s1Inner * S1CV_RATIO_DEFAULT *
                       (fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
            n2Offset = n2oIdx * (fBaseParams.g * fBaseParams.d);
            gOffset = goIdx * fBaseParams.d;
        } else if (fBaseParams.layoutType == INPUT_FORMAT_BS2N2GD) {
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.d);
            s1Offset =
                s1oIdx * fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
            n2Offset = n2oIdx * (fBaseParams.g * fBaseParams.d);
            gOffset = goIdx * fBaseParams.d;
        }
        currentDqOffset = bOffset + n2Offset + gOffset + s1Offset;
        // caculate dk dv offset
        if (fBaseParams.layoutType == INPUT_FORMAT_BN2GS2D) {
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.s2 * fBaseParams.d);
            n2Offset = n2oIdx * (fBaseParams.s2 * fBaseParams.d);
            s2Offset = s2oIdx * fBaseParams.s2Inner * fBaseParams.d;
        } else if (fBaseParams.layoutType == INPUT_FORMAT_S2BN2GD) {
            s2Offset = s2oIdx * fBaseParams.s2Inner * (fBaseParams.b * fBaseParams.n2 * fBaseParams.d);
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.d);
            n2Offset = n2oIdx * fBaseParams.d;
        } else if (fBaseParams.layoutType == INPUT_FORMAT_BS2N2GD) {
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.s2 * fBaseParams.d);
            s2Offset = s2oIdx * fBaseParams.s2Inner * (fBaseParams.n2 * fBaseParams.d);
            n2Offset = n2oIdx * fBaseParams.d;
        }
        currentDkDvOffset = bOffset + n2Offset + s2Offset;
    }
}

} // namespace

// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// Migrated from ASC. Preserve the documented local
// BAND origin and empty-batch support corrections when reimporting ASC.
// Source: attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_varlen_regbase.cpp
// SHA256: 4fd887cae30ee387310c2e323209de0edf9c6bce1e6e99009e82f633e9c3fd25
// Source: attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp
// SHA256: cb140d6d29ee900ff1bbc4a1ebaa1217c265698a1f3a2c05843d782a470dab5d
// Source: attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_common_regbase.cpp
// SHA256: 1b53a86de6273de021585b37b53a1e4c81bf9851061360566ba7386c7f6f1873

#define FAG_PLAN_TRACE(...) ((void)0)

void Scheduler::CalcleTNDDeterParam()
{
    if (fBaseParams.layoutType != INPUT_FORMAT_TND) {
        return;
    }

    // 如果b小于deterPrefixThreshold，传完整的前缀和，否则按步长切分传部分
    if (fBaseParams.b > fBaseParams.deterPrefixThreshold) {
        fBaseParams.deterPrefixStep = CeilDivideBy(fBaseParams.b, fBaseParams.deterPrefixThreshold);
    }

    std::fill(std::begin(fBaseParams.deterPrefix0), std::end(fBaseParams.deterPrefix0), static_cast<int64_t>(0));
    std::fill(std::begin(fBaseParams.deterPrefix1), std::end(fBaseParams.deterPrefix1), static_cast<int64_t>(0));
    std::fill(std::begin(fBaseParams.deterPrefix2), std::end(fBaseParams.deterPrefix2), static_cast<int64_t>(0));
    std::fill(std::begin(fBaseParams.deterColStart), std::end(fBaseParams.deterColStart), static_cast<int64_t>(0));
    fBaseParams.tndLineDeter = 0;
    fBaseParams.tndGqaScheduleMode = DeterParamRegbase::SCHEDULE_DEFAULT;
    fBaseParams.tndGqaHeadPairGroup = 0;

    if (TndMath::PreferTNDLineDeter(fBaseParams)) {
        ApplyTNDLineDeterParam(fBaseParams, TndMath::BuildTNDLineDeter(fBaseParams));
        FAG_PLAN_TRACE("CalcleTNDDeterParam", "TND deterMaxRound is %ld.", fBaseParams.deterMaxRound);
        return;
    }

    CalcleTNDDenseDeterParam();
    CalcleTNDCausalDeterParam();
    CalcleTNDBandDeterParam();
    FAG_PLAN_TRACE("CalcleTNDDeterParam", "TND deterMaxRound is %ld.", fBaseParams.deterMaxRound);
}

void Scheduler::CalcleTNDDenseDeterParam()
{
    if (fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_DENSE)) {
        return;
    }
    DeterPrefixData deterPrefixData;
    int64_t s1Max = 0;
    int64_t s2Max = 0;
    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t actualS1Outer = CeilDivideBy(fBaseParams.actualSeqQlen[i], fBaseParams.s1Inner * fBaseParams.s1CvRatio);
        int64_t actualS2Outer =
            CeilDivideBy(fBaseParams.actualSeqKvlen[i], fBaseParams.s2Inner * fBaseParams.s2CvRatio);
        deterPrefixData.deterPrefix.push_back(deterPrefixData.deterPrefix.back() +
                                              fBaseParams.actualSeqQlen[i] * fBaseParams.actualSeqKvlen[i]);
        deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() + actualS1Outer * actualS2Outer);
        deterPrefixData.deterPrefixAlign.push_back(
            deterPrefixData.deterPrefixAlign.back() +
            fBaseParams.actualSeqQlen[i] *
                AlignTo(fBaseParams.actualSeqKvlen[i], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));
        s1Max = actualS1Outer > s1Max ? actualS1Outer : s1Max;
        s2Max = actualS2Outer > s2Max ? actualS2Outer : s2Max;
        deterPrefixData.mNewList.push_back(actualS1Outer);
        deterPrefixData.nNewList.push_back(actualS2Outer);
    }
    int64_t totalArea = deterPrefixData.prefix1.back() * fBaseParams.n1;
    fBaseParams.deterMaxRound =
        std::max({CeilDivideBy(totalArea, static_cast<int64_t>(fBaseParams.aicNum)), s1Max * fBaseParams.g, s2Max});

    deterPrefixData.prefix0 = SliceVector(deterPrefixData.prefix1, fBaseParams.deterPrefixStep);
    deterPrefixData.deterPrefix = SliceVector(deterPrefixData.deterPrefix, fBaseParams.deterPrefixStep);
    deterPrefixData.deterPrefixAlign = SliceVector(deterPrefixData.deterPrefixAlign, fBaseParams.deterPrefixStep);
    std::copy(deterPrefixData.prefix0.begin(), deterPrefixData.prefix0.end(), fBaseParams.deterPrefix0);
    std::copy(deterPrefixData.deterPrefix.begin(), deterPrefixData.deterPrefix.end(), fBaseParams.deterPrefix);
    std::copy(deterPrefixData.deterPrefixAlign.begin(), deterPrefixData.deterPrefixAlign.end(),
              fBaseParams.deterPrefixAlign);
    // MHA-only boundary: GQA exclusive scheduling is unreachable.
    deterPrefixData.prefix1.push_back(fBaseParams.deterMaxRound);
    CalcleTNDDenseBns2DeterParam(deterPrefixData);
    return;
}

void Scheduler::CalcleTNDCausalDeterParam()
{
    if (fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL)) {
        return;
    }
    fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
    fallbackReasons |= static_cast<int64_t>(Fallback::CAUSAL);
    int64_t m0Max{0}, m1Max{0}, m2Max{0};
    DeterPrefixData deterPrefixData;
    CalcleTNDCausalDeterPrefix(deterPrefixData, m0Max, m1Max, m2Max);

    deterPrefixData.deterPrefix = SliceVector(deterPrefixData.deterPrefix, fBaseParams.deterPrefixStep);
    deterPrefixData.deterPrefixAlign = SliceVector(deterPrefixData.deterPrefixAlign, fBaseParams.deterPrefixStep);
    std::copy(deterPrefixData.deterPrefix.begin(), deterPrefixData.deterPrefix.end(), fBaseParams.deterPrefix);
    std::copy(deterPrefixData.deterPrefixAlign.begin(), deterPrefixData.deterPrefixAlign.end(),
              fBaseParams.deterPrefixAlign);

    CalcleTNDCausalDeterParamNormal(deterPrefixData, m0Max, m1Max, m2Max);
}

void Scheduler::CalcleTNDBandDeterParam()
{
    if (fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_BAND)) {
        return;
    }
    int64_t N11 = fBaseParams.n1 / fBaseParams.aicNum;
    int64_t N12 = fBaseParams.n1 % fBaseParams.aicNum;
    int64_t mnMax = 0;

    DeterPrefixData deterPrefixData;
    CalcleTNDBandDeterPrefix(deterPrefixData, N11, mnMax);

    if (N11 > 0 && fBaseParams.g == 1) {
        int64_t R0 = deterPrefixData.prefix0.back();
        deterPrefixData.prefix0 = SliceVector(deterPrefixData.prefix0, fBaseParams.deterPrefixStep);
        deterPrefixData.prefix0.push_back(R0);
    }

    // 将最大轮次append在了prefix的最后，需要的时候可以直接取用，形式更简洁
    // prefix0的是乘了N1的结果，也可以不乘，乘了后二分查找不用额外申请空间
    int64_t R1 =
        fBaseParams.g == 1 ?
            (N12 > 0 ?
                 std::max(CeilDivideBy(deterPrefixData.prefix1.back() * N12, static_cast<int64_t>(fBaseParams.aicNum)),
                          mnMax) :
                 0) :
            std::max(
                CeilDivideBy(deterPrefixData.prefix1.back() * fBaseParams.n1, static_cast<int64_t>(fBaseParams.aicNum)),
                mnMax);
    std::vector<int64_t> slicePrefix1 = SliceVector(deterPrefixData.prefix1, fBaseParams.deterPrefixStep);
    deterPrefixData.prefix1.push_back(R1);
    slicePrefix1.push_back(R1);

    fBaseParams.deterMaxRound += deterPrefixData.prefix0.back();
    fBaseParams.deterMaxRound += slicePrefix1.back();
    deterPrefixData.deterPrefix = SliceVector(deterPrefixData.deterPrefix, fBaseParams.deterPrefixStep);
    deterPrefixData.deterPrefixAlign = SliceVector(deterPrefixData.deterPrefixAlign, fBaseParams.deterPrefixStep);
    std::copy(deterPrefixData.deterPrefix.begin(), deterPrefixData.deterPrefix.end(), fBaseParams.deterPrefix);
    std::copy(deterPrefixData.deterPrefixAlign.begin(), deterPrefixData.deterPrefixAlign.end(),
              fBaseParams.deterPrefixAlign);
    std::copy(deterPrefixData.prefix0.begin(), deterPrefixData.prefix0.end(), fBaseParams.deterPrefix0);
    std::copy(slicePrefix1.begin(), slicePrefix1.end(), fBaseParams.deterPrefix1);

    CalcleTNDBandBns2DeterParam(deterPrefixData);
}

void Scheduler::CalcleTNDDenseBns2DeterParam(DeterPrefixData& deterPrefixData)
{
    if (fBaseParams.splitAxis != SplitAxisEnum::BN2S2) {
        return;
    }

    // 最多允许coreNum列分给不同的核fBaseParams.deterMaxRound
    if (!SupportTNDBns2(deterPrefixData, fBaseParams.deterMaxRound)) {
        fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
        fallbackReasons |= static_cast<int64_t>(Fallback::SUPPORT_BN2S2);
        return;
    }

    // BNS2分核按顺序分核，存在前后两核收尾分同一列的情况，计算可能分开的列
    std::vector<std::pair<uint64_t, uint64_t>> syncRounds, syncRoundRanges;
    std::fill(std::begin(fBaseParams.startNeedSyncRound), std::end(fBaseParams.startNeedSyncRound),
              static_cast<uint64_t>(0));
    std::fill(std::begin(fBaseParams.endNeedSyncRound), std::end(fBaseParams.endNeedSyncRound),
              static_cast<uint64_t>(0));
    std::fill(std::begin(fBaseParams.separateDkOffset), std::end(fBaseParams.separateDkOffset),
              static_cast<int64_t>(-1));
    CalcleTNDDenseDeterSplitDkOffset(deterPrefixData, syncRounds, syncRoundRanges);
    std::copy(std::begin(fBaseParams.separateDkOffset), std::end(fBaseParams.separateDkOffset),
              std::begin(fBaseParams.deterPrefix2));

    CalcleTNDDeterSyncRounds(syncRounds, syncRoundRanges);
}

void Scheduler::CalcleTNDCausalDeterPrefix(DeterPrefixData& deterPrefixData, int64_t& m0Max, int64_t& m1Max,
                                           int64_t& m2Max)
{
    int64_t N12 = fBaseParams.g == 1 ? fBaseParams.n2 % fBaseParams.aicNum % NUM_TWO : fBaseParams.n2 % NUM_TWO;
    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t actualS1Outer = CeilDivideBy(fBaseParams.actualSeqQlen[i], fBaseParams.s1Inner * fBaseParams.s1CvRatio);
        int64_t actualS2Outer =
            CeilDivideBy(fBaseParams.actualSeqKvlen[i], fBaseParams.s2Inner * fBaseParams.s2CvRatio);
        deterPrefixData.deterPrefix.push_back(deterPrefixData.deterPrefix.back() +
                                              fBaseParams.actualSeqQlen[i] * fBaseParams.actualSeqKvlen[i]);
        deterPrefixData.deterPrefixAlign.push_back(
            deterPrefixData.deterPrefixAlign.back() +
            fBaseParams.actualSeqQlen[i] *
                AlignTo(fBaseParams.actualSeqKvlen[i], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));

        // left_up_causal场景下，如果m<n, 需要将n裁剪为m
        if (actualS1Outer < actualS2Outer) {
            actualS2Outer = actualS1Outer;
        }

        m0Max = std::max(m0Max, fBaseParams.g * (NUM_TWO * actualS1Outer - actualS2Outer + 1));
        deterPrefixData.prefix0.push_back(deterPrefixData.prefix0.back() +
                                          (NUM_TWO * actualS1Outer - actualS2Outer + 1) * actualS2Outer);

        if (N12 > 0) {
            deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() +
                                              (actualS1Outer - (actualS2Outer + 1) / NUM_TWO + 1) *
                                                  (actualS2Outer / NUM_TWO));
            if (fBaseParams.g == 1 || (actualS2Outer >= NUM_TWO && fBaseParams.g != 1)) {
                m1Max = std::max(m1Max, fBaseParams.g * (actualS1Outer - (actualS2Outer + 1) / NUM_TWO + 1));
            }

            deterPrefixData.prefix2.push_back(deterPrefixData.prefix2.back() +
                                              (actualS1Outer - actualS2Outer / NUM_TWO) *
                                                  ((actualS2Outer + 1) / NUM_TWO));
            m2Max = std::max(m2Max, fBaseParams.g * (actualS1Outer - actualS2Outer / NUM_TWO));
        }
    }
}

void Scheduler::CalcleTNDCausalDeterParamNormal(DeterPrefixData& deterPrefixData, const int64_t m0Max,
                                                const int64_t m1Max, const int64_t m2Max)
{
    int64_t N11 = fBaseParams.n2 % fBaseParams.aicNum / NUM_TWO;
    int64_t N12 = fBaseParams.n2 % fBaseParams.aicNum % NUM_TWO;
    int64_t prefix0Max1 = deterPrefixData.prefix0.back() / NUM_TWO * (fBaseParams.n2 / fBaseParams.aicNum);
    int64_t prefix0Max2 = std::max(
        CeilDivideBy(deterPrefixData.prefix0.back() * N11 * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
        m0Max);
    deterPrefixData.prefix0 = SliceVector(deterPrefixData.prefix0, fBaseParams.deterPrefixStep);
    deterPrefixData.prefix0.push_back(prefix0Max1);
    fBaseParams.deterMaxRound += prefix0Max1;
    if (N11 > 0) {
        deterPrefixData.prefix0.push_back(prefix0Max2);
        fBaseParams.deterMaxRound += prefix0Max2;
    } else {
        deterPrefixData.prefix0.push_back(0);
    }
    std::copy(deterPrefixData.prefix0.begin(), deterPrefixData.prefix0.end(), fBaseParams.deterPrefix0);

    if (N12 > 0) {
        int64_t r1 = std::max(
            CeilDivideBy(deterPrefixData.prefix1.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
            m1Max);
        int64_t r2 = std::max(
            CeilDivideBy(deterPrefixData.prefix2.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
            m2Max);
        deterPrefixData.prefix1 = SliceVector(deterPrefixData.prefix1, fBaseParams.deterPrefixStep);
        deterPrefixData.prefix2 = SliceVector(deterPrefixData.prefix2, fBaseParams.deterPrefixStep);
        deterPrefixData.prefix1.push_back(r1);
        deterPrefixData.prefix2.push_back(r2);
        fBaseParams.deterMaxRound += deterPrefixData.prefix1.back();
        fBaseParams.deterMaxRound += deterPrefixData.prefix2.back();
        std::copy(deterPrefixData.prefix1.begin(), deterPrefixData.prefix1.end(), fBaseParams.deterPrefix1);
        std::copy(deterPrefixData.prefix2.begin(), deterPrefixData.prefix2.end(), fBaseParams.deterPrefix2);
    }
}

void Scheduler::CalcleTNDBandDeterPrefix(DeterPrefixData& deterPrefixData, int64_t N11, int64_t& mnMax)
{
    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t m, n, p, q, mNew, nNew;
        m = CeilDivideBy(fBaseParams.actualSeqQlen[i], fBaseParams.s1Inner * fBaseParams.s1CvRatio);
        n = CeilDivideBy(fBaseParams.actualSeqKvlen[i], fBaseParams.s2Inner * fBaseParams.s2CvRatio);
        deterPrefixData.deterPrefix.push_back(deterPrefixData.deterPrefix.back() +
                                              fBaseParams.actualSeqQlen[i] * fBaseParams.actualSeqKvlen[i]);
        deterPrefixData.deterPrefixAlign.push_back(
            deterPrefixData.deterPrefixAlign.back() +
            fBaseParams.actualSeqQlen[i] *
                AlignTo(fBaseParams.actualSeqKvlen[i], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));

        int64_t actualCalcS1Token, actualCalcS2Token;
        CalcleActualToken(fBaseParams, i, actualCalcS1Token, actualCalcS2Token);
        p = CeilDivideBy(actualCalcS1Token, fBaseParams.s1Inner * fBaseParams.s1CvRatio) + 1;
        q = CeilDivideBy(actualCalcS2Token, fBaseParams.s2Inner * fBaseParams.s2CvRatio) + 1;
        p = p > m ? m : p;
        q = q > n ? n : q;

        // 负数场景变换
        if (p < 0) {
            n = n + p;
            q = p + q;
            p = 1;
        } else if (q < 0) {
            m = m + q;
            p = p + q;
            q = 1;
        }
        if (p + q <= m) {
            int64_t L1{q - 1}, L2{std::min(n - q + 1, m + NUM_TWO - p - q)},
                L3{std::max(static_cast<int64_t>(0), std::min(p + n - m - 1, p + q - NUM_TWO))};
            mNew = L3 == 0 ? p + q + L2 - NUM_TWO : m;
            nNew = L1 + L2 + L3;
            mnMax = n <= m || fBaseParams.g == 1 ? std::max({mnMax, (p + q - 1) * fBaseParams.g}) :
                                                   std::max({mnMax, mNew * fBaseParams.g, p + q - 1});
            deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() + std::min(mNew, nNew) * (p + q - 1));
        } else {
            mNew = m;
            nNew = std::min(m - 1 + q, n);
            if (p + q <= n) {
                mnMax = std::max({mnMax, mNew * fBaseParams.g, p + q - 1});
                deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() + mNew * (p + q - 1));
            } else {
                mnMax = std::max({mnMax, mNew * fBaseParams.g, nNew});
                deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() + mNew * nNew);
            }
        }

        // 注意，这里更新后的mNewList和nNewList是可以替代list_m和list_n的，它们的引入是避免空白行/列
        deterPrefixData.mNewList.push_back(mNew);
        deterPrefixData.nNewList.push_back(nNew);
        deterPrefixData.pNewList.push_back(p);
        deterPrefixData.qNewList.push_back(q);
        if (N11 > 0 && fBaseParams.g == 1) {
            int64_t R0 =
                deterPrefixData.prefix0.back() +
                (mNew * nNew - (mNew - p) * (mNew - p + 1) / NUM_TWO - (nNew - q) * (nNew - q + 1) / NUM_TWO) * N11;
            deterPrefixData.prefix0.push_back(R0);
        }
    }
}

void Scheduler::CalcleTNDBandBns2DeterParam(DeterPrefixData& deterPrefixData)
{
    if (fBaseParams.splitAxis != SplitAxisEnum::BN2S2) {
        return;
    }

    // 最多允许coreNum列分给不同的核
    if (!SupportTNDBns2(deterPrefixData, deterPrefixData.prefix1.back())) {
        fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
        fallbackReasons |= static_cast<int64_t>(Fallback::SUPPORT_BN2S2);
        FAG_PLAN_TRACE("CalcleTNDBandBns2DeterParam", "Not support BNS2, change to BN2GS1S2.");
        return;
    }

    // BNS2分核按顺序分核，存在前后两核收尾分同一列的情况，计算可能分开的列
    std::vector<std::pair<uint64_t, uint64_t>> syncRounds, syncRoundRanges;
    std::fill(std::begin(fBaseParams.separateDkOffset), std::end(fBaseParams.separateDkOffset),
              static_cast<int64_t>(-1));
    std::fill(std::begin(fBaseParams.startNeedSyncRound), std::end(fBaseParams.startNeedSyncRound),
              static_cast<uint64_t>(0));
    std::fill(std::begin(fBaseParams.endNeedSyncRound), std::end(fBaseParams.endNeedSyncRound),
              static_cast<uint64_t>(0));

    CalcleTNDBandDeterSplitDkOffset(deterPrefixData, syncRounds, syncRoundRanges);
    std::copy(std::begin(fBaseParams.separateDkOffset), std::end(fBaseParams.separateDkOffset),
              std::begin(fBaseParams.deterPrefix2));

    CalcleTNDDeterSyncRounds(syncRounds, syncRoundRanges);
}

void Scheduler::CalcleTNDDenseDeterSplitDkOffset(DeterPrefixData& deterPrefixData,
                                                 std::vector<std::pair<uint64_t, uint64_t>>& syncRounds,
                                                 std::vector<std::pair<uint64_t, uint64_t>>& syncRoundRanges)
{
    int64_t precoreLastBatchStartRound = 0;
    for (uint32_t coreId = 0; coreId < CORE_LIST_NUM; coreId++) {
        if (fBaseParams.deterMaxRound == 0 || coreId > fBaseParams.aicNum - 1) {
            continue;
        }
        const int64_t* kvValue = kvEnds.data();

        TndBandDeterRoundInfo tndBandDeterRoundInfo;
        for (int64_t round = fBaseParams.deterMaxRound; round > 0; round--) {
            auto oriCoordinateInfo = CalTNDDenseIndex<static_cast<uint32_t>(DeterSparseType::DETER_DENSE)>(
                deterPrefixData, coreId + 1, round, fBaseParams.n1);
            int64_t w, x, y;
            std::tie(w, x, y) = oriCoordinateInfo;

            int64_t batchId = CeilDivideBy(w, fBaseParams.n1);
            if (w == -1 || batchId > fBaseParams.b) {
                continue;
            }
            int64_t m = deterPrefixData.mNewList[batchId - 1];

            if (fBaseParams.separateDkOffset[coreId] == -1 && x < m && round == fBaseParams.deterMaxRound) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
            SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
        }
        if (coreId != 0) {
            uint64_t startSyncRound = precoreLastBatchStartRound;
            uint64_t endSyncRound = tndBandDeterRoundInfo.coreFirstBatchLastRound;
            if (startSyncRound > endSyncRound) {
                syncRounds.push_back(std::make_pair(startSyncRound, endSyncRound));
            } else {
                syncRoundRanges.push_back(std::make_pair(startSyncRound, endSyncRound));
            }
        }
        if (coreId == 0) {
            precoreLastBatchStartRound = 1; // 如果在0-1核上涉及切BN轴，那么0核的起点就是第一个roundid = 1
        } else {
            precoreLastBatchStartRound = tndBandDeterRoundInfo.coreLastBatchStartRound;
        }
    }
}

bool Scheduler::SupportTNDBns2(DeterPrefixData& deterPrefixData, int64_t round)
{
    for (int64_t b = 0; b < fBaseParams.b; b++) {
        int64_t m = deterPrefixData.mNewList[b];
        int64_t n = deterPrefixData.nNewList[b];
        // Dense empty batches contain no split columns. Keep the existing
        // sparse/no-remainder guard; do not change sparse admission here.
        if ((fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_DENSE) &&
             (m == 0 || n == 0)) ||
            (m == 0 && round == 0)) {
            continue;
        }
        // Zero-work BAND batches have no columns to split. Keep their physical
        // cu offsets and prefixes, but do not reject BN2S2 for their empty area.
        if (fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND) &&
            (fBaseParams.actualSeqQlen[b] == 0 || fBaseParams.actualSeqKvlen[b] == 0)) {
            continue;
        }
        if ((round / Gcd(m, round)) >= n) {
            continue;
        }
        return false;
    }
    return true;
}

void Scheduler::CalcleTNDBandDeterSplitDkOffset(DeterPrefixData& deterPrefixData,
                                                std::vector<std::pair<uint64_t, uint64_t>>& syncRounds,
                                                std::vector<std::pair<uint64_t, uint64_t>>& syncRoundRanges)
{
    int64_t precoreLastBatchStartRound = 0;
    int64_t N11 = fBaseParams.n1 / fBaseParams.aicNum;
    int64_t N12 = fBaseParams.n1 % fBaseParams.aicNum;
    for (uint32_t coreId = 0; coreId < CORE_LIST_NUM; coreId++) {
        if (deterPrefixData.prefix1.back() == 0 || coreId >= fBaseParams.aicNum - 1) {
            continue;
        }
        TndBandDeterRoundInfo tndBandDeterRoundInfo;
        for (uint64_t round = deterPrefixData.prefix1.back(); round > 0; round--) {
            auto oriCoordinateInfo = CalTNDDenseIndex<static_cast<uint32_t>(DeterSparseType::DETER_BAND)>(
                deterPrefixData, coreId + 1, round, fBaseParams.n1 % static_cast<int64_t>(fBaseParams.aicNum));
            int64_t w, x, y;
            std::tie(w, x, y) = oriCoordinateInfo;
            int64_t batchId = CeilDivideBy(w, N12);
            if (w == -1 || batchId > fBaseParams.b) {
                continue;
            }

            int64_t m = deterPrefixData.mNewList[batchId - 1];
            int64_t n = deterPrefixData.nNewList[batchId - 1];
            int64_t p = deterPrefixData.pNewList[batchId - 1];
            int64_t q = deterPrefixData.qNewList[batchId - 1];

            int64_t wTail = w % N12;
            wTail = wTail != 0 ? wTail : N12;
            w = (CeilDivideBy(w, N12) - 1) * fBaseParams.n1 + N11 * fBaseParams.aicNum + wTail;

            std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> coordinateInfo =
                std::make_tuple(m, n, p, q, x, y, w, coreId, round);
            UpdateSeparateDkOffset(coordinateInfo, tndBandDeterRoundInfo);
        }

        if (coreId != 0) {
            uint64_t startSyncRound = precoreLastBatchStartRound + deterPrefixData.prefix0.back();
            uint64_t endSyncRound = tndBandDeterRoundInfo.coreFirstBatchLastRound + deterPrefixData.prefix0.back();
            if (startSyncRound > endSyncRound) {
                syncRounds.push_back(std::make_pair(startSyncRound, endSyncRound));
            } else {
                syncRoundRanges.push_back(std::make_pair(startSyncRound, endSyncRound));
            }
        }
        precoreLastBatchStartRound = tndBandDeterRoundInfo.coreLastBatchStartRound;
    }
}

void Scheduler::UpdateSeparateDkOffsetLargeM(
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo,
    TndBandDeterRoundInfo& tndBandDeterRoundInfo)
{
    const int64_t* kvValue = kvEnds.data();
    int64_t m, n, p, q, x, y, w, coreId, round;
    std::tie(m, n, p, q, x, y, w, coreId, round) = coordinateInfo;
    if (n >= m) {
        if (y - q + 1 <= x && x <= p + y - 1) {
            if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
            SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
        } else {
            y = AbsCeil((x - (p + y - 1)), (p + q - 1)) * (p + q - 1) + y;
            bool isValid = 1 <= y && y <= n;
            if (fBaseParams.separateDkOffset[coreId] == -1 && isValid && IsSeparateS2(coordinateInfo)) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
            if (isValid) {
                SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
            }
        }
    } else {
        // 优先处理拼出来后有空白格的情况
        if (q + 1 <= y && y <= std::min(n, p + NUM_TWO * q - NUM_TWO)) {
            if (x != p + q - 1 && fBaseParams.separateDkOffset[coreId] == -1) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
        } else {
            if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
        }
        SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
    }
}

void Scheduler::UpdateSeparateDkOffsetSmallM(
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo,
    TndBandDeterRoundInfo& tndBandDeterRoundInfo)
{
    const int64_t* kvValue = kvEnds.data();
    int64_t m, n, p, q, x, y, w, coreId, round;
    std::tie(m, n, p, q, x, y, w, coreId, round) = coordinateInfo;
    if (p + q <= n) {
        if (x - p + 1 <= y && y <= x + q - 1) {
            if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
            SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
        } else if (y < x - p + 1 && 1 <= y + p + q - 1 && y + p + q - 1 <= n) {
            y = y + p + q - 1;
            if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
            SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
        }
    } else {
        if (x - p + 1 <= y && y <= x + q - 1) {
            if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
            SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
        }
    }
}

void Scheduler::UpdateSeparateDkOffset(
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo,
    TndBandDeterRoundInfo& tndBandDeterRoundInfo)
{
    int64_t m, p, q;
    std::tie(m, std::ignore, p, q, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) = coordinateInfo;
    if (p + q <= m) {
        UpdateSeparateDkOffsetLargeM(coordinateInfo, tndBandDeterRoundInfo);
    } else {
        UpdateSeparateDkOffsetSmallM(coordinateInfo, tndBandDeterRoundInfo);
    }
}

bool Scheduler::IsSeparateS2(
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo)
{
    int64_t m, n, p, q, x, y;
    std::tie(m, n, p, q, x, y, std::ignore, std::ignore, std::ignore) = coordinateInfo;
    // 补充的 is_max 运算
    bool isSeparate = true;

    if (p + q <= m && m > n) {
        // 这里要注意None的情况
        if (y <= q) {
            if (x - y >= p - 1) {
                isSeparate = false;
            }
            // 有空白块的情况交给上面处理了，因为有可能R_max坐标是None
        } else if (y > std::min(n, p + NUM_TWO * q - NUM_TWO)) {
            x = AbsCeil((y - (q + x - 1)), (p + q - 1)) * (p + q - 1) + x;
            if (x == m || x < y - q + 1) {
                isSeparate = false;
            }
        }
    } else if (p + q <= m && n >= m) {
        if (y <= std::min(n, m + 1 - p)) {
            if (x - y == p - 1) {
                isSeparate = false;
            }
        } else {
            if (x == m) {
                isSeparate = false;
            }
        }
    } else {
        if (y <= m - p) {
            if (x - y == p - 1) {
                isSeparate = false;
            }
        } else {
            if (x == m) {
                isSeparate = false;
            }
        }
    }
    return isSeparate;
}

int64_t Scheduler::GetKeyOffset(const int64_t* kvValue, int64_t w, int64_t y)
{
    int64_t batchId = CeilDivideBy(w, fBaseParams.n2) - 1;
    int64_t n2Idx = w - batchId * fBaseParams.n2 - 1;
    int64_t s2Idx = y - 1;

    if (fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND)) {
        // CalcleTNDBandDeterPrefix removes leading invalid KV columns when
        // p < 0. Its y stays in that reduced coordinate system, whereas
        // the kernel restores the column origin before forming keyOffset.
        // Use the same original p (including signed CeilDivideBy) so that
        // shared columns match SpecialS2Index instead of writing as owners.
        int64_t actualS1Token, actualS2Token;
        CalcleActualToken(fBaseParams, batchId, actualS1Token, actualS2Token);
        const int64_t cubeM = fBaseParams.s1Inner * fBaseParams.s1CvRatio;
        const int64_t m = CeilDivideBy(fBaseParams.actualSeqQlen[batchId], cubeM);
        const int64_t p = std::min(CeilDivideBy(actualS1Token, cubeM) + 1, m);
        s2Idx += std::max(-p, static_cast<int64_t>(0));
    }

    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    int64_t seqKvLenPrefix = batchId == 0 ? 0 : kvValue[batchId - 1];
    bOffset = seqKvLenPrefix * fBaseParams.n2 * fBaseParams.d;
    n2Offset = n2Idx * fBaseParams.d;
    s2Offset = s2Idx * fBaseParams.s2Inner * fBaseParams.n2 * fBaseParams.d;
    return bOffset + n2Offset + s2Offset;
}

void Scheduler::CalcleTNDDeterSyncRounds(std::vector<std::pair<uint64_t, uint64_t>>& syncRounds,
                                         std::vector<std::pair<uint64_t, uint64_t>>& syncRoundRanges)
{
    if (syncRounds.size() + syncRoundRanges.size() > CORE_LIST_NUM) {
        fBaseParams.startNeedSyncRound[0] = 1;
        fBaseParams.endNeedSyncRound[0] = std::numeric_limits<uint64_t>::max();
        FAG_PLAN_TRACE("CalcleTNDBandDeterParam", "All rounds need sync!");
        return;
    }
    std::vector<uint64_t> needSyncRoundsTmp = CalculateSyncRound(syncRounds);
    std::vector<uint64_t> needSyncRounds;
    for (uint64_t needSyncRound : needSyncRoundsTmp) {
        bool isValid = true;
        for (auto needSyncRoundRange : syncRoundRanges) {
            if (needSyncRound >= needSyncRoundRange.first && needSyncRound <= needSyncRoundRange.second) {
                isValid = false;
                break;
            }
        }
        if (isValid) {
            needSyncRounds.push_back(needSyncRound);
        }
    }

    for (uint32_t i = 0; i < needSyncRounds.size(); i++) {
        fBaseParams.startNeedSyncRound[i] = needSyncRounds[i];
        fBaseParams.endNeedSyncRound[i] = needSyncRounds[i];
    }
    for (uint32_t i = 0; i < syncRoundRanges.size(); i++) {
        auto syncRoundPair = syncRoundRanges[i];
        fBaseParams.startNeedSyncRound[i + needSyncRounds.size()] = syncRoundPair.first;
        fBaseParams.endNeedSyncRound[i + needSyncRounds.size()] = syncRoundPair.second;
    }

    uint64_t allNeedSyncLoopNums = 0;
    for (uint64_t loopIdx = 0; static_cast<int64_t>(loopIdx) < fBaseParams.deterMaxRound; loopIdx++) {
        for (uint32_t i = 0; i < CORE_LIST_NUM; i++) {
            if (fBaseParams.endNeedSyncRound[i] == 0) {
                break;
            }
            if (loopIdx >= fBaseParams.startNeedSyncRound[i] && loopIdx <= fBaseParams.endNeedSyncRound[i]) {
                allNeedSyncLoopNums++;
                break;
            }
        }
    }
    if (fBaseParams.deterMaxRound < static_cast<int64_t>(allNeedSyncLoopNums) * NUM_TWO) {
        fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
        fallbackReasons |= static_cast<int64_t>(Fallback::SYNC_COST);
    }
}

std::vector<uint64_t> Scheduler::CalculateSyncRound(std::vector<std::pair<uint64_t, uint64_t>> syncRounds)
{
    if (syncRounds.size() == 0) {
        return {};
    }
    // Each pair is (next writer's first round, previous writer's last round),
    // with first > second. The kernel barrier executes AFTER that round's FIX.
    // Therefore a cut must satisfy second <= cut < first; using first lets the
    // next writer race the preceding core's final atomic dQ update.
    // A (1, 0) pair has no preceding work. Keep its old startup barrier at 1:
    // zero is the serialized end-of-list sentinel, not a usable barrier entry.
    if (syncRounds.size() == 1) {
        return {std::max(syncRounds[0].first - 1, uint64_t{1})};
    }
    uint64_t minStartSyncRound = std::max(syncRounds[0].first - 1, uint64_t{1});
    for (auto syncRound : syncRounds) {
        const uint64_t latestSafeCut = std::max(syncRound.first - 1, uint64_t{1});
        minStartSyncRound = std::min(latestSafeCut, minStartSyncRound);
    }

    std::vector<std::pair<uint64_t, uint64_t>> smallSyncRounds;
    for (auto syncRound : syncRounds) {
        if (syncRound.second > minStartSyncRound) {
            smallSyncRounds.push_back(syncRound);
        }
    }
    std::vector<uint64_t> needSyncRounds = CalculateSyncRound(smallSyncRounds);
    needSyncRounds.push_back(minStartSyncRound);
    return needSyncRounds;
}

void Scheduler::SetCoreRoundInfo(TndBandDeterRoundInfo& tndBandDeterRoundInfo, uint64_t round, int64_t batchId)
{
    if (batchId != tndBandDeterRoundInfo.lastBatchId && tndBandDeterRoundInfo.lastBatchId != 0 &&
        tndBandDeterRoundInfo.coreLastBatchStartRound == 0) {
        tndBandDeterRoundInfo.coreLastBatchStartRound = tndBandDeterRoundInfo.lastValidRound;
    }
    if (batchId != tndBandDeterRoundInfo.lastBatchId) {
        tndBandDeterRoundInfo.coreFirstBatchLastRound = round;
    }
    tndBandDeterRoundInfo.lastValidRound = round;
    tndBandDeterRoundInfo.lastBatchId = batchId;
}

void Scheduler::CalcTNDSwizzleParam()
{
    if (fBaseParams.b >= TND_SWIZZLE_PREFIX_NUM) {
        return;
    }
    for (int64_t bIdx = 0; bIdx < fBaseParams.b; bIdx++) {
        int64_t s1OuterTmp = (fBaseParams.actualSeqQlen[bIdx] + fBaseParams.s1Inner * S1CV_RATIO_DEFAULT - 1) /
                             (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT);
        int64_t s2OuterTmp = (fBaseParams.actualSeqKvlen[bIdx] + fBaseParams.s2Inner * S2CV_RATIO_DEFAULT - 1) /
                             (fBaseParams.s2Inner * S2CV_RATIO_DEFAULT);
        tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1] =
            tndBaseInfo.tndS2BlockPrefixSum[bIdx] +
            CeilDivideBy(fBaseParams.n2 * fBaseParams.g * s2OuterTmp, static_cast<int64_t>(fBaseParams.aicNum)) *
                s1OuterTmp;
        FAG_PLAN_TRACE("CalcTNDSwizzleParam", " bIdx = %ld: tndS2BlockPrefixSum = %ld", bIdx + 1,
                       tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1]);
    }
}

bool Scheduler::GetBlockInfoOfBNS4TND()
{
    std::vector<std::vector<int64_t>> totalBlockInfo(fBaseParams.b, std::vector<int64_t>(TOTAL_BLOCK_DIMENSION));
    std::vector<std::vector<float>> acturalBlockInfo(fBaseParams.b + NUM_TWO, std::vector<float>(fBaseParams.s2Outer));
    std::vector<std::vector<uint64_t>> validBlockCount(fBaseParams.b, std::vector<uint64_t>(fBaseParams.s2Outer));
    FillBlockInfoLoadBalance(totalBlockInfo, acturalBlockInfo);
    FillColumnValidBlockCount(validBlockCount);

    float maxBlockNumPerCore = BinarySearchMaxBlockNumPerCore(acturalBlockInfo);

    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];

    if (!CaclePerCoreBlockInfo(totalBlockInfo, acturalBlockInfo, validBlockCount, maxBlockNumPerCore, blockStarts,
                               blockEnds)) {
        return false;
    }

    for (int64_t c = 0; c < fBaseParams.blockOuter; c++) {
        FAG_PLAN_TRACE("GetBlockInfoOfBNS4TND", "blockNum[%ld], blockStarts = %ld , blockEnds = %ld", c, blockStarts[c],
                       blockEnds[c]);
    }

    for (uint32_t c = fBaseParams.blockOuter; c < CORE_LIST_NUM; c++) {
        blockStarts[c] = 0;
        blockEnds[c] = 0;
    }
    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));
    fBaseParams.maxValidBBLen = std::ceil(maxBlockNumPerCore);
    return true;
}

float Scheduler::BinarySearchMaxBlockNumPerCore(const std::vector<std::vector<float>>& acturalBlockInfo)
{
    float left = acturalBlockInfo[fBaseParams.b + 1][0];
    float right = acturalBlockInfo[fBaseParams.b][0];
    float mid = 0;
    while (left < right - 1) {
        mid = (left + right) / NUM_TWO;
        if (IsPossible(acturalBlockInfo, mid)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return right;
}

void Scheduler::FillColumnValidBlockCount(std::vector<std::vector<uint64_t>>& validBlockCount)
{
    for (int64_t b = 0; b < fBaseParams.b; b++) {
        int64_t actualS1Outer = (fBaseParams.actualSeqQlen[b] + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[b];
        for (int64_t s2oIdx = 0; s2oIdx < fBaseParams.s2Outer; s2oIdx++) {
            int64_t s2IdxLeft = fBaseParams.cvS2Inner * s2oIdx;
            if (s2IdxLeft >= actualS2Len) {
                continue;
            }
            int64_t s2IdxRight = std::min(s2IdxLeft + fBaseParams.cvS2Inner, actualS2Len);
            for (int64_t s1oIdx = 0; s1oIdx < actualS1Outer; s1oIdx++) {
                if (fBaseParams.attenMaskOptional == EMPTY_TENSOR ||
                    CheckUnpadSparseLeftAndRight(s1oIdx, s2IdxLeft, s2IdxRight, b)) {
                    validBlockCount[b][s2oIdx]++;
                }
            }
        }
    }
}

bool Scheduler::CaclePerCoreBlockInfo(const std::vector<std::vector<int64_t>>& totalBlockInfo,
                                      const std::vector<std::vector<float>>& acturalBlockInfo,
                                      const std::vector<std::vector<uint64_t>>& validBlockCount,
                                      const float maxBlockNumPerCore, int64_t (&blockStarts)[CORE_LIST_NUM],
                                      int64_t (&blockEnds)[CORE_LIST_NUM])
{
    float currentSum = 0;
    uint64_t currentValidBlockCount = 0;
    uint64_t maxValidBlockCount = 0;
    int64_t coreIdx = 0;
    uint64_t tndS1S2PrefixSumTmp = 0;
    uint64_t tndS1S2AlignPrefixSumTmp = 0;
    uint64_t tndPrefixSumTmp = 0;
    bool isSetSwizzleParam = fBaseParams.b < TND_SWIZZLE_PREFIX_NUM;
    for (int64_t b = 0; b < fBaseParams.b; b++) {
        for (int64_t n = 0; n < fBaseParams.n2 * fBaseParams.g; n++) {
            int64_t actualS1Outer = (fBaseParams.actualSeqQlen[b] + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
            for (int64_t j = 0; j < fBaseParams.s2Outer; j++) {
                float num = acturalBlockInfo[b][j];
                uint64_t blockCount = validBlockCount[b][j];
                if (coreIdx >= CORE_LIST_NUM) {
                    FAG_PLAN_TRACE("GetBlockInfoOfBNS4TND", " Not support BN2S2.");
                    return false;
                } else if (currentSum + num > maxBlockNumPerCore) {
                    FAG_PLAN_TRACE("GetBlockInfoOfBNS4TND", " blockIdx = %ld: actualBlock = %f", coreIdx, currentSum);
                    int64_t preBatchBlockNum = b == 0 ? 0 : totalBlockInfo[b - 1][1];
                    int64_t preNGBlockNum = n * totalBlockInfo[b][0];
                    int64_t preS2BlockNum = j * actualS1Outer;
                    blockEnds[coreIdx] = preBatchBlockNum + preNGBlockNum + preS2BlockNum;
                    blockStarts[coreIdx + 1] = blockEnds[coreIdx];
                    maxValidBlockCount = std::max(maxValidBlockCount, currentValidBlockCount);
                    coreIdx += 1;
                    currentSum = num;
                    currentValidBlockCount = blockCount;
                    tndBaseInfo.tndStartBIdx[coreIdx] = b;
                    tndBaseInfo.tndS1S2PrefixSum[coreIdx] = tndS1S2PrefixSumTmp;
                    tndBaseInfo.tndS1S2AlignPrefixSum[coreIdx] = tndS1S2AlignPrefixSumTmp;
                    tndBaseInfo.tndPrefixSum[coreIdx] = tndPrefixSumTmp;
                } else {
                    currentSum += num;
                    currentValidBlockCount += blockCount;
                }
            }
        }
        tndS1S2PrefixSumTmp += (fBaseParams.actualSeqQlen[b] * fBaseParams.actualSeqKvlen[b]);
        tndS1S2AlignPrefixSumTmp +=
            (fBaseParams.actualSeqQlen[b] *
             AlignTo(fBaseParams.actualSeqKvlen[b], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));
        int64_t s1OuterTmp = (fBaseParams.actualSeqQlen[b] + fBaseParams.s1Inner * S1CV_RATIO_DEFAULT - 1) /
                             (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT);
        int64_t s2OuterTmp = (fBaseParams.actualSeqKvlen[b] + fBaseParams.s2Inner * S2CV_RATIO_DEFAULT - 1) /
                             (fBaseParams.s2Inner * S2CV_RATIO_DEFAULT);
        tndPrefixSumTmp += (s1OuterTmp * s2OuterTmp);
        if (isSetSwizzleParam) {
            SetTndSwizzleParam(b, s1OuterTmp, s2OuterTmp);
        }
    }
    FAG_PLAN_TRACE("GetBlockInfoOfBNS4TND", " blockIdx = %ld: actualBlock = %f", coreIdx, currentSum);
    blockStarts[0] = 0;
    blockEnds[coreIdx] = totalBlockInfo[fBaseParams.b - 1][1];
    maxValidBlockCount = std::max(maxValidBlockCount, currentValidBlockCount);
    fBaseParams.blockOuter = coreIdx + 1;
    tndBaseInfo.normalMaxValidBlockCount = maxValidBlockCount;
    return true;
}

void Scheduler::SetTndSwizzleParam(int64_t bIdx, int64_t s1OuterTmp, int64_t s2OuterTmp)
{
    int64_t realS1OuterTmp = s1OuterTmp;
    int64_t realS2OuterTmp = s2OuterTmp;
    // when enable tnd swizzle, fill tnd swizzle info
    if (fBaseParams.sparseType == static_cast<uint8_t>(SparseType::DENSE)) {
        tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1] =
            tndBaseInfo.tndS2BlockPrefixSum[bIdx] +
            CeilDivideBy(s2OuterTmp * fBaseParams.n2 * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)) *
                s1OuterTmp;
    } else if (fBaseParams.sparseType == static_cast<uint8_t>(SparseType::CASUAL)) {
        // 处理无效列和无效行场景
        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) && s1OuterTmp < s2OuterTmp) {
            realS2OuterTmp = s1OuterTmp;
        } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) &&
                   s1OuterTmp > s2OuterTmp) {
            realS1OuterTmp = s2OuterTmp;
        }
        int64_t halfN2g = (fBaseParams.n2 * fBaseParams.g) >> 1;
        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
            int64_t n1 = (realS2OuterTmp << 1) - realS1OuterTmp + 1;
            tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1] =
                tndBaseInfo.tndS2BlockPrefixSum[bIdx] +
                CeilDivideBy(n1 * halfN2g, static_cast<int64_t>(fBaseParams.aicNum)) * realS1OuterTmp;
        } else {
            int64_t m1 = (realS1OuterTmp << 1) - realS2OuterTmp + 1;
            tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1] =
                tndBaseInfo.tndS2BlockPrefixSum[bIdx] +
                CeilDivideBy(realS2OuterTmp * halfN2g, static_cast<int64_t>(fBaseParams.aicNum)) * m1;
        }
    } else if (fBaseParams.sparseType == static_cast<uint8_t>(SparseType::BAND)) {
        int64_t actualCalcS1Token, actualCalcS2Token;
        CalcleActualToken(fBaseParams, bIdx, actualCalcS1Token, actualCalcS2Token);
        int64_t p = CeilDivideBy(actualCalcS1Token, fBaseParams.s1Inner * fBaseParams.s1CvRatio) + 1;
        int64_t q = CeilDivideBy(actualCalcS2Token, fBaseParams.s2Inner * fBaseParams.s2CvRatio) + 1;
        p = p > s1OuterTmp ? s1OuterTmp : p;
        q = q > s2OuterTmp ? s2OuterTmp : q;
        if (p + q <= s1OuterTmp) {
            tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1] =
                tndBaseInfo.tndS2BlockPrefixSum[bIdx] +
                CeilDivideBy(s2OuterTmp * fBaseParams.n2 * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)) *
                    (p + q - 1);
        } else {
            tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1] =
                tndBaseInfo.tndS2BlockPrefixSum[bIdx] +
                CeilDivideBy(s2OuterTmp * fBaseParams.n2 * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)) *
                    s1OuterTmp;
        }
    }
    tndBaseInfo.tndSwizzleS1S2PrefixSum[bIdx + 1] =
        tndBaseInfo.tndSwizzleS1S2PrefixSum[bIdx] +
        (fBaseParams.actualSeqQlen[bIdx] * fBaseParams.actualSeqKvlen[bIdx]);
    tndBaseInfo.tndSwizzleS1S2AlignPrefixSum[bIdx + 1] =
        tndBaseInfo.tndSwizzleS1S2AlignPrefixSum[bIdx] +
        (fBaseParams.actualSeqQlen[bIdx] *
         AlignTo(fBaseParams.actualSeqKvlen[bIdx], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));
    FAG_PLAN_TRACE("GetBlockInfoOfBNS4TND", " bIdx = %ld: tndS2BlockPrefixSum = %ld, tndSwizzleS1S2PrefixSum = %d",
                   bIdx + 1, tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1], tndBaseInfo.tndSwizzleS1S2PrefixSum[bIdx + 1]);
}

bool Scheduler::IsPossible(const std::vector<std::vector<float>>& acturalBlockInfo, const float possibleMax)
{
    float currentSum = 0;
    uint64_t needCoreNum = 1;
    int64_t n2g = fBaseParams.n2 * fBaseParams.g;
    int64_t bn2g = fBaseParams.b * n2g;
    if (fBaseParams.isBn2MultiBlk) {
        for (int64_t i = 0; i < bn2g; i++) {
            int64_t b = i / n2g;
            float blkNumPerBN = acturalBlockInfo[b][fBaseParams.s2Outer];
            if (currentSum + blkNumPerBN > possibleMax) {
                needCoreNum += 1;
                currentSum = blkNumPerBN;
            } else {
                currentSum += blkNumPerBN;
            }
            if (needCoreNum > fBaseParams.aicNum) {
                return false;
            }
        }
    } else {
        for (int64_t i = 0; i < bn2g; i++) {
            int64_t b = i / n2g;
            for (int64_t j = 0; j < fBaseParams.s2Outer; j++) {
                float num = acturalBlockInfo[b][j];
                if (currentSum + num > possibleMax) {
                    needCoreNum += 1;
                    currentSum = num;
                } else {
                    currentSum += num;
                }
                if (needCoreNum > fBaseParams.aicNum) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool Scheduler::CheckUnpadSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft, int64_t s2IdxRight, int64_t bIdx)
{
    int64_t actualS1Len = fBaseParams.actualSeqQlen[bIdx];
    int64_t actualS2Len = fBaseParams.actualSeqKvlen[bIdx];
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
        int64_t s2IgnoredEndLen =
            actualS1Len - static_cast<int64_t>(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (s1oDimIdx + 1));
        int64_t s2EndLen = 0;
        if (actualS2Len > s2IgnoredEndLen) {
            s2EndLen = actualS2Len - s2IgnoredEndLen;
        }
        s2EndLen = std::min(std::max(s2EndLen, static_cast<int64_t>(fBaseParams.prefixN[bIdx])), actualS2Len);
        bool isValid = s2IdxLeft < s2EndLen;
        return isValid;
    }
    int64_t actualCalcS1Token = fBaseParams.s1Token;
    int64_t actualCalcS2Token = fBaseParams.s2Token;
    // sparse_mode == band 或者 RIGHT_DOWN_CASUAL时，token以右下角为基本，需要校正
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) &&
        bIdx != fBaseParams.bandIdx) {
        actualCalcS1Token = static_cast<int64_t>(INT32_MAX) + actualS1Len - actualS2Len;
        actualCalcS2Token = static_cast<int64_t>(0) - actualS1Len + actualS2Len;
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL) &&
               bIdx != fBaseParams.bandIdx) {
        actualCalcS1Token = INT32_MAX;
        actualCalcS2Token = 0;
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
               fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) ||
               (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) &&
                bIdx == fBaseParams.bandIdx) ||
               (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL) &&
                bIdx == fBaseParams.bandIdx)) {
        actualCalcS1Token = fBaseParams.s1Token + actualS1Len - actualS2Len;
        actualCalcS2Token = fBaseParams.s2Token - actualS1Len + actualS2Len;
    }
    int64_t s2SparseLeft =
        std::max(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * s1oDimIdx - actualCalcS1Token, static_cast<int64_t>(0));
    s2SparseLeft = AlignTo(s2SparseLeft, ALIGN64);
    int64_t s2SparseRight = AlignTo(
        std::min(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (s1oDimIdx + 1), fBaseParams.s1) + actualCalcS2Token,
        static_cast<int64_t>(64));
    s2SparseRight = std::min(s2SparseRight, actualS2Len);
    bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
    return isValid;
}

bool Scheduler::IsValidUnpad(int64_t blockIdx)
{
    int64_t resbaseIdx = blockIdx;
    for (int64_t bIdx = 0; bIdx < fBaseParams.b; bIdx++) {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[bIdx];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[bIdx];
        int64_t s1OuterTmp =
            (actualS1Len + fBaseParams.s1Inner * S1CV_RATIO_DEFAULT - 1) / (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT);
        int64_t s2OuterTmp =
            (actualS2Len + fBaseParams.s2Inner * S2CV_RATIO_DEFAULT - 1) / (fBaseParams.s2Inner * S2CV_RATIO_DEFAULT);
        int64_t totalBaseIdx = fBaseParams.n2 * fBaseParams.g * s1OuterTmp * s2OuterTmp;
        if (resbaseIdx < totalBaseIdx) {
            int64_t gDimTail = resbaseIdx % (s1OuterTmp * s2OuterTmp);
            int64_t s2oDimIdx = gDimTail / s1OuterTmp;
            int64_t s1oDimIdx = gDimTail % s1OuterTmp;
            int64_t s2IdxLeft = s2oDimIdx * fBaseParams.s2Inner * S2CV_RATIO_DEFAULT;
            int64_t s2IdxRight = std::min((s2oDimIdx + 1) * fBaseParams.s2Inner * S2CV_RATIO_DEFAULT, actualS2Len);
            if (fBaseParams.attenMaskOptional != EMPTY_TENSOR) {
                return CheckSparseLeftAndRight(s1oDimIdx, s2IdxLeft, s2IdxRight, bIdx);
            } else {
                return true;
            }
        } else {
            resbaseIdx -= totalBaseIdx;
        }
    }
    return false;
}

ge::graphStatus Scheduler::GetSparseUnpadBlockInfo()
{
    std::vector<std::vector<std::vector<int64_t>>> calculatedBlockInfo(
        fBaseParams.b,
        std::vector<std::vector<int64_t>>(fBaseParams.s2Outer, std::vector<int64_t>(CALCULATED_BLOCK_DIMENSION)));
    std::vector<std::vector<int64_t>> totalBlockInfo(fBaseParams.b, std::vector<int64_t>(TOTAL_BLOCK_DIMENSION));
    FillBlockInfo(calculatedBlockInfo, totalBlockInfo);

    // block split
    int64_t fusedOuter = calculatedBlockInfo[fBaseParams.b - 1][0][SUM_ALL];
    if (fusedOuter == 0) {
        fBaseParams.blockOuter = 0;
        fBaseParams.blockFactor = 0;
        fBaseParams.maxValidBBLen = 0;
        std::fill(std::begin(fBaseParams.blockStarts), std::end(fBaseParams.blockStarts), 0);
        std::fill(std::begin(fBaseParams.blockEnds), std::end(fBaseParams.blockEnds), 0);
        return ge::GRAPH_SUCCESS;
    }
    int64_t blockFactor = (fusedOuter + fBaseParams.aicNum - 1) / fBaseParams.aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

    FAG_PLAN_TRACE("GetSparseUnpadBlockInfo", " fusedOuter = %ld: blockFactor = %ld, blockOuter = %ld", fusedOuter,
                   blockFactor, blockOuter);
    fBaseParams.blockOuter = blockOuter;
    fBaseParams.blockFactor = blockFactor;
    fBaseParams.maxValidBBLen = blockFactor;
    int64_t bIdx = 0;
    int64_t bTail = 0;
    int64_t n2Idx = 0;
    int64_t n2Tail = 0;
    int64_t gIdx = 0;
    int64_t gTail = 0;
    int64_t s1oIdx = 0;
    int64_t s2oIdx = 0;
    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    blockStarts[0] = 0;
    blockEnds[blockOuter - 1] = totalBlockInfo[fBaseParams.b - 1][1];
    int64_t s1OuterTmp = 0;
    FAG_PLAN_TRACE("GetSparseUnpadBlockInfo", "Load balancing calculation results in TND scenario:");
    for (int64_t c = 1; c < blockOuter; c++) {
        int64_t currentIdx = std::min(c * blockFactor, fusedOuter);
        uint64_t tndS1S2PrefixSumTmp = 0;
        uint64_t tndS1S2AlignPrefixSumTmp = 0;
        uint64_t tndPrefixSumTmp = 0;
        for (int64_t b = 0; b < fBaseParams.b; b++) {
            if (calculatedBlockInfo[b][0][SUM_ALL] > currentIdx) {
                bIdx = b;
                auto s1os2o = calculatedBlockInfo[b][fBaseParams.s2Outer - 1][SUM_S1S2];
                auto gs1os2o = s1os2o * fBaseParams.g;
                bTail = (b == 0) ? currentIdx : currentIdx - calculatedBlockInfo[b - 1][0][SUM_ALL];
                n2Idx = bTail / gs1os2o;
                n2Tail = bTail % gs1os2o;
                gIdx = n2Tail / s1os2o;
                gTail = n2Tail % s1os2o;

                GetUnpadS1S2OuterIndex(s1oIdx, s2oIdx, gTail, b, calculatedBlockInfo);
                s1OuterTmp = (fBaseParams.actualSeqQlen[b] + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;

                tndBaseInfo.tndStartBIdx[c] = b;
                tndBaseInfo.tndS1S2PrefixSum[c] = tndS1S2PrefixSumTmp;
                tndBaseInfo.tndS1S2AlignPrefixSum[c] = tndS1S2AlignPrefixSumTmp;
                tndBaseInfo.tndPrefixSum[c] = tndPrefixSumTmp;
                break;
            } else {
                tndS1S2PrefixSumTmp += (fBaseParams.actualSeqQlen[b] * fBaseParams.actualSeqKvlen[b]);
                tndS1S2AlignPrefixSumTmp +=
                    (fBaseParams.actualSeqQlen[b] *
                     AlignTo(fBaseParams.actualSeqKvlen[b], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));
                int64_t s1Outer = (fBaseParams.actualSeqQlen[b] + fBaseParams.s1Inner * S1CV_RATIO_DEFAULT - 1) /
                                  (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT);
                int64_t s2Outer = (fBaseParams.actualSeqKvlen[b] + fBaseParams.s2Inner * S2CV_RATIO_DEFAULT - 1) /
                                  (fBaseParams.s2Inner * S2CV_RATIO_DEFAULT);
                tndPrefixSumTmp += (s1Outer * s2Outer);
            }
        }
        if (bIdx == 0) {
            blockStarts[c] = (n2Idx * fBaseParams.g + gIdx) * totalBlockInfo[bIdx][0] + s2oIdx * s1OuterTmp + s1oIdx;
        } else {
            blockStarts[c] = totalBlockInfo[bIdx - 1][1] + (n2Idx * fBaseParams.g + gIdx) * totalBlockInfo[bIdx][0] +
                             s2oIdx * s1OuterTmp + s1oIdx;
        }

        blockEnds[c - 1] = blockStarts[c];
    }

    for (int64_t c = 0; c < blockOuter; c++) {
        FAG_PLAN_TRACE("GetSparseUnpadBlockInfo", "blockNum[%ld], blockStarts = %ld , blockEnds = %ld ", c,
                       blockStarts[c], blockEnds[c]);
    }

    for (uint32_t c = static_cast<uint32_t>(blockOuter); c < CORE_LIST_NUM; c++) {
        blockStarts[c] = 0;
        blockEnds[c] = 0;
    }
    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

    return ge::GRAPH_SUCCESS;
}

void Scheduler::FillBlockInfo(std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo,
                              std::vector<std::vector<int64_t>>& totalBlockInfo)
{
    FAG_PLAN_TRACE("FillBlockInfo", " Starting load balancing calculation in TND scenario");
    FAG_PLAN_TRACE("FillBlockInfo", "SparseMode %u, find band index %u", fBaseParams.sparseMode, fBaseParams.bandIdx);
    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[i];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[i];

        auto actualS1Outer = (actualS1Len + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
        auto actualS2Outer = (actualS2Len + fBaseParams.cvS2Inner - 1) / fBaseParams.cvS2Inner;
        totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;

        CalValidUnpadBlockInfo(i, calculatedBlockInfo);

        if (i == 0) {
            calculatedBlockInfo[0][0][SUM_ALL] =
                fBaseParams.n2 * fBaseParams.g * calculatedBlockInfo[0][fBaseParams.s2Outer - 1][SUM_S1S2];
            totalBlockInfo[0][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[0][0];
        } else {
            calculatedBlockInfo[i][0][SUM_ALL] =
                fBaseParams.n2 * fBaseParams.g * calculatedBlockInfo[i][fBaseParams.s2Outer - 1][SUM_S1S2] +
                calculatedBlockInfo[i - 1][0][SUM_ALL];
            totalBlockInfo[i][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
        }
        FAG_PLAN_TRACE("FillBlockInfo", "Up to b idx = %ld , a total of %ld blocks that need to be calculated", i,
                       calculatedBlockInfo[i][0][SUM_ALL]);
    }
}

void Scheduler::CalValidUnpadBlockInfo(int64_t batchIdx,
                                       std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo)
{
    int64_t actualS1Len = fBaseParams.actualSeqQlen[batchIdx];
    int64_t actualS2Len = fBaseParams.actualSeqKvlen[batchIdx];
    // 对unpad场景的token值做二次校正
    // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
    int64_t actualCalcS1Token, actualCalcS2Token;
    CalcleActualToken(fBaseParams, batchIdx, actualCalcS1Token, actualCalcS2Token);

    // unpad 场景下s2Outer是按照最大的s2计算得到的
    for (int64_t j = 0; j < fBaseParams.s2Outer; j++) {
        if (fBaseParams.cvS2Inner * j >= actualS2Len) {
            calculatedBlockInfo[batchIdx][j][BEGIN_IDX] = 0;
            calculatedBlockInfo[batchIdx][j][END_IDX] = 0;
        } else {
            int64_t leftIntersectionPoint =
                std::max(int64_t(fBaseParams.cvS2Inner * j) - actualCalcS2Token, int64_t{0});
            if (leftIntersectionPoint > int64_t(actualS1Len)) {
                calculatedBlockInfo[batchIdx][j][BEGIN_IDX] =
                    (actualS1Len + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
            } else {
                calculatedBlockInfo[batchIdx][j][BEGIN_IDX] = leftIntersectionPoint / fBaseParams.s1CvInner;
            }
            int64_t cvBlockTail = fBaseParams.cvS2Inner * (j + 1) > actualS2Len ?
                                      actualS2Len - fBaseParams.cvS2Inner * j :
                                      fBaseParams.cvS2Inner;
            calculatedBlockInfo[batchIdx][j][END_IDX] =
                int64_t(std::min(int64_t(actualS1Len),
                                 std::max(fBaseParams.cvS2Inner * j + cvBlockTail + actualCalcS1Token, int64_t{0})) +
                        fBaseParams.s1CvInner - 1) /
                fBaseParams.s1CvInner;
        }

        int64_t tmpLength =
            calculatedBlockInfo[batchIdx][j][END_IDX] > calculatedBlockInfo[batchIdx][j][BEGIN_IDX] ?
                calculatedBlockInfo[batchIdx][j][END_IDX] - calculatedBlockInfo[batchIdx][j][BEGIN_IDX] :
                0;
        if (j == 0) {
            calculatedBlockInfo[batchIdx][j][SUM_S1S2] = tmpLength;
        } else {
            calculatedBlockInfo[batchIdx][j][SUM_S1S2] = calculatedBlockInfo[batchIdx][j - 1][SUM_S1S2] + tmpLength;
        }

        calculatedBlockInfo[batchIdx][j][SUM_ALL] = 0; // 初始化清零

        FAG_PLAN_TRACE("FillBlockInfo", " s2Outer idx = %ld: Begin = %ld, End = %ld, Sum_S1S2 = %ld", j,
                       calculatedBlockInfo[batchIdx][j][BEGIN_IDX], calculatedBlockInfo[batchIdx][j][END_IDX],
                       calculatedBlockInfo[batchIdx][j][SUM_S1S2]);
    }
}

void Scheduler::GetUnpadS1S2OuterIndex(int64_t& s1oIdx, int64_t& s2oIdx, int64_t gTail, int64_t bIdx,
                                       std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo)
{
    int64_t s1oTail = 0;
    for (int64_t i = 0; i < fBaseParams.s2Outer; i++) {
        if (calculatedBlockInfo[bIdx][i][SUM_S1S2] > gTail) {
            s2oIdx = i;
            s1oTail = (i == 0) ? gTail : gTail - calculatedBlockInfo[bIdx][i - 1][SUM_S1S2];
            s1oIdx = calculatedBlockInfo[bIdx][i][BEGIN_IDX] + s1oTail;
            break;
        }
    }
}

void Scheduler::FillBlockInfoLoadBalance(std::vector<std::vector<int64_t>>& totalBlockInfo,
                                         std::vector<std::vector<float>>& acturalBlockInfo)
{
    acturalBlockInfo[fBaseParams.b][0] = 0;
    acturalBlockInfo[fBaseParams.b + 1][0] = 0;
    FAG_PLAN_TRACE("FillBlockInfoLoadBalance", "SparseMode %u, find band index %u", fBaseParams.sparseMode,
                   fBaseParams.bandIdx);

    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[i];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[i];

        auto actualS1Outer = (actualS1Len + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
        auto actualS2Outer = (actualS2Len + fBaseParams.cvS2Inner - 1) / fBaseParams.cvS2Inner;
        totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;
        // 针对S为0的场景，pre中增加initGm为0的操作
        if (totalBlockInfo[i][0] == 0) {
            fBaseParams.sValueZeroUnderTND = true;
        }

        // 对unpad场景的token值做二次校正
        // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
        int64_t actualCalcS1Token, actualCalcS2Token;
        CalcleActualToken(fBaseParams, i, actualCalcS1Token, actualCalcS2Token);

        FAG_PLAN_TRACE(
            "FillBlockInfoLoadBalance",
            " b idx = %ld: actualS1Len = %ld, actualS2Len = %ld, actualCalcS1Token = %ld, actualCalcS2Token = %ld", i,
            actualS1Len, actualS2Len, actualCalcS1Token, actualCalcS2Token);

        // unpad 场景下s2Outer是按照最大的s2计算得到的
        for (int64_t j = 0; j < fBaseParams.s2Outer; j++) {
            if (fBaseParams.cvS2Inner * j >= actualS2Len) {
                acturalBlockInfo[i][j] = 0;
            } else {
                int64_t leftIntersectionPoint =
                    std::max(int64_t(fBaseParams.cvS2Inner * j) - actualCalcS2Token, int64_t{0});
                int64_t cvBlockTail = fBaseParams.cvS2Inner * (j + 1) > actualS2Len ?
                                          actualS2Len - fBaseParams.cvS2Inner * j :
                                          fBaseParams.cvS2Inner;

                float acturalS1Begin = static_cast<float>(
                    leftIntersectionPoint > int64_t(actualS1Len) ? actualS1Len : leftIntersectionPoint);
                float acturalS1End = static_cast<float>(
                    std::min(int64_t(actualS1Len),
                             std::max(fBaseParams.cvS2Inner * j + cvBlockTail + actualCalcS1Token, int64_t{0})));
                if (fBaseParams.splitAxis == SplitAxisEnum::BN2S2 && (acturalS1Begin >= acturalS1End)) {
                    fBaseParams.isInvalidCol = true;
                }
                float acturalS1Num = acturalS1Begin > acturalS1End ? 0 : acturalS1End - acturalS1Begin;
                float acturalS2Num = static_cast<float>(cvBlockTail);
                acturalBlockInfo[i][j] = acturalS1Num / static_cast<float>(fBaseParams.s1CvInner) +
                                         acturalS2Num / static_cast<float>(fBaseParams.cvS2Inner);

                acturalBlockInfo[fBaseParams.b][0] += acturalBlockInfo[i][j] * fBaseParams.n2 * fBaseParams.g;
                acturalBlockInfo[fBaseParams.b + 1][0] =
                    acturalBlockInfo[fBaseParams.b + 1][0] < acturalBlockInfo[i][j] ?
                        acturalBlockInfo[i][j] :
                        acturalBlockInfo[fBaseParams.b + 1][0];
            }
        }

        if (i == 0) {
            totalBlockInfo[0][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[0][0];
        } else {
            totalBlockInfo[i][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
        }
    }
}

uint8_t Scheduler::GetSparseType()
{
    // DENSE: 1）非sparse；2）ALL_MASK；3）NO_MASK & preToken>=Sq & nextToken>=Skv
    bool denseCondition = !fBaseParams.isSparse ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK)) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token >= fBaseParams.s2);

    bool casualCondition = false;
    bool bandCondition = false;
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        casualCondition = (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL)) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token == 0) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token == 0);
        // 仅支持N1为偶数，分核时按照N1维度拼接性能最优，如果N1为奇数存在负载不均问题。
        casualCondition = casualCondition && ((fBaseParams.n1 % MULT_BASE == 0));
        if (fBaseParams.sparseMode != static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
            casualCondition = casualCondition && tndBaseInfo.isS1GreaterThanS2;
        } else {
            casualCondition = casualCondition && tndBaseInfo.isS1LessThanS2;
            if (!fBaseParams.isS1S2Same) {
                casualCondition = casualCondition && (fBaseParams.s1 % fBaseParams.s1CvInner == 0);
            }
        }

        bandCondition = fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) ||
                        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND);
        bandCondition = bandCondition && fBaseParams.s1Token >= 0 && fBaseParams.s2Token >= 0 && fBaseParams.isS1S2Same;
    } else {
        // CASUAL: 1）LEFT_UP_CASUAL；2）RIGHT_DOWN_CASUAL；3）NO_MASK & preToken>=Sq & nextToken=0；4）BAND &
        // preToken>=Sq & nextToken=0
        casualCondition = (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) &&
                           fBaseParams.s1 <= fBaseParams.s2) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token == 0) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) &&
                           fBaseParams.s1 <= fBaseParams.s2) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token == 0);

        // BAND: 1）NO_MASK剩余场景；2）BAND剩余场景；3）LEFT_UP_CAUSAL剩余场景；4）RIGHT_DOWN_CAUSAL剩余场景
        bandCondition = fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) ||
                        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) ||
                        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
                        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL);
    }
    FAG_PLAN_TRACE(
        "GetSparseType",
        "denseCondition = %d, casualCondition = %d, bandCondition = %d, isS1GreaterThanS2 = %d, isS1LessThanS2 = %d",
        denseCondition, casualCondition, bandCondition, tndBaseInfo.isS1GreaterThanS2, tndBaseInfo.isS1LessThanS2);

    if (denseCondition) {
        return static_cast<uint8_t>(SparseType::DENSE);
    } else if (casualCondition) {
        return static_cast<uint8_t>(SparseType::CASUAL);
    } else if (bandCondition) {
        return static_cast<uint8_t>(SparseType::BAND);
    } else {
        // 超L2优化暂不支持的sparse场景
        return static_cast<uint8_t>(SparseType::UNSUPPORTED);
    }
}

bool Scheduler::CheckExceedL2Cache()
{
    std::array<int64_t, CORE_LIST_NUM> dqOffset;
    std::array<int64_t, CORE_LIST_NUM> dkDvOffset;
    std::array<int64_t, CORE_LIST_NUM> loopIdx;
    std::set<int> dqOffsetSet;
    std::set<int> dkDvOffsetSet;
    uint64_t usedl2CacheSize = 0;
    int64_t calcNum = 0;
    int32_t inputSize = FP16_BYTES;
    std::fill(std::begin(loopIdx), std::end(loopIdx), static_cast<int64_t>(0));

    // Admitted fp16/bf16 both use FP16_BYTES.

    bool isExceed = false;
    while (calcNum < fBaseParams.maxValidBBLen) {
        for (uint16_t cBlockIdx = 0; cBlockIdx < fBaseParams.blockOuter; cBlockIdx++) {
            while (!IsValid(fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx]) &&
                   (fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx] < fBaseParams.blockEnds[cBlockIdx])) {
                loopIdx[cBlockIdx]++;
            }
            if (fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx] >= fBaseParams.blockEnds[cBlockIdx]) {
                dqOffset[cBlockIdx] = OUTINDEX;
                dkDvOffset[cBlockIdx] = OUTINDEX;
                continue;
            }
            int64_t validBlockIdx = fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx];
            GetOffset(fBaseParams, dqOffset[cBlockIdx], dkDvOffset[cBlockIdx], validBlockIdx);
            loopIdx[cBlockIdx]++;

            if (dqOffsetSet.find(dqOffset[cBlockIdx]) == dqOffsetSet.end()) {
                dqOffsetSet.insert(dqOffset[cBlockIdx]);
                // qSize + dxSize + dqSize(btyes) + ySize，3 means q, dx and y
                usedl2CacheSize += (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * fBaseParams.d * inputSize * NUM_THREE +
                                    fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * fBaseParams.d * FP32_BYTES);
            }
            if (dkDvOffsetSet.find(dkDvOffset[cBlockIdx]) == dkDvOffsetSet.end()) {
                dkDvOffsetSet.insert(dkDvOffset[cBlockIdx]);
                // kSize + vSize + dkSize + dvSize(btyes)，2 means k/dk and v/dv
                usedl2CacheSize += (fBaseParams.s2Inner * fBaseParams.d * inputSize * NUM_TWO +
                                    fBaseParams.s2Inner * fBaseParams.d * FP32_BYTES * NUM_TWO);
            }
            if (usedl2CacheSize > fBaseParams.l2CacheSize) {
                isExceed = true;
                break;
            }
        }
        if (isExceed) {
            break;
        }
        calcNum++;
    }
    if (!isExceed) {
        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK) ||
            fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK)) {
            if (fBaseParams.attenMaskShapeType == static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKBN2GS1S2)) {
                usedl2CacheSize += (fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.s2);
            } else if (fBaseParams.attenMaskShapeType == static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKBS1S2)) {
                usedl2CacheSize += (fBaseParams.b * fBaseParams.s1 * fBaseParams.s2);
            } else if (fBaseParams.attenMaskShapeType == static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKS1S2)) {
                usedl2CacheSize += (fBaseParams.s1 * fBaseParams.s2);
            }
        } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
                   fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
                   fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND)) {
            usedl2CacheSize += COMPRESS_ATTEN_MASK_SIZE;
        }
        isExceed = usedl2CacheSize > fBaseParams.l2CacheSize;
    }

    return isExceed;
}

#undef FAG_PLAN_TRACE
#ifdef _MSC_VER
#pragma warning(pop)
#endif
} // namespace asc

namespace {
ScheduleError ReadCu(const int32_t* cu, size_t size, const RouteInputs& route, bool query,
                     std::vector<int64_t>& lengths)
{
    if (cu == nullptr || size != static_cast<size_t>(route.batch) + 1) {
        return ScheduleError::CU_SIZE;
    }
    if (cu[0] != 0) {
        return ScheduleError::CU_ORIGIN;
    }
    int64_t maximum = 0;
    for (size_t i = 1; i < size; ++i) {
        if (cu[i] < cu[i - 1]) {
            return ScheduleError::CU_ORDER;
        }
        const int64_t length = static_cast<int64_t>(cu[i]) - cu[i - 1];
        lengths.push_back(length);
        maximum = std::max(maximum, length);
    }
    if (cu[size - 1] != (query ? route.totalQ : route.totalKv)) {
        return ScheduleError::CU_TOTAL;
    }
    if (maximum != (query ? route.maxQ : route.maxKv)) {
        return ScheduleError::CU_MAX;
    }
    return ScheduleError::OK;
}
} // namespace

ScheduleError BuildInitialSchedule(const ScheduleInputs& inputs, const int32_t* cuQ, size_t cuQSize,
                                   const int32_t* cuKv, size_t cuKvSize, ScheduleResult& result)
{
    result = ScheduleResult{};
    const auto& r = inputs.route;
    if (Admit(r) != Admission::ELIGIBLE) {
        return ScheduleError::ADMISSION;
    }
    if (inputs.aicNum <= 0 || inputs.aicNum > MAX_CORES || inputs.aivNum != 2 * inputs.aicNum) {
        return ScheduleError::CORE_COUNT;
    }
    if (r.batch > 2048) {
        return ScheduleError::BATCH_CAPACITY;
    }
    if (inputs.windowLeft < -1 || inputs.windowRight < -1 || inputs.windowLeft > INT32_MAX ||
        inputs.windowRight > INT32_MAX) {
        return ScheduleError::WINDOW;
    }
    // cu is int32; also bound products used by ASC's area, head and address arithmetic.
    // Include alignment and folded causal rectangles, which can exceed raw area.
    int64_t bound = 0;
    if (r.totalQ > INT32_MAX || r.totalKv > INT32_MAX || !CheckedMul(r.totalQ + 128 * r.batch, r.maxKv + 128, bound) ||
        !CheckedMul(bound, r.headsQ, bound) || !CheckedMul(bound, 4, bound) ||
        !CheckedMul(r.totalKv, r.headsQ, bound) || !CheckedMul(bound, AlignHead(r.dimQ), bound)) {
        return ScheduleError::ARITHMETIC_OVERFLOW;
    }
    asc::Scheduler planner;
    auto& p = planner.fBaseParams;
    auto error = ReadCu(cuQ, cuQSize, r, true, p.actualSeqQlen);
    if (error != ScheduleError::OK) {
        return error;
    }
    error = ReadCu(cuKv, cuKvSize, r, false, p.actualSeqKvlen);
    if (error != ScheduleError::OK) {
        return error;
    }
    planner.kvEnds.assign(cuKv + 1, cuKv + cuKvSize);
    p.b = r.batch;
    p.s1 = r.maxQ;
    p.s2 = r.maxKv;
    p.n1 = p.n2 = r.headsQ;
    p.d = r.dimQ;
    p.aicNum = static_cast<uint64_t>(inputs.aicNum);
    p.l2CacheSize = inputs.l2Bytes;
    p.s1CvInner = std::min(int64_t{128}, p.s1);
    p.cvS2Inner = std::min(int64_t{128}, p.s2);
    p.s1Outer = (p.s1 + p.s1CvInner - 1) / p.s1CvInner;
    p.s2Outer = (p.s2 + p.cvS2Inner - 1) / p.cvS2Inner;
    p.isSparse = inputs.hasMask;
    p.attenMaskOptional = inputs.hasMask ? 1 : 0;
    p.attenMaskShapeType = inputs.maskShapeType;
    p.sparseMode = static_cast<uint32_t>(r.maskMode);
    p.isS1S2Same = p.actualSeqQlen == p.actualSeqKvlen;
    p.sValueZeroUnderTND = p.actualSeqQlen.front() == 0 || p.actualSeqKvlen.front() == 0;
    p.s1Token = inputs.windowLeft == -1 ? INT32_MAX : inputs.windowLeft;
    p.s2Token = inputs.windowRight == -1 ? INT32_MAX : inputs.windowRight;
    // ASC ProcessTokensInfo, restricted to the admitted sparse modes and TND.
    if (p.sparseMode == 3) {
        p.s1Token = INT32_MAX;
        p.s2Token = 0;
    }
    if (!inputs.hasMask) {
        p.s1Token = p.s2Token = INT32_MAX;
    }
    // ASC GetDeterSparseTilingKey in the same restricted domain.
    if (!inputs.hasMask || (p.sparseMode == 0 && p.s1Token >= p.s1 && p.s2Token >= p.s2)) {
        p.deterSparseType = 2;
    } else if ((p.sparseMode == 0 && p.s1Token >= p.s1 && p.s2Token <= 0 && p.s2Token > -128) ||
               (p.sparseMode == 3 && p.isS1S2Same)) {
        p.deterSparseType = 3;
    } else {
        p.deterSparseType = 4;
    }
    std::fill(std::begin(p.separateDkOffset), std::end(p.separateDkOffset), -1);
    for (int64_t i = 0; i < p.b; ++i) {
        result.hasZeroSequence |= p.actualSeqQlen[i] == 0 || p.actualSeqKvlen[i] == 0;
        planner.tndBaseInfo.isS1GreaterThanS2 &= p.actualSeqQlen[i] >= p.actualSeqKvlen[i];
        planner.tndBaseInfo.isS1LessThanS2 &= p.actualSeqQlen[i] <= p.actualSeqKvlen[i];
    }
    planner.tndBaseInfo.isSeqExistZero = result.hasZeroSequence;
    p.sparseType = planner.GetSparseType();
    // No EOD encoding is accepted. Repeated final cu endpoints are ordinary empty
    // sequences, not the old EOD tail-zero marker; keep all batches in the prefix.
    planner.CalcTNDSwizzleParam();
    planner.CalcleTNDDeterParam();
    if (p.tndLineDeter != 0) {
        planner.fallbackReasons |= static_cast<int64_t>(Fallback::LINE);
    }
    result.state = std::move(p);
    result.tnd = std::move(planner.tndBaseInfo);
    result.fallbackReasons = planner.fallbackReasons;
    return ScheduleError::OK;
}

ScheduleError BuildSchedule(const ScheduleInputs& inputs, const int32_t* cuQ, size_t cuQSize, const int32_t* cuKv,
                            size_t cuKvSize, ScheduleResult& result)
{
    if (inputs.l2Bytes == 0) {
        result = ScheduleResult{};
        return ScheduleError::L2_CAPACITY;
    }
    const auto error = BuildInitialSchedule(inputs, cuQ, cuQSize, cuKv, cuKvSize, result);
    if (error != ScheduleError::OK) {
        return error;
    }
    asc::Scheduler planner;
    planner.fBaseParams = std::move(result.state);
    planner.tndBaseInfo = std::move(result.tnd);
    planner.kvEnds.assign(cuKv + 1, cuKv + cuKvSize);
    planner.fallbackReasons = result.fallbackReasons;
    auto& p = planner.fBaseParams;
    // The initial split is proven BN2S2: BN2 and layout-normalization branches in
    // DoSparse are unreachable. Preserve the saturation gate after deter planning.
    const bool retainedBn2s2 =
        p.splitAxis == asc::SplitAxisEnum::BN2S2 && planner.GetBlockInfoOfBNS4TND() && p.blockOuter >= p.aicNum;
    if (!retainedBn2s2) {
        if (p.splitAxis == asc::SplitAxisEnum::BN2S2) {
            planner.fallbackReasons |= static_cast<int64_t>(Fallback::UNSATURATED);
        }
        p.splitAxis = asc::SplitAxisEnum::BN2GS1S2;
        planner.GetSparseUnpadBlockInfo();
    }
    // CheckIsLargeInvalidBlk is false for the admitted mask 0/3/4 domain.
    p.enableSwizzle = planner.CheckExceedL2Cache() && p.blockOuter == p.aicNum;
    const bool supported = p.deterSparseType == 2 && asc::IsTndDeterSwizzleSupported(p);
    // A zero-work Dense batch contributes no round to the monotone prefix.
    // Its physical cu endpoints remain in the plan; pre clears the original
    // direct-output rows. Column ownership and per-round dQ fences are unchanged.
    const bool emptyDenseDirect = retainedBn2s2 && inputs.route.maskMode == 0 &&
                                  CanCompileDirectStore(AlignHead(inputs.route.dimQ), AlignHead(inputs.route.dimV));
    p.isSwizzle = p.enableSwizzle && supported && p.b < asc::TND_SWIZZLE_PREFIX_NUM &&
                  (!planner.tndBaseInfo.isSeqExistZero || emptyDenseDirect) && result.tailZeroCount == 0;
    if (p.splitAxis == asc::SplitAxisEnum::BN2S2 && p.isSwizzle) {
        asc::ConfigureTndDeterBn2S2Swizzle(p, planner.tndBaseInfo);
    }
    result.state = std::move(p);
    result.tnd = std::move(planner.tndBaseInfo);
    result.fallbackReasons = planner.fallbackReasons;
    return ScheduleError::OK;
}
} // namespace optiling::fag_deter

namespace optiling::fag_deter {
namespace {
int64_t Ceil(int64_t value, int64_t divisor)
{
    return value / divisor + (value % divisor != 0);
}

template <class T, size_t N>
void Copy(PlanWords& plan, Field field, const T (&source)[N])
{
    for (size_t i = 0; i < N; ++i) {
        At(plan, field, i) = static_cast<int64_t>(source[i]);
    }
}

void PutRegion(PlanWords& plan, Field offset, Field bytes, const Region& region)
{
    At(plan, offset) = region.offset;
    At(plan, bytes) = region.bytes;
}

void FillIdentity(const PlanInputs& in, PlanWords& plan)
{
    const auto& s = in.schedule;
    const auto& r = s.route;
    At(plan, Field::compiledFamily) = At(plan, Field::initialSplit) = static_cast<int64_t>(Split::BN2S2);
#define PUT(field, value) At(plan, Field::field) = static_cast<int64_t>(value)
    PUT(batch, r.batch);
    PUT(totalQ, r.totalQ);
    PUT(totalKv, r.totalKv);
    PUT(maxQ, r.maxQ);
    PUT(maxKv, r.maxKv);
    PUT(headsQ, r.headsQ);
    PUT(headsKv, r.headsKv);
    PUT(dimQ, r.dimQ);
    PUT(dimV, r.dimV);
    PUT(alignQ, AlignHead(r.dimQ));
    PUT(alignV, AlignHead(r.dimV));
    PUT(supportedDtypeMask, SUPPORTED_DTYPE_MASK);
    PUT(maskMode, r.maskMode);
    PUT(windowLeft, s.windowLeft);
    PUT(windowRight, s.windowRight);
    PUT(sink, in.sink);
    PUT(aicNum, s.aicNum);
    PUT(aivNum, s.aivNum);
    PUT(l2Bytes, s.l2Bytes);
    PUT(reservedPlanMode, 0);
    PUT(libapiBytes, 0);
    PUT(ubBytes, in.ubBytes);
    PUT(maskPresent, s.hasMask);
    PUT(maskShapeType, s.maskShapeType);
    PUT(effectiveLayout, asc::INPUT_FORMAT_TND);
    PUT(ascWriteFormat, in.ascWriteFormat);
    PUT(writeFormat, WriteFormat::ND);
#undef PUT
}

void FillSchedule(const ScheduleResult& result, PlanWords& plan)
{
    const auto& p = result.state;
    const auto& t = result.tnd;
    const Split split = p.splitAxis == asc::SplitAxisEnum::BN2S2 ? Split::BN2S2 : Split::GS;
    const Schedule kind = p.isSwizzle            ? Schedule::DENSE_SWIZZLE :
                          p.tndLineDeter         ? Schedule::LINE :
                          p.deterSparseType == 2 ? Schedule::DENSE :
                          p.deterSparseType == 3 ? Schedule::CAUSAL :
                                                   Schedule::BAND;
    At(plan, Field::logicalSplit) = static_cast<int64_t>(split);
    At(plan, Field::scheduleKind) = static_cast<int64_t>(kind);
    At(plan, Field::fallbackReasons) = result.fallbackReasons;
    At(plan, Field::tailZeroCount) = result.tailZeroCount;
    At(plan, Field::hasZeroSequence) = result.hasZeroSequence;
#define PUT(field, value) At(plan, Field::field) = static_cast<int64_t>(p.value)
    PUT(sparseType, sparseType);
    PUT(deterType, deterSparseType);
    PUT(s1Token, s1Token);
    PUT(s2Token, s2Token);
    PUT(s1Inner, s1Inner);
    PUT(s2Inner, s2Inner);
    PUT(s1CvRatio, s1CvRatio);
    PUT(s2CvRatio, s2CvRatio);
    PUT(isSwizzle, isSwizzle);
    PUT(lineDeter, tndLineDeter);
    PUT(maxRound, deterMaxRound);
    PUT(blockOuter, blockOuter);
    PUT(blockFactor, blockFactor);
    PUT(prefixStep, deterPrefixStep);
    PUT(invalidRow, isInvalidRow);
    PUT(invalidCol, isInvalidCol);
    PUT(sValueZeroUnderTnd, sValueZeroUnderTND);
    PUT(s1Outer, s1Outer);
    PUT(s2Outer, s2Outer);
#undef PUT
    // coreDivide remains zero for the MHA TND branch of ASC CalcleDeterParam.
    Copy(plan, Field::blockStarts, p.blockStarts);
    Copy(plan, Field::blockEnds, p.blockEnds);
    Copy(plan, Field::startBatch, t.tndStartBIdx);
    Copy(plan, Field::startNeedSyncRound, p.startNeedSyncRound);
    Copy(plan, Field::endNeedSyncRound, p.endNeedSyncRound);
    Copy(plan, Field::separateDkOffset, p.separateDkOffset);
    Copy(plan, Field::deterPrefix, p.deterPrefix);
    Copy(plan, Field::deterPrefixAlign, p.deterPrefixAlign);
    Copy(plan, Field::deterPrefix0, p.deterPrefix0);
    Copy(plan, Field::deterPrefix1, p.deterPrefix1);
    Copy(plan, Field::deterPrefix2, p.deterPrefix2);
    Copy(plan, Field::swizzleAreaPrefix, t.tndSwizzleS1S2PrefixSum);
    Copy(plan, Field::swizzleAlignAreaPrefix, t.tndSwizzleS1S2AlignPrefixSum);
    Copy(plan, Field::swizzleRoundPrefix, t.tndS2BlockPrefixSum);
    // ASC CalDeterMaxLoopNum reads the swizzle prefix's final round, not the
    // legacy dense bound retained in baseDeterParam after swizzle selection.
    if (p.isSwizzle) {
        At(plan, Field::maxRound) = static_cast<int64_t>(t.tndS2BlockPrefixSum[p.b]);
    }
    if (split == Split::BN2S2 && !p.isSwizzle) {
        for (const auto offset : p.separateDkOffset) {
            At(plan, Field::specialCount) += offset >= 0;
        }
    }
}

bool FillPrePost(const PlanInputs& in, const ScheduleResult& result, PlanWords& plan)
{
    const auto& r = in.schedule.route;
    const auto& p = result.state;
    const int64_t align = AlignHead(r.dimV);
    const bool sfmg = DecidePreSfmg(r, align, in.sink, r.maxKv, p.s1Outer);
    At(plan, Field::enablePreSfmg) = sfmg;
    const int64_t blockOuter = p.blockOuter != 0 ? p.blockOuter : in.schedule.aicNum;
    int64_t preCores = 0;
    if (sfmg) {
        const int64_t rows = r.totalQ * r.headsQ;
        const int64_t normal = Ceil(rows, blockOuter * 2);
        preCores = Ceil(rows, normal);
        const int64_t loop = PreSfmgWindowRows(align);
        if (loop <= 0) {
            return false;
        }
        At(plan, Field::sfmgRows) = rows;
        // CalculateSoftmaxGrad writes one FP32 scalar per BNS row. The 8x
        // workspace/output-buffer reserve is not the stored row stride:
        // CopyInMaxSumWithSfmg explicitly divides the BNS8 offset by eight.
        At(plan, Field::sfmgStride) = 1;
        At(plan, Field::sfmgUsedCoreNum) = preCores;
        At(plan, Field::sfmgNormalCoreRows) = normal;
        At(plan, Field::sfmgTailCoreRows) = rows - (preCores - 1) * normal;
        At(plan, Field::sfmgLoopRows) = loop;
        At(plan, Field::sfmgDyBufferLen) = PreSfmgInputBytes(align);
        At(plan, Field::sfmgOutputBufferLen) = PreSfmgOutputBytes(align);
    } else {
        int64_t area = 0;
        for (size_t b = 0; b < p.actualSeqQlen.size(); ++b) {
            area += p.actualSeqQlen[b] * p.actualSeqKvlen[b];
        }
        const int64_t mask = Ceil(area * r.headsQ, 32) * 32;
        preCores = mask == 0 ? blockOuter : Ceil(mask, Ceil(Ceil(mask, blockOuter), 32) * 32);
    }
    At(plan, Field::postBaseElements) = 128 * 128;
    auto ranges = [&](int64_t elements, Field preFactor, Field preTotal, Field postFactor, Field postTotal) {
        const int64_t factor = Ceil(elements, preCores);
        At(plan, preFactor) = factor;
        At(plan, preTotal) = Ceil(elements, factor);
        At(plan, postTotal) = elements;
        At(plan, postFactor) = Ceil(Ceil(elements, 128 * 128), blockOuter * 2);
    };
    ranges(r.totalQ * r.headsQ * r.dimQ, Field::qPreBlockFactor, Field::qPreBlockTotal, Field::qPostBlockFactor,
           Field::qPostBlockTotal);
    ranges(r.totalKv * r.headsKv * r.dimQ, Field::kPreBlockFactor, Field::kPreBlockTotal, Field::kPostBlockFactor,
           Field::kPostBlockTotal);
    ranges(r.totalKv * r.headsKv * r.dimV, Field::vPreBlockFactor, Field::vPreBlockTotal, Field::vPostBlockFactor,
           Field::vPostBlockTotal);
    if (in.sink) {
        const int64_t reduce = r.batch * p.s1Outer * 2 * p.s2Outer;
        const int64_t elements = reduce * r.headsQ;
        const int64_t padded = (elements + 512) / 512 * 512;
        const int64_t factor = Ceil(padded, preCores);
        At(plan, Field::sinkS1Outer) = p.s1Outer * 2;
        At(plan, Field::sinkS2Outer) = p.s2Outer;
        At(plan, Field::sinkReduceAxis) = reduce;
        At(plan, Field::sinkPreBlockFactor) = factor;
        At(plan, Field::sinkPreBlockTotal) = Ceil(padded, factor);
        At(plan, Field::sinkPostBlockFactor) = Ceil(r.headsQ, blockOuter * 2);
        At(plan, Field::sinkPostBlockTotal) = r.headsQ;
    }
    return true;
}

bool ValidateScheduleFields(const PlanWords& plan, const PlanInputs& in)
{
    const auto& r = in.schedule.route;
    auto value = [&](Field f) { return At(plan, f); };
    const bool bn = value(Field::logicalSplit) == static_cast<int64_t>(Split::BN2S2);
    constexpr Field flags[] = {Field::isSwizzle,  Field::lineDeter,          Field::hasZeroSequence, Field::invalidRow,
                               Field::invalidCol, Field::sValueZeroUnderTnd, Field::enablePreSfmg};
    for (auto f : flags) {
        if (value(f) != 0 && value(f) != 1) {
            return false;
        }
    }
    const auto kind = value(Field::isSwizzle)      ? Schedule::DENSE_SWIZZLE :
                      value(Field::lineDeter)      ? Schedule::LINE :
                      value(Field::deterType) == 2 ? Schedule::DENSE :
                      value(Field::deterType) == 3 ? Schedule::CAUSAL :
                                                     Schedule::BAND;
    if (value(Field::scheduleKind) != static_cast<int64_t>(kind) ||
        value(Field::effectiveLayout) != asc::INPUT_FORMAT_TND || value(Field::coreDivide) != 0 ||
        value(Field::s1Inner) != 64 || value(Field::s2Inner) != 128 || value(Field::s1CvRatio) != 2 ||
        value(Field::s2CvRatio) != 1 || value(Field::s1Outer) != Ceil(r.maxQ, 128) ||
        value(Field::s2Outer) != Ceil(r.maxKv, 128) || value(Field::prefixStep) != Ceil(r.batch, 128) ||
        value(Field::blockFactor) < 0 || value(Field::sparseType) < 0 || value(Field::sparseType) > 3 ||
        value(Field::tailZeroCount) < 0 || value(Field::tailZeroCount) > r.batch || value(Field::fallbackReasons) < 0 ||
        (value(Field::fallbackReasons) & ~31LL) != 0 || value(Field::ascWriteFormat) < -1 ||
        value(Field::ascWriteFormat) > 1 ||
        (value(Field::lineDeter) && (bn || value(Field::isSwizzle) || value(Field::deterType) == 2)) ||
        // ASC permits TND dense swizzle after either retained BN2S2 or GS
        // fallback. Both use the same coordinate decoder and round prefix;
        // logicalSplit independently selects workspace and commit ordering.
        (value(Field::isSwizzle) && (r.batch > 128 || value(Field::deterType) != 2))) {
        return false;
    }
    // Only the sampled prefix portion is monotone. ASC appends phase round
    // counts after it, and BN2S2 reuses prefix2 as a special-column table.
    const int64_t samples = r.batch / value(Field::prefixStep) + 1;
    if (samples + 2 > PREFIX_CAPACITY) {
        return false;
    }
    int64_t prefix0Start = 0;
    if (value(Field::lineDeter)) {
        // ASC's line schedule handles single-tile tasks first. prefix0 stores
        // absolute round boundaries, so its origin is that phase's round count.
        // StoreTNDPrefix appends the exact single-task total after the samples,
        // including when prefixStep does not divide the batch count.
        const int64_t singleTasks = At(plan, Field::deterPrefix1, samples);
        if (singleTasks < 0 || in.schedule.aicNum <= 0) {
            return false;
        }
        prefix0Start = Ceil(singleTasks, in.schedule.aicNum);
    }
    auto monotone = [&](Field f, int64_t count) {
        if (value(f) != (f == Field::deterPrefix0 ? prefix0Start : 0)) {
            return false;
        }
        for (int64_t i = 1; i < count; ++i) {
            if (At(plan, f, i) < At(plan, f, i - 1)) {
                return false;
            }
        }
        return true;
    };
    for (auto f : {Field::deterPrefix, Field::deterPrefixAlign, Field::deterPrefix0, Field::deterPrefix1}) {
        if (!monotone(f, samples)) {
            return false;
        }
    }
    // A GS fallback from BN2S2 retains the unused special table in prefix2.
    // Only the non-line causal decoder reads it as a sampled prefix.
    if (!bn && value(Field::deterType) == 3 && !value(Field::lineDeter) && !monotone(Field::deterPrefix2, samples)) {
        return false;
    }
    if (value(Field::isSwizzle)) {
        if (value(Field::maxRound) != At(plan, Field::swizzleRoundPrefix, r.batch)) {
            return false;
        }
        for (auto f : {Field::swizzleAreaPrefix, Field::swizzleAlignAreaPrefix, Field::swizzleRoundPrefix}) {
            if (!monotone(f, r.batch + 1)) {
                return false;
            }
        }
    }
    int64_t kvElements = 0;
    if (!CheckedMul(r.totalKv, r.headsKv, kvElements) || !CheckedMul(kvElements, r.dimQ, kvElements)) {
        return false;
    }
    int64_t specialCount = 0;
    for (int64_t i = 0; i < MAX_CORES; ++i) {
        const auto offset = At(plan, Field::separateDkOffset, i);
        if (At(plan, Field::startBatch, i) < 0 || At(plan, Field::startBatch, i) >= r.batch || offset < -1 ||
            offset >= kvElements || (offset >= 0 && offset % r.dimQ != 0)) {
            return false;
        }
        specialCount += offset >= 0;
        // deterPrefix2 is the final kernel special-column table. Swizzle
        // invalidates it while leaving the earlier separateDkOffset intact.
        if (bn && At(plan, Field::deterPrefix2, i) != (value(Field::isSwizzle) ? -1 : offset)) {
            return false;
        }
    }
    return value(Field::specialCount) == (bn && !value(Field::isSwizzle) ? specialCount : 0);
}

bool ValidatePrePostFields(const PlanWords& plan, const PlanInputs& in)
{
    const auto& r = in.schedule.route;
    auto value = [&](Field f) { return At(plan, f); };
    const int64_t workers = (value(Field::blockOuter) ? value(Field::blockOuter) : in.schedule.aicNum) * 2;
    if (workers <= 0 || workers > in.schedule.aivNum || value(Field::postBaseElements) != 16384) {
        return false;
    }
    auto pre = [&](int64_t elements, Field factor, Field total) {
        int64_t coverage = 0;
        return value(factor) > 0 && value(factor) <= elements && value(total) > 0 && value(total) <= workers &&
               value(total) == Ceil(elements, value(factor)) && CheckedMul(value(factor), in.schedule.aivNum, coverage);
    };
    const Field ranges[][4] = {
        {Field::qPreBlockFactor, Field::qPreBlockTotal, Field::qPostBlockFactor, Field::qPostBlockTotal},
        {Field::kPreBlockFactor, Field::kPreBlockTotal, Field::kPostBlockFactor, Field::kPostBlockTotal},
        {Field::vPreBlockFactor, Field::vPreBlockTotal, Field::vPostBlockFactor, Field::vPostBlockTotal}};
    const int64_t tokens[] = {r.totalQ, r.totalKv, r.totalKv};
    const int64_t dims[] = {r.dimQ, r.dimQ, r.dimV};
    for (int i = 0; i < 3; ++i) {
        int64_t elements = 0;
        if (!CheckedMul(tokens[i], r.headsQ, elements) || !CheckedMul(elements, dims[i], elements) ||
            value(ranges[i][3]) != elements || !pre(elements, ranges[i][0], ranges[i][1]) ||
            value(ranges[i][2]) != Ceil(Ceil(elements, 16384), workers)) {
            return false;
        }
    }
    const bool sfmg = DecidePreSfmg(r, AlignHead(r.dimV), in.sink, r.maxKv, value(Field::s1Outer));
    if (value(Field::enablePreSfmg) != static_cast<int64_t>(sfmg)) {
        return false;
    }
    if (sfmg) {
        int64_t rows = 0;
        if (!CheckedMul(r.totalQ, r.headsQ, rows)) {
            return false;
        }
        const int64_t normal = Ceil(rows, workers), used = Ceil(rows, normal);
        const int64_t loop = PreSfmgWindowRows(AlignHead(r.dimV));
        if (loop <= 0 || value(Field::sfmgRows) != rows || value(Field::sfmgStride) != 1 ||
            value(Field::sfmgUsedCoreNum) != used || value(Field::sfmgNormalCoreRows) != normal ||
            value(Field::sfmgTailCoreRows) != rows - (used - 1) * normal || value(Field::sfmgLoopRows) != loop ||
            value(Field::sfmgDyBufferLen) != PreSfmgInputBytes(AlignHead(r.dimV)) ||
            value(Field::sfmgOutputBufferLen) != PreSfmgOutputBytes(AlignHead(r.dimV))) {
            return false;
        }
    } else {
        for (auto f :
             {Field::sfmgRows, Field::sfmgStride, Field::sfmgUsedCoreNum, Field::sfmgNormalCoreRows,
              Field::sfmgTailCoreRows, Field::sfmgLoopRows, Field::sfmgDyBufferLen, Field::sfmgOutputBufferLen}) {
            if (value(f) != 0) {
                return false;
            }
        }
    }
    if (in.sink) {
        int64_t elements = 0;
        if (!CheckedMul(value(Field::sinkReduceAxis), r.headsQ, elements) ||
            elements > std::numeric_limits<int64_t>::max() - 512 ||
            !pre((elements + 512) / 512 * 512, Field::sinkPreBlockFactor, Field::sinkPreBlockTotal) ||
            value(Field::sinkPostBlockFactor) != Ceil(r.headsQ, workers) ||
            value(Field::sinkPostBlockTotal) != r.headsQ) {
            return false;
        }
    } else {
        for (auto f : {Field::sinkS1Outer, Field::sinkS2Outer, Field::sinkReduceAxis, Field::sinkPreBlockFactor,
                       Field::sinkPreBlockTotal, Field::sinkPostBlockFactor, Field::sinkPostBlockTotal}) {
            if (value(f) != 0) {
                return false;
            }
        }
    }
    return true;
}
} // namespace

PlanResult BuildPlan(const PlanInputs& in, const int32_t* cuQ, size_t cuQSize, const int32_t* cuKv, size_t cuKvSize,
                     int64_t workspaceCapacity, PlanWords& plan)
{
    ScheduleResult result;
    const auto error = BuildSchedule(in.schedule, cuQ, cuQSize, cuKv, cuKvSize, result);
    if (error != ScheduleError::OK) {
        InitializePlan(plan);
        return {PlanError::SCHEDULE, error};
    }
    return BuildPlanFromSchedule(in, result, workspaceCapacity, plan);
}

PlanResult BuildPlanFromSchedule(const PlanInputs& in, const ScheduleResult& result, int64_t workspaceCapacity,
                                 PlanWords& plan)
{
    InitializePlan(plan);
    if (in.schedule.l2Bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return {PlanError::RESOURCE, ScheduleError::OK};
    }
    int64_t capacity = 0;
    if (!WorkspaceCapacity(in, capacity) || workspaceCapacity < capacity) {
        return {PlanError::WORKSPACE, ScheduleError::OK};
    }
    FillIdentity(in, plan);
    FillSchedule(result, plan);
    if (!FillPrePost(in, result, plan)) {
        return {PlanError::RESOURCE, ScheduleError::OK};
    }
    WorkspaceLayout layout;
    if (!MakeWorkspace(in, static_cast<Split>(At(plan, Field::logicalSplit)), At(plan, Field::enablePreSfmg) != 0,
                       layout)) {
        return {PlanError::WORKSPACE, ScheduleError::OK};
    }
    At(plan, Field::workspaceBytes) = layout.bytes;
    At(plan, Field::scratchStride) = layout.scratchStride;
    PutRegion(plan, Field::dqOffset, Field::dqBytes, layout.dq);
    PutRegion(plan, Field::dkOffset, Field::dkBytes, layout.dk);
    PutRegion(plan, Field::dvOffset, Field::dvBytes, layout.dv);
    PutRegion(plan, Field::sfmgOffset, Field::sfmgBytes, layout.sfmg);
    PutRegion(plan, Field::sinkOffset, Field::sinkBytes, layout.sink);
    PutRegion(plan, Field::specialOffset, Field::specialBytes, layout.special);
    // Validate while INVALID. READY is the last write by the AICPU producer.
    if (!ValidatePlan(plan, in, workspaceCapacity)) {
        return {PlanError::CONTRACT, ScheduleError::OK};
    }
    At(plan, Field::status) = static_cast<int64_t>(PlanStatus::READY);
    return {};
}

bool ValidatePlan(const PlanWords& plan, const PlanInputs& in, int64_t workspaceCapacity)
{
    auto header = plan;
    At(header, Field::status) = static_cast<int64_t>(PlanStatus::READY);
    if (!HasReadyHeader(header.data(), PLAN_BYTES) ||
        (At(plan, Field::status) != static_cast<int64_t>(PlanStatus::INVALID) &&
         At(plan, Field::status) != static_cast<int64_t>(PlanStatus::READY)) ||
        At(plan, Field::writeFormat) != static_cast<int64_t>(WriteFormat::ND) || At(plan, Field::workspaceBytes) <= 0 ||
        At(plan, Field::workspaceBytes) > workspaceCapacity || At(plan, Field::deterType) < 2 ||
        At(plan, Field::deterType) > 4 || At(plan, Field::blockOuter) < 0 ||
        At(plan, Field::blockOuter) > in.schedule.aicNum || At(plan, Field::prefixStep) <= 0 ||
        At(plan, Field::maxRound) < 0) {
        return false;
    }
    PlanWords identity;
    InitializePlan(identity);
    FillIdentity(in, identity);
    // Explicit fields avoid hashes/collisions and avoid comparing scheduling data
    // against an independently recomputed plan in the consumer.
    constexpr Field fields[] = {Field::batch,       Field::totalQ,     Field::totalKv,     Field::maxQ,
                                Field::maxKv,       Field::headsQ,     Field::headsKv,     Field::dimQ,
                                Field::dimV,        Field::alignQ,     Field::alignV,      Field::supportedDtypeMask,
                                Field::maskMode,    Field::windowLeft, Field::windowRight, Field::sink,
                                Field::aicNum,      Field::aivNum,     Field::l2Bytes,     Field::reservedPlanMode,
                                Field::libapiBytes, Field::ubBytes,    Field::maskPresent, Field::maskShapeType};
    for (auto field : fields) {
        if (At(plan, field) != At(identity, field)) {
            return false;
        }
    }
    constexpr std::pair<Field, Field> regions[] = {
        {Field::dqOffset, Field::dqBytes},     {Field::dkOffset, Field::dkBytes},
        {Field::dvOffset, Field::dvBytes},     {Field::sfmgOffset, Field::sfmgBytes},
        {Field::sinkOffset, Field::sinkBytes}, {Field::specialOffset, Field::specialBytes}};
    int64_t end = 0;
    for (const auto& [offset, bytes] : regions) {
        const int64_t size = At(plan, bytes), begin = At(plan, offset);
        if (size < 0 || begin < 0) {
            return false;
        }
        if (size == 0) {
            continue;
        }
        if (begin < end || begin % 512 != 0 || size % 512 != 0 || begin > At(plan, Field::workspaceBytes) ||
            size > At(plan, Field::workspaceBytes) - begin) {
            return false;
        }
        end = begin + size;
    }
    const int64_t sfmg = At(plan, Field::enablePreSfmg);
    WorkspaceLayout layout;
    if ((sfmg != 0 && sfmg != 1) ||
        !MakeWorkspace(in, static_cast<Split>(At(plan, Field::logicalSplit)), sfmg != 0, layout) ||
        At(plan, Field::workspaceBytes) != layout.bytes || At(plan, Field::scratchStride) != layout.scratchStride) {
        return false;
    }
    const Region expected[] = {layout.dq, layout.dk, layout.dv, layout.sfmg, layout.sink, layout.special};
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
        if (At(plan, regions[i].first) != expected[i].offset || At(plan, regions[i].second) != expected[i].bytes) {
            return false;
        }
    }
    if (in.sink) {
        // ASC pre padding is in FP32 elements, whereas the named workspace
        // region is padded in bytes. It can enter the not-yet-consumed special
        // region (BN2S2) or reserved tail (GS), but must stay in user workspace.
        const auto& r = in.schedule.route;
        const int64_t s1 = (r.maxQ + 127) / 128 * 2;
        const int64_t s2 = (r.maxKv + 127) / 128;
        int64_t axis = 0, elements = 0, paddedBytes = 0;
        if (!CheckedMul(r.batch, s1, axis) || !CheckedMul(axis, s2, axis) || !CheckedMul(axis, r.headsQ, elements) ||
            elements > std::numeric_limits<int64_t>::max() - 512 ||
            !CheckedMul((elements + 512) / 512 * 512, 4, paddedBytes) || At(plan, Field::sinkS1Outer) != s1 ||
            At(plan, Field::sinkS2Outer) != s2 || At(plan, Field::sinkReduceAxis) != axis ||
            paddedBytes > layout.bytes - layout.sink.offset) {
            return false;
        }
    }
    for (size_t i = 0; i < static_cast<size_t>(MAX_CORES); ++i) {
        if (At(plan, Field::blockStarts, i) < 0 || At(plan, Field::blockEnds, i) < At(plan, Field::blockStarts, i) ||
            At(plan, Field::startNeedSyncRound, i) < 0 || At(plan, Field::endNeedSyncRound, i) < -1 ||
            At(plan, Field::separateDkOffset, i) < -1) {
            return false;
        }
    }
    return ValidateScheduleFields(plan, in) && ValidatePrePostFields(plan, in);
}
} // namespace optiling::fag_deter

namespace optiling::fag_deter {
PlanResult BuildMetadataPlans(const PlanInputs& inputs, const int32_t* cuQ, size_t cuQSize, const int32_t* cuKv,
                              size_t cuKvSize, MetadataWords& output)
{
    output.fill(0);
    output[0] = METADATA_MAGIC;
    output[1] = METADATA_VERSION;
    output[2] = METADATA_BYTES;
    output[4] = PLAN_VERSION;
    output[5] = METADATA_ARRAY_WORDS;
    output[6] = METADATA_SCALAR_WORDS;
    output[7] = METADATA_BANK_COUNT;
    output[8] = METADATA_SINK_VARIANTS;
    output[9] = SUPPORTED_DTYPE_MASK;
    output[10] = METADATA_BANK_WORDS;
    for (size_t mask = 0; mask < METADATA_BANK_COUNT; ++mask) {
        // mask_mode=0 never consumes a mask tensor; leave its unused bank invalid.
        if (mask != 0 && inputs.schedule.route.maskMode == 0) {
            continue;
        }
        PlanInputs current = inputs;
        current.schedule.hasMask = mask != 0;
        current.schedule.maskShapeType = mask != 0 ? 2 : 99;
        ScheduleResult schedule;
        const auto error = BuildSchedule(current.schedule, cuQ, cuQSize, cuKv, cuKvSize, schedule);
        if (error != ScheduleError::OK) {
            return {PlanError::SCHEDULE, error};
        }
        const size_t bank = METADATA_HEADER_WORDS + mask * METADATA_BANK_WORDS;
        for (size_t sink = 0; sink < METADATA_SINK_VARIANTS; ++sink) {
            current.sink = sink != 0;
            int64_t capacity = 0;
            if (!WorkspaceCapacity(current, capacity)) {
                return {PlanError::WORKSPACE, ScheduleError::OK};
            }
            PlanWords plan;
            const auto result = BuildPlanFromSchedule(current, schedule, capacity, plan);
            if (result.error != PlanError::OK) {
                return result;
            }
            if (sink == 0) {
                std::copy(plan.begin() + METADATA_ARRAY_BEGIN, plan.begin() + METADATA_ARRAY_END,
                          output.begin() + bank);
            } else if (!std::equal(plan.begin() + METADATA_ARRAY_BEGIN, plan.begin() + METADATA_ARRAY_END,
                                   output.begin() + bank)) {
                return {PlanError::CONTRACT, ScheduleError::OK};
            }
            const size_t scalar = bank + METADATA_ARRAY_WORDS + sink * METADATA_SCALAR_WORDS;
            std::copy(plan.begin(), plan.begin() + METADATA_ARRAY_BEGIN, output.begin() + scalar);
            std::copy(plan.begin() + METADATA_ARRAY_END, plan.end(), output.begin() + scalar + METADATA_ARRAY_BEGIN);
        }
    }
    output[3] = static_cast<int64_t>(PlanStatus::READY);
    return {};
}
} // namespace optiling::fag_deter
