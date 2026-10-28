/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <sstream>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "err/ops_err.h"
#include "mhc_pre_sinkhorn_backward_arch35_tiling.h"
#include "mhc/mhc_pre_sinkhorn_backward/op_kernel/arch35/mhc_pre_sinkhorn_backward_key_arch35.h"

namespace optiling {
using MhcPreSinkhornBackwardNs::SplitMode;

namespace {
constexpr int64_t MAX_N = 8;
constexpr int64_t MIN_C = 128;
constexpr int64_t MAX_C_VALUE = 100000;
constexpr int64_t C_ALIGNMENT = 128;
constexpr int64_t ITERATIONS = 20;
constexpr int64_t M_TILE = 256;
constexpr int64_t MK_BLOCK_ROWS = 1024;
constexpr int64_t K_TILE = 256;
constexpr int64_t C_TILE = 256;
constexpr int64_t C0_ALIGN = 8;
constexpr int64_t CUBE_ALIGN = 16;
constexpr int64_t ROW_TILE = 32;
constexpr int32_t DETERMINISTIC_LEVEL = 3;
constexpr uint64_t WORKSPACE_ALIGN = 32;
constexpr uint32_t INPUT_X = 3;
constexpr uint32_t SUM_OUT = 10;
constexpr uint32_t NORM_OUT = 11;
constexpr uint32_t OUTPUT_GRAD_X = 0;
constexpr uint32_t OUTPUT_GRAD_PHI = 1;
constexpr uint32_t OUTPUT_GRAD_ALPHA = 2;
constexpr uint32_t OUTPUT_GRAD_BIAS = 3;

// MM1/MM2 Cube配置。
constexpr int64_t MM_ROWS_PER_AIV = 16;
constexpr int64_t MM_TILE_ROWS = 2 * MM_ROWS_PER_AIV;
constexpr int64_t X_L1_SLOTS = 2;
constexpr int64_t L1_TOTAL_BYTES = 512 * 1024;
constexpr int64_t MM_BASE_N_MAX = 128;
constexpr int64_t MM_BASE_N_MID = 64;
constexpr int64_t MM_BASE_N_MIN = 32;
constexpr int64_t MM_DEPTH_B1 = 2; // phi B1乒乓深度
constexpr int64_t MM_DB_L0A = 2;   // L0A/B乒乓深度
constexpr int64_t MM_DB_L0C = 2;   // L0C乒乓深度

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

int64_t RoundUp(int64_t value, int64_t align)
{
    return CeilDiv(value, align) * align;
}

} // namespace

bool MhcPreSinkhornBackwardArch35Tiling::IsCapable()
{
    return true;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_->GetNodeName(), "Ascend950 platform information is null"),
                return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    aicNum_ = platform.GetCoreNumAic();
    aivNum_ = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ASize_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0BSize_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize_);
    systemWorkspaceBytes_ = platform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0 || ubSize_ == 0,
                OP_LOGE(context_->GetNodeName(), "Invalid platform resources: aic=%lu, aiv=%lu, ub=%lu", aicNum_,
                        aivNum_, ubSize_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(l1Size_ == 0 || l0ASize_ == 0 || l0BSize_ == 0 || l0CSize_ == 0,
                OP_LOGE(context_->GetNodeName(), "Invalid platform cache resources: l1=%lu, l0a=%lu, l0b=%lu, l0c=%lu",
                        l1Size_, l0ASize_, l0BSize_, l0CSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool MhcPreSinkhornBackwardArch35Tiling::CheckTensorShape(uint32_t index, const char* name, const int64_t* dims,
                                                          size_t rank, bool logError) const
{
    auto inputShape = context_->GetInputShape(index);
    if (inputShape == nullptr) {
        if (logError) {
            OP_LOGE(context_->GetNodeName(), "%s shape is null", name);
        }
        return false;
    }
    const auto& shape = inputShape->GetOriginShape();
    if (shape.GetDimNum() != rank) {
        if (logError) {
            OP_LOGE(context_->GetNodeName(), "%s must have %zu dimensions, got %zu", name, rank, shape.GetDimNum());
        }
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        if (shape.GetDim(i) != dims[i]) {
            if (logError) {
                OP_LOGE(context_->GetNodeName(), "%s dim[%zu] must be %ld, got %ld", name, i, dims[i], shape.GetDim(i));
            }
            return false;
        }
    }
    return true;
}

bool MhcPreSinkhornBackwardArch35Tiling::CheckOutputShape(uint32_t index, const char* name, const int64_t* dims,
                                                          size_t rank, bool logError) const
{
    auto outputShape = context_->GetOutputShape(index);
    if (outputShape == nullptr) {
        if (logError) {
            OP_LOGE(context_->GetNodeName(), "%s output shape is null", name);
        }
        return false;
    }
    const auto& shape = outputShape->GetOriginShape();
    if (shape.GetDimNum() != rank) {
        if (logError) {
            OP_LOGE(context_->GetNodeName(), "%s must have %zu dimensions, got %zu", name, rank, shape.GetDimNum());
        }
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        if (shape.GetDim(i) != dims[i]) {
            if (logError) {
                OP_LOGE(context_->GetNodeName(), "%s dim[%zu] must be %ld, got %ld", name, i, dims[i], shape.GetDim(i));
            }
            return false;
        }
    }
    return true;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::CheckDataTypes() const
{
    auto xDesc = context_->GetInputDesc(INPUT_X);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "x descriptor is null"), return ge::GRAPH_FAILED);
    ge::DataType xType = xDesc->GetDataType();
    OP_CHECK_IF(xType != ge::DT_BF16 && xType != ge::DT_FLOAT16,
                OP_LOGE(context_->GetNodeName(), "x must be BF16 or FP16, got dtype %d", static_cast<int32_t>(xType)),
                return ge::GRAPH_FAILED);

    for (uint32_t index = 0; index <= NORM_OUT; ++index) {
        auto desc = context_->GetInputDesc(index);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context_->GetNodeName(), "input %u descriptor is null", index),
                    return ge::GRAPH_FAILED);
        ge::DataType expected = index == 0 || index == INPUT_X ? xType : ge::DT_FLOAT;
        OP_CHECK_IF(desc->GetDataType() != expected,
                    OP_LOGE(context_->GetNodeName(), "input %u dtype must be %d, got %d", index,
                            static_cast<int32_t>(expected), static_cast<int32_t>(desc->GetDataType())),
                    return ge::GRAPH_FAILED);
    }
    for (uint32_t index = OUTPUT_GRAD_X; index <= OUTPUT_GRAD_BIAS; ++index) {
        auto desc = context_->GetOutputDesc(index);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context_->GetNodeName(), "output %u descriptor is null", index),
                    return ge::GRAPH_FAILED);
        ge::DataType expected = index == OUTPUT_GRAD_X ? xType : ge::DT_FLOAT;
        OP_CHECK_IF(desc->GetDataType() != expected,
                    OP_LOGE(context_->GetNodeName(), "output %u dtype must be %d, got %d", index,
                            static_cast<int32_t>(expected), static_cast<int32_t>(desc->GetDataType())),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::GetShapeAttrsInfo()
{
    tilingData_ = context_->GetTilingData<MhcPreSinkhornBackwardArch35TilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "tiling data is null"),
                return ge::GRAPH_FAILED);
    auto xShapeInfo = context_->GetInputShape(INPUT_X);
    OP_CHECK_IF(xShapeInfo == nullptr, OP_LOGE(context_->GetNodeName(), "x shape is null"), return ge::GRAPH_FAILED);
    const auto& xShape = xShapeInfo->GetOriginShape();
    xRank_ = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF(xRank_ != 3 && xRank_ != 4,
                OP_LOGE(context_->GetNodeName(), "x must be 3D [T,N,C] or 4D [B,S,N,C], got rank %ld", xRank_),
                return ge::GRAPH_FAILED);

    const int64_t batchRank = xRank_ - 2;
    int64_t batchDims[2] = {xShape.GetDim(0), xRank_ == 4 ? xShape.GetDim(1) : 1};
    const int64_t n = xShape.GetDim(xRank_ - 2);
    const int64_t c = xShape.GetDim(xRank_ - 1);
    OP_CHECK_IF(batchDims[0] < 0 || batchDims[1] < 0,
                OP_LOGE(context_->GetNodeName(), "batch dimensions must be non-negative, got %ld and %ld", batchDims[0],
                        batchDims[1]),
                return ge::GRAPH_FAILED);
    const int64_t bs = xRank_ == 4 ? batchDims[0] * batchDims[1] : batchDims[0];
    OP_CHECK_IF(n <= 0 || n > MAX_N,
                OP_LOGE(context_->GetNodeName(), "n must be > 0 and <= %ld, but got %ld", MAX_N, n),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(c < MIN_C || c >= MAX_C_VALUE || c % C_ALIGNMENT != 0,
                OP_LOGE(context_->GetNodeName(), "c must be >= %ld, < %ld and divisible by %ld, but got %ld", MIN_C,
                        MAX_C_VALUE, C_ALIGNMENT, c),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDataTypes() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "MhcPreSinkhornBackward dtype validation failed"),
                return ge::GRAPH_FAILED);

    const int64_t hcMix = n * n + 2 * n;
    const int64_t k = n * c;
    int64_t gradHin[3] = {batchDims[0], batchDims[1], c};
    int64_t gradHPost[3] = {batchDims[0], batchDims[1], n};
    int64_t gradHRes[4] = {batchDims[0], batchDims[1], n, n};
    int64_t gradHResFlat[3] = {batchDims[0], batchDims[1], n * n};
    int64_t xDims[4] = {batchDims[0], batchDims[1], n, c};
    int64_t phiDims[2] = {hcMix, k};
    int64_t alphaDims[1] = {3};
    int64_t biasDims[1] = {hcMix};
    int64_t hPre[3] = {batchDims[0], batchDims[1], n};
    int64_t qDims[3] = {batchDims[0], batchDims[1], hcMix};
    int64_t invDims[3] = {batchDims[0], batchDims[1], 1};
    int64_t sumDims[4] = {2 * ITERATIONS, batchDims[0], batchDims[1], n};
    int64_t normDims[5] = {2 * ITERATIONS, batchDims[0], batchDims[1], n, n};
    if (xRank_ == 3) {
        gradHin[1] = c;
        gradHPost[1] = n;
        gradHRes[1] = n;
        gradHRes[2] = n;
        gradHResFlat[1] = n * n;
        xDims[1] = n;
        xDims[2] = c;
        hPre[1] = n;
        qDims[1] = hcMix;
        invDims[1] = 1;
        sumDims[1] = batchDims[0];
        sumDims[2] = n;
        normDims[1] = batchDims[0];
        normDims[2] = n;
        normDims[3] = n;
    }

    const size_t dataRank = static_cast<size_t>(batchRank + 1);
    const size_t xExpectedRank = static_cast<size_t>(xRank_);
    bool gradHResOk = CheckTensorShape(2, "grad_h_res", gradHRes, xExpectedRank, false) ||
                      CheckTensorShape(2, "grad_h_res", gradHResFlat, dataRank, false);
    if (!gradHResOk) {
        auto gradHResShape = context_->GetInputShape(2);
        OP_LOGE(context_->GetNodeName(),
                "grad_h_res must be [.., %ld, %ld] (rank %zu) or [.., %ld] (rank %zu), got rank %zu", n, n,
                xExpectedRank, n * n, dataRank,
                gradHResShape == nullptr ? static_cast<size_t>(0) : gradHResShape->GetOriginShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(!CheckTensorShape(0, "grad_hin", gradHin, dataRank) ||
                    !CheckTensorShape(1, "grad_h_post", gradHPost, dataRank) ||
                    !CheckTensorShape(INPUT_X, "x", xDims, xExpectedRank) || !CheckTensorShape(4, "phi", phiDims, 2) ||
                    !CheckTensorShape(5, "alpha", alphaDims, 1) || !CheckTensorShape(6, "bias", biasDims, 1) ||
                    !CheckTensorShape(7, "h_pre", hPre, dataRank) ||
                    !CheckTensorShape(8, "hc_before_norm", qDims, dataRank) ||
                    !CheckTensorShape(9, "inv_rms", invDims, dataRank) ||
                    !CheckTensorShape(SUM_OUT, "sum_out", sumDims, dataRank + 1) ||
                    !CheckTensorShape(NORM_OUT, "norm_out", normDims, dataRank + 2),
                OP_LOGE(context_->GetNodeName(), "MhcPreSinkhornBackward input shape validation failed"),
                return ge::GRAPH_FAILED);

    int64_t gradXDims[4] = {batchDims[0], batchDims[1], n, c};
    int64_t gradPhiDims[2] = {hcMix, k};
    int64_t gradAlphaDims[1] = {3};
    int64_t gradBiasDims[1] = {hcMix};
    if (xRank_ == 3) {
        gradXDims[1] = n;
        gradXDims[2] = c;
    }
    OP_CHECK_IF(!CheckOutputShape(OUTPUT_GRAD_X, "grad_x", gradXDims, xExpectedRank) ||
                    !CheckOutputShape(OUTPUT_GRAD_PHI, "grad_phi", gradPhiDims, 2) ||
                    !CheckOutputShape(OUTPUT_GRAD_ALPHA, "grad_alpha", gradAlphaDims, 1) ||
                    !CheckOutputShape(OUTPUT_GRAD_BIAS, "grad_bias", gradBiasDims, 1),
                OP_LOGE(context_->GetNodeName(), "MhcPreSinkhornBackward output shape validation failed"),
                return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attributes are null"), return ge::GRAPH_FAILED);
    auto hcEps = attrs->GetAttrPointer<float>(0);
    tilingData_->hcEps = hcEps == nullptr ? 1.0e-6f : *hcEps;
    OP_CHECK_IF(!std::isfinite(tilingData_->hcEps) || tilingData_->hcEps <= 0.0f,
                OP_LOGE(context_->GetNodeName(), "hc_eps must be finite and positive"), return ge::GRAPH_FAILED);

    tilingData_->bs = bs;
    tilingData_->n = n;
    tilingData_->c = c;
    tilingData_->k = k;
    tilingData_->hcMix = hcMix;
    tilingData_->hcMixAlign = RoundUp(hcMix, C0_ALIGN);
    tilingData_->iterTimes = ITERATIONS;
    tilingData_->normEps = 0.0f;
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcPreSinkhornBackwardArch35Tiling::AlignWorkspace(uint64_t value) const
{
    return (value + WORKSPACE_ALIGN - 1) / WORKSPACE_ALIGN * WORKSPACE_ALIGN;
}

void MhcPreSinkhornBackwardArch35Tiling::SetMmConfig()
{
    const int64_t hcMixAlignRows = RoundUp(tilingData_->hcMix, CUBE_ALIGN);
    // 在L0B容量内优先使用较大的baseN，并保留双缓冲。
    int64_t baseN = MM_BASE_N_MAX;
    int64_t dbL0A = MM_DB_L0A;
    if (hcMixAlignRows * MM_BASE_N_MAX * sizeof(float) * MM_DB_L0A * 2 > static_cast<int64_t>(l0BSize_)) {
        if (hcMixAlignRows * MM_BASE_N_MID * sizeof(float) * MM_DB_L0A * 2 <= static_cast<int64_t>(l0BSize_)) {
            baseN = MM_BASE_N_MID;
        } else if (hcMixAlignRows * MM_BASE_N_MID * sizeof(float) * 1 * 2 <= static_cast<int64_t>(l0BSize_)) {
            baseN = MM_BASE_N_MID;
            dbL0A = 1;
        } else {
            baseN = MM_BASE_N_MIN;
        }
    }
    tilingData_->mmConfigMm1.baseN = static_cast<uint32_t>(baseN);
    tilingData_->mmConfigMm1.l1K = K_TILE;
    tilingData_->mmConfigMm1.depthB1 = MM_DEPTH_B1;
    tilingData_->mmConfigMm1.dbL0A = static_cast<uint32_t>(dbL0A);
    tilingData_->mmConfigMm1.dbL0C = MM_DB_L0C;
    tilingData_->mmConfigMm2 = tilingData_->mmConfigMm1;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::ValidateMmConfig()
{
    constexpr int64_t elementSize = sizeof(float);
    const int64_t hcMixAlignRows = RoundUp(tilingData_->hcMix, CUBE_ALIGN);
    const int64_t mmBaseN = tilingData_->mmConfigMm1.baseN;
    OP_CHECK_IF(K_TILE % mmBaseN != 0,
                OP_LOGE(context_->GetNodeName(), "invalid MM config: K_TILE=%ld, baseN=%ld", K_TILE, mmBaseN),
                return ge::GRAPH_FAILED);

    // L1布局：[GQ缓存波][phi B1乒乓槽][X双缓冲槽]。
    const int64_t phiSlotBytes = hcMixAlignRows * K_TILE * elementSize;
    const int64_t phiRegionBytes = MM_DEPTH_B1 * phiSlotBytes;
    const int64_t xRegionBytes = X_L1_SLOTS * MM_TILE_ROWS * K_TILE * elementSize;
    const int64_t gqWaveBytes = MM_TILE_ROWS * hcMixAlignRows * elementSize;
    const int64_t l1NeededBytes = phiRegionBytes + xRegionBytes + gqWaveBytes;
    OP_CHECK_IF(l1NeededBytes > L1_TOTAL_BYTES || static_cast<int64_t>(l1Size_) < L1_TOTAL_BYTES,
                OP_LOGE(context_->GetNodeName(), "MM config exceeds L1: need=%ld, budget=%ld, platform=%lu",
                        l1NeededBytes, L1_TOTAL_BYTES, l1Size_),
                return ge::GRAPH_FAILED);
    tilingData_->l1SingleBufferElems = static_cast<uint32_t>(phiSlotBytes / elementSize);
    tilingData_->l1BOffsetElems = static_cast<uint32_t>((L1_TOTAL_BYTES - xRegionBytes - phiRegionBytes) / elementSize);

    // L0A/L0B：MM1与MM2各自独立乒乓池，单槽取两侧最大需求。
    const int64_t mm1ASlotBytes = MM_TILE_ROWS * hcMixAlignRows * elementSize;               // GQ
    const int64_t mm1BSlotBytes = hcMixAlignRows * mmBaseN * elementSize;                    // phi分块
    const int64_t mm2ASlotBytes = hcMixAlignRows * MM_TILE_ROWS * elementSize;               // GQ^T
    const int64_t mm2BSlotBytes = RoundUp(MM_TILE_ROWS, CUBE_ALIGN) * mmBaseN * elementSize; // X分块
    const int64_t l0ABSingleBufferBytes =
        std::max(std::max(mm1ASlotBytes, mm1BSlotBytes), std::max(mm2ASlotBytes, mm2BSlotBytes));
    const int64_t l0ABUsedBytes = 2 * static_cast<int64_t>(tilingData_->mmConfigMm1.dbL0A) * l0ABSingleBufferBytes;

    // L0C：MM1乒乓槽 + MM2按N分块累加槽。
    const int64_t mm2L0cChunks = K_TILE / mmBaseN;
    const int64_t mm1CSlotBytes = RoundUp(MM_TILE_ROWS, CUBE_ALIGN) * mmBaseN * elementSize;
    const int64_t mm2CSlotBytes = hcMixAlignRows * mmBaseN * elementSize;
    const int64_t l0CSingleBufferBytes = std::max(mm1CSlotBytes, mm2CSlotBytes);
    const int64_t l0CUsedBytes = (MM_DB_L0C + mm2L0cChunks) * l0CSingleBufferBytes;

    OP_CHECK_IF(l0ABUsedBytes > static_cast<int64_t>(l0ASize_) || l0ABUsedBytes > static_cast<int64_t>(l0BSize_) ||
                    l0CUsedBytes > static_cast<int64_t>(l0CSize_),
                OP_LOGE(context_->GetNodeName(), "MM config exceeds platform memory: L0A/B=%ld/%lu/%lu, L0C=%ld/%lu",
                        l0ABUsedBytes, l0ASize_, l0BSize_, l0CUsedBytes, l0CSize_),
                return ge::GRAPH_FAILED);
    tilingData_->l0ABUsedBytes = static_cast<uint32_t>(l0ABUsedBytes);
    tilingData_->l0ABSingleBufferElems = static_cast<uint32_t>(l0ABSingleBufferBytes / elementSize);
    tilingData_->l0CUsedBytes = static_cast<uint32_t>(l0CUsedBytes);
    tilingData_->l0CSingleBufferElems = static_cast<uint32_t>(l0CSingleBufferBytes / elementSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::DoOpTiling()
{
    const int64_t bs = tilingData_->bs;
    const int64_t k = tilingData_->k;
    const bool empty = bs == 0;
    int64_t mBlockCount = empty ? 1 : std::min<int64_t>(aicNum_, CeilDiv(bs, M_TILE));
    OP_CHECK_IF(mBlockCount <= 0, OP_LOGE(context_->GetNodeName(), "mBlockCount is zero"), return ge::GRAPH_FAILED);
    int64_t kCoreCount = empty ? 1 : static_cast<int64_t>(aicNum_) / mBlockCount;
    const bool deterministic = context_->GetDeterministicLevel() == DETERMINISTIC_LEVEL;
    if (deterministic) {
        kCoreCount = 1;
    } else if (!empty) {
        // Keep K parallelism when M exceeds the cache size; the kernel streams M in bounded chunks.
        const int64_t maxMBlocks = std::max<int64_t>(1, static_cast<int64_t>(aicNum_) / 2);
        const int64_t mTarget = std::min(maxMBlocks, std::max<int64_t>(1, CeilDiv(bs, MK_BLOCK_ROWS)));
        // 选择用核数最多的M分块数；用核数相同时优先较大的M块。
        int64_t bestUsed = -1;
        for (int64_t m = mTarget; m >= 1; --m) {
            const int64_t cores = aicNum_ / m;
            const int64_t kSize = RoundUp(CeilDiv(k, cores), K_TILE);
            const int64_t used = m * CeilDiv(k, kSize);
            if (used > bestUsed) {
                bestUsed = used;
                mBlockCount = m;
                kCoreCount = cores;
            }
            if (bestUsed >= static_cast<int64_t>(aicNum_)) {
                break;
            }
        }
    }
    const int64_t mBlockRows = empty ? 0 : RoundUp(CeilDiv(bs, mBlockCount), ROW_TILE);
    const int64_t kBlockSize = RoundUp(CeilDiv(k, kCoreCount), K_TILE);
    const int64_t kBlockCount = CeilDiv(k, kBlockSize);
    const bool mkSplit = !empty && !deterministic;

    tilingKey_ = deterministic ? MhcPreSinkhornBackwardNs::TILING_KEY_M_SPLIT_DETERMINISTIC :
                 mkSplit       ? MhcPreSinkhornBackwardNs::TILING_KEY_MK_SPLIT :
                                 MhcPreSinkhornBackwardNs::TILING_KEY_M_SPLIT;
    tilingData_->splitMode = static_cast<int64_t>(mkSplit ? SplitMode::MK_SPLIT : SplitMode::M_SPLIT);
    tilingData_->deterministic = deterministic ? 1 : 0;
    tilingData_->mBlockCount = mBlockCount;
    tilingData_->kBlockCount = mkSplit ? kBlockCount : 1;
    tilingData_->mBlockRows = mBlockRows;
    tilingData_->kBlockSize = mkSplit ? kBlockSize : k;

    tilingData_->rowFactor = ROW_TILE;
    tilingData_->rowLoop = CeilDiv(mBlockRows, tilingData_->rowFactor);
    tilingData_->tailRowFactor = empty ? 0 : (bs % ROW_TILE == 0 ? ROW_TILE : bs % ROW_TILE);
    tilingData_->cFactor = C_TILE;
    tilingData_->cLoop = CeilDiv(tilingData_->c, C_TILE);
    tilingData_->tailCFactor = tilingData_->c % C_TILE == 0 ? C_TILE : tilingData_->c % C_TILE;

    tilingData_->mL1Size = std::min<int64_t>(M_TILE, mBlockRows);
    const int64_t mSizeForKL1 = deterministic || empty ? M_TILE : tilingData_->mL1Size;
    tilingData_->kL1Size = std::max<int64_t>(128, std::min<int64_t>(1024, (128 * 256) / mSizeForKL1) / 128 * 128);
    tilingData_->mm1TileM = ROW_TILE;
    tilingData_->mm1TileK = K_TILE;
    tilingData_->mm1LoopK = CeilDiv(tilingData_->kBlockSize, K_TILE);
    tilingData_->usedAic = empty ? 1 : (mkSplit ? mBlockCount * kBlockCount : mBlockCount);
    tilingData_->usedPrepAiv = mkSplit ? std::min<int64_t>(aivNum_, tilingData_->usedAic * 2) : 0;
    tilingData_->gradPhiTileK = K_TILE;
    tilingData_->gradPhiLoopK = tilingData_->mm1LoopK;
    tilingData_->reduceOwnerCount = mBlockCount;
    tilingData_->maxMWave = CeilDiv(mBlockRows, ROW_TILE);

    SetMmConfig();
    if (ValidateMmConfig() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::GetWorkspaceSize()
{
    tilingData_->offGQ = 0;
    tilingData_->offGR = 0;
    tilingData_->offRowPart = 0;
    tilingData_->offMm1Ring = 0;
    tilingData_->offGradPhiPart = 0;
    tilingData_->userWorkspaceBytes = 0;
    if (tilingData_->splitMode == static_cast<int64_t>(SplitMode::MK_SPLIT)) {
        uint64_t offset = 0;
        tilingData_->offGQ = offset;
        offset = AlignWorkspace(offset + tilingData_->bs * tilingData_->hcMix * sizeof(float));
        tilingData_->offGR = offset;
        offset = AlignWorkspace(offset + tilingData_->bs * sizeof(float));
        tilingData_->offRowPart = offset;
        offset = AlignWorkspace(offset + tilingData_->usedPrepAiv * (tilingData_->hcMix + 3) * sizeof(float));
        tilingData_->offGradPhiPart = offset;
        if (tilingData_->mBlockCount > 1) {
            offset = AlignWorkspace(offset + tilingData_->kBlockCount * tilingData_->mBlockCount *
                                                 tilingData_->gradPhiLoopK * tilingData_->hcMix *
                                                 tilingData_->gradPhiTileK * sizeof(float));
        }
        tilingData_->userWorkspaceBytes = offset;
    }
    workspaceBytes_ = systemWorkspaceBytes_ + tilingData_->userWorkspaceBytes;
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcPreSinkhornBackwardArch35Tiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::PostTiling()
{
    OP_CHECK_IF(tilingKey_ == 3002, OP_LOGE(context_->GetNodeName(), "tiling key 3002 is not supported"),
                return ge::GRAPH_FAILED);
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(std::min<uint64_t>(tilingData_->usedAic, aicNum_));
    context_->SetScheduleMode(1);
    context_->SetDynUBufSize(ubSize_);
    size_t* workspaceSizes = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaceSizes == nullptr, OP_LOGE(context_->GetNodeName(), "workspace size output is null"),
                return ge::GRAPH_FAILED);
    workspaceSizes[0] = workspaceBytes_;
    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornBackwardArch35Tiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(),
            "Ascend950 reimplementation: key=%lu, M=%ld, K=%ld, split=%ld, deterministic=%ld, "
            "mBlocks=%ld, kBlocks=%ld, mRows=%ld, kBlock=%ld, cFactor=%ld, userWorkspace=%lu",
            tilingKey_, tilingData_->bs, tilingData_->k, tilingData_->splitMode, tilingData_->deterministic,
            tilingData_->mBlockCount, tilingData_->kBlockCount, tilingData_->mBlockRows, tilingData_->kBlockSize,
            tilingData_->cFactor, tilingData_->userWorkspaceBytes);
    OP_LOGI(context_->GetNodeName(),
            "MM config: baseN=%u, l1K=%u, depthB1=%u, dbL0=[%u,%u], l1BOffset=%u, l1Slot=%u, "
            "l0AB=[%u,%u], l0C=[%u,%u]",
            tilingData_->mmConfigMm1.baseN, tilingData_->mmConfigMm1.l1K, tilingData_->mmConfigMm1.depthB1,
            tilingData_->mmConfigMm1.dbL0A, tilingData_->mmConfigMm1.dbL0C, tilingData_->l1BOffsetElems,
            tilingData_->l1SingleBufferElems, tilingData_->l0ABUsedBytes, tilingData_->l0ABSingleBufferElems,
            tilingData_->l0CUsedBytes, tilingData_->l0CSingleBufferElems);
}

REGISTER_OPS_TILING_TEMPLATE(MhcPreSinkhornBackward, MhcPreSinkhornBackwardArch35Tiling, 0);
} // namespace optiling
