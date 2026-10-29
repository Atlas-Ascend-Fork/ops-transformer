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
 * \file dsa_dense_tiling.cpp
 * \brief
 */

#include "dsa_dense_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "err/ops_err.h"

#include <cmath>
#include <algorithm>

using namespace ge;
using namespace AscendC;

namespace optiling {
namespace {
//// constants ////
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t DB = 2;
constexpr uint32_t CAUSAL_MODE = 1;
constexpr uint32_t CORE_LIST_NUM = 50;

// cv切块size
constexpr uint32_t CUBE_S1_TILE = 128;
constexpr uint32_t CUBE_S2_TILE = 1024;
constexpr uint32_t VEC_S1_TILE = 8;
constexpr uint32_t VEC_S2_TILE = 1024;

// input index
constexpr uint32_t Q_POS = 0;
constexpr uint32_t K_POS = 1;
constexpr uint32_t MAX_POS = 2;
constexpr uint32_t SUM_POS = 3;
constexpr uint32_t Q_LEN_POS = 4;
constexpr uint32_t KV_LEN_POS = 5;
constexpr uint32_t MASK_POS = 6;

constexpr uint32_t Q_INDEX_POS = 7;
constexpr uint32_t K_INDEX_POS = 8;
constexpr uint32_t WEIGHTS_POS = 9;
constexpr uint32_t MAX_INDEX_POS = 10;
constexpr uint32_t SUM_INDEX_POS = 11;

// output index
constexpr uint32_t LOSS_POS = 0;
constexpr uint32_t GRAD_Q_POS = 1;
constexpr uint32_t GRAD_K_POS = 2;
constexpr uint32_t GRAD_WEIGHTS_POS = 3;

// attr index
constexpr uint32_t ATTR_SPARSE_MODE_POS = 0;
constexpr uint32_t ATTR_SCALE_POS = 1;

// func tools
template <typename T1, typename T2>
static T2 CeilDivide(const T1 a, const T2 b)
{
    return static_cast<T2>((a + static_cast<T1>(b) - 1) / static_cast<T1>(b));
}

uint64_t GetSizePlatForm(const platform_ascendc::CoreMemType memType, platform_ascendc::PlatformAscendC ascendcPlatform)
{
    uint64_t size;
    ascendcPlatform.GetCoreMemSize(memType, size);
    return size;
}

struct PlatFormInfo {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    uint32_t aicNum;
    uint32_t aivNum;

    PlatFormInfo() = default;

    explicit PlatFormInfo(platform_ascendc::PlatformAscendC ascendcPlatform)
        : ubSize(GetSizePlatForm(platform_ascendc::CoreMemType::UB, ascendcPlatform)),
          l1Size(GetSizePlatForm(platform_ascendc::CoreMemType::L1, ascendcPlatform)),
          l0CSize(GetSizePlatForm(platform_ascendc::CoreMemType::L0_C, ascendcPlatform)),
          l0ASize(GetSizePlatForm(platform_ascendc::CoreMemType::L0_A, ascendcPlatform)),
          l0BSize(GetSizePlatForm(platform_ascendc::CoreMemType::L0_B, ascendcPlatform)),
          aicNum(ascendcPlatform.GetCoreNumAic()),
          aivNum(ascendcPlatform.GetCoreNumAiv())
    {}
};
} // namespace

class DsaDenseTiling {
public:
    DsaDenseTiling() {}
    ge::graphStatus Init(const gert::TilingContext* context, const platform_ascendc::PlatformAscendC& ascendcPlatform);
    ge::graphStatus DoTiling(gert::TilingContext* context);

    DsaDenseTilingData tilingData;

protected:
    ge::graphStatus DoLibTiling(const gert::TilingContext* context);
    void SetWorkspace(gert::TilingContext* context);
    void SaveTilingInfo(gert::TilingContext* context);

private:
    PlatFormInfo ascInfo_;

    uint32_t B_;
    uint32_t N_;
    uint32_t NI_;
    uint32_t D_;
    uint32_t DI_;
    uint64_t T1_;
    uint64_t T2_;
    uint64_t maxS1_ = 0;
    uint64_t maxS2_ = 0;
    float scale_;
};

ge::graphStatus DsaDenseTiling::Init(const gert::TilingContext* context,
                                     const platform_ascendc::PlatformAscendC& ascendcPlatform)
{
    // parse ascendc platform information
    static const PlatFormInfo ascInfo(ascendcPlatform);
    ascInfo_ = ascInfo;

    // parse shape
    const gert::Shape& qShape = context->GetInputShape(Q_POS)->GetStorageShape();
    const gert::Shape& kShape = context->GetInputShape(K_POS)->GetStorageShape();
    const gert::Shape& maxShape = context->GetInputShape(MAX_POS)->GetStorageShape();
    const gert::Shape& sumShape = context->GetInputShape(SUM_POS)->GetStorageShape();
    const gert::Shape& qLenShape = context->GetInputShape(Q_LEN_POS)->GetStorageShape();
    const gert::Shape& kvLenShape = context->GetInputShape(KV_LEN_POS)->GetStorageShape();
    const gert::Shape& maskShape = context->GetInputShape(MASK_POS)->GetStorageShape();

    const gert::Shape& qIndexShape = context->GetInputShape(Q_INDEX_POS)->GetStorageShape();
    const gert::Shape& kIndexShape = context->GetInputShape(K_INDEX_POS)->GetStorageShape();
    const gert::Shape& weightsShape = context->GetInputShape(WEIGHTS_POS)->GetStorageShape();
    const gert::Shape& maxIndexShape = context->GetInputShape(MAX_INDEX_POS)->GetStorageShape();
    const gert::Shape& sumIndexShape = context->GetInputShape(SUM_INDEX_POS)->GetStorageShape();

    const gert::Shape& lossShape = context->GetOutputShape(LOSS_POS)->GetStorageShape();
    const gert::Shape& gradQShape = context->GetOutputShape(GRAD_Q_POS)->GetStorageShape();
    const gert::Shape& gradKShape = context->GetOutputShape(GRAD_K_POS)->GetStorageShape();
    const gert::Shape& gradWeightsShape = context->GetOutputShape(GRAD_WEIGHTS_POS)->GetStorageShape();

    // check dim
    if (qShape.GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "Layout of q is not TND");
        return ge::GRAPH_FAILED;
    }
    if (kShape.GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "Layout of k is not TND");
        return ge::GRAPH_FAILED;
    }
    if (maxShape.GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "Layout of max is not NT");
        return ge::GRAPH_FAILED;
    }
    if (sumShape.GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "Layout of sum is not NT");
        return ge::GRAPH_FAILED;
    }
    if (qLenShape.GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "Dim of q_len is not 1");
        return ge::GRAPH_FAILED;
    }
    if (kvLenShape.GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "Dim of kv_len is not 1");
        return ge::GRAPH_FAILED;
    }
    if (maskShape.GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "Dim of mask is not 2");
        return ge::GRAPH_FAILED;
    }
    if (qIndexShape.GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "Layout of q_index is not TND");
        return ge::GRAPH_FAILED;
    }
    if (kIndexShape.GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "Layout of k_index is not TD");
        return ge::GRAPH_FAILED;
    }
    if (weightsShape.GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "Layout of weights is not TN");
        return ge::GRAPH_FAILED;
    }
    if (maxIndexShape.GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "Layout of max_index is not T");
        return ge::GRAPH_FAILED;
    }
    if (sumIndexShape.GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "Layout of sum_index is not T");
        return ge::GRAPH_FAILED;
    }

    if (lossShape.GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "Dim of loss is not 1");
        return ge::GRAPH_FAILED;
    }
    if (gradQShape.GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "Layout of grad q is not TND");
        return ge::GRAPH_FAILED;
    }
    if (gradKShape.GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "Layout of grad k is not TD");
        return ge::GRAPH_FAILED;
    }
    if (gradWeightsShape.GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "Layout of grad weights is not NT");
        return ge::GRAPH_FAILED;
    }
    B_ = qLenShape.GetDim(0) - 1;
    T1_ = qShape.GetDim(0);
    N_ = qShape.GetDim(1);
    D_ = qShape.GetDim(2);
    T2_ = kShape.GetDim(0);
    NI_ = qIndexShape.GetDim(1);
    DI_ = qIndexShape.GetDim(2);

    // parse attrs
    auto attrs = context->GetAttrs();
    int sparseMode = *(attrs->GetAttrPointer<int>(ATTR_SPARSE_MODE_POS));
    if (sparseMode != CAUSAL_MODE) {
        OP_LOGE(context->GetNodeName(), "Only support causal mode.");
        return ge::GRAPH_PARAM_INVALID;
    }
    scale_ = *(attrs->GetAttrPointer<float>(ATTR_SCALE_POS));

    // shape value check
    if (kShape.GetDim(1) != N_ || kShape.GetDim(2) != D_) {
        OP_LOGE(context->GetNodeName(), "Shape mismatch: k and q have incompatible dimensions");
        return ge::GRAPH_FAILED;
    }
    if (maxShape.GetDim(0) != N_ || maxShape.GetDim(1) != T1_) {
        OP_LOGE(context->GetNodeName(), "Shape of max tensor is not [N T1]");
        return ge::GRAPH_FAILED;
    }
    if (sumShape.GetDim(0) != N_ || sumShape.GetDim(1) != T1_) {
        OP_LOGE(context->GetNodeName(), "Shape of sum tensor is not [N T1]");
        return ge::GRAPH_FAILED;
    }
    if (kvLenShape.GetDim(0) != (B_ + 1)) {
        OP_LOGE(context->GetNodeName(), "Shape of kv_len tensor is not [B+1,]");
        return ge::GRAPH_FAILED;
    }
    if (maskShape.GetDim(0) != CUBE_S1_TILE || maskShape.GetDim(1) != CUBE_S1_TILE) {
        OP_LOGE(context->GetNodeName(), "Shape of mask tensor is not [CUBE_S1_TILE CUBE_S1_TILE]");
        return ge::GRAPH_FAILED;
    }

    if (kIndexShape.GetDim(0) != T2_ || kIndexShape.GetDim(1) != DI_) {
        OP_LOGE(context->GetNodeName(), "Shape of k_index tensor is not [T2 DI]");
        return ge::GRAPH_FAILED;
    }
    if (weightsShape.GetDim(0) != T1_ || weightsShape.GetDim(1) != NI_) {
        OP_LOGE(context->GetNodeName(), "Shape of weights tensor is not [T1 NI]");
        return ge::GRAPH_FAILED;
    }
    if (maxIndexShape.GetDim(0) != T1_) {
        OP_LOGE(context->GetNodeName(), "Shape of max_index tensor is not [T1]");
        return ge::GRAPH_FAILED;
    }
    if (sumIndexShape.GetDim(0) != T1_) {
        OP_LOGE(context->GetNodeName(), "Shape of sum_index tensor is not [T1]");
        return ge::GRAPH_FAILED;
    }

    if (lossShape.GetDim(0) != 1) {
        OP_LOGE(context->GetNodeName(), "Shape of loss tensor is not [1]");
        return ge::GRAPH_FAILED;
    }
    if (gradQShape.GetDim(0) != T1_ || gradQShape.GetDim(1) != NI_ || gradQShape.GetDim(2) != DI_) {
        OP_LOGE(context->GetNodeName(), "Shape mismatch: grad_q and q_index have incompatible dimensions");
        return ge::GRAPH_FAILED;
    }
    if (gradKShape.GetDim(0) != T2_ || gradKShape.GetDim(1) != DI_) {
        OP_LOGE(context->GetNodeName(), "Shape mismatch: grad_k and k_index have incompatible dimensions");
        return ge::GRAPH_FAILED;
    }
    if (gradWeightsShape.GetDim(0) != NI_ || gradWeightsShape.GetDim(1) != T1_) {
        OP_LOGE(context->GetNodeName(), "Shape mismatch: grad_weights and weights have incompatible dimensions");
        return ge::GRAPH_FAILED;
    }

    // check head num
    if (NI_ > CUBE_S1_TILE) {
        OP_LOGE(context->GetNodeName(), "Not support NI > CUBE_S1_TILE currently");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// TODO: 考虑db? depth等信息？
ge::graphStatus DsaDenseTiling::DoLibTiling(const gert::TilingContext* context)
{
    matmul_tiling::MatmulApiTiling mm1;
    matmul_tiling::MatmulApiTiling mm2;
    matmul_tiling::MatmulApiTiling mm3;

    // setting mm1
    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                 false);
    mm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                 true);
    mm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm1.SetOrgShape(maxS1_, maxS2_, D_); // tiling阶段随意设置，kernel运行时再设置为准确值
    mm1.SetShape(CUBE_S1_TILE, CUBE_S2_TILE, D_);
    mm1.SetBias(false);
    mm1.SetFixSplit(CUBE_S1_TILE, 128, -1); // TODO: 这里是影响应能的关键，可参考FAG
    if (mm1.GetTiling(tilingData.mm1TilingData) == -1) {
        OP_LOGE(context->GetNodeName(), "mm1 getTiling failed.");
        return ge::GRAPH_FAILED;
    }
    tilingData.mm1TilingData.set_shareMode(0);
    tilingData.mm1TilingData.set_shareL1Size(ascInfo_.l1Size);
    tilingData.mm1TilingData.set_shareL0CSize(ascInfo_.l0CSize);

    // setting mm2
    mm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                 true);
    mm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                 false);
    mm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm2.SetOrgShape(maxS2_, DI_, maxS1_); // tiling阶段随意设置，kernel运行时再设置为准确值
    mm2.SetShape(CUBE_S2_TILE, DI_, CUBE_S1_TILE);
    mm2.SetBias(false);
    mm2.SetFixSplit(-1, -1, -1);
    if (mm2.GetTiling(tilingData.mm2TilingData) == -1) {
        OP_LOGE(context->GetNodeName(), "mm2 getTiling failed.");
        return ge::GRAPH_FAILED;
    }
    tilingData.mm2TilingData.set_shareMode(0);
    tilingData.mm2TilingData.set_shareL1Size(ascInfo_.l1Size);
    tilingData.mm2TilingData.set_shareL0CSize(ascInfo_.l0CSize);

    // setting mm3
    mm3.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                 false);
    mm3.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                 false);
    mm3.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm3.SetOrgShape(maxS1_, DI_, maxS2_); // tiling阶段随意设置，kernel运行时再设置为准确值
    mm3.SetShape(CUBE_S1_TILE, DI_, CUBE_S2_TILE);
    mm3.SetBias(false);
    mm3.SetFixSplit(-1, -1, -1);
    if (mm3.GetTiling(tilingData.mm3TilingData) == -1) {
        OP_LOGE(context->GetNodeName(), "mm3 getTiling failed.");
        return ge::GRAPH_FAILED;
    }
    tilingData.mm3TilingData.set_shareMode(0);
    tilingData.mm3TilingData.set_shareL1Size(ascInfo_.l1Size);
    tilingData.mm3TilingData.set_shareL0CSize(ascInfo_.l0CSize);

    return ge::GRAPH_SUCCESS;
}

void DsaDenseTiling::SetWorkspace(gert::TilingContext* context)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    uint64_t qkSpace = CUBE_S1_TILE * CUBE_S2_TILE * sizeof(float) * DB;    // P、Y两个计算图复用
    uint64_t pSpace = CUBE_S1_TILE * CUBE_S2_TILE * sizeof(float);          // 保存P
    uint64_t ySpace = CUBE_S1_TILE * CUBE_S2_TILE * sizeof(float);          // 保存Y
    uint64_t reluSpace = CUBE_S1_TILE * CUBE_S2_TILE * sizeof(float) * NI_; // 保存S'
    // uint64_t reluGradSpace = CUBE_S1_TILE * CUBE_S2_TILE * sizeof(float) * DB; // 复用qkSpace
    uint64_t dwSpace = T1_ * NI_ * sizeof(float);
    // kl loss workspace复用qkSpace
    uint64_t dkSpace = T2_ * DI_ * sizeof(float);
    uint64_t dqSpace = T1_ * NI_ * DI_ * sizeof(float);

    workspaces[0] =
        SYS_WORKSPACE_SIZE + ascInfo_.aivNum * (qkSpace + pSpace + ySpace + reluSpace + dkSpace) + dwSpace + dqSpace;

    tilingData.baseParams.set_qkSpaceBytes(qkSpace);
    tilingData.baseParams.set_pSpaceBytes(pSpace);
    tilingData.baseParams.set_ySpaceBytes(ySpace);
    tilingData.baseParams.set_reluSpaceBytes(reluSpace);
    tilingData.baseParams.set_dwSpaceBytes(dwSpace);
    tilingData.baseParams.set_dkSpaceBytes(dkSpace);
    tilingData.baseParams.set_dqSpaceBytes(dqSpace);
}

void DsaDenseTiling::SaveTilingInfo(gert::TilingContext* context)
{
    context->SetBlockDim(ascInfo_.aicNum); // mix算子按aic num发射《aicore》
    context->SetTilingKey(0);
    context->SetScheduleMode(1); // 1: batchmode模式

    tilingData.baseParams.set_B(B_);
    tilingData.baseParams.set_N(N_);
    tilingData.baseParams.set_NI(NI_);
    tilingData.baseParams.set_D(D_);
    tilingData.baseParams.set_DI(DI_);
    tilingData.baseParams.set_T1(T1_);
    tilingData.baseParams.set_T2(T2_);
    tilingData.baseParams.set_maxS1(maxS1_);
    tilingData.baseParams.set_maxS2(maxS2_);
    tilingData.baseParams.set_scale(scale_);
    tilingData.baseParams.set_cubeS1Tile(CUBE_S1_TILE);
    tilingData.baseParams.set_cubeS2Tile(CUBE_S2_TILE);
    tilingData.baseParams.set_vecS1Tile(VEC_S1_TILE);
    tilingData.baseParams.set_vecS2Tile(VEC_S2_TILE);
    tilingData.baseParams.set_aivNum(ascInfo_.aivNum);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

ge::graphStatus DsaDenseTiling::DoTiling(gert::TilingContext* context)
{
    if (DoLibTiling(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "DoLibTiling failed");
        return ge::GRAPH_FAILED;
    }

    SetWorkspace(context);

    SaveTilingInfo(context);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingDsaDense(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("DsaDense", "Context is nullptr."), return ge::GRAPH_FAILED);
    DsaDenseTiling tilingObj;
    platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
    if (tilingObj.Init(context, ascendcPlatform) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling Init() failed.");
        return ge::GRAPH_FAILED;
    }
    return tilingObj.DoTiling(context);
}

IMPL_OP_OPTILING(DsaDense).Tiling(TilingDsaDense);

} // namespace optiling
