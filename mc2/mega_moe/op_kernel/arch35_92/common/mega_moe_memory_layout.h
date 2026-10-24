/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_MEMORY_LAYOUT_H
#define MEGA_MOE_MEMORY_LAYOUT_H

#include <cstdint>

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__) || defined(__DAV_310R6_CUBE__) || \
    defined(__DAV_310R6_VEC__) || defined(__DAV_920R1_CUBE__) || defined(__DAV_920R1_VEC__) || defined(__NPU_ARCH__)
#define HOST_DEVICE __forceinline__[aicore]
#else
#define HOST_DEVICE inline
#endif

namespace MegaMoeImpl {

// 本头只计算容量和偏移，供容量申请、tiling 校验和核内地址绑定共同使用。
namespace MemoryLayoutDetail {
constexpr int64_t ALIGN_32 = 32;
constexpr int64_t ALIGN_256 = 256;
constexpr int64_t ALIGN_512 = 512;
constexpr int64_t INT_CACHELINE = 16;
constexpr int64_t INT32_PER_32B = ALIGN_32 / sizeof(int32_t);
constexpr int64_t BITS_PER_BYTE = 8;
constexpr int64_t MXFP_DIVISOR_SIZE = 64;
constexpr int64_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t L1_TILE_M_256 = 256;
constexpr int64_t ACTIVATION_N_HALF = 2;
constexpr int64_t DEDUP_PREPASS_MAX_CHUNKS = 128;
constexpr int64_t DEDUP_ROW_DESC_HEADER_INT32 = 2;
constexpr int64_t MAX_INT16_TOPK_INDEX_COUNT = 1LL << 15;

HOST_DEVICE int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

HOST_DEVICE int64_t CeilAlign(int64_t value, int64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}
} // namespace MemoryLayoutDetail

constexpr int64_t EXCEPTION_DUMP_REGION_SIZE = 60 * 1024LL;
constexpr int64_t SIZE_INT_8 = 1;
constexpr int64_t SIZE_INT_32 = 4;
constexpr int64_t SIZE_BF_16 = 2;
constexpr int64_t INVALID_WORKSPACE_OFFSET = -1;
constexpr uint32_t MTE_RING_WAVE_COUNT = 3U;

// 原同步布局：前 48 KiB 是 rank 槽位，随后 12 KiB 是各 AIV 的同步计数区。
// URMA 的 rank 槽位超过预留区时动态扩展；MTE 保持原 60 KiB 固定布局。
constexpr int64_t PEERMEM_MIN_RANK_SYNC_SIZE = 1024 * 48LL;
constexpr int64_t PEERMEM_DATA_OFFSET = 1024 * 60LL;
// MTE count 区固定预留 2048 个 512B 槽，每槽首个 int32 有效。
constexpr int64_t PEERMEM_MTE_COUNT_REGION_SIZE = 512LL * 2048LL;
constexpr int64_t PEERMEM_SYNC_COUNT_REGION_SIZE = PEERMEM_DATA_OFFSET - PEERMEM_MIN_RANK_SYNC_SIZE;
constexpr int64_t PEERMEM_SYNC_SLOT_SIZE = static_cast<int64_t>(MemoryLayoutDetail::INT_CACHELINE) * sizeof(int32_t);

// 通信拓扑：决定窗口数据区按 MTE 还是 URMA 布局推进。
constexpr int64_t TOPO_TYPE_MTE = 0U;
constexpr int64_t TOPO_TYPE_URMA = 1U;
constexpr int64_t TOPO_TYPE_URMA_ROCE = 2U;

HOST_DEVICE bool IsLayeredTopology(int64_t topo)
{
    return topo == TOPO_TYPE_URMA || topo == TOPO_TYPE_URMA_ROCE;
}

HOST_DEVICE int64_t CalcUrmaRankSyncSize(int64_t epWorldSize)
{
    int64_t rankSyncSize = epWorldSize * PEERMEM_SYNC_SLOT_SIZE;
    return rankSyncSize > PEERMEM_MIN_RANK_SYNC_SIZE ? rankSyncSize : PEERMEM_MIN_RANK_SYNC_SIZE;
}

// ============================ 各区尺寸（host / device 共用） ============================

// URMA layered 路径仍使用 mask 槽；MTE WAVE 不再经过该布局。
HOST_DEVICE int64_t CalcDispatchMaskAlignSize(int64_t numMaxTokensPerRank, int64_t topK)
{
    int64_t sendTotalNum = numMaxTokensPerRank * topK;
    int64_t alignedRouteCount = MemoryLayoutDetail::CeilAlign(sendTotalNum * static_cast<int64_t>(sizeof(int32_t)),
                                                              MemoryLayoutDetail::ALIGN_256) /
                                static_cast<int64_t>(sizeof(int32_t));
    return MemoryLayoutDetail::CeilAlign(alignedRouteCount / static_cast<int64_t>(MemoryLayoutDetail::BITS_PER_BYTE),
                                         MemoryLayoutDetail::ALIGN_32);
}

// 所有 Rank 使用一致的 numMaxTokensPerRank，因此可据此统一选择 int16 或 int32 编码。
HOST_DEVICE bool UseInt16TopkIndex(int64_t numMaxTokensPerRank, int64_t topK)
{
    return topK > 0 && numMaxTokensPerRank >= 0 &&
           numMaxTokensPerRank * topK <= MemoryLayoutDetail::MAX_INT16_TOPK_INDEX_COUNT;
}

HOST_DEVICE int64_t CalcTopkIndexTypeBytes(int64_t numMaxTokensPerRank, int64_t topK)
{
    return UseInt16TopkIndex(numMaxTokensPerRank, topK) ? static_cast<int64_t>(sizeof(int16_t)) :
                                                          static_cast<int64_t>(sizeof(int32_t));
}

// 每个 (localExpert, srcRank) 槽直接保存有序 topkIndex。
// 同一 token 的 topK expert id 不重复，因此单专家从一张源卡最多接收 numMaxTokensPerRank 个 index。
// route index 接收区不携带槽尾 count；MTE count 使用独立的 512B 间隔表。
HOST_DEVICE int64_t CalcRouteIndexRecvSize(int64_t routeIndexAlignSize, int64_t moeExpertPerRank, int64_t epWorldSize)
{
    return MemoryLayoutDetail::CeilAlign(moeExpertPerRank * epWorldSize * routeIndexAlignSize,
                                         MemoryLayoutDetail::ALIGN_512);
}

HOST_DEVICE int64_t CalcMaskRecvSize(int64_t maskAlignSize, int64_t moeExpertPerRank, int64_t epWorldSize)
{
    int64_t maskSlotSize = maskAlignSize + MemoryLayoutDetail::ALIGN_32;
    return MemoryLayoutDetail::CeilAlign(moeExpertPerRank * epWorldSize * maskSlotSize, MemoryLayoutDetail::ALIGN_512);
}

// URMA 的独立 raw count 表仍按 [localExpert][sourceRank] 连续排列；不使用 MTE slot 布局。
HOST_DEVICE int64_t CalcExpertCountRecvSize(int64_t moeExpertPerRank, int64_t epWorldSize)
{
    return MemoryLayoutDetail::CeilAlign(moeExpertPerRank * epWorldSize * static_cast<int64_t>(sizeof(int32_t)),
                                         MemoryLayoutDetail::ALIGN_512);
}

// dedupMode 的值就是门控后实际开启的去重模式：
// 0 = dispatch 关、combine 关；1 = dispatch 关、combine 开；
// 2 = dispatch 开、combine 关；3 = dispatch 开、combine 开；4 = URMA Server Combine。
// 去重判断由本头统一提供，host 和 kernel 不再各自比较模式数值。
constexpr int32_t DEDUP_MODE_SERVER_COMBINE = 4;
HOST_DEVICE constexpr bool IsDispatchDedupOn(int32_t dedupMode)
{
    return dedupMode == 2 || dedupMode == 3;
}

HOST_DEVICE constexpr bool IsCombineDedupOn(int32_t dedupMode)
{
    return dedupMode == 1 || dedupMode == 3;
}

HOST_DEVICE constexpr bool IsServerCombineDedupOn(int32_t dedupMode)
{
    return dedupMode == DEDUP_MODE_SERVER_COMBINE;
}

HOST_DEVICE int64_t CalcServerCombineReadyRowBytes(int64_t capacity)
{
    // 每个 token 一份 uint64 位图，末尾另存 count+1，整行按 32B 对齐。
    return MemoryLayoutDetail::CeilAlign((capacity + 1) * static_cast<int64_t>(sizeof(uint64_t)),
                                         MemoryLayoutDetail::ALIGN_32);
}

HOST_DEVICE int64_t CalcServerCombineTokenListStrideBytes(int64_t capacity)
{
    // 每个源 Server 紧凑保存 int32 count+1 和最多 capacity 个 token ID。
    return (capacity + 1) * static_cast<int64_t>(sizeof(int32_t));
}

// 单 token 的量化记录字节数 = 量化数据 + mx scale（prefetch 场景再拼 topk 权重），按 32B 对齐。
HOST_DEVICE int64_t CalcQuantTokenScaleBytes(int64_t h, uint32_t elemsPerByte, int64_t topK, bool topkWeightsPrefetch)
{
    uint32_t mxScaleNum =
        MemoryLayoutDetail::CeilDiv(static_cast<uint32_t>(h), static_cast<uint32_t>(MemoryLayoutDetail::ALIGN_32));
    uint32_t dataBytes = MemoryLayoutDetail::CeilAlign(static_cast<uint32_t>(h) / elemsPerByte,
                                                       static_cast<uint32_t>(MemoryLayoutDetail::ALIGN_256)) *
                         static_cast<uint32_t>(sizeof(int8_t));
    uint32_t scaleBytes = mxScaleNum * static_cast<uint32_t>(sizeof(int8_t));
    uint32_t tokenScaleBytes =
        MemoryLayoutDetail::CeilAlign(dataBytes + scaleBytes, static_cast<uint32_t>(MemoryLayoutDetail::ALIGN_32));
    if (topkWeightsPrefetch) {
        uint32_t weightBytes =
            MemoryLayoutDetail::CeilAlign(static_cast<uint32_t>(topK) * static_cast<uint32_t>(sizeof(float)),
                                          static_cast<uint32_t>(MemoryLayoutDetail::ALIGN_32));
        tokenScaleBytes = MemoryLayoutDetail::CeilAlign(tokenScaleBytes + weightBytes,
                                                        static_cast<uint32_t>(MemoryLayoutDetail::ALIGN_32));
    }
    return static_cast<int64_t>(tokenScaleBytes);
}

// 去重 rowDesc 每行的 int32 数：头 2 个 + 成员三元组 3*topK 个，向上对齐到 8 个 int32（32B），
// 使描述符批量搬运天然 32B 对齐。字段语义见 mega_moe_constants.h 的 DEDUP_ROW_* 注释。
HOST_DEVICE int64_t CalcDedupRowDescStrideInt32(int64_t topK)
{
    int64_t rawInt32 = static_cast<int64_t>(MemoryLayoutDetail::DEDUP_ROW_DESC_HEADER_INT32) + 3LL * topK;
    return MemoryLayoutDetail::CeilAlign(rawInt32, static_cast<int64_t>(MemoryLayoutDetail::INT32_PER_32B));
}

// combine 接收区单 token 记录字节数：须与 kernel InitCombineBuffers 的 combine record 布局一致。
HOST_DEVICE int64_t CalcCombineTokenBytes(int64_t h, int64_t yDtypeSize, bool isQuantCombine)
{
    if (!isQuantCombine) {
        return h * yDtypeSize;
    }
    int64_t tokenStorageBytes = MemoryLayoutDetail::CeilAlign(h, MemoryLayoutDetail::ALIGN_256);
    int64_t scaleCount = MemoryLayoutDetail::CeilDiv(h, static_cast<int64_t>(MemoryLayoutDetail::MXFP_SCALE_GROUP_NUM));
    int64_t storedScaleBytes =
        MemoryLayoutDetail::CeilAlign(scaleCount, static_cast<int64_t>(MemoryLayoutDetail::MXFP_MULTI_BASE_SIZE));
    return MemoryLayoutDetail::CeilAlign(tokenStorageBytes + storedScaleBytes, MemoryLayoutDetail::ALIGN_32);
}

// CCL 通信数据及协议状态的布局参数，不依赖 tiling 类型。
struct CclCommLayoutParams {
    int64_t numMaxTokensPerRank;
    int64_t topK;
    int64_t h;
    int64_t moeExpertPerRank;
    int64_t epWorldSize;
    int64_t yDtypeSize;
    uint32_t elemsPerByte;
    bool topkWeightsPrefetch;
    bool isQuantCombine;
    int64_t topoType;
    int64_t serverNum;
    bool serverCombineReduce{false};
};

// 纯布局只记录区域的偏移和容量，核内适配层再绑定 GM 地址。
struct MemoryRegionLayout {
    int64_t offset{-1};
    int64_t size{0};
};

// CCL 通信布局的尺寸与偏移，容量申请、主机校验和核内地址绑定共用。
template <typename Region>
struct CclCommLayoutBase {
    Region rankSync;
    Region syncCount;
    Region maskRecv;
    Region expertCountRecv;
    Region quantTokenScale;
    Region dispatchReceive;
    Region dispatchFlag;
    Region sourceReady;
    Region contributionReady;
    Region combineSend;
    Region contribution;
    Region combineReceive;
    int64_t totalSize{0};

    HOST_DEVICE Region AppendRegion(int64_t bytes)
    {
        Region region{totalSize, bytes};
        totalSize += bytes;
        return region;
    }
};

template <typename Region = MemoryRegionLayout>
HOST_DEVICE CclCommLayoutBase<Region> CalcMteCclCommLayout(const CclCommLayoutParams& params)
{
    CclCommLayoutBase<Region> sizes{};
    sizes.rankSync = sizes.AppendRegion(PEERMEM_MIN_RANK_SYNC_SIZE);
    sizes.syncCount = sizes.AppendRegion(PEERMEM_SYNC_COUNT_REGION_SIZE);
    const int64_t routeIndexAlignSize = MemoryLayoutDetail::CeilAlign(
        params.numMaxTokensPerRank * CalcTopkIndexTypeBytes(params.numMaxTokensPerRank, params.topK),
        MemoryLayoutDetail::ALIGN_32);
    const int64_t maskRecvSize =
        CalcRouteIndexRecvSize(routeIndexAlignSize, params.moeExpertPerRank, params.epWorldSize);
    sizes.expertCountRecv = sizes.AppendRegion(PEERMEM_MTE_COUNT_REGION_SIZE);
    sizes.maskRecv = sizes.AppendRegion(maskRecvSize);
    const int64_t tokenScaleBytes =
        CalcQuantTokenScaleBytes(params.h, params.elemsPerByte, params.topK, params.topkWeightsPrefetch);
    const int64_t dispatchRecordAreaSize =
        MemoryLayoutDetail::CeilAlign(params.numMaxTokensPerRank * tokenScaleBytes, MemoryLayoutDetail::ALIGN_512);
    sizes.quantTokenScale = sizes.AppendRegion(dispatchRecordAreaSize);
    const int64_t combineTokenBytes = CalcCombineTokenBytes(params.h, params.yDtypeSize, params.isQuantCombine);
    const int64_t combineSendSize = MemoryLayoutDetail::CeilAlign(
        params.numMaxTokensPerRank * params.topK * combineTokenBytes, MemoryLayoutDetail::ALIGN_512);
    sizes.combineSend = sizes.AppendRegion(combineSendSize);
    return sizes;
}

template <typename Region = MemoryRegionLayout>
HOST_DEVICE CclCommLayoutBase<Region> CalcUrmaCclCommLayout(const CclCommLayoutParams& params)
{
    CclCommLayoutBase<Region> sizes{};
    sizes.rankSync = sizes.AppendRegion(CalcUrmaRankSyncSize(params.epWorldSize));
    sizes.syncCount = sizes.AppendRegion(PEERMEM_SYNC_COUNT_REGION_SIZE);
    sizes.totalSize = MemoryLayoutDetail::CeilAlign(sizes.totalSize, MemoryLayoutDetail::ALIGN_512);
    const int64_t maskRecvSize = CalcMaskRecvSize(CalcDispatchMaskAlignSize(params.numMaxTokensPerRank, params.topK),
                                                  params.moeExpertPerRank, params.epWorldSize);
    sizes.maskRecv = sizes.AppendRegion(maskRecvSize);
    sizes.expertCountRecv = sizes.AppendRegion(CalcExpertCountRecvSize(params.moeExpertPerRank, params.epWorldSize));
    const int64_t tokenScaleBytes = CalcQuantTokenScaleBytes(params.h, params.elemsPerByte, params.topK,
                                                             params.topkWeightsPrefetch || params.serverCombineReduce);
    const int64_t relayRecordBytes = MemoryLayoutDetail::CeilAlign(tokenScaleBytes, MemoryLayoutDetail::ALIGN_512);
    const int64_t relayDataSize = MemoryLayoutDetail::CeilAlign(
        params.numMaxTokensPerRank * relayRecordBytes * params.serverNum, MemoryLayoutDetail::ALIGN_512);
    const int64_t readyRowBytes = params.serverCombineReduce ?
                                      CalcServerCombineReadyRowBytes(params.numMaxTokensPerRank) :
                                      params.numMaxTokensPerRank * static_cast<int64_t>(sizeof(uint64_t));
    const int64_t relayFlagSize =
        MemoryLayoutDetail::CeilAlign(params.serverNum * readyRowBytes, MemoryLayoutDetail::ALIGN_512);
    sizes.dispatchReceive = sizes.AppendRegion(relayDataSize);
    sizes.dispatchFlag = sizes.AppendRegion(relayFlagSize);
    const int64_t combineTokenBytes = CalcCombineTokenBytes(params.h, params.yDtypeSize, params.isQuantCombine);
    if (params.serverCombineReduce) {
        // 三个状态区连续分配，供 Layered 入口统一清零。
        sizes.sourceReady = sizes.AppendRegion(
            MemoryLayoutDetail::CeilAlign(params.serverNum * readyRowBytes, MemoryLayoutDetail::ALIGN_512));
        sizes.contributionReady = sizes.AppendRegion(MemoryLayoutDetail::CeilAlign(
            params.serverNum * params.numMaxTokensPerRank * params.topK * static_cast<int64_t>(sizeof(uint64_t)),
            MemoryLayoutDetail::ALIGN_512));
        const int64_t tokenBytes = params.h * SIZE_BF_16;
        sizes.contribution = sizes.AppendRegion(MemoryLayoutDetail::CeilAlign(
            params.serverNum * params.numMaxTokensPerRank * params.topK * tokenBytes, MemoryLayoutDetail::ALIGN_512));
        sizes.combineReceive = sizes.AppendRegion(MemoryLayoutDetail::CeilAlign(
            params.serverNum * params.numMaxTokensPerRank * tokenBytes, MemoryLayoutDetail::ALIGN_512));
    } else {
        const int64_t combineSendSize = MemoryLayoutDetail::CeilAlign(
            params.numMaxTokensPerRank * params.topK * combineTokenBytes, MemoryLayoutDetail::ALIGN_512);
        sizes.combineSend = sizes.AppendRegion(combineSendSize);
    }
    return sizes;
}

template <typename Region = MemoryRegionLayout>
HOST_DEVICE CclCommLayoutBase<Region> CalcCclCommLayout(const CclCommLayoutParams& params)
{
    return params.topoType == TOPO_TYPE_MTE ? CalcMteCclCommLayout<Region>(params) :
                                              CalcUrmaCclCommLayout<Region>(params);
}

using CclCommLayout = CclCommLayoutBase<MemoryRegionLayout>;

// 各阶段的循环配置同时用于容量规划与核内寻址，不再嵌套判断总开关。
struct CclStageRingConfig {
    bool dispatch{false};
    bool gmm1{false};
    bool activation{false};
    bool gmm2{false};
    uint64_t rows{0};
};

// CCL 本地计算数据及辅助状态的布局参数；未知特征由容量申请侧取安全上界。
struct CclComputeLayoutParams {
    CclStageRingConfig ring{};
    int64_t topoType{TOPO_TYPE_MTE};
    int64_t maxOutputSize{0};
    int64_t h{0};
    int64_t hiddenDim{0};
    uint32_t activationOutputDivisor{MemoryLayoutDetail::ACTIVATION_N_HALF};
    int64_t bs{0};               // 本卡真实 token 数。
    int64_t moeExpertPerRank{0}; // 本卡参与 topK 路由的专家数，与 weight1 的专家数一致。
    int64_t epWorldSize{0};
    int64_t aicNum{0};
    int64_t sharedExpertNum{0};   // 独立 dense 路径的共享专家数，不进入路由 expert id 空间。
    int64_t mGroupsPerWave{0};    // 每个 Wave 消费的 256 行 M 分组预算。
    int64_t maxTilesPerExpert{0}; // GMM1 状态区每专家容量，覆盖完整 N 方向调度。
    int64_t combineSyncSlotCountPerExpert{0}; // 每专家预留的 GMM2 到 Combine 同步槽数。
    int64_t serverNum{1};
    int64_t numMaxTokensPerRank{0}; // 全卡一致的单卡 token 数上界，用于通信槽和 mask 的几何。
    int64_t topK{0};
    int32_t dedupMode{0};
    bool combineQuant{false};
    bool topkWeightsPrefetch{false};
    bool dispatchFp4{false};
    bool reserveGmm1Output{false};
    bool reserveCumsum{false};
    bool needsGmmEpilogue{false};
    bool sharedInputIndependent{false};
    uint32_t sharedInputElementsPerByte{1};
};

// GMM1 每核至少覆盖四个逻辑 N tile，GMM2 每核至少覆盖一个。
HOST_DEVICE int64_t CalcMemoryMGroupsPerWave(int64_t hiddenDim, int64_t h, int64_t aicNum)
{
    if (hiddenDim == 0 || h == 0 || aicNum == 0) {
        return 1;
    }
    const int64_t gmm1Groups = MemoryLayoutDetail::CeilDiv(
        aicNum * 4, MemoryLayoutDetail::CeilDiv(hiddenDim, MemoryLayoutDetail::L1_TILE_M_256));
    const int64_t gmm2Groups =
        MemoryLayoutDetail::CeilDiv(aicNum, MemoryLayoutDetail::CeilDiv(h, MemoryLayoutDetail::L1_TILE_M_256));
    return gmm1Groups > gmm2Groups ? gmm1Groups : gmm2Groups;
}

HOST_DEVICE int64_t CalcMemoryMaxTilesPerExpert(int64_t maxOutputSize, int64_t hiddenDim)
{
    return MemoryLayoutDetail::CeilAlign(MemoryLayoutDetail::CeilDiv(maxOutputSize, MemoryLayoutDetail::L1_TILE_M_256) *
                                             MemoryLayoutDetail::CeilDiv(hiddenDim, MemoryLayoutDetail::L1_TILE_M_256),
                                         MemoryLayoutDetail::INT_CACHELINE);
}

// CCL 本地计算区从通信布局尾部开始，MTE 另保留头部异常记录区域。
HOST_DEVICE int64_t CalcCclComputeOffset(const CclCommLayoutParams& params)
{
    const int64_t exceptionBytes = params.topoType == TOPO_TYPE_MTE ? EXCEPTION_DUMP_REGION_SIZE : 0;
    return MemoryLayoutDetail::CeilAlign(exceptionBytes + CalcCclCommLayout(params).totalSize,
                                         MemoryLayoutDetail::ALIGN_512);
}

HOST_DEVICE int64_t CalcMemoryRingRows(int64_t mGroupsPerWave)
{
    return static_cast<int64_t>(MTE_RING_WAVE_COUNT) * mGroupsPerWave * MemoryLayoutDetail::L1_TILE_M_256;
}

// 准入规则仅在此处维护；布局与核内寻址直接使用对应阶段的配置。
HOST_DEVICE CclStageRingConfig CalcCclStageRingConfig(const CclComputeLayoutParams& params)
{
    CclStageRingConfig ring{};
    ring.rows = CalcMemoryRingRows(params.mGroupsPerWave);
    const bool canReuse = params.topoType == TOPO_TYPE_MTE && !params.combineQuant && params.mGroupsPerWave > 0 &&
                          params.maxOutputSize > static_cast<int64_t>(ring.rows);
    // Combine 去重按完整专家消费，专家内缺少逐 tile 进度约束，Dispatch 输入也需全量保存。
    ring.dispatch = canReuse && !IsDispatchDedupOn(params.dedupMode) && !IsCombineDedupOn(params.dedupMode);
    ring.gmm1 = canReuse;
    ring.activation = canReuse;
    ring.gmm2 = canReuse && !IsCombineDedupOn(params.dedupMode);
    return ring;
}

HOST_DEVICE int64_t CalcSharedActivationFlagElementsPerExpert(int64_t tokenNum)
{
    return MemoryLayoutDetail::CeilDiv(tokenNum, MemoryLayoutDetail::L1_TILE_M_256) * MemoryLayoutDetail::INT_CACHELINE;
}

// 仅描述 CCL 本地计算区域相对 cclComputePtr 的字节偏移，不持有或构造地址。
struct CclComputeLayout {
    int64_t dispatchRevDataOffset{INVALID_WORKSPACE_OFFSET};
    int64_t dispatchRevScaleOffset{INVALID_WORKSPACE_OFFSET};
    int64_t dispatchRevWeightsOffset{INVALID_WORKSPACE_OFFSET};
    int64_t activationQuantDataOffset{INVALID_WORKSPACE_OFFSET};
    int64_t activationQuantScaleOffset{INVALID_WORKSPACE_OFFSET};
    int64_t expertRecvTokenCountOffset{INVALID_WORKSPACE_OFFSET};
    int64_t metaInfoOffset{INVALID_WORKSPACE_OFFSET};
    int64_t flagActivationToGmm2Offset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedActivationToGmm2Offset{INVALID_WORKSPACE_OFFSET};
    int64_t flagDispatchToGmm1Offset{INVALID_WORKSPACE_OFFSET};
    int64_t flagSendCntCalToUpdParamsOffset{INVALID_WORKSPACE_OFFSET};
    int64_t flagGmmToEpilogueOffset{INVALID_WORKSPACE_OFFSET};
    // 仅 MTE 启用共享专家时使用：AIV1 子组分别在 topk 有效下标发送完成、跨卡握手完成后同步。
    int64_t flagAiv1TopkValidIndexSyncOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm2ReadyOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm2CombineSyncCounterOffset{INVALID_WORKSPACE_OFFSET};
    int64_t cumsumInfoOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm1MmadResOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm2MmadResOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertResultOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertGmm1OutOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertInputOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertInputDataOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertInputScaleOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertActivationDataOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertActivationScaleOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm1TileStatusOffset{INVALID_WORKSPACE_OFFSET}; // GMM1 tile 就绪状态位区（仅 prefetch 软同步分配）
    int64_t sharedExpertGmm2TileCounterOffset{INVALID_WORKSPACE_OFFSET};
    int64_t dedupPrepassReadyOffset{INVALID_WORKSPACE_OFFSET}; // 去重 prepass 完成计数（flag 区内，自动清零）
    int64_t dedupRowDescOffset{INVALID_WORKSPACE_OFFSET};      // 去重 rowDesc 表（仅 MTE 且 dedupMode!=0）
    int64_t maskSlotOffset{INVALID_WORKSPACE_OFFSET};          // urma发送mask临时GM
    int64_t dispatchRelaySendQueueOffset{INVALID_WORKSPACE_OFFSET}; // 按目标 Server 划分的一级中继发送队列
    int64_t dispatchRemoteReadyFlagSnapshotOffset{
        INVALID_WORKSPACE_OFFSET}; // 各逻辑核的固定 256-token 远端就绪标志窗口
    int64_t tokenIdsOffset{INVALID_WORKSPACE_OFFSET};
    int64_t tokenBitmapsOffset{INVALID_WORKSPACE_OFFSET};
    int64_t combineRelayCompletedCountOffset{INVALID_WORKSPACE_OFFSET};
    int64_t serverCombineResetBytes{0};

    // 连续 flag 通知区（自 flagActivationToGmm2Offset 起）的 int32 元素总数。
    // 在分配处顺手记账，保证 ResetSyncStatus 的清零范围与分配范围恒同源。
    int64_t flagResetElementCount{0};
    // prefetch 软同步 GMM1 tile 状态区的 int32 元素总数（未分配时为 0）。
    int64_t gmm1TileStatusElementCount{0};

    int64_t workspaceSize{0};
    HOST_DEVICE explicit CclComputeLayout(const CclComputeLayoutParams& params)
    {
        Initialize(params, params.serverNum);
    }

private:
    HOST_DEVICE void Initialize(const CclComputeLayoutParams& params, int64_t serverNum)
    {
        workspaceSize = 0;
        // MTE 与 URMA 分别保持既有分区顺序。
        if (params.topoType == TOPO_TYPE_MTE) {
            InitializeMte(params);
        } else {
            InitializeUrma(params, serverNum);
        }
    }

    HOST_DEVICE void InitializeMte(const CclComputeLayoutParams& params)
    {
        InitializeDispatchBuffers(params);
        // tkw=0 且 combine 去重：dispatch 把 LAST 行整 token 权重段落 revWeights 区、merge 读回乘权，
        // MTE 同样依赖本区。漏分配则 dispatchRevWeightsPtr 为 nullptr，首 launch 即 MTE invalid GM。
        // tkw=1 与去重关闭态在 MTE 下不使用本区，保持既有布局不变。
        if (params.topkWeightsPrefetch == 0 && IsCombineDedupOn(params.dedupMode)) {
            InitializeDispatchRevWeights(params);
        }
        // 每个 AIC 的计数跨度，以 int32_t 元素计。
        int64_t expertRecvTokenCountStride = static_cast<int64_t>(MemoryLayoutDetail::CeilAlign(
            params.moeExpertPerRank, static_cast<uint32_t>(MemoryLayoutDetail::INT_CACHELINE)));
        int64_t expertRecvTokenCountBytes =
            expertRecvTokenCountStride * static_cast<int64_t>(sizeof(int32_t)) * params.aicNum;
        InitializeActivationBuffers(params);
        InitializeRoutingMetadata(params, expertRecvTokenCountBytes);
        InitializeMteSyncFlags(params);
        const bool needsReloadCumsum = params.reserveCumsum;
        InitializeGmmIntermediateBuffers(params, needsReloadCumsum, true);
        InitializeGmm1TileStatus(params);
        InitializeDedupRowDesc(params);
        if (params.sharedExpertNum > 0) {
            InitializeSharedExpertGmmOutputs(params);
            if (params.sharedInputIndependent) {
                InitializeMteSharedExpertInput(params);
            }
            InitializeSharedExpertActivationBuffers(params);
        }
    }

    HOST_DEVICE void InitializeUrma(const CclComputeLayoutParams& params, int64_t serverNum)
    {
        InitializeDispatchBuffers(params);
        InitializeDispatchRevWeights(params);
        const int64_t expertRecvTokenCountStride =
            MemoryLayoutDetail::CeilAlign(params.moeExpertPerRank, MemoryLayoutDetail::INT_CACHELINE);
        int64_t expertRecvTokenCountBytes = expertRecvTokenCountStride * SIZE_INT_32 * params.aicNum;
        InitializeActivationBuffers(params);
        InitializeRoutingMetadata(params, expertRecvTokenCountBytes);
        InitializeUrmaSyncFlags(params);
        // URMA 所有 GMM 模式均保留 dispatch 累积计数。
        InitializeGmmIntermediateBuffers(params, true, false);
        InitializeGmm1TileStatus(params);
        InitializeDedupRowDesc(params);
        InitializeUrmaDispatchBuffers(params, serverNum);
        if (IsServerCombineDedupOn(params.dedupMode)) {
            InitializeServerCombineBuffers(params, serverNum);
        }
        if (params.sharedExpertNum > 0) {
            InitializeSharedExpertGmmOutputs(params);
            InitializeUrmaSharedExpertInput(params);
            InitializeSharedExpertActivationBuffers(params);
        }
    }

    HOST_DEVICE void InitializeServerCombineBuffers(const CclComputeLayoutParams& params, int64_t serverNum)
    {
        const int64_t capacity = params.numMaxTokensPerRank;
        combineRelayCompletedCountOffset = workspaceSize;
        // 每个 Server 的完成计数独占 64B，仅本段在入口统一清零。
        workspaceSize += MemoryLayoutDetail::CeilAlign(serverNum * MemoryLayoutDetail::INT_CACHELINE * SIZE_INT_32,
                                                       MemoryLayoutDetail::ALIGN_512);
        serverCombineResetBytes = workspaceSize - combineRelayCompletedCountOffset;
        tokenIdsOffset = workspaceSize;
        workspaceSize += MemoryLayoutDetail::CeilAlign(serverNum * CalcServerCombineTokenListStrideBytes(capacity),
                                                       MemoryLayoutDetail::ALIGN_512);
        tokenBitmapsOffset = workspaceSize;
        workspaceSize += MemoryLayoutDetail::CeilAlign(serverNum * capacity * static_cast<int64_t>(sizeof(uint32_t)),
                                                       MemoryLayoutDetail::ALIGN_512);
    }

    HOST_DEVICE void InitializeDispatchBuffers(const CclComputeLayoutParams& params)
    {
        const int64_t dataRows = params.ring.dispatch ? static_cast<int64_t>(params.ring.rows) : params.maxOutputSize;
        dispatchRevDataOffset = workspaceSize;
        // dispatchRevData 按 GMM1 激活的实际物理字节数分配：FP8 每字节 1 个元素，FP4 每字节 2 个元素。
        const int64_t dispatchDataElementsPerByte = params.dispatchFp4 ? 2LL : 1LL;
        const int64_t dispatchDataBytesPerToken = static_cast<int64_t>(params.h) / dispatchDataElementsPerByte;
        workspaceSize +=
            MemoryLayoutDetail::CeilAlign(dataRows * dispatchDataBytesPerToken, MemoryLayoutDetail::ALIGN_512);
        dispatchRevScaleOffset = workspaceSize;

        int64_t dispatchScaleElementsPerToken =
            MemoryLayoutDetail::CeilDiv(static_cast<int64_t>(params.h),
                                        static_cast<int64_t>(MemoryLayoutDetail::MXFP_DIVISOR_SIZE)) *
            MemoryLayoutDetail::MXFP_MULTI_BASE_SIZE;
        workspaceSize += MemoryLayoutDetail::CeilAlign(SIZE_INT_8 * dataRows * dispatchScaleElementsPerToken,
                                                       MemoryLayoutDetail::ALIGN_512);
    }

    HOST_DEVICE void InitializeDispatchRevWeights(const CclComputeLayoutParams& params)
    {
        // dispatch 接收侧权重区，两种场景共用、MTE 与 URMA 两条初始化路径都必须调用：
        // tkw=1 的权重预取；tkw=0 且 combine 去重——dispatch 侧把 LAST 行
        // 整 token 的 topK 权重段落在这里，merge 一次连续读取后按成员 topkIndex 乘权。
        const bool dedupCombineWeights = params.topkWeightsPrefetch == 0 && IsCombineDedupOn(params.dedupMode);
        if (params.topkWeightsPrefetch == 1 || dedupCombineWeights) {
            dispatchRevWeightsOffset = workspaceSize;
            uint32_t weightAlignBytes =
                MemoryLayoutDetail::CeilAlign(static_cast<uint32_t>(params.topK * sizeof(float)),
                                              static_cast<uint32_t>(MemoryLayoutDetail::ALIGN_32));
            workspaceSize += MemoryLayoutDetail::CeilAlign(
                static_cast<int64_t>(params.maxOutputSize) * static_cast<int64_t>(weightAlignBytes),
                MemoryLayoutDetail::ALIGN_512);
        }
    }

    HOST_DEVICE void InitializeActivationBuffers(const CclComputeLayoutParams& params)
    {
        const int64_t dataRows = params.ring.activation ? static_cast<int64_t>(params.ring.rows) : params.maxOutputSize;
        activationQuantDataOffset = workspaceSize;
        workspaceSize += MemoryLayoutDetail::CeilAlign(
            SIZE_INT_8 * dataRows * params.hiddenDim / params.activationOutputDivisor, MemoryLayoutDetail::ALIGN_512);

        activationQuantScaleOffset = workspaceSize;
        workspaceSize += MemoryLayoutDetail::CeilAlign(
            SIZE_INT_8 * dataRows *
                MemoryLayoutDetail::CeilDiv(static_cast<int64_t>(params.hiddenDim / params.activationOutputDivisor),
                                            static_cast<int64_t>(MemoryLayoutDetail::MXFP_DIVISOR_SIZE)) *
                MemoryLayoutDetail::MXFP_MULTI_BASE_SIZE,
            MemoryLayoutDetail::ALIGN_512);
    }

    HOST_DEVICE void InitializeRoutingMetadata(const CclComputeLayoutParams& params, int64_t expertRecvTokenCountBytes)
    {
        expertRecvTokenCountOffset = workspaceSize;
        workspaceSize += MemoryLayoutDetail::CeilAlign(expertRecvTokenCountBytes, MemoryLayoutDetail::ALIGN_512);

        metaInfoOffset = workspaceSize;
        workspaceSize += MemoryLayoutDetail::CeilAlign(params.maxOutputSize * MemoryLayoutDetail::ALIGN_32,
                                                       MemoryLayoutDetail::ALIGN_512);
    }

    HOST_DEVICE void InitializeMteSyncFlags(const CclComputeLayoutParams& params)
    {
        // 标量通知区连续分配，保证统一清零范围。
        int64_t flagRegionBeginOffset = workspaceSize;
        int64_t moeExpertCount = static_cast<int64_t>(params.moeExpertPerRank);
        flagActivationToGmm2Offset = workspaceSize;
        int64_t wavesPerExpert = MemoryLayoutDetail::CeilDiv(static_cast<int64_t>(params.maxOutputSize),
                                                             static_cast<int64_t>(MemoryLayoutDetail::L1_TILE_M_256));
        workspaceSize += SIZE_INT_32 * moeExpertCount * wavesPerExpert * MemoryLayoutDetail::INT_CACHELINE;
        if (params.sharedExpertNum > 0) {
            sharedActivationToGmm2Offset = workspaceSize;
            workspaceSize += SIZE_INT_32 * CalcSharedActivationFlagElementsPerExpert(static_cast<int64_t>(params.bs)) *
                             static_cast<int64_t>(params.sharedExpertNum);
        }
        InitializeRoutingFlags(params);
        // 去重 prepass 完成标记（flag 连续区内，随 ResetSyncStatus 每 launch 自动清零）：
        // 槽 0 = 每核 done 计数（AIV1 完成全部认领项后 +1；combine 消费 desc 前等其到齐）；
        // 槽 1+item = 该工作项 ready（dispatch 每批只等自己覆盖的 (srcRank,chunk) 项）。
        // 按 worldSize*MemoryLayoutDetail::DEDUP_PREPASS_MAX_CHUNKS 上界分配，每槽独占一个 cache line。
        if (params.dedupMode != 0) {
            dedupPrepassReadyOffset = workspaceSize;
            workspaceSize +=
                SIZE_INT_32 * MemoryLayoutDetail::INT_CACHELINE *
                (1 + static_cast<int64_t>(params.epWorldSize) * MemoryLayoutDetail::DEDUP_PREPASS_MAX_CHUNKS);
        }
        if (params.needsGmmEpilogue || !params.combineQuant) {
            flagGmmToEpilogueOffset = workspaceSize;
            workspaceSize += static_cast<int64_t>(params.aicNum) * MemoryLayoutDetail::INT_CACHELINE * SIZE_INT_32;
        }
        // 量化 Combine 在完整专家结束后等待每个 AIC 的完成标记；combine 去重下非量化路径的
        // 合并驱动行也要等组内成员所在专家的 GMM2 全部落地，同样需要该 per-expert 标记区。
        if (params.combineQuant || IsCombineDedupOn(params.dedupMode)) {
            gmm2ReadyOffset = workspaceSize;
            workspaceSize += SIZE_INT_32 * moeExpertCount * params.aicNum * MemoryLayoutDetail::INT_CACHELINE;
        }
        if (params.sharedExpertNum > 0) {
            sharedExpertGmm2TileCounterOffset = workspaceSize;
            int64_t tokenGroupCount = MemoryLayoutDetail::CeilDiv(
                static_cast<int64_t>(params.bs), static_cast<int64_t>(MemoryLayoutDetail::L1_TILE_M_256));
            workspaceSize += SIZE_INT_32 * tokenGroupCount * static_cast<int64_t>(params.sharedExpertNum) *
                             MemoryLayoutDetail::INT_CACHELINE;

            flagAiv1TopkValidIndexSyncOffset = workspaceSize;
            workspaceSize += static_cast<int64_t>(params.aicNum) * MemoryLayoutDetail::INT_CACHELINE * SIZE_INT_32;
        }
        flagResetElementCount = (workspaceSize - flagRegionBeginOffset) / SIZE_INT_32;
    }

    HOST_DEVICE void InitializeUrmaSyncFlags(const CclComputeLayoutParams& params)
    {
        // 标量通知区连续分配，保证统一清零范围。
        int64_t flagRegionBeginOffset = workspaceSize;
        int64_t moeExpertCount = static_cast<int64_t>(params.moeExpertPerRank);
        flagActivationToGmm2Offset = workspaceSize;
        // Layered token wave 为专家内每个 256 行分组保留激活完成标志。
        const int64_t maxGroupsPerExpert =
            MemoryLayoutDetail::CeilDiv(params.maxOutputSize, MemoryLayoutDetail::L1_TILE_M_256);
        workspaceSize += SIZE_INT_32 * moeExpertCount * maxGroupsPerExpert * MemoryLayoutDetail::INT_CACHELINE;
        InitializeRoutingFlags(params);
        if (params.needsGmmEpilogue) {
            flagGmmToEpilogueOffset = workspaceSize;
            workspaceSize += static_cast<int64_t>(params.aicNum) * MemoryLayoutDetail::INT_CACHELINE * SIZE_INT_32;
        }
        gmm2CombineSyncCounterOffset = workspaceSize;
        workspaceSize += static_cast<int64_t>(params.combineSyncSlotCountPerExpert) * moeExpertCount *
                         MemoryLayoutDetail::INT_CACHELINE * SIZE_INT_32;
        flagResetElementCount = (workspaceSize - flagRegionBeginOffset) / SIZE_INT_32;
    }

    HOST_DEVICE void InitializeRoutingFlags(const CclComputeLayoutParams& params)
    {
        int64_t moeExpertCount = static_cast<int64_t>(params.moeExpertPerRank);
        int64_t maxWavesPerExpert = MemoryLayoutDetail::CeilDiv(
            static_cast<int64_t>(params.maxOutputSize), static_cast<int64_t>(MemoryLayoutDetail::L1_TILE_M_256));
        int64_t waveFlagSlotsPerExpert = maxWavesPerExpert * static_cast<int64_t>(MemoryLayoutDetail::INT_CACHELINE);
        flagDispatchToGmm1Offset = workspaceSize;
        workspaceSize += SIZE_INT_32 * moeExpertCount * waveFlagSlotsPerExpert;

        // 每(expert, aiCore)单独占一个cache_line
        flagSendCntCalToUpdParamsOffset = workspaceSize;
        workspaceSize += SIZE_INT_32 * MemoryLayoutDetail::INT_CACHELINE * moeExpertCount * params.aicNum;
    }

    HOST_DEVICE void InitializeGmmIntermediateBuffers(const CclComputeLayoutParams& params, bool reserveCumsum,
                                                      bool alignGmm2Output)
    {
        const int64_t dataRows = params.ring.gmm1 ? static_cast<int64_t>(params.ring.rows) : params.maxOutputSize;
        if (reserveCumsum) {
            // cumsumInfo：逐核备份 cumsum 状态，每核 moeExpertPerRank × epWorldSize 个 int32。
            cumsumInfoOffset = workspaceSize;
            workspaceSize += MemoryLayoutDetail::CeilAlign(
                                 static_cast<int64_t>(SIZE_INT_32 * params.moeExpertPerRank * params.epWorldSize),
                                 MemoryLayoutDetail::ALIGN_32) *
                             params.aicNum;
        }
        if (params.reserveGmm1Output) {
            // 支持复用的 MTE 路径按循环容量分配，其余路径保留全量 GMM1 结果。
            gmm1MmadResOffset = workspaceSize;
            workspaceSize += SIZE_BF_16 * dataRows * params.hiddenDim;
        }
        // Combine 去重使用全量逻辑行，其余可复用路径使用循环物理行；结果始终只有一份。
        const int64_t gmm2Rows = params.ring.gmm2 ? static_cast<int64_t>(params.ring.rows) : params.maxOutputSize;
        int64_t outputBytes = SIZE_BF_16 * gmm2Rows * params.h;
        if (alignGmm2Output) {
            workspaceSize =
                MemoryLayoutDetail::CeilAlign(workspaceSize, static_cast<int64_t>(MemoryLayoutDetail::ALIGN_512));
            outputBytes =
                MemoryLayoutDetail::CeilAlign(outputBytes, static_cast<int64_t>(MemoryLayoutDetail::ALIGN_512));
        }
        gmm2MmadResOffset = workspaceSize;
        workspaceSize += outputBytes;
    }

    // 去重 rowDesc 表：每个接收行一条定长描述符。无需清零——prepass 在 dispatch 读取前
    // 覆盖写全部有效行，超出实际接收计数的行不会被读取。
    HOST_DEVICE void InitializeDedupRowDesc(const CclComputeLayoutParams& params)
    {
        if (params.topoType == TOPO_TYPE_MTE && params.dedupMode != 0) {
            dedupRowDescOffset = workspaceSize;
            int64_t rowDescStrideInt32 = CalcDedupRowDescStrideInt32(static_cast<int64_t>(params.topK));
            workspaceSize += MemoryLayoutDetail::CeilAlign(
                SIZE_INT_32 * rowDescStrideInt32 * static_cast<int64_t>(params.maxOutputSize),
                MemoryLayoutDetail::ALIGN_512);
        }
    }

    HOST_DEVICE void InitializeGmm1TileStatus(const CclComputeLayoutParams& params)
    {
        // GMM1 tile 状态位区（仅 prefetch 路径分配，用于 AIC→AIV 软同步）。
        // 每个 tile 状态和末尾 allDone 状态都独占一条 64B cache line。
        gmm1TileStatusOffset = INVALID_WORKSPACE_OFFSET;
        if (params.topkWeightsPrefetch == 1) {
            gmm1TileStatusOffset = workspaceSize;
            int64_t statusSlots = static_cast<int64_t>(params.moeExpertPerRank) * params.maxTilesPerExpert + 1;
            gmm1TileStatusElementCount = statusSlots * MemoryLayoutDetail::INT_CACHELINE;
            int64_t statusBytes = SIZE_INT_32 * statusSlots * MemoryLayoutDetail::INT_CACHELINE;
            workspaceSize += MemoryLayoutDetail::CeilAlign(statusBytes, MemoryLayoutDetail::ALIGN_512);
        }
    }

    HOST_DEVICE void InitializeUrmaDispatchBuffers(const CclComputeLayoutParams& params, int64_t serverNum)
    {
        maskSlotOffset = workspaceSize;
        // 远端 mask 槽的 count 位于全卡共用的容量尾部，本地发送暂存必须使用相同槽宽。
        const int64_t maskAlignSize = CalcDispatchMaskAlignSize(params.numMaxTokensPerRank, params.topK);
        int64_t maskSlotSize = maskAlignSize + static_cast<int64_t>(MemoryLayoutDetail::ALIGN_32); // mask + 32B count

        workspaceSize += MemoryLayoutDetail::CeilAlign(
            static_cast<int64_t>(params.moeExpertPerRank) * params.epWorldSize * maskSlotSize,
            static_cast<int64_t>(MemoryLayoutDetail::ALIGN_512));

        dispatchRelaySendQueueOffset = workspaceSize;
        int64_t serverWorkspaceBytes = static_cast<int64_t>(MemoryLayoutDetail::ALIGN_32) +
                                       static_cast<int64_t>(params.bs) * MemoryLayoutDetail::ALIGN_32;
        workspaceSize += MemoryLayoutDetail::CeilAlign(static_cast<int64_t>(serverNum) * serverWorkspaceBytes,
                                                       static_cast<int64_t>(MemoryLayoutDetail::ALIGN_512));

        dispatchRemoteReadyFlagSnapshotOffset = workspaceSize;
        // 每个逻辑核只缓存当前 256-token 窗口；稀疏 token 按命中的窗口分块读取。
        int64_t flagSnapshotBytes = static_cast<int64_t>(MemoryLayoutDetail::L1_TILE_M_256) * sizeof(uint64_t);
        int64_t relayFlagSnapshotBytesPerBlock =
            MemoryLayoutDetail::CeilAlign(flagSnapshotBytes, static_cast<int64_t>(MemoryLayoutDetail::ALIGN_512));
        workspaceSize += static_cast<int64_t>(params.aicNum) * relayFlagSnapshotBytesPerBlock;
    }

    HOST_DEVICE void InitializeSharedExpertGmmOutputs(const CclComputeLayoutParams& params)
    {
        // sharedExpertResult：共享专家 GMM2 输出 [sharedExpertNum × bs × h]。
        sharedExpertResultOffset = workspaceSize;
        workspaceSize += MemoryLayoutDetail::CeilAlign(SIZE_BF_16 * params.bs * params.sharedExpertNum * params.h,
                                                       MemoryLayoutDetail::ALIGN_512);
        // sharedExpertGmm1Out：共享专家 GMM1 输出 [sharedExpertNum × bs × hiddenDim]。
        sharedExpertGmm1OutOffset = workspaceSize;
        workspaceSize += MemoryLayoutDetail::CeilAlign(
            SIZE_BF_16 * params.bs * params.sharedExpertNum * params.hiddenDim, MemoryLayoutDetail::ALIGN_512);
    }

    HOST_DEVICE void InitializeMteSharedExpertInput(const CclComputeLayoutParams& params)
    {
        // MTE 共享专家使用逐 token 交织数据 [Align256(data) | Align32(scale)]，仅独立量化时分配。
        sharedExpertInputOffset = workspaceSize;
        uint32_t elementsPerByte = params.sharedInputElementsPerByte;
        int64_t recordBytes = CalcQuantTokenScaleBytes(static_cast<int64_t>(params.h), elementsPerByte, 0LL, false);
        workspaceSize += MemoryLayoutDetail::CeilAlign(static_cast<int64_t>(params.bs) * recordBytes,
                                                       static_cast<int64_t>(MemoryLayoutDetail::ALIGN_512));
    }

    HOST_DEVICE void InitializeUrmaSharedExpertInput(const CclComputeLayoutParams& params)
    {
        // URMA layered kernel 仍使用连续 data/scale，两块地址契约保持不变。
        sharedExpertInputDataOffset = workspaceSize;
        workspaceSize +=
            MemoryLayoutDetail::CeilAlign(SIZE_INT_8 * params.bs * params.h, MemoryLayoutDetail::ALIGN_512);
        sharedExpertInputScaleOffset = workspaceSize;
        workspaceSize += MemoryLayoutDetail::CeilAlign(
            SIZE_INT_8 * params.bs *
                MemoryLayoutDetail::CeilDiv(static_cast<uint32_t>(params.h),
                                            static_cast<uint32_t>(MemoryLayoutDetail::MXFP_SCALE_GROUP_NUM)) *
                MemoryLayoutDetail::MXFP_MULTI_BASE_SIZE,
            MemoryLayoutDetail::ALIGN_512);
    }

    HOST_DEVICE void InitializeSharedExpertActivationBuffers(const CclComputeLayoutParams& params)
    {
        // 共享专家激活输出宽度随激活类型变化：门控激活减半，GELU 保留完整宽度。
        sharedExpertActivationDataOffset = workspaceSize;
        workspaceSize += MemoryLayoutDetail::CeilAlign(
            SIZE_INT_8 * params.bs * params.sharedExpertNum * params.hiddenDim / params.activationOutputDivisor,
            MemoryLayoutDetail::ALIGN_512);
        // sharedExpertActivationScale：每个 MXFP scale group 保存两个 multi-base 分量。
        sharedExpertActivationScaleOffset = workspaceSize;
        workspaceSize += MemoryLayoutDetail::CeilAlign(
            SIZE_INT_8 * params.bs * params.sharedExpertNum *
                MemoryLayoutDetail::CeilDiv(static_cast<int64_t>(params.hiddenDim / params.activationOutputDivisor),
                                            static_cast<int64_t>(MemoryLayoutDetail::MXFP_DIVISOR_SIZE)) *
                MemoryLayoutDetail::MXFP_MULTI_BASE_SIZE,
            MemoryLayoutDetail::ALIGN_512);
    }
};

} // namespace MegaMoeImpl

#endif // MEGA_MOE_MEMORY_LAYOUT_H
