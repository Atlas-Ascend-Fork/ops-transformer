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
 * \file quant_sparse_lightning_indexer_common_arch22.h
 * \brief
 */
#ifndef QUANT_SPARSE_LIGHTNING_INDEXER_COMMON_H
#define QUANT_SPARSE_LIGHTNING_INDEXER_COMMON_H

namespace QSLICommon {

// candidate (two-level topk) 模式, 与 host 侧 tiling.h 定义保持一致
// 本算子(consumer)模式固定为 2; SOURCE/OFF 不可达, 仅为与 host 常量表对齐保留
inline constexpr uint32_t CANDIDATE_MODE_SOURCE = 1; // 不可达: 第一级(source)由主算子承载
inline constexpr uint32_t CANDIDATE_MODE_CONSUMER = 2; // 本算子唯一有效模式: 输入候选块索引, 候选内选topk
inline constexpr uint32_t CANDIDATE_MODE_OFF = 3; // 不可达

// 候选 tile 粒度：64 个候选块 × 8 token = 512 token（与 DSL 及 QSLI metadata 的
// CandidateTileTokens=512 一致）。候选模式下 metadata 的 s2Start/s2End 均以本粒度计。
inline constexpr uint32_t CAND_TILE_BLOCKS = 64;
inline constexpr uint32_t CAND_TILE_TOKENS = CAND_TILE_BLOCKS * 8;

// 候选 staging ring（AIV 聚集 K/kscale → staging GM → AIC 连续读取），移植自
// vllm-ascend zj-qsli 的 D3a/D3b。910B 跨核 flag 语义：同一 flagId 上多个 AIV 的 set
// 合并成一次可等待通知（AIV→AIC），AIC 的一次 set 广播给两个 AIV（AIC→AIV）。
// flagId 只有 4 bit 且 11..14 被 AscendC 保留；本 kernel 已占用 0/1/2
// （syncV1C1/syncV0C1/syncC1V0），ring 相关取 3/4/6/8/10/15。
inline constexpr uint32_t CAND_RING_SLOTS = 8; // staging 槽数（流水深度，槽号 = 全局 tile 序号 % 本值）
inline constexpr uint32_t CAND_READY_FLAG = 6;   // AIV→AIC：本 tile 的两个半区 staging 已写完
inline constexpr uint32_t CAND_FREE_FLAG = 8;    // AIC→AIV：空槽回执（AIC 启动预置 RING_SLOTS 个）
inline constexpr uint32_t CAND_WFREE_FLAG = 3;   // AIC→AIV：W 槽回收（预置 2 个，W 按任务交替双槽）
inline constexpr uint32_t CAND_WREADY_FLAG = 10; // AIV→AIC：本任务 W（w×qscale 广播）就绪
inline constexpr uint32_t CAND_SCORE_READY_FLAG = 15; // AIC→AIV：某 token 的全部候选 tile 分数已写
inline constexpr uint32_t CAND_AIC_DONE_FLAG = 4; // AIC→AIV：主循环结束，分数写已落地（核尾排空用）
// 候选分数区每任务的槽位数：4 行（s1Base）× 最多 32 个候选 tile = 128 行 512-float。
// 按任务序号奇偶双缓冲：AIV 跨任务延迟消费上一任务分数时，本任务写不会覆盖未读数据。
inline constexpr uint32_t CAND_SCORE_TILES_PER_TASK = 128;
// 候选 kscale 辅助区：每 tile 64 块 × 16 half（前 8 half 有效、后 8 填充），AIV gather 时
// 顺手写入，消费时一次 DMA 读回，取代"每 tile 64 次 16B DMA + 64 次 block_table 标量读"。
inline constexpr uint32_t CAND_AUX_SCALE_HALVES_PER_TILE = CAND_TILE_BLOCKS * 16U;
// aux 区奇偶周期 4（不是 2）：aux 只由单个 AIV 写、无 score 那样的 READY 链约束，
// 周期 2 时不消费的那侧 AIV 会在对侧延迟读之前把下一任务写进同 parity 区（vllm 的 X6 根因）。
inline constexpr uint32_t CAND_AUX_PERIOD = 4;
inline constexpr uint32_t CAND_PENDING_SCORE_SLOTS = 8; // 跨任务延迟消费队列深度

// 与tiling的layout保持一致
enum class LI_LAYOUT : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

template <typename Q_T, typename K_T, typename QK_T, typename SCORE_T, typename OUT_T,
          const bool PAGE_ATTENTION = false, LI_LAYOUT Q_LAYOUT_T = LI_LAYOUT::BSND,
          LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BBND, typename... Args>
struct QSLIType {
    using queryType = Q_T;
    using keyType = K_T;
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_LAYOUT layout = Q_LAYOUT_T;
    static constexpr LI_LAYOUT keyLayout = K_LAYOUT_T;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actS2SizeOrig = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorKeyScaleOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;
    uint64_t candidateOutOffset;
    uint64_t outputIdxOffsetCoreOffset; // A15: output_idx_offset 的 batch 级前缀 (行 x kHeadNum 布局)
    // 候选驱动（按 512 token 候选 tile 执行时）：本任务首行在 candidate_block_indices/length
    // 中的全局行号、本任务负责的候选 tile 区间 [candidateTileBegin, candidateTileEnd)、
    // 整行候选 tile 数（分数/辅助区寻址基准）与任务序号（W/分数/aux 区奇偶基准）
    uint32_t candidateRowBase = 0;
    uint32_t candidateTileIdx = 0;
    uint32_t candidateWeightOffset = 0;
    uint32_t candidateTileBegin = 0;
    uint32_t candidateTileEnd = 0;
    uint32_t candidateMaxTiles = 0;
    uint32_t candidateTaskSeq = 0;
    bool candidateNeedSpill = false;
    uint32_t candidateWsBase = 0;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isValid = false;
    bool isNeedLD = false;
    uint32_t saveWorkSpaceIdx = 0;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    // 无效索引
    static constexpr int INVALID_IDX = -1;

    // CUBE和VEC的核间同步EventID
    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;

    // 基本块大小
    uint32_t mBaseSize = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t sparseCount;             // topK选取大小
    uint64_t kSeqSize = 0ULL;         // kv最大S长度
    uint64_t qSeqSize = 1ULL;         // q最大S长度
    uint32_t kCacheBlockSize = 0;     // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    uint32_t keyStride0 = 0;          // key 第0维真实 stride (0=非PA/旧调用, 紧凑兜底; 对齐 arch35 A11)
    uint32_t keyDequantScaleStride0 = 0; // k_scale 第0维真实 stride (同上)
    LI_LAYOUT outputLayout;              // 输出的格式
    bool attenMaskFlag = false;
    uint32_t cmpRatio = 1; // 压缩率

    uint32_t actualLenQDims = 0U;     // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;      // KV 的actualSeqLength 的维度
    uint32_t cmpResiduaKLenDims = 0U; // cmpResidualK的维度
    bool isAccumSeqS1 = false;        // 是否累加模式
    bool isAccumSeqS2 = false;        // 是否累加模式

    // candidate (two-level topk)
    uint32_t candidateMode = 3U; // 本算子恒为 2(consumer); 初值仅占位, 由 tiling 下发覆盖
    uint32_t candidateTopkBlocks = 2048U;
    uint32_t candidateBlockSize = 8U;
    bool candidateDriven = false; // 传入非空 candidate_block_length 时按 512 token 候选 tile 执行

    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    uint32_t coreEnable = 0U;
};

struct LdSplitCoreInfo {
    bool isLdCoreEnable = false;    // 当前核是否参与规约任务
    uint32_t saveWorkSpaceIdx = 0U; // 存放LD参数的地址
    uint32_t bn2Idx = 0U;           // 归约任务
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t mIdx = 0U;
    uint32_t workspaceIdx = 0U; // 当前AIV核上规约任务的索引
    uint32_t workspaceNum = 0U; // 当前AIV核上规约任务的S2切分数量
    uint32_t mStart = 0U;
    uint32_t mNum = 0U;
    uint64_t indiceOutCoreOffset = 0U; // 最终输出索引搬出Topk的初始偏移地址
};

template <typename T1, typename T2>
__aicore__ inline T1 Align(T1 num, T2 rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
} // namespace QSLICommon

#endif // QUANT_SPARSE_LIGHTNING_INDEXER_COMMON_H
