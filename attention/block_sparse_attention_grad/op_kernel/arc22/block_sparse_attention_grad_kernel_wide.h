/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_sparse_attention_grad_kernel_wide.h
 * \brief Block Sparse Attention Grad Wide Kernel Implementation
 */

#ifndef BSAG_KERNEL_WIDE_H
#define BSAG_KERNEL_WIDE_H

#include "block_sparse_attention_grad_kernel_common.h"
#include "attn_infra/gemm/block/block_mmad_bsag_wide.hpp"
#include "attn_infra/epilogue/block/block_epilogue_wide_softmax.hpp"

// ============================ wide 执行体算法总览 ============================
// 输入：Q/K/V/dO/O/LSE/sparse mask（BNSD，D=128）   输出：dQ/dK/dV（FP32）
//   [Step 1] S  = Q·Kᵀ                AIC RunWideCube1        → FP32 NZ workspace
//   [Step 2] dP = dO·Vᵀ               AIC RunWideCube1（与 Step 1 同一次 KV 扫描）
//   [Step 3] D  = rowsum(dO∘O)        AIV ComputeWideRowDelta → (N, s1, 8)
//   [Step 4] P  = exp(scale·S − LSE) ┐ AIV RunWideVectorPacket → 低精度 zN
//   [Step 5] dS = P·(dP − D)         ┘
//   [Step 6] dQ = dS·K                AIC RunWideDq           （首写覆盖，后续原子累加）
//   [Step 7] dV = Pᵀ·dO ┐             AIC RunWideDkvPair      （恒原子累加）
//   [Step 8] dK = dSᵀ·Q ┘
// 跨核令牌：AIC --ready(CUBE2VEC 5/6)--> AIV --done(VEC2CUBE 7/8)--> AIC；收尾 POST = 9。
// 三级任务：work item（Host cyclic 划分）→ packet（≤8 边 / ≤8 KV 列，两侧逐字节一致）
//           → edge/tile（边下标即 workspace tile 槽号）。
// 同步分层（各同步点均带 [SYNC-*] 标记，便于检索）：
//   [SYNC-跨核]     SyncAll / CrossCoreSetFlag / WaitEvent，只出现在两个执行入口；
//   [SYNC-L1别名]   SyncExtendedL1Reuse，在低 L1 复用点插入顺序保证；
//   [SYNC-pipe栅栏] PipeBarrier<PIPE_ALL>，跨 pipe 阶段边界；
//   [SYNC-同核DMA]  AIV 计算 delta 后的 set_flag/wait_flag(MTE3→MTE2)。
// ============================================================================

namespace BSA::WideDetail {
using NpuArch::GemmCoord;
using NpuArch::Gemm::Block::WideCacheMode;

template <class Kernel>
class WideExecutor {
private:
    using Params = typename Kernel::Params;

public:
    // 对外接口：构造及 AIC/AIV 执行入口。
    /**
     * @brief 绑定原 kernel 和调用参数，构造独立的 wide 执行器。
     * @param[in] kernel 提供公共类型及 I1 prefix 读取接口的原 kernel 对象。
     * @param[in] params 本次调用的输入输出、workspace 和 tiling 地址集合。
     * @note 两个参数均保存为引用，其生命周期须覆盖执行器的使用期；
     *       缓冲使用和流水事件在对应的 AIC/AIV 执行入口内建立。
     */
    __aicore__ inline WideExecutor(Kernel& kernel, const Params& params)
        : kernel_(kernel),
          params_(params)
    {}

    /**
     * @brief AIC 主流程：按 work item -> packet -> edge 三级组织 Cube1 与 dQ/dKV 计算。
     * @param params Kernel 传入的输入输出地址与 workspace。
     * @param tilingData Host 下发的 tiling（含 wide 的容量、workspace 与任务划分字段）。
     * @note 本函数只在 scheduleMode==WIDE 时被 kernel.h 调用。
     * @note 与 AIV 侧的 packet 划分必须完全一致，任何新增的"提前 continue"都要双侧同步。
     * @note 流程框架（阶段编号只在本函数内有效）：
     *       阶段一 建立 wide 事件与配置 → 阶段二 准备 work item 并常驻预载 Q/dO →
     *       阶段三 生产当前 packet 的 S/dP（[Step 1,2]）并发布 ready 令牌 →
     *       阶段四 消费上一 packet 的 dQ/dKV（[Step 6,7,8]，晚一个 packet 以掩盖跨核等待）→
     *       阶段五 排空最后一个 packet → 阶段六 回收本核令牌并通知 POST。
     * @note 同一时刻存在两个 packet 描述（packets[2]）：stageIndex 为当前生产槽，
     *       previousStageIndex 为待消费槽；两者在 [0,1] 之间交替。
     */
    __aicore__ inline void ProcessWideAic(const Params& params, __gm__ BlockSparseAttentionGradTilingData* tilingData)
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t workLength = tilingData->taskNumPerCore + (tilingData->tailTaskNum > coreIdx ? 1 : 0);

        WideAicTensorViews tensors;
        const WideWorkspaceLayout ws(tilingData);
        BindWideAicTensorViews(params, ws, tensors);
        TaskInfo current;
        InitializeWorkItem(tilingData, coreIdx, current);

        BlockMmadBSAG1 cube1Mmad(resource);
        BlockMmadBSAG2 queryGradMmad(resource, 0, WIDE_PINGPONG_OFFSET_2, WideCacheMode::RESIDENT_K);
        BlockMmadBSAG3 residentQueryMmad(resource, WIDE_L1_OFFSET * 7 / 2, WIDE_PINGPONG_OFFSET_4,
                                         WideCacheMode::RESIDENT_QUERY);
        const MaskLayoutInfo maskLayout = GetMaskLayout(tilingData);
        uint32_t cubeOperandStage = 0;
        WideQMask queryGradWrittenMask = 0;
        uint32_t processedWorks = 0;

        // 阶段一：建立 wide 资源事件并加载配置。
        AscendC::SyncAll<false>(); // [SYNC-跨核]
        SeedWideFlags();
        AscendC::SetLoadDataPaddingValue<uint64_t>(0);
        AscendC::SetNdParaImpl(0x1);
        while (processedWorks < workLength) {
            // 阶段二：准备物理 Q 分片和 mask 扫描，预载本 work 常驻 Q/dO。
            WideWorkState work;
            PrepareWideWorkItem(tilingData, current, maskLayout, processedWorks + 1 < workLength, work);
            residentQueryMmad.AcquireFullPreloadReuse();
            PreloadWideQueryOperands(tilingData, work, tensors, residentQueryMmad);
            residentQueryMmad.FinishFullPreload();

            WidePacketInfo packets[2];
            // 每个 stage 一份 tile 缓存：消费的是上一 packet 的缓存，故必须按 stage 分开存。
            WideTileCache tileCaches[2];
            uint32_t stageIndex = 0;
            uint32_t previousStageIndex = 0;
            bool hasPendingGradient = false;
            queryGradWrittenMask = 0;
            while (work.nextKvIndex < work.kvScanLimit) {
                const uint64_t workspaceBase = ws.stageBase(coreIdx, stageIndex);
                WidePacketInfo& packet = packets[stageIndex];
                // packet 由两侧用同一份输入重建；空 packet 只推进游标，不切换 stage。
                BuildWidePacket(params, current, tilingData, workspaceBase, work, packet);
                if (packet.edgeCount == 0) {
                    continue;
                }
                ComputeWideTileCache(current, work.queries, packet, tilingData, tileCaches[stageIndex]);

                // 阶段三：生产当前 packet 的 S/dP，随后通知 AIV。
                SyncExtendedL1Reuse();            // [SYNC-L1别名]
                AscendC::PipeBarrier<PIPE_ALL>(); // [SYNC-pipe栅栏]
                // ---- [Step 1,2] S = Q·Kᵀ，dP = dO·Vᵀ（写入本 stage 的 FP32 NZ 槽）----
                RunWideCube1(packet, tileCaches[stageIndex], tilingData, tensors, cube1Mmad, cubeOperandStage,
                             residentQueryMmad);
                // ready 令牌走 PIPE_FIX 队列：令牌可见即 S/dP 已落到 GM。
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(WideReadyFlag(stageIndex)); // [SYNC-跨核]

                // 阶段四：当前 Cube1 发射后，消费上一 packet 的梯度。
                if (hasPendingGradient) {
                    AscendC::WaitEvent(WideDoneFlag(previousStageIndex)); // [SYNC-跨核]
                    SyncExtendedL1Reuse();                                // [SYNC-L1别名]
                    // ---- [Step 6] dQ = dS·K（首写覆盖，后续原子累加）----
                    RunWideDq(packets[previousStageIndex], tileCaches[previousStageIndex], tilingData, tensors,
                              queryGradMmad, queryGradWrittenMask);
                    SyncExtendedL1Reuse(); // [SYNC-L1别名]
                    // ---- [Step 7,8] dV = Pᵀ·dO，dK = dSᵀ·Q（恒原子累加）----
                    RunWideDkvPair(packets[previousStageIndex], tileCaches[previousStageIndex], tilingData, tensors,
                                   queryGradMmad, residentQueryMmad);
                }
                previousStageIndex = stageIndex;
                stageIndex = 1 - stageIndex;
                hasPendingGradient = true;
            }
            // 阶段五：排空最后一个 packet，再归还常驻 Q/dO 槽。
            // 与阶段四的消费段同构：保留这份"等待 → dQ → dKV"的展开形态，
            // 使跨核同步框架在主流程里可直接审查（不额外封装隐藏屏障的梯度函数）。
            if (hasPendingGradient) {
                AscendC::WaitEvent(WideDoneFlag(previousStageIndex)); // [SYNC-跨核]
                SyncExtendedL1Reuse();                                // [SYNC-L1别名]
                // ---- [Step 6] dQ = dS·K ----
                RunWideDq(packets[previousStageIndex], tileCaches[previousStageIndex], tilingData, tensors,
                          queryGradMmad, queryGradWrittenMask);
                SyncExtendedL1Reuse(); // [SYNC-L1别名]
                // ---- [Step 7,8] dV = Pᵀ·dO，dK = dSᵀ·Q ----
                RunWideDkvPair(packets[previousStageIndex], tileCaches[previousStageIndex], tilingData, tensors,
                               queryGradMmad, residentQueryMmad);
                SyncExtendedL1Reuse(); // [SYNC-L1别名]
            }
            residentQueryMmad.ReleaseFullPreloadReuse();
            ++processedWorks;
            if (processedWorks < workLength) {
                current = work.nextWork;
            }
        }
        // 阶段六：回收入口令牌，通知 POST。
        WaitWideFlag();
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(WIDE_CUBE2POST); // [SYNC-跨核]
    }

    /**
     * @brief AIV 主流程：按 work item -> packet 消费 S/dP，产出 P/dS，并回写完成令牌。
     * @param params Kernel 传入的输入输出地址与 workspace。
     * @param tilingData Host 下发的 tiling。
     * @note AIV 与 AIC 是 1:2 映射（两个 AIV 子核服务一个 AIC），故 coreIdx = blockIdx/2。
     * @note packet 的构造与 AIC 完全一致；AIV 不生成 packet 的语义，只消费相同的划分结果。
     * @note 流程框架：阶段二 准备 work item 并计算 delta（[Step 3]）→
     *       阶段三 逐 packet 等待 ready、计算 P/dS（[Step 4,5]）、发布 done 令牌 →
     *       阶段四 结束 raw stage 复用与 wide softmax 流水。
     * @note AIV 没有"延迟一个 packet"：packet 一就绪即消费；stageIndex 只用于选择令牌与 workspace 槽。
     */
    __aicore__ inline void ProcessWideAiv(const Params& params, __gm__ BlockSparseAttentionGradTilingData* tilingData)
    {
        const uint32_t vecCoreIdx = AscendC::GetBlockIdx();
        const uint32_t coreIdx = vecCoreIdx / 2;
        const uint32_t subBlockIdx = vecCoreIdx % 2;
        const uint32_t workLength = tilingData->taskNumPerCore + (tilingData->tailTaskNum > coreIdx ? 1 : 0);

        TaskInfo current;
        InitializeWorkItem(tilingData, coreIdx, current);
        const WideWorkspaceLayout ws(tilingData);
        GM_ADDR sfmgGm = params.workspace + ws.rowDelta();
        SfmgParams sfmgParams(params.dout, params.out, params.actualQseqlen, sfmgGm, params.tiling);
        EpilogueFAGSfmg rowDeltaOp(sfmgParams);
        EpilogueFAGOp softmaxOp;
        softmaxOp.BeginWidePipeline(params.softmaxLse, sfmgGm, params.tiling);
        const MaskLayoutInfo maskLayout = GetMaskLayout(tilingData);
        uint32_t processedWorks = 0;

        while (processedWorks < workLength) {
            // 阶段二：准备共享任务描述，计算本子核各 Q 分片的 delta。
            WideWorkState work;
            PrepareWideWorkItem(tilingData, current, maskLayout, processedWorks + 1 < workLength, work);
            // ---- [Step 3] D = rowsum(dO∘O)（两个子核按行分半）----
            ComputeWideRowDelta(tilingData, work, subBlockIdx, rowDeltaOp);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);  // [SYNC-同核DMA]
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); // [SYNC-同核DMA]

            WidePacketInfo packet;
            WideTileInfoCache tileCache;
            uint32_t stageIndex = 0;
            // 阶段三：按与 AIC 相同的规则消费 raw packet，回写 P/dS。
            softmaxOp.BeginWideRawStageReuse();
            while (work.nextKvIndex < work.kvScanLimit) {
                const uint64_t workspaceBase = ws.stageBase(coreIdx, stageIndex);
                BuildWidePacket(params, current, tilingData, workspaceBase, work, packet);
                if (packet.edgeCount == 0) {
                    continue;
                }
                ComputeWideTileInfoCache(current, work.queries, packet, tilingData, tileCache);
                AscendC::WaitEvent(WideReadyFlag(stageIndex)); // [SYNC-跨核]
                // ---- [Step 4,5] P = exp(scale·S − LSE)，dS = P·(dP − D) ----
                RunWideVectorPacket(params, ws, packet, tileCache, work, subBlockIdx, softmaxOp);
                AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(WideDoneFlag(stageIndex)); // [SYNC-跨核]
                stageIndex = 1 - stageIndex;
            }
            softmaxOp.EndWideRawStageReuse();
            ++processedWorks;
            if (processedWorks < workLength) {
                current = work.nextWork;
            }
        }
        // 阶段四：结束 wide softmax 流水。
        softmaxOp.EndWidePipeline();
    }

private:
    // 内部类型与任务、packet 描述。
    using ArchTag = typename Kernel::ArchTag;
    using ElementInput = typename Kernel::ElementInput;
    using ElementA1 = typename Kernel::ElementA1;
    // —— 类型别名 ——
    // 三个 MMAD 都包装为 WideBlockMmad 以获得宽缓存与 zN 累加能力：
    //   BSAG1 -> Cube1（STREAMED 缓存），
    //   BSAG2 -> dQ 与 P/dS 预载（RESIDENT_K：8 个常驻 K 槽），
    //   BSAG3 -> dKV 与 Q/dO 常驻槽（RESIDENT_QUERY）。
    using BlockMmadBSAG1 = NpuArch::Gemm::Block::WideBlockMmad<typename Kernel::BlockMmadBSAG1>;
    using BlockMmadBSAG2 = NpuArch::Gemm::Block::WideBlockMmad<typename Kernel::BlockMmadBSAG2>;
    using BlockMmadBSAG3 = NpuArch::Gemm::Block::WideBlockMmad<typename Kernel::BlockMmadBSAG3>;
    using EpilogueFAGSfmg = typename Kernel::EpilogueFAGSfmg;
    using SfmgParams = typename Kernel::SfmgParams;
    // wide 专用 softmax：独立 UB 双 stage，不改变 legacy softmax 的缓冲与事件。
    using EpilogueFAGOp = NpuArch::Epilogue::Block::WideSoftmax<ElementInput>;
    using SfmParams = typename EpilogueFAGOp::Params;
    using LayoutA1 = typename Kernel::LayoutA1;
    using LayoutB1 = typename Kernel::LayoutB1;
    using LayoutA2 = typename Kernel::LayoutA2;
    using LayoutB2 = typename Kernel::LayoutB2;
    using LayoutC2 = typename Kernel::LayoutC2;
    using LayoutB3 = typename Kernel::LayoutB3;
    using LayoutC3 = typename Kernel::LayoutC3;

    // 任务/分片坐标。随处理粒度逐级细化：
    //   work item 级只填 batch/head/Q 起点与序列长度；
    //   Q task 级由 WideQInfo 保存 Q 分片信息；
    //   tile 级再补 KV 维与 workspace 槽位（MakeWideTile）。
    struct TaskInfo {
        // batchIndex      : work item 所属 batch。
        // queryHeadIndex  : Q head 下标（GQA/MQA 下 KV head 由它换算）。
        // queryStart      : 分片首行的绝对位置（整条 Q 序列内的 s1 下标，不是 batch 内偏移）。
        // queryBlockIndex : queryStart 所在的用户 Q block 下标，用于 prefix 与 mask 寻址。
        uint32_t batchIndex, queryHeadIndex, queryStart, queryBlockIndex;
        // queryRows      : 本 tile 的有效 Q 行数（已按 prefix 裁剪）。
        // keyValueRows   : 本 tile 的有效 KV 列数（已按 prefix 裁剪）。
        // queryLength    : 整条 Q 序列长度（maxQSeqlen）。
        // keyValueLength : 整条 KV 序列长度（maxKvSeqlen）。
        uint32_t queryRows, keyValueRows, queryLength, keyValueLength;
        // queryLocalIndex      : Q 在 work item 内的局部下标（0..1），决定常驻 dO/Q 槽号与 L0C 槽号奇偶。
        // validKeyValueColumns : 交给 softmax 的有效 KV 列数（当前等于 keyValueRows）。
        uint32_t queryLocalIndex, validKeyValueColumns;
        // queryOffset     : Q/dO 在各自 tensor 中的元素偏移。
        // keyValueOffset  : K/V 在各自 tensor 中的元素偏移（tile 级会再叠加 KV 基础块偏移）。
        // workspaceOffset : S/dP/P/dS 在本核 workspace 中的元素偏移（已含核基址与 stage）。
        uint64_t queryOffset, keyValueOffset, workspaceOffset; // 各 tensor 的元素偏移。
    };

    // 一个 Q task（Q 分片）的物理信息，由 BuildWideWorkItem 生成、MakeWideTile 消费。
    struct WideQInfo {
        // 分片起点在 Q/dO tensor 中的元素偏移；queryRows==0 表示该分片被 prefix 裁空。
        uint64_t queryOffset;
        // 分片首行的绝对位置（整条 Q 序列内的 s1 下标）。
        uint32_t queryStart;
        // 首行所在的用户 Q block 下标。
        uint32_t queryBlockIndex;
        // 分片有效行数（prefix 裁剪后）；0 表示该 Q 在本 work item 内无需计算。
        uint32_t queryRows;
    };

    // 一个 packet 内按 KV 列聚合的边区间；packet 以紧凑边坐标同时支持 KV 优先的
    // Cube1/dKV 遍历与 Q 优先的 dQ/Vector 遍历。
    struct WideKvDesc {
        // 该 KV 列对应的基础块下标（basicKVBlockSize 粒度）。
        uint32_t kvBasicIdx;
        // 该 KV 列在 packet.edgeCode 中的首条边下标。
        uint16_t edgeBegin;
        // 该 KV 列的边数。
        uint16_t edgeCount;
    };

    // Q 优先索引：只登记有有效边的 Q，并给出其在 qOrder 中的连续区间；
    // 单 packet 至多八条边，故四个字段都可用单字节。
    struct WideQDesc {
        // Q 在 work item 内的局部下标（0..WIDE_MAX_Q_TASKS-1）。
        uint8_t qLocal;
        // 该 Q 在 qOrder 中的首条边下标。
        uint8_t edgeBegin;
        // 该 Q 的边数（同一个 Q 的边在 qOrder 中连续）。
        uint8_t edgeCount;
        // 对齐填充，保证结构体无隐藏空隙。
        uint8_t reserved;
    };

    // 一个 packet：满足"边数 <= edgeCapacity 且唯一 KV 列 <= WIDE_MAX_KV_COLUMNS"
    // 的一组边，外加两侧共用的遍历索引与下一 packet 的游标。
    // AIC 与 AIV 用同一个 BuildWidePacket 逐字节一致地重建它，因此本结构体内
    // 不得出现依赖核类型或核号的字段。
    struct WidePacketInfo {
        // 压缩边坐标：位 [0,8) 为 qLocal，位 [8,24) 为 kvLocal（packet.kv 的下标）。
        // 边的存储顺序 = 发现顺序（KV 优先）；其下标同时是 workspace 的 tile 槽号，
        // 因此任何对 edgeCode 的重排都必须同步调整 tile 的 workspaceOffset。
        uint32_t edgeCode[WIDE_MAX_EDGES];
        // 按 KV 列聚合的索引，供 Cube1/dKV 的 KV 外层循环使用。
        WideKvDesc kv[WIDE_MAX_KV_COLUMNS];
        // edgeCode 的下标序列，按 qLocal 稳定排序，供 dQ/Vector 做 Q 优先遍历。
        uint16_t qOrder[WIDE_MAX_EDGES];
        // Q 优先索引；长度取 WIDE_MAX_EDGES（Q 数必然不超过边数）。
        WideQDesc activeQ[WIDE_MAX_EDGES];
        // 本 packet 的实际边数；为 0 表示无有效边；调用方仍更新扫描游标，但不切换 stage 或发送令牌。
        uint32_t edgeCount;
        // 本 packet 的唯一 KV 列数（= kv 的有效长度）。
        uint32_t uniqueKvCount;
        // 本 packet 的有效 Q 数（= activeQ 的有效长度）。
        uint32_t activeQCount;
        // 下一 packet 的 KV 列游标（因边数或 KV 列数达上限而中断时的续扫位置）。
        uint32_t nextKvStart;
        // 下一 packet 的 Q 游标：KV 列未扫完时，指向该列内下一个未消费的 Q。
        uint32_t nextQStart;
        // 本 packet 所在 stage 的 workspace 元素基址（= coreIdx*coreElements + stage*stageElements）。
        uint64_t workspaceBase;
    };

    // mask 窗口缓存：sparse mask 按 (batch, head, qBlock, kvBlock) 展开为字节；
    // 这里把"每个 Q task 一行"的 4 个字节惰性载入，并在同一 work item 内跨 packet
    // 复用，避免每条边都做一次跨步 GM 读；窗口宽度与 Q task 数量无关。
    struct WideMaskScanState {
        // 每个 Q task 一个 4 字节窗口；仅当 loadedMaskWindows 对应位为 1 时有效。
        uint32_t maskWindows[WIDE_MAX_Q_TASKS];
        // 位 i 表示 maskWindows[i] 已装填，覆盖 [maskWindowStart, maskWindowEnd)。
        WideQMask loadedMaskWindows = 0;
        // 当前窗口覆盖的起始 KV 基础块下标（4 的倍数）。
        uint32_t maskWindowStart = ~0U;
        // 当前窗口的结束 KV 基础块下标（不含；不超过该 Q 的 qLimits）。
        uint32_t maskWindowEnd = 0;
    };

    // sparse mask 的 strides，单位为字节（调用方按字节寻址）。
    struct MaskLayoutInfo {
        // 一行 mask 的字节数（= maxKvSeqlen 向上取整到 blockShapeY 的块数）。
        uint32_t kvBlockCount;
        // 相邻 Q head 之间的字节步长。
        uint64_t headStride;
        // 相邻 batch 之间的字节步长。
        uint64_t batchStride;
    };

    // softmax 侧的 LSE/D 元数据缓存槽数（对应 block_epilogue_wide_softmax.hpp 的
    // WIDE_Q_METADATA_SLOTS）；一个 packet 内参与计算的 Q 数不得超过该槽数。
    constexpr static uint32_t WIDE_Q_METADATA_SLOTS = 4;
    static_assert(WIDE_MAX_Q_TASKS <= WIDE_Q_METADATA_SLOTS, "wide Q metadata slots insufficient");

    // workspace 分区（与 Host 的写入顺序严格一致，顺序不可调换）：
    //   S(FP32) | dP(FP32) | P(lowp) | dS(lowp) | dQ | dK | dV | D(rowsum(dO∘O))
    // 元素偏移在 FP32 面与低精度面之间共用：FP32 面用 sizeof(float)，低精度面用 sizeof(ElementInput)。
    // 本结构只保存各分区的字节数，不持有 GM 指针，也不改变任何同步状态。
    struct WideWorkspaceLayout {
        __aicore__ inline explicit WideWorkspaceLayout(const __gm__ BlockSparseAttentionGradTilingData* tilingData)
            : rawBytes_(tilingData->wideRawPlaneBytes),
              sBytes_(tilingData->sOutSize),
              dpBytes_(tilingData->dPOutSize),
              dqBytes_(tilingData->dQOutSize),
              dkBytes_(tilingData->dKOutSize),
              dvBytes_(tilingData->dVOutSize),
              coreElements_(tilingData->wideWorkspaceCoreElements),
              stageElements_(tilingData->wideWorkspaceStageElements)
        {}

        // 各分区相对 workspace 起点的字节偏移：本文件内 workspace 布局的唯一权威定义。
        __aicore__ inline uint64_t score() const
        {
            return 0;
        }
        __aicore__ inline uint64_t probabilityGrad() const
        {
            return sBytes_;
        }
        __aicore__ inline uint64_t probability() const
        {
            return rawBytes_;
        }
        __aicore__ inline uint64_t scoreGrad() const
        {
            return sBytes_ + rawBytes_;
        }
        __aicore__ inline uint64_t queryGrad() const
        {
            return sBytes_ + dpBytes_;
        }
        __aicore__ inline uint64_t keyGrad() const
        {
            return queryGrad() + dqBytes_;
        }
        __aicore__ inline uint64_t valueGrad() const
        {
            return keyGrad() + dkBytes_;
        }
        // D = rowsum(dO∘O) 的分区起点，也是 workspace 中最后一个分区。
        __aicore__ inline uint64_t rowDelta() const
        {
            return valueGrad() + dvBytes_;
        }

        // S/dP 的元素数；P/dS 与它们元素数相同，故共用该长度（若精度面元素宽度不同须分别计算）。
        __aicore__ inline uint64_t rawPlaneElements() const
        {
            return rawBytes_ / sizeof(float);
        }
        __aicore__ inline uint64_t queryGradElements() const
        {
            return dqBytes_ / sizeof(float);
        }
        __aicore__ inline uint64_t keyGradElements() const
        {
            return dkBytes_ / sizeof(float);
        }
        __aicore__ inline uint64_t valueGradElements() const
        {
            return dvBytes_ / sizeof(float);
        }

        // 本核、本 stage 的 workspace 元素基址（每个 stage 一份独立 tile 槽）。
        __aicore__ inline uint64_t stageBase(uint32_t coreIndex, uint32_t stageIndex) const
        {
            return static_cast<uint64_t>(coreIndex) * coreElements_ +
                   static_cast<uint64_t>(stageIndex) * stageElements_;
        }
        // 一条边（tile）在 FP32 面 / 低精度面上的字节地址。
        __aicore__ inline uint64_t rawByteOf(uint64_t tileOffset) const
        {
            return tileOffset * sizeof(float);
        }
        __aicore__ inline uint64_t lowpByteOf(uint64_t tileOffset) const
        {
            return rawBytes_ + tileOffset * sizeof(ElementInput);
        }

    private:
        uint64_t rawBytes_;
        uint64_t sBytes_;
        uint64_t dpBytes_;
        uint64_t dqBytes_;
        uint64_t dkBytes_;
        uint64_t dvBytes_;
        uint32_t coreElements_;
        uint32_t stageElements_;
    };

    // 一条边（= 一个 workspace tile 槽）的只读派生视图：只保留四个执行体真正用到的 6 个字段。
    // 若直接缓存 TaskInfo，每 packet 需 8 * sizeof(TaskInfo) 字节；本结构把开销压到 8 * 36 字节。
    struct WideTileSlot {
        uint64_t keyValueOffset;  // KV 序列内的元素偏移（K/V 输入）
        uint64_t workspaceOffset; // 本边在 workspace 中的 tile 槽起点（元素）
        uint64_t queryOffset;     // Q 序列内的元素偏移（dQ 输出）
        uint32_t queryRows;       // 该 Q 分片的有效行数
        uint32_t keyValueRows;    // 该 KV 列的有效行数
        uint32_t queryLocalIndex; // Q 在本 work item 内的下标（决定常驻槽与 L0C 槽奇偶）
    };

    // 一个 packet 内所有边的 tile 坐标缓存（下标 = 边下标 = workspace tile 槽号）。
    struct WideTileCache {
        WideTileSlot slots[WIDE_MAX_EDGES];
    };

    // 两个 AIV 子核对同一 tile 的行切分。契约：两个子核的 executeRows 之和等于 queryRows，
    // 两区间不重叠，子核 0 取前 ceil(rows/2) 行（奇数行时上半区多一行）。
    struct WideSubBlockRows {
        uint32_t firstHalfRows; // 子核 0 的行数，也是子核 1 的行起点偏移
        uint32_t executeRows;   // 本子核实际执行的行数，0 表示本子核不参与该分片
    };

    __aicore__ inline WideSubBlockRows WideSplitRows(uint32_t queryRows, uint32_t subBlockIndex) const
    {
        const uint32_t firstHalfRows = (queryRows + 1) / 2;
        return WideSubBlockRows{firstHalfRows, subBlockIndex == 0 ? firstHalfRows : queryRows - firstHalfRows};
    }

    // 元数据槽 = 参与计算的 Q 在本 packet 内的序号（按 activeQ 顺序），不是 Q 的绝对下标。
    // 必须与 softmax 侧 PrefetchWide/ComputeWide 的槽号推导保持一致。
    __aicore__ inline uint32_t WideMetadataSlot(uint32_t activeQueryOrdinal) const
    {
        return activeQueryOrdinal & (WIDE_Q_METADATA_SLOTS - 1U);
    }

    // 跨核 stage 令牌：ready = CUBE2VEC 5/6，done = VEC2CUBE 7/8（事件号定义在同名 common 头）。
    // 本函数只做 stage 到事件号的映射；收发动作仍显式写在两个执行入口里，便于检索跨核同步点。
    __aicore__ inline uint32_t WideReadyFlag(uint32_t stageIndex) const
    {
        return stageIndex == 0 ? WIDE_CUBE2VEC_STAGE0 : WIDE_CUBE2VEC_STAGE1;
    }
    __aicore__ inline uint32_t WideDoneFlag(uint32_t stageIndex) const
    {
        return stageIndex == 0 ? WIDE_VEC2CUBE_STAGE0 : WIDE_VEC2CUBE_STAGE1;
    }

    // 一次 AIC 调用的 GM 视图。该结构仅聚合原有 GlobalTensor，不复制数据。
    struct WideAicTensorViews {
        AscendC::GlobalTensor<ElementInput> query;
        AscendC::GlobalTensor<ElementInput> key;
        AscendC::GlobalTensor<ElementInput> value;
        AscendC::GlobalTensor<ElementA1> outputGrad;
        AscendC::GlobalTensor<float> rawScore;
        AscendC::GlobalTensor<float> rawProbabilityGrad;
        AscendC::GlobalTensor<ElementInput> probability;
        AscendC::GlobalTensor<ElementInput> scoreGrad;
        AscendC::GlobalTensor<float> queryGrad;
        AscendC::GlobalTensor<float> keyGrad;
        AscendC::GlobalTensor<float> valueGrad;
    };

    // 一个 work item 的 Q 描述、扫描上界与跨 packet 游标。
    struct WideWorkState {
        WideQInfo queries[WIDE_MAX_Q_TASKS];
        uint32_t queryCount;
        TaskInfo nextWork;
        uint32_t queryKvLimits[WIDE_MAX_Q_TASKS];
        uint64_t maskRowOffsets[WIDE_MAX_Q_TASKS];
        uint32_t kvScanLimit;
        uint32_t nextKvIndex;
        uint32_t nextQueryIndex;
        WideMaskScanState maskScanState;
    };

    // 物理 Q 任务推进与 tile 坐标生成。
    /**
     * @brief 取本核第一个 work item 的坐标与地址。
     * @param tilingData Host 下发的 beginBatch / beginHead / beginQSeqOffset 起点表与序列长度。
     * @param coreIndex AIC 核号（AIV 侧传 coreIdx = blockIdx/2，与 AIC 一一对应）。
     * @param task 输出：batch/head/Q 起点、序列长度与 Q/KV 元素偏移。
     * @note 起点表由 Host 的 AssignWideCoreTasks 按 cyclic 规则填写（起点 coreIndex、步长
     *       wideTaskStride）；KV 偏移按 GQA 的 numHeads/kvHeads 折算到对应 KV head。
     * @note wide 仅接收 BNSD，输入长度与 head 映射已由 Host 校验，这里不做重复校验。
     */
    __aicore__ inline void InitializeWorkItem(const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                              uint32_t coreIndex, TaskInfo& task) const
    {
        task.batchIndex = tilingData->beginBatch[coreIndex];
        task.queryHeadIndex = tilingData->beginHead[coreIndex];
        task.queryStart = tilingData->beginQSeqOffset[coreIndex];
        task.queryLength = tilingData->maxQSeqlen;
        task.keyValueLength = tilingData->maxKvSeqlen;
        // Q：先把 (batch, head) 压成全局 head 下标，再按 (head, s1) 展平并乘 headDim。
        const uint64_t headIndex = static_cast<uint64_t>(task.batchIndex) * tilingData->numHeads + task.queryHeadIndex;
        task.queryOffset = (headIndex * task.queryLength + task.queryStart) * tilingData->headDim;
        // KV：GQA/MQA 下多个 Q head 共享一个 KV head，用 numHeads/kvHeads 折算（Host 已保证整除）。
        task.keyValueOffset = (static_cast<uint64_t>(task.batchIndex) * tilingData->kvHeads +
                               task.queryHeadIndex / (tilingData->numHeads / tilingData->kvHeads)) *
                              task.keyValueLength * tilingData->headDim;
    }

    /**
     * @brief 将连续物理 Q 分片聚合为 work item，并生成下一个 work item 的起点。
     * @param tilingData 聚合容量、物理 block 尺寸和 cyclic 步长。
     * @param current 当前 work item 的物理起点，不能用 prefix 长度推进它。
     * @param qInfos 输出 Q 分片的物理地址和有效行数。
     * @param nextWork 输出下一 work item 起点。
     * @param needNext 当前核是否还有下一个 work item。
     * @return 当前 work item 的实际 Q task 数。
     * @note 流程框架：段1 取容量（wideQTasksPerWorkItem）→
     *       段2 前向聚合物理 Q 分片（CollectWideQueryFragments：用户 block 末尾、128 行、序列末尾
     *       三重截断，每片查一次 prefix）→ 段3 后向反解下一 work item 起点（AdvanceWideWorkItem，
     *       仅在 needNext 时执行）。
     *
     * @note 本函数是"设备侧重算 Host 划分"的入口：两个 helper 的推导必须与 Host 的
     *       AssignWideCoreTasks 保持一致。
     * @note 段3 在设备侧重算 Host 的任务划分：workItemsPerHead / firstQTask 的推导必须与 Host 的
     *       AssignWideCoreTasks 保持一致；wideTaskStride 由 Host 下发为 blockDim_（AIC 核数）。
     */
    __aicore__ inline uint32_t BuildWideWorkItem(const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                                 const TaskInfo& current, WideQInfo* qInfos, TaskInfo& nextWork,
                                                 bool needNext) const
    {
        // 段1+段2：容量由 Host 下发（wideQTasksPerWorkItem，常量 WIDE_COMPACT_Q_TASKS = 2）。
        const uint32_t capacity = tilingData->wideQTasksPerWorkItem;
        const uint32_t qCount = CollectWideQueryFragments(tilingData, current, qInfos, capacity);
        // 段3：只有还存在下一个 work item 时才反解（尾次调用省一次除法链）。
        if (needNext) {
            nextWork = AdvanceWideWorkItem(tilingData, current, capacity);
        }
        return qCount;
    }

    // 段2：前向聚合本 work item 的物理 Q 分片，返回实际分片数。
    // 三重截断：不跨用户 block、不超过 128 行、不超过序列末尾；每片查一次 prefix 得到有效行数。
    // 契约：qInfos 至少能容纳 capacity 个元素；queryRows 可能小于 physicalRows（被 mask 裁短），
    //       而 queryStart 始终按 physicalRows 前进（分片边界只由物理布局决定）。
    __aicore__ inline uint32_t CollectWideQueryFragments(const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                                         const TaskInfo& current, WideQInfo* qInfos,
                                                         uint32_t capacity) const
    {
        const uint32_t blockX = tilingData->blockShapeX;
        uint32_t queryStart = current.queryStart;
        uint32_t qCount = 0;
        while (qCount < capacity && queryStart < current.queryLength) {
            // 分片不跨用户 block：先截到 block 末尾，再截到 128 行与序列末尾。
            const uint32_t fragmentOffset = queryStart % blockX;
            uint32_t physicalRows = blockX - fragmentOffset;
            physicalRows = physicalRows < 128 ? physicalRows : 128;
            physicalRows =
                physicalRows < current.queryLength - queryStart ? physicalRows : current.queryLength - queryStart;
            WideQInfo& q = qInfos[qCount++];
            q.queryOffset =
                current.queryOffset + static_cast<uint64_t>(queryStart - current.queryStart) * tilingData->headDim;
            q.queryStart = queryStart;
            q.queryBlockIndex = queryStart / blockX;
            // fragmentOffset 让 mask 行内偏移与物理起点对齐（mask 关闭时该值不参与计算）。
            q.queryRows = WidePrefixLength(tilingData, current.batchIndex, current.queryHeadIndex, q.queryBlockIndex, 0,
                                           physicalRows, fragmentOffset);
            queryStart += physicalRows;
        }
        return qCount;
    }

    // 段3：后向反解下一个 work item 的起点，返回新的 work item 描述。
    // 先反解出 (全局 head 下标, 该 head 内的 work item 序号)，按 cyclic 步长推进，再正解回
    // (batch, head, queryStart)。之所以能反解，是因为 Host 的 AssignWideCoreTasks 把任务按
    // "每个 head 有 workItemsPerHead 个 work item"展平后做等步长划分，故 work item 序号与
    // (head, Q 分片) 之间是双射。
    __aicore__ inline TaskInfo AdvanceWideWorkItem(const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                                   const TaskInfo& current, uint32_t capacity) const
    {
        const uint32_t blockX = tilingData->blockShapeX;
        // 一个用户 Q block 被切成几个 128 行的物理分片（blockShapeX <= 128 时为 1，== 256 时为 2）。
        const uint32_t fragmentsPerBlock = (blockX + 127) / 128;
        const uint32_t fullBlocks = current.queryLength / blockX;
        const uint32_t qTasks = fullBlocks * fragmentsPerBlock + (current.queryLength % blockX + 127) / 128;
        const uint32_t workItemsPerHead = (qTasks + capacity - 1) / capacity;
        const uint32_t firstQTask = current.queryStart / blockX * fragmentsPerBlock + current.queryStart % blockX / 128;
        const uint64_t headIndex =
            static_cast<uint64_t>(current.batchIndex) * tilingData->numHeads + current.queryHeadIndex;
        // nextIndex：全局 work item 序号。wideTaskStride 由 Host 在 wide 分支下发为 blockDim_
        // （等于 AIC 核数），因而同一核上相邻两个 work item 恰相隔一整轮，这正是本文件用
        // "+ 步长"完成推进、而无需记录每核分区表的原因。
        const uint64_t nextIndex = headIndex * workItemsPerHead + firstQTask / capacity + tilingData->wideTaskStride;
        const uint64_t nextHead = nextIndex / workItemsPerHead;
        const uint32_t nextQTask = (nextIndex % workItemsPerHead) * capacity;
        TaskInfo nextWork = current;
        nextWork.batchIndex = nextHead / tilingData->numHeads;
        nextWork.queryHeadIndex = nextHead % tilingData->numHeads;
        // 由 head 内的 work item 序号还原 Q 行偏移：分片序号 -> block 下标 + 块内 128 行偏移。
        nextWork.queryStart = nextQTask / fragmentsPerBlock * blockX + nextQTask % fragmentsPerBlock * 128;
        nextWork.queryOffset = (nextHead * current.queryLength + nextWork.queryStart) * tilingData->headDim;
        nextWork.keyValueOffset = (static_cast<uint64_t>(nextWork.batchIndex) * tilingData->kvHeads +
                                   nextWork.queryHeadIndex / (tilingData->numHeads / tilingData->kvHeads)) *
                                  current.keyValueLength * tilingData->headDim;
        return nextWork;
    }

    // sparse mask 有效前缀长度的唯一入口（三处调用点语义各不相同）：
    //   1) CollectWideQueryFragments：把物理 Q 行数裁成有效前缀行数；
    //   2) MakeWideTile：把 KV 列行数裁到该 Q 的有效前缀；
    //   3) BuildWidePacket：判断某 KV 列是否已越出该 Q 的有效前缀（整列跳过）。
    // 契约：mask 关闭（hasPerBlockMask == 0）时 kernel_ 直接返回 physicalLength，不读 GM；
    //       mask 开启时每次调用读取一个 GM 标量，故列扫描循环里必须先取上界再比较，避免重复调用。
    //       两种情况下 Host 都保证"物理长度 >= 有效前缀"，返回值只可能小于等于 physicalLength。
    __aicore__ inline uint32_t WidePrefixLength(const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                                uint32_t batchIndex, uint32_t queryHeadIndex, uint32_t blockIndex,
                                                uint32_t kvSideFlag, uint32_t physicalLength,
                                                uint32_t fragmentOffset = 0) const
    {
        return kernel_.GetEffectivePrefixLength(params_, tilingData, batchIndex, queryHeadIndex, blockIndex, kvSideFlag,
                                                physicalLength, fragmentOffset);
    }

    /**
     * @brief 由边下标派生一个 tile：Q 维取对应 Q 分片，KV 维取该边的 KV 基础块。
     * @param workInfo 当前 work item。
     * @param qInfos Q 分片数组。
     * @param packet 当前 packet（提供边编码与 KV 聚合信息）。
     * @param edgeIndex 边下标，同时决定 workspace 的 tile 槽位。
     * @param tilingData Host tiling（提供 basicKVBlockSize 与 headDim）。
     * @return 完整的 tile 坐标；AIC 与 AIV 必须用同一个 edgeIndex 调用本函数。
     * @note KV 行数先按"是否最后一个基础块"取整块或余量，再用 prefix 裁剪；
     *       workspaceOffset 取 packet.workspaceBase 之后的第 edgeIndex 个 WIDE_TILE_ELEMENTS 槽。
     * @note 流程框架：拆边编码 → 继承 work item 坐标并覆盖 Q 侧字段 → 算该 KV 列行数（末块取余量）→
     *       prefix 裁剪 KV 列 → 叠加 KV 基础块元素偏移 → 定 workspace tile 槽 →
     *       定常驻槽号（queryLocalIndex）与 softmax 有效列数。
     */
    __aicore__ inline TaskInfo MakeWideTile(const TaskInfo& workInfo, const WideQInfo* qInfos,
                                            const WidePacketInfo& packet, uint32_t edgeIndex,
                                            const __gm__ BlockSparseAttentionGradTilingData* tilingData) const
    {
        const uint32_t edgeCode = packet.edgeCode[edgeIndex];
        const uint32_t qLocal = WideEdgeQ(edgeCode);
        const uint32_t kvLocal = WideEdgeKv(edgeCode);
        const uint32_t kvBasicIdx = packet.kv[kvLocal].kvBasicIdx;
        TaskInfo tile = workInfo;
        const WideQInfo& query = qInfos[qLocal];
        tile.queryOffset = query.queryOffset;
        tile.queryStart = query.queryStart;
        tile.queryBlockIndex = query.queryBlockIndex;
        tile.queryRows = query.queryRows;
        // 该 KV 序列按 basicKVBlockSize 切成多少块，用于判断末块（末块行数可能不足一块）。
        const uint32_t kvTileCount =
            (tile.keyValueLength + tilingData->basicKVBlockSize - 1) / tilingData->basicKVBlockSize;
        tile.keyValueRows = kvBasicIdx + 1 < kvTileCount ?
                                tilingData->basicKVBlockSize :
                                tile.keyValueLength - kvBasicIdx * tilingData->basicKVBlockSize;
        // 按 prefix 合同裁剪 KV 列（与 BuildWidePacket 的扫描使用同一接口，保证结论一致）。
        tile.keyValueRows =
            WidePrefixLength(tilingData, tile.batchIndex, tile.queryHeadIndex, kvBasicIdx, 1, tile.keyValueRows);
        tile.keyValueOffset += static_cast<uint64_t>(kvBasicIdx) * tilingData->basicKVBlockSize * tilingData->headDim;
        // 每条边独占一个 tile 槽，槽号就是边下标（因此 reorder 边必须同步改这里）。
        tile.workspaceOffset = packet.workspaceBase + static_cast<uint64_t>(edgeIndex) * WIDE_TILE_ELEMENTS;
        // queryLocalIndex 决定常驻 dO/Q 槽号与 L0C 槽号奇偶；validKeyValueColumns 供 softmax 使用。
        tile.queryLocalIndex = qLocal;
        tile.validKeyValueColumns = tile.keyValueRows;
        return tile;
    }

    // 把一个 TaskInfo 投影为执行体需要的 6 个字段。
    __aicore__ inline WideTileSlot MakeWideTileSlot(const TaskInfo& workInfo, const WideQInfo* qInfos,
                                                    const WidePacketInfo& packet, uint32_t edgeIndex,
                                                    const __gm__ BlockSparseAttentionGradTilingData* tilingData) const
    {
        const TaskInfo tile = MakeWideTile(workInfo, qInfos, packet, edgeIndex, tilingData);
        WideTileSlot slot;
        slot.keyValueOffset = tile.keyValueOffset;
        slot.workspaceOffset = tile.workspaceOffset;
        slot.queryOffset = tile.queryOffset;
        slot.queryRows = tile.queryRows;
        slot.keyValueRows = tile.keyValueRows;
        slot.queryLocalIndex = tile.queryLocalIndex;
        return slot;
    }

    // packet 构造完成后调用一次：为本 packet 的每条边算一次 tile 坐标。
    // 之后四个执行体只读 slots[]，不再重复调用 MakeWideTile（mask 开启时可省掉每条边一次 GM 标量读）。
    __aicore__ inline void ComputeWideTileCache(const TaskInfo& workInfo, const WideQInfo* qInfos,
                                                const WidePacketInfo& packet,
                                                const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                                WideTileCache& cache) const
    {
        for (uint32_t edge = 0; edge < packet.edgeCount; ++edge) {
            cache.slots[edge] = MakeWideTileSlot(workInfo, qInfos, packet, edge, tilingData);
        }
    }

    // AIV 侧的 tile 缓存：FillWideSfmParams 需要完整的 tile 坐标（queryStart / batch / head /
    // validKeyValueColumns），故这里直接缓存 TaskInfo，而不是 AIC 用的投影结构。
    struct WideTileInfoCache {
        TaskInfo slots[WIDE_MAX_EDGES];
    };

    __aicore__ inline void ComputeWideTileInfoCache(const TaskInfo& workInfo, const WideQInfo* qInfos,
                                                    const WidePacketInfo& packet,
                                                    const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                                    WideTileInfoCache& cache) const
    {
        for (uint32_t edge = 0; edge < packet.edgeCount; ++edge) {
            cache.slots[edge] = MakeWideTile(workInfo, qInfos, packet, edge, tilingData);
        }
    }

    // 稀疏 mask 寻址与 packet 构造；AIC/AIV 共用同一规则。
    /**
     * @brief 计算 sparse mask 的三个字节 stride。
     * @param tilingData Host tiling（提供 maxQSeqlen/maxKvSeqlen/blockShapeX/blockShapeY/numHeads）。
     * @return kvBlockCount 为一行字节数，headStride/batchStride 为 head/batch 的字节步长。
     * @note 这里用的是序列上界（max*Seqlen）而非当前序列长度，与 Host 的 mask 布局一致。
     */
    __aicore__ inline MaskLayoutInfo GetMaskLayout(const __gm__ BlockSparseAttentionGradTilingData* tilingData) const
    {
        const uint32_t qBlockCount = (tilingData->maxQSeqlen + tilingData->blockShapeX - 1) / tilingData->blockShapeX;
        const uint32_t kvBlockCount = (tilingData->maxKvSeqlen + tilingData->blockShapeY - 1) / tilingData->blockShapeY;
        MaskLayoutInfo layout;
        layout.kvBlockCount = kvBlockCount;
        layout.headStride = static_cast<uint64_t>(qBlockCount) * kvBlockCount;
        layout.batchStride = static_cast<uint64_t>(tilingData->numHeads) * layout.headStride;
        return layout;
    }

    /**
     * @brief 读取 mask 的 1..4 个连续字节，按小端序拼成一个 uint32 窗口。
     * @param params Kernel 的 Params（提供 blockSparseMask 基址）。
     * @param absoluteOffset 起始字节偏移（由 mask 行起点加列偏移得到）。
     * @param byteCount 字节数，1..4；调用方保证不跨行。
     * @return 低 byteCount 个字节有效，高位为 0。
     * @note 对齐且恰好 4 字节时用一次 32 位读；否则逐字节读，避免非对齐访问。
     * @note 调用方保证窗口位于一行内且不超过四字节。
     */
    __aicore__ inline uint32_t LoadMaskByteWindow(const Params& params, uint64_t absoluteOffset,
                                                  uint32_t byteCount) const
    {
        __gm__ uint8_t* mask = reinterpret_cast<__gm__ uint8_t*>(params.blockSparseMask);
        uint32_t values = 0;
        uint32_t copiedBytes = 0;
        while (copiedBytes < byteCount) {
            const uint64_t offset = absoluteOffset + copiedBytes;
            const uint32_t remaining = byteCount - copiedBytes;
            // 只有「4 字节对齐且剩余恰好 4 字节」才走整字读；其余情况逐字节拼接（兼容非对齐）。
            if ((offset & 3ULL) == 0 && remaining == sizeof(uint32_t)) {
                const uint32_t word = *reinterpret_cast<__gm__ uint32_t*>(mask + offset);
                values |= word << (copiedBytes * 8U);
                copiedBytes += sizeof(uint32_t);
            } else {
                values |= static_cast<uint32_t>(mask[offset]) << (copiedBytes * 8U);
                ++copiedBytes;
            }
        }
        return values;
    }

    /**
     * @brief 取压缩边坐标中的 qLocal。
     * @param edgeCode 边编码，位 [0,8) 为 qLocal，位 [8,24) 为 kvLocal。
     * @return Q 在 work item 内的局部下标。
     */
    __aicore__ inline uint32_t WideEdgeQ(uint32_t edgeCode) const
    {
        return edgeCode & 0xffU;
    }

    /**
     * @brief 取压缩边坐标中的 kvLocal。
     * @param edgeCode 边编码。
     * @return KV 列在 packet.kv 中的下标。
     */
    __aicore__ inline uint32_t WideEdgeKv(uint32_t edgeCode) const
    {
        return (edgeCode >> 8) & 0xffffU;
    }

    /**
     * @brief 从 KV/Q 二维游标收集最多八条有效边，同时建立 Q 优先索引。
     * @param params 输入 tensor 地址。
     * @param workInfo 当前 work item 的 batch/head 和物理地址。
     * @param tilingData packet 容量、prefix 合同和物理 tile 尺寸。
     * @param workspaceBase 当前核、当前 stage 的 workspace 元素偏移。
     * @param[in,out] work 当前 work item 的 Q 扫描上界、mask 行偏移和 KV/Q 游标。
     * @param packet 输出边、双遍历索引及下一 packet 游标。
     * @note AIC/AIV 必须使用相同输入重建 packet；空 packet 仍更新扫描游标，
     *       调用方仅在存在有效边时切换 stage 并收发跨核令牌。
     * @note 流程框架：阶段A 初始化 packet 头与续扫起点 → 阶段B 按 KV 优先收集边
     *       （B0 续扫起点纠正 → B1 跨列纠正与 prefix 跳列 → B2 mask 窗口按需换页 → B3 登记 KV 分组）→
     *       阶段C 构造 Q 优先索引（BuildWideQIndex）→ 阶段D 提交扫描游标（CommitWidePacketCursor）。
     *       阶段B2/B3 的换页与按需加载分别由 EnsureWideMaskWindow / LoadWideMaskWindow 承担。
     * @note 两处容量检查含义不同：KV 列循环条件里的检查是"进入下一列前提前退出"（省一次 mask 读），
     *       Q 内循环里的检查是"列内建边的硬上限"（保证不越界写 edgeCode），两者都不能删。
     * @note 两处"收敛"含义不同：maskWindowEnd 收敛到 kvScanLimit（全体 Q 的扫描上界），
     *       单个 Q 的读取长度收敛到 queryKvLimits[qLocal]。
     */
    __aicore__ inline void BuildWidePacket(const Params& params, const TaskInfo& workInfo,
                                           const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                           uint64_t workspaceBase, WideWorkState& work, WidePacketInfo& packet) const
    {
        // ---- 阶段A：初始化 packet 头与续扫起点（空 packet 也走这里，游标推进不能跳过）----
        packet.edgeCount = 0;
        packet.uniqueKvCount = 0;
        packet.activeQCount = 0;
        packet.nextKvStart = work.nextKvIndex;
        packet.nextQStart = work.nextQueryIndex;
        packet.workspaceBase = workspaceBase;
        // edgeCapacity 由 Host 下发（当前等于 WIDE_MAX_EDGES）；packet 的容量约束有两条：
        // 边数不超过 edgeCapacity（列内硬上限），唯一 KV 列数不超过 WIDE_MAX_KV_COLUMNS（列数上限）。
        const uint32_t edgeCapacity = tilingData->widePacketEdgeCapacity;
        // 每个 Q 行缓存四列；保持 KV 优先顺序，用移位判断替代逐边 GM 读取。
        constexpr uint32_t MASK_WINDOW_BYTES = sizeof(uint32_t);

        // ---- 阶段B：按 KV 优先收集边 ----
        // B0 续扫起点纠正：KV 列还有可扫的，但上一个 packet 已把 Q 扫完时，前进到下一个 KV 列。
        uint32_t kvBasicIdx = work.nextKvIndex;
        uint32_t qScanStart = work.nextQueryIndex;
        if (qScanStart >= work.queryCount && kvBasicIdx < work.kvScanLimit) {
            ++kvBasicIdx;
            qScanStart = 0;
            SetWidePacketCursor(packet, kvBasicIdx, qScanStart);
        }
        while (kvBasicIdx < work.kvScanLimit && packet.edgeCount < edgeCapacity &&
               packet.uniqueKvCount < WIDE_MAX_KV_COLUMNS) {
            // B1 跨列纠正：AIC/AIV 使用相同的逐 Q head prefix 值，保证 packet 与事件数量一致；
            //    该列整列被 prefix 裁掉时直接跳过（此处是列循环的第一道提前退出，可省一次 mask 读）。
            if (WidePrefixLength(tilingData, workInfo.batchIndex, workInfo.queryHeadIndex, kvBasicIdx, 1,
                                 tilingData->basicKVBlockSize) == 0) {
                ++kvBasicIdx;
                qScanStart = 0;
                SetWidePacketCursor(packet, kvBasicIdx, qScanStart);
                continue;
            }
            // B2 mask 窗口按需换页；随后在 Q 循环内按需加载该 Q 的 1..4 个字节。
            EnsureWideMaskWindow(work, kvBasicIdx, MASK_WINDOW_BYTES);
            // 本 KV 列的局部下标；新列在扫描完成后才登记进 packet.kv。
            const uint32_t kvLocal = packet.uniqueKvCount;
            const uint32_t columnBegin = packet.edgeCount;
            uint32_t qLocal = qScanStart;
            const uint32_t maskShift = (kvBasicIdx - work.maskScanState.maskWindowStart) * 8U;
            for (; qLocal < work.queryCount; ++qLocal) {
                // 该 Q 的扫描上界由 PrepareWideWorkItem 预先算好（= min(有效前缀列数, kvScanLimit)）。
                if (kvBasicIdx >= work.queryKvLimits[qLocal]) {
                    continue;
                }
                LoadWideMaskWindow(params, work, qLocal, MASK_WINDOW_BYTES);
                // mask 非零才建立一条边：用移位取该 KV 列对应的那一个字节，避免逐边 GM 读。
                if (((work.maskScanState.maskWindows[qLocal] >> maskShift) & 0xffU) == 0) {
                    continue;
                }
                // 列内建边的硬上限：写满 edgeCode 就必须收手（不能删，否则越界写）。
                if (packet.edgeCount == edgeCapacity) {
                    break;
                }
                // 边编码：(kvLocal << 8) | qLocal；边下标即 workspace tile 槽号。
                packet.edgeCode[packet.edgeCount] = qLocal | (kvLocal << 8);
                ++packet.edgeCount;
            }

            // B3 登记 KV 分组：本 KV 列至少建立了一条边，才登记该列（否则该列不进 packet.kv）。
            if (packet.edgeCount != columnBegin) {
                WideKvDesc& kv = packet.kv[packet.uniqueKvCount++];
                kv.kvBasicIdx = kvBasicIdx;
                kv.edgeBegin = static_cast<uint16_t>(columnBegin);
                kv.edgeCount = static_cast<uint16_t>(packet.edgeCount - columnBegin);
            }

            if (qLocal < work.queryCount) {
                // 密集 KV 列可跨 packet；从未消费的 Q 边续扫，dKV 用原子写汇总各部分。
                SetWidePacketCursor(packet, kvBasicIdx, qLocal);
                break;
            }
            ++kvBasicIdx;
            qScanStart = 0;
            SetWidePacketCursor(packet, kvBasicIdx, qScanStart);
        }

        // ---- 阶段C：构造 Q 优先索引（qOrder + activeQ 连续区间）----
        BuildWideQIndex(packet);
        // ---- 阶段D：把 packet 游标提交为本 work item 的当前游标 ----
        CommitWidePacketCursor(work, packet);
    }

    // packet 续扫游标：packet.next* 表示"下一个 packet 从哪继续"，work.next* 表示本 work item
    // 当前的扫描位置。两者必须成对更新；任何只改其一的写法都会让两侧失步（跨核事件数不匹配）。
    __aicore__ inline void SetWidePacketCursor(WidePacketInfo& packet, uint32_t kvBasicIdx, uint32_t qScanStart) const
    {
        packet.nextKvStart = kvBasicIdx;
        packet.nextQStart = qScanStart;
    }

    __aicore__ inline void CommitWidePacketCursor(WideWorkState& work, const WidePacketInfo& packet) const
    {
        work.nextKvIndex = packet.nextKvStart;
        work.nextQueryIndex = packet.nextQStart;
    }

    // mask 窗口按需换页：窗口按 4 个 KV 基础块对齐，末尾收敛到 kvScanLimit（全体 Q 的扫描上界）。
    // 契约：换页后必须把 loadedMaskWindows 清零——窗口基址变了，已缓存的那 4 个字节全部失效。
    __aicore__ inline void EnsureWideMaskWindow(WideWorkState& work, uint32_t kvBasicIdx, uint32_t windowBytes) const
    {
        if (kvBasicIdx >= work.maskScanState.maskWindowStart && kvBasicIdx < work.maskScanState.maskWindowEnd) {
            return;
        }
        work.maskScanState.maskWindowStart = kvBasicIdx & ~(windowBytes - 1U);
        work.maskScanState.maskWindowEnd = work.maskScanState.maskWindowStart + windowBytes;
        if (work.maskScanState.maskWindowEnd > work.kvScanLimit) {
            work.maskScanState.maskWindowEnd = work.kvScanLimit;
        }
        work.maskScanState.loadedMaskWindows = 0;
    }

    // 为一个 Q 加载它在当前窗口内的 1..4 个 mask 字节（已加载则直接复用，不重复读 GM）。
    // 单个 Q 的读取长度收敛到 queryKvLimits[qLocal]；窗口起点之后不足 4 字节时按实际长度读。
    __aicore__ inline void LoadWideMaskWindow(const Params& params, WideWorkState& work, uint32_t qLocal,
                                              uint32_t windowBytes) const
    {
        const WideQMask qWindowBit = static_cast<WideQMask>(1) << qLocal;
        if ((work.maskScanState.loadedMaskWindows & qWindowBit) != 0) {
            return;
        }
        const uint32_t qWindowCandidateEnd = work.maskScanState.maskWindowStart + windowBytes;
        const uint32_t qWindowEnd =
            qWindowCandidateEnd < work.queryKvLimits[qLocal] ? qWindowCandidateEnd : work.queryKvLimits[qLocal];
        work.maskScanState.maskWindows[qLocal] =
            LoadMaskByteWindow(params, work.maskRowOffsets[qLocal] + work.maskScanState.maskWindowStart,
                               qWindowEnd - work.maskScanState.maskWindowStart);
        work.maskScanState.loadedMaskWindows |= qWindowBit;
    }

    // 阶段C：按 Q 稳定排序边索引，并聚合出每个有效 Q 的连续边区间。
    // qOrder 是"qOrder 位次 -> 边下标"的排列；activeQ 只登记有边的 Q，故每个 activeQ 至少一条边。
    __aicore__ inline void BuildWideQIndex(WidePacketInfo& packet) const
    {
        for (uint32_t edge = 0; edge < packet.edgeCount; ++edge) {
            packet.qOrder[edge] = static_cast<uint16_t>(edge);
        }
        // 插入排序：边数最多 8，代价可忽略；稳定排序保证同一个 Q 内的 KV 顺序不变。
        for (uint32_t ordered = 1; ordered < packet.edgeCount; ++ordered) {
            const uint16_t key = packet.qOrder[ordered];
            const uint32_t keyQ = WideEdgeQ(packet.edgeCode[key]);
            uint32_t insert = ordered;
            while (insert > 0) {
                const uint16_t previous = packet.qOrder[insert - 1];
                if (WideEdgeQ(packet.edgeCode[previous]) <= keyQ) {
                    break;
                }
                packet.qOrder[insert] = previous;
                --insert;
            }
            packet.qOrder[insert] = key;
        }
        for (uint32_t ordered = 0; ordered < packet.edgeCount; ++ordered) {
            const uint32_t qLocal = WideEdgeQ(packet.edgeCode[packet.qOrder[ordered]]);
            // qOrder 已按 Q 排好，故同一个 Q 的边必然连续：换 Q 时新开一个 desc，否则只累加计数。
            if (packet.activeQCount == 0 || packet.activeQ[packet.activeQCount - 1].qLocal != qLocal) {
                WideQDesc& desc = packet.activeQ[packet.activeQCount++];
                desc.qLocal = static_cast<uint8_t>(qLocal);
                desc.edgeBegin = static_cast<uint8_t>(ordered);
                desc.edgeCount = 1;
                desc.reserved = 0;
            } else {
                ++packet.activeQ[packet.activeQCount - 1].edgeCount;
            }
        }
    }

    // AIC 本核资源事件：初始化、结束回收与 L1 复用。
    /**
     * @brief 预置所有"先等待后释放"型令牌，使第一次 WaitFlag 立即通过。
     * @note 三类资源各预置一次，与 WaitWideFlag 一一对应；循环结束时计数必须归零：
     *       - M_MTE1 0..3   : L0A/L0B 操作数槽；
     *       - FIX_M  0..1   : L0C 累加器槽；
     *       - MTE1_MTE2 0..7: L1 复用槽。其中 0/1 属 Cube1（cube1Mmad 的 pingpongFlagOffset=0），
     *         2/3 属 P/dS（WIDE_PINGPONG_OFFSET_2），4/5 属 dO/Q（WIDE_PINGPONG_OFFSET_4），
     *         6 为预留入口令牌，7 供 SyncExtendedL1Reuse 做 L1 别名屏障。
     * @note 任何新增的"按槽位索引"的事件，都必须同时在 WaitWideFlag 中回收，
     *       否则最后一次 WaitFlag 会悬空，导致核退出时事件未清零。
     * @note 槽位分配表（预置与回收必须一一对应，共 14 个）：
     *       M_MTE1    0..3 ：L0A/L0B 操作数槽（三个 MMAD 的 pingpongFlagOffset）；
     *       FIX_M     0/1  ：L0C 累加器槽；
     *       MTE1_MTE2 0/1  ：Cube1 的 L1 槽；2/3：P/dS；4/5：dO/Q；6：预留入口令牌；
     *       MTE1_MTE2 7    ：SyncExtendedL1Reuse 的 L1 别名屏障（wait→set→wait→set）。
     * @note 流程框架：入口预置 → 全程按槽复用 → WaitWideFlag 一次性回收 → 计数归零。
     */
    __aicore__ inline void SeedWideFlags() const
    {
        // 与 WaitWideFlag 的回收清单逐项镜像：共 14 个令牌，两侧顺序必须一致。
        // M_MTE1 0..3：L0A/L0B 操作数槽（三个 MMAD 的 pingpongFlagOffset）。
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        // FIX_M 0/1：L0C 累加器槽。
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        // MTE1_MTE2 0/1：Cube1 的 L1 槽；2/3：P/dS；4/5：dO/Q；6：入口令牌；7：L1 别名屏障。
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    }

    /**
     * @brief 在 AIC 执行结束时等待资源归还，回收 SeedWideFlags 预置的本核令牌。
     * @note 仅在全部 work item 和最后一个 packet 的梯度计算结束后调用一次，
     *       回收操作数槽、L0C 累加器和 L1 复用槽；不负责等待 AIV 的 packet 完成令牌。
     * @note 流程框架：按 SeedWideFlags 的顺序逐槽回收 14 个令牌，回收后计数归零。
     */
    __aicore__ inline void WaitWideFlag() const
    {
        // MTE1_MTE2：0/1 保护 Cube1，2/3 保护 P/dS，4/5 保护 dO/Q，6 为入口令牌，7 保护 L1 别名。
        // 顺序与 SeedWideFlags 一致，14 个令牌在这里各回收一次，回收后计数归零。
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    }

    /**
     * @brief 在 MTE1_MTE2 队列上插入一次栅栏，且保持 7 号令牌计数不变。
     * @note 形态为 wait->set->wait->set（同一事件号）：第一个 wait 吃掉令牌，随后的 set
     *       把它放回，第二个 wait 立刻通过，再 set 回去留给下一次调用。净效果是在
     *       MTE1（L1->L0 读）与 MTE2（GM->L1 写）之间插入一次顺序保证，
     *       使"低 L1 的上一轮读取"确定结束后才允许覆盖（P/dS、K、Q/dO 的复用点）。
     * @note SeedWideFlags 已预置 7 号令牌，故首次调用即可通过，不会死等。
     */
    __aicore__ inline void SyncExtendedL1Reuse() const
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    }

    // AIC 计算：Cube1 生成 S/dP，梯度阶段依次执行 dQ 和 dKV。
    /**
     * @brief 按 KV 分组计算 S=Q*K^T 和 dP=dO*V^T，写入当前 packet 的 FP32 NZ 槽。
     * @param[in] packet 有效边、KV 分组及 workspace stage 的描述。
     * @param[in] workInfo 当前 work item 的 batch、Q head 和物理序列地址。
     * @param[in] qInfos 当前 work item 的 Q 分片坐标、元素偏移及有效行数。
     * @param[in] tilingData Host 下发的 head 维度、tile 尺寸和 prefix 元数据。
     * @param[in,out] tensors K/V 输入、S/dP workspace 与常驻 Q/dO 的 GM 视图组。
     * @param[in,out] cube1Mmad 负责流式 K/V 预载和 S/dP 计算的 MMAD 服务。
     * @param[in,out] cubeOperandStage 两次矩阵乘共用的 L0 操作数双缓冲槽号。
     * @param[in,out] residentQueryMmad 提供已预载且在 work item 内保持有效的 Q/dO 常驻槽。
     * @note 调用前 Q/dO 预载须完成；同一 KV 的 K/V 在本 packet 内复用，
     *       由其最后一条边归还流式槽令牌，raw ready 令牌由调用方发布。
     * @note 流程框架（[Step 1,2]）：阶段A 每个唯一 KV 列预载一次 K/V（holdForConsumer = true）→
     *       阶段B 该列逐边依次计算 S = Q·Kᵀ 与 dP = dO·Vᵀ，两次 MMAD 共用 cubeOperandStage 作 L0 槽号。
     */
    __aicore__ inline void RunWideCube1(const WidePacketInfo& packet, const WideTileCache& tiles,
                                        const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                        WideAicTensorViews& tensors, BlockMmadBSAG1& cube1Mmad,
                                        uint32_t& cubeOperandStage, BlockMmadBSAG3& residentQueryMmad)
    {
        const uint32_t headDim = tilingData->headDim;
        for (uint32_t kvLocal = 0; kvLocal < packet.uniqueKvCount; ++kvLocal) {
            const WideKvDesc& kv = packet.kv[kvLocal];
            const WideTileSlot& firstTile = tiles.slots[kv.edgeBegin];
            // 每个 KV 列只预载一次 K/V；holdForConsumer=true 表示槽令牌留给最后一条边释放。
            LayoutB1 layoutB(headDim, firstTile.keyValueRows, headDim);
            GemmCoord preloadShape{firstTile.queryRows, firstTile.keyValueRows, headDim};
            // 槽 0 = K（配合 Q 算 S），槽 1 = V（配合 dO 算 dP）；覆盖前等该槽最后一次 MTE1 读取。
            cube1Mmad.PreloadB(tensors.key[firstTile.keyValueOffset], layoutB, preloadShape, 0, true);
            cube1Mmad.PreloadB(tensors.value[firstTile.keyValueOffset], layoutB, preloadShape, 1, true);
            const uint32_t edgeEnd = kv.edgeBegin + kv.edgeCount;
            for (uint32_t edge = kv.edgeBegin; edge < edgeEnd; ++edge) {
                // 前一轮 MMAD 完成后才能复用操作数。
                AscendC::PipeBarrier<PIPE_M>();
                const WideTileSlot& tile = tiles.slots[edge];
                GemmCoord shape{tile.queryRows, tile.keyValueRows, headDim};
                // A 操作数取常驻槽：Q 在 dO 槽 + WIDE_QUERY_CACHE_OFFSET，dO 在低位槽。
                // 两条 MMAD 依次产出 S 与 dP，并共用 cubeOperandStage 作为 L0 槽号与 M 事件号。
                auto localQ = residentQueryMmad.GetCachedB(tile.queryLocalIndex + WIDE_QUERY_CACHE_OFFSET);
                auto localDo = residentQueryMmad.GetCachedB(tile.queryLocalIndex);
                cube1Mmad.WithCachedBNz(0, tensors.rawScore[tile.workspaceOffset], shape, cubeOperandStage, localQ,
                                        edge + 1 == edgeEnd);
                cube1Mmad.WithCachedBNz(1, tensors.rawProbabilityGrad[tile.workspaceOffset], shape, cubeOperandStage,
                                        localDo, edge + 1 == edgeEnd);
            }
        }
    }

    /**
     * @brief 按 Q 分组累加当前 packet 的 dS*K，更新 FP32 dQ workspace。
     * @param[in] packet KV 缓存索引、Q 优先边序列及有效 Q 分组。
     * @param[in] workInfo 当前 work item 的物理坐标和输入元素偏移。
     * @param[in] qInfos 当前 work item 的 Q 分片信息，决定 dQ 行范围和输出位置。
     * @param[in] tilingData head 维度、基础 KV 块大小及 prefix 元数据。
     * @param[in,out] tensors K/dS 输入、dQ workspace 的 GM 视图组。
     * @param[in,out] queryGradMmad 提供常驻 K、dS 双缓冲及 L0C 累加的 MMAD 服务。
     * @param[in,out] queryGradWrittenMask 已写回 dQ 的 Q 位图，跨 packet 保留、跨 work item 清零。
     * @note 调用前须等待对应 AIV 完成令牌；同一 Q 首次写回覆盖输出，
     *       后续 packet 使用原子累加，完成写回后更新 queryGradWrittenMask。
     * @note 流程框架（[Step 6] dQ = dS·K）：阶段A 每个唯一 KV 列预载一次 K（FinishFullPreload 同步一次）→
     *       阶段B 按 activeQ 分组归约：dS 双槽滚动预载（预载 i 与 i+2），L0C 槽 = qLocal&1，
     *       initC 仅首边为真（L0C 首写）、finalC 仅末边为真（触发 Fixpipe）→
     *       阶段C 写回并置位 queryGradWrittenMask。
     * @note 契约：activeQ 只登记有边的 Q，故每个 activeQ 至少一条边，末片 tile 必然被赋值。
     */
    __aicore__ inline void RunWideDq(const WidePacketInfo& packet, const WideTileCache& tiles,
                                     const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                     WideAicTensorViews& tensors, BlockMmadBSAG2& queryGradMmad,
                                     WideQMask& queryGradWrittenMask)
    {
        const uint32_t headDim = tilingData->headDim;

        // 最多八个 KV 列各加载一次 K；Q 优先归约通过 kvLocal 复用 K 缓存。
        for (uint32_t kvLocal = 0; kvLocal < packet.uniqueKvCount; ++kvLocal) {
            const uint32_t edge = packet.kv[kvLocal].edgeBegin;
            const WideTileSlot& tile = tiles.slots[edge];
            // 每个唯一 KV 列只向 l1B 载入一次 K，之后按 Q 优先顺序复用（即「K 常驻」的含义）。
            LayoutB2 layoutB(tile.keyValueRows, headDim, headDim);
            GemmCoord shape{tile.queryRows, headDim, tile.keyValueRows};
            queryGradMmad.PreloadFullB(tensors.key[tile.keyValueOffset], layoutB, shape, kvLocal);
        }
        // 一批 PreloadFullB 只在此处做一次 MTE2->MTE1 同步。
        queryGradMmad.FinishFullPreload();

        for (uint32_t activeQ = 0; activeQ < packet.activeQCount; ++activeQ) {
            // 按槽位配对的 MTE/M/FIX 事件保护 dQ 操作数和累加器复用。
            const WideQDesc& qDesc = packet.activeQ[activeQ];
            const uint32_t qLocal = qDesc.qLocal;
            const uint32_t qBegin = qDesc.edgeBegin;
            const uint32_t edgeCount = qDesc.edgeCount;
            WideTileSlot lastTile;
            // 先用 dS 预载前两个 L1A 槽（ping-pong 起点），后续边在消费时滚动预载 i+2。
            const uint32_t initial = edgeCount < 2 ? edgeCount : 2;
            for (uint32_t i = 0; i < initial; ++i) {
                const uint32_t edge = packet.qOrder[qBegin + i];
                const WideTileSlot& tile = tiles.slots[edge];
                GemmCoord shape{tile.queryRows, headDim, tile.keyValueRows};
                queryGradMmad.PreloadAccumANz(tensors.scoreGrad[tile.workspaceOffset], shape, i);
            }
            for (uint32_t i = 0; i < edgeCount; ++i) {
                // 前一轮 MMAD 完成后才能复用操作数。
                AscendC::PipeBarrier<PIPE_M>();
                const uint32_t slot = i & 1U;
                const uint32_t edge = packet.qOrder[qBegin + i];
                const uint32_t kvLocal = WideEdgeKv(packet.edgeCode[edge]);
                const WideTileSlot& tile = tiles.slots[edge];
                LayoutA2 layoutA(tile.queryRows, tile.keyValueRows);
                LayoutB2 layoutB(tile.keyValueRows, headDim);
                GemmCoord shape{tile.queryRows, headDim, tile.keyValueRows};
                // slot：L1A 的 ping-pong 槽（i&1）；kvLocal：常驻 K 槽；qLocal&1：L0C 槽。
                // initC 仅首条边为真（L0C 首写），finalC 仅末条边为真（触发 Fixpipe）。
                queryGradMmad.AccumulatePreloadedA(slot, kvLocal, layoutA, layoutB, shape, i == 0, i + 1 == edgeCount,
                                                   qLocal & 1U);
                lastTile = tile;
                // 预载两条之后的 dS 分片，使「L1->L0 读」与「GM->L1 写」重叠。
                if (i + 2 < edgeCount) {
                    const uint32_t nextEdge = packet.qOrder[qBegin + i + 2];
                    const WideTileSlot& nextTile = tiles.slots[nextEdge];
                    GemmCoord nextShape{nextTile.queryRows, headDim, nextTile.keyValueRows};
                    queryGradMmad.PreloadAccumANz(tensors.scoreGrad[nextTile.workspaceOffset], nextShape, slot);
                }
            }
            LayoutC2 layoutC(lastTile.queryRows, headDim, headDim);
            GemmCoord outShape{lastTile.queryRows, headDim, lastTile.keyValueRows};
            const WideQMask qBit = static_cast<WideQMask>(1) << qLocal;
            // 首次运算该 Q 时直接写，后续 packet 再次遇到该 Q 时改为原子累加；
            // queryGradWrittenMask 在 work item 内累计，跨 packet 有效（同一 Q 的 KV 可能被拆到多个 packet）。
            queryGradMmad.FlushAccumulator(tensors.queryGrad[lastTile.queryOffset], layoutC, outShape,
                                           (queryGradWrittenMask & qBit) != 0, qLocal & 1U);
            queryGradWrittenMask |= qBit;
        }
    }

    // dKV 一路（lane 0 = dV = Pᵀ·dO，lane 1 = dK = dSᵀ·Q）的"归还 → 转置累加 → 滚动预载"三件套。
    // 两路唯一差异：L1A/L0C 槽号 = lane、常驻 B 槽号（Q 比 dO 高 WIDE_QUERY_CACHE_OFFSET）、
    // 下一轮预载的数据源（lane 0 用 P，lane 1 用 dS）。
    // A 从外部低 L1 的 zN 直接转置装入 L0A（本对象不读自己的 l1A）；B 取常驻 dO/Q 槽。
    // initC 仅该列首边为真（L0C 首写），finalC 仅该列末边为真（触发 Fixpipe 原子写回）。
    __aicore__ inline void AccumulateOneGrad(BlockMmadBSAG2& queryGradMmad, BlockMmadBSAG3& residentQueryMmad,
                                             const WideAicTensorViews& tensors, const WideTileSlot& tile,
                                             const WideTileSlot* nextTile, uint32_t lane, uint32_t headDim,
                                             const LayoutB3& layoutB, const GemmCoord& shape, bool initC, bool finalC)
    {
        // 上一轮的 A 已就绪：等 MTE2->MTE1 后把该槽交还，供下一轮预载覆盖。
        queryGradMmad.FinishAccumPreloadForExternal(lane);
        residentQueryMmad.AccumulateTransposeFromExternalZNL1(
            queryGradMmad.GetAccumL1A(lane), WIDE_PINGPONG_OFFSET_2 + lane,
            tile.queryLocalIndex + (lane == 0 ? 0 : WIDE_QUERY_CACHE_OFFSET), layoutB, shape, lane, initC, finalC);
        if (nextTile != nullptr) {
            const GemmCoord nextSourceShape{nextTile->queryRows, headDim, nextTile->keyValueRows};
            // 滚动预载下一条边的 A（覆盖刚刚归还的槽）。
            if (lane == 0) {
                queryGradMmad.PreloadAccumANz(tensors.probability[nextTile->workspaceOffset], nextSourceShape, 0);
            } else {
                queryGradMmad.PreloadAccumANz(tensors.scoreGrad[nextTile->workspaceOffset], nextSourceShape, 1);
            }
        }
    }

    /**
     * @brief 按 KV 分组计算 dV=P^T*dO 和 dK=dS^T*Q，并原子累加到 FP32 workspace。
     * @param[in] packet KV 优先的有效边分组及其 workspace 槽描述。
     * @param[in] workInfo 当前 work item 的物理坐标及 Q/KV 输入元素偏移。
     * @param[in] qInfos 当前 work item 的 Q 分片信息，用于定位常驻 Q/dO 槽。
     * @param[in] tilingData head 维度、tile 尺寸和 prefix 元数据。
     * @param[in,out] tensors P/dS 输入、dK/dV workspace 的 GM 视图组。
     * @param[in,out] queryGradMmad 提供低 L1 中 P/dS 的双槽预载及跨服务张量访问。
     * @param[in,out] residentQueryMmad 提供常驻 Q/dO、P/dS 转置计算和 dK/dV 累加写回。
     * @note 在 dQ 阶段后调用；控制函数负责在 dQ 与 dKV 之间插入 L1 复用屏障，
     *       同一 KV 在 packet 内归约 Q 边，跨 packet 和 work item 始终原子累加。
     * @note 流程框架（[Step 7,8]）按 KV 列执行：阶段A 预载 P（低 L1 槽 0）与 dS（槽 1）→
     *       阶段B 逐边两路：槽 0 转置算 dV = Pᵀ·dO（B = 常驻 dO，L0C 槽 0），
     *       槽 1 转置算 dK = dSᵀ·Q（B = 常驻 Q，L0C 槽 1），两路各自滚动预载下一条边 →
     *       阶段C 列尾两次原子写回（dV 走槽 0、dK 走槽 1）。
     * @note 两路共用 initC/finalC：initC 仅该列首边为真，finalC 仅该列末边为真。
     */
    __aicore__ inline void RunWideDkvPair(const WidePacketInfo& packet, const WideTileCache& tiles,
                                          const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                          WideAicTensorViews& tensors, BlockMmadBSAG2& queryGradMmad,
                                          BlockMmadBSAG3& residentQueryMmad)
    {
        const uint32_t headDim = tilingData->headDim;
        for (uint32_t kvLocal = 0; kvLocal < packet.uniqueKvCount; ++kvLocal) {
            const WideKvDesc& kv = packet.kv[kvLocal];
            const uint32_t edgeEnd = kv.edgeBegin + kv.edgeCount;
            WideTileSlot lastTile;
            const WideTileSlot& firstTile = tiles.slots[kv.edgeBegin];
            GemmCoord firstSourceShape{firstTile.queryRows, headDim, firstTile.keyValueRows};
            // P/dS 以 zN[Q,KV] 流经低 L1，在 L1->L0A 时转置，计算 P^T*dO、dS^T*Q。
            // l1A 槽 0 放 P（算 dV = P^T*dO），槽 1 放 dS（算 dK = dS^T*Q）；
            // 槽号与 WIDE_PINGPONG_OFFSET_2 的事件 2/3 一一对应。
            queryGradMmad.PreloadAccumANz(tensors.probability[firstTile.workspaceOffset], firstSourceShape, 0);
            queryGradMmad.PreloadAccumANz(tensors.scoreGrad[firstTile.workspaceOffset], firstSourceShape, 1);
            for (uint32_t edge = kv.edgeBegin; edge < edgeEnd; ++edge) {
                AscendC::PipeBarrier<PIPE_M>();
                const WideTileSlot& tile = tiles.slots[edge];
                LayoutB3 layoutB(tile.queryRows, headDim, headDim);
                GemmCoord shape{tile.keyValueRows, headDim, tile.queryRows};
                const bool initC = edge == kv.edgeBegin;
                const bool finalC = edge + 1 == edgeEnd;
                // 下一条边只在本列还有边时才存在；用指针表达"没有下一条"，避免越界取 slots[edgeCount]。
                const WideTileSlot* nextTile = edge + 1 < edgeEnd ? &tiles.slots[edge + 1] : nullptr;
                // 槽 0 = dV 路（A 取 P），槽 1 = dK 路（A 取 dS）；两路各自"归还槽 → 转置累加 → 预载下一条"。
                AccumulateOneGrad(queryGradMmad, residentQueryMmad, tensors, tile, nextTile, 0, headDim, layoutB, shape,
                                  initC, finalC);
                AccumulateOneGrad(queryGradMmad, residentQueryMmad, tensors, tile, nextTile, 1, headDim, layoutB, shape,
                                  initC, finalC);
                lastTile = tile;
            }
            LayoutC3 layoutC(lastTile.keyValueRows, headDim, headDim);
            GemmCoord outShape{lastTile.keyValueRows, headDim, lastTile.queryRows};
            // dV 与 dK 均按原子累加写回：同一条 KV 列可能被多个 packet 分摊，
            // 且 dKV 的 Q 维归约跨 packet，无法保证单次独占写。
            residentQueryMmad.FlushAccumulator(tensors.valueGrad[lastTile.keyValueOffset], layoutC, outShape, true, 0);
            residentQueryMmad.FlushAccumulator(tensors.keyGrad[lastTile.keyValueOffset], layoutC, outShape, true, 1);
        }
    }

    // AIV tile 参数组装。
    /**
     * @brief 组装一片 S/dP/P/dS 的 softmax 参数（宽 softmax 的 Params 为 13 个字段的聚合体）。
     * @param tileParams 输出参数。
     * @param params Kernel 的 Params（提供 workspace 基址）。
     * @param tile 本片 tile 的坐标与偏移。
     * @param ws workspace 分区布局（提供各分区字节偏移与元素宽度换算）。
     * @param subBlockIdx AIV 子核下标（0/1），决定本子核处理的上/下半区行。
     * @param metadataSlot 本 Q 使用的 LSE/D 缓存槽（0..3）。
     * @note 两个 AIV 子核按行对分同一 tile：子核 0 取前 ceil(rows/2) 行，子核 1 取剩余行；
     *       行的切分必须与 AIC 侧无关（AIC 不参与 softmax），但两侧的 tile 坐标必须一致。
     * @note 流程框架：算本子核行区间（firstHalfRows / executeRow）→ 算 FP32 面与低精度面两个字节偏移 →
     *       13 字段聚合赋值。
     * @note 三个"行数"含义不同：queryRows = 本子核实际执行行数；matrixRows = tile 物理行数（供 stride）；
     *       rowOffset = 本子核在 tile 内的行偏移。
     * @note 聚合体必须整体赋值（覆盖全部 13 个字段），不得只改其中几个字段。
     */
    __aicore__ inline void FillWideSfmParams(SfmParams& tileParams, const Params& params, const TaskInfo& tile,
                                             const WideWorkspaceLayout& ws, uint32_t subBlockIdx,
                                             uint32_t metadataSlot) const
    {
        // 行分半：奇数行时上半区多一行（子核 0 承担 ceil(rows/2)）。
        const WideSubBlockRows rows = WideSplitRows(tile.queryRows, subBlockIdx);
        const uint64_t firstHalfRows = rows.firstHalfRows;
        const uint64_t executeRow = rows.executeRows;
        const uint64_t curVecCoreS1Idx = tile.queryStart + subBlockIdx * firstHalfRows;
        // FP32 S/dP 保留 NZ tile 起点，由 Vector 内部追加 rowOffset*C0。
        const uint64_t rawByteOffset = ws.rawByteOf(tile.workspaceOffset);
        // P/dS 保持 zN 格式；两个 AIV 子核分别写自己的行范围。
        const uint64_t lowpByteOffset = ws.lowpByteOf(tile.workspaceOffset);
        // 聚合初始化的字段顺序与 WideSoftmax::Params 的声明顺序严格一致：
        //   rawScore, rawProbabilityGrad, probability, scoreGrad,
        //   queryRows, columnCount, validColumns, batchIndex, queryHeadIndex,
        //   queryStart, metadataSlot, matrixRows, rowOffset
        tileParams = {params.workspace + rawByteOffset,
                      params.workspace + ws.probabilityGrad() + rawByteOffset,
                      params.workspace + lowpByteOffset,
                      params.workspace + ws.probabilityGrad() + lowpByteOffset,
                      static_cast<uint32_t>(executeRow),
                      tile.keyValueRows,
                      tile.validKeyValueColumns,
                      tile.batchIndex,
                      tile.queryHeadIndex,
                      static_cast<uint32_t>(curVecCoreS1Idx),
                      metadataSlot,
                      tile.queryRows,
                      static_cast<uint32_t>(subBlockIdx * firstHalfRows)};
    }

    /**
     * @brief 绑定 AIC 使用的输入与 workspace GM 视图。
     * @param[in] params 输入输出地址和 workspace 基址。
     * @param[in] ws workspace 分区布局（各分区字节偏移的唯一来源）。
     * @param[out] tensors 本次 AIC 调用的 GM 视图组。
     * @note 仅绑定地址与长度，不搬移数据，也不改变任何同步状态。
     * @note 流程框架：先绑定 4 个输入视图，再按 workspace 分区顺序绑定 7 个 workspace 视图。
     * @note workspace 分区顺序（与 Host 写入顺序严格一致，不可调换）：
     *       S(FP32) | dP(FP32) | P(lowp) | dS(lowp) | dQ | dK | dV | D(rowDelta)。
     * @note rawPlaneElements 同时用于 FP32 面与低精度面：两者元素数相同（FP32 为 4B、低精度为 2B，
     *       但元素个数一致）；若将来低精度面的元素数改变，此处必须改为分别计算。
     */
    __aicore__ inline void BindWideAicTensorViews(const Params& params, const WideWorkspaceLayout& ws,
                                                  WideAicTensorViews& tensors) const
    {
        tensors.query.SetGlobalBuffer((__gm__ ElementInput*)params.q);
        tensors.key.SetGlobalBuffer((__gm__ ElementInput*)params.k);
        tensors.value.SetGlobalBuffer((__gm__ ElementInput*)params.v);
        tensors.outputGrad.SetGlobalBuffer((__gm__ ElementA1*)params.dout);
        // FP32 面（S/dP）与低精度面（P/dS）元素数相同，故共用同一长度。
        const uint64_t planeElements = ws.rawPlaneElements();
        tensors.rawScore.SetGlobalBuffer((__gm__ float*)(params.workspace + ws.score()), planeElements);
        tensors.rawProbabilityGrad.SetGlobalBuffer((__gm__ float*)(params.workspace + ws.probabilityGrad()),
                                                   planeElements);
        tensors.probability.SetGlobalBuffer((__gm__ ElementInput*)(params.workspace + ws.probability()), planeElements);
        tensors.scoreGrad.SetGlobalBuffer((__gm__ ElementInput*)(params.workspace + ws.scoreGrad()), planeElements);
        tensors.queryGrad.SetGlobalBuffer((__gm__ float*)(params.workspace + ws.queryGrad()), ws.queryGradElements());
        tensors.keyGrad.SetGlobalBuffer((__gm__ float*)(params.workspace + ws.keyGrad()), ws.keyGradElements());
        tensors.valueGrad.SetGlobalBuffer((__gm__ float*)(params.workspace + ws.valueGrad()), ws.valueGradElements());
    }

    /**
     * @brief 准备当前 work item 的 Q 分片、扫描边界和 mask 游标。
     * @param[in] tilingData Host 下发的任务与块尺寸。
     * @param[in] currentWork 当前物理起点。
     * @param[in] maskLayout sparse mask 字节步长。
     * @param[in] hasNextWork 是否需要推导下一 work item。
     * @param[out] work 完整写入当前 work item 的共享扫描状态。
     * @note 只计算地址和标量，不启动异步搬运或同步操作。
     * @note 流程框架：聚合 Q 分片 → 逐 Q 算 KV 扫描上界与 mask 行偏移 → 取全体上界的最大值 →
     *       游标归零并复位 mask 扫描状态。
     * @note queryKvLimits = 0 表示该 Q 分片被 prefix 裁空（契约分支，后续函数据此跳过计算）。
     * @note kvScanLimit = max(queryKvLimits)，是 packet 循环 while (kvBasicIdx < kvScanLimit) 的上界。
     */
    __aicore__ inline void PrepareWideWorkItem(const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                               const TaskInfo& currentWork, const MaskLayoutInfo& maskLayout,
                                               bool hasNextWork, WideWorkState& work) const
    {
        work.queryCount = BuildWideWorkItem(tilingData, currentWork, work.queries, work.nextWork, hasNextWork);
        const uint32_t kvBlockCount =
            (currentWork.keyValueLength + tilingData->blockShapeY - 1) / tilingData->blockShapeY;
        work.kvScanLimit = 0;
        for (uint32_t queryIndex = 0; queryIndex < work.queryCount; ++queryIndex) {
            work.queryKvLimits[queryIndex] = work.queries[queryIndex].queryRows == 0 ? 0 : kvBlockCount;
            work.maskRowOffsets[queryIndex] =
                static_cast<uint64_t>(currentWork.batchIndex) * maskLayout.batchStride +
                static_cast<uint64_t>(currentWork.queryHeadIndex) * maskLayout.headStride +
                static_cast<uint64_t>(work.queries[queryIndex].queryBlockIndex) * maskLayout.kvBlockCount;
            if (work.queryKvLimits[queryIndex] > work.kvScanLimit) {
                work.kvScanLimit = work.queryKvLimits[queryIndex];
            }
        }
        work.nextKvIndex = 0;
        work.nextQueryIndex = 0;
        work.maskScanState = WideMaskScanState{};
    }

    /**
     * @brief 常驻预载当前 work item 的 dO 和 Q。
     * @param[in] tilingData headDim 等预载布局参数。
     * @param[in] work 当前 Q 分片描述。
     * @param[in] tensors Q/dO GM 视图。
     * @param[in,out] residentQueryMmad 常驻 Q/dO 的 MMAD 服务。
     * @note 获取和完成常驻槽的事件由 AIC 主流程负责；本方法只发起 DMA。
     */
    __aicore__ inline void PreloadWideQueryOperands(const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                                    const WideWorkState& work, WideAicTensorViews& tensors,
                                                    BlockMmadBSAG3& residentQueryMmad) const
    {
        for (uint32_t queryLocal = 0; queryLocal < work.queryCount; ++queryLocal) {
            const WideQInfo& query = work.queries[queryLocal];
            if (query.queryRows == 0) {
                continue;
            }
            const LayoutB3 layout(query.queryRows, tilingData->headDim, tilingData->headDim);
            const GemmCoord shape{128, tilingData->headDim, query.queryRows};
            residentQueryMmad.PreloadFullB(tensors.outputGrad[query.queryOffset], layout, shape, queryLocal);
            residentQueryMmad.PreloadFullB(tensors.query[query.queryOffset], layout, shape,
                                           queryLocal + WIDE_QUERY_CACHE_OFFSET);
        }
    }

    /**
     * @brief 计算本 AIV 子核负责的 delta 行。
     * @param[in] tilingData headDim 用于元素偏移换算行偏移。
     * @param[in] work 当前 Q 分片描述。
     * @param[in] subBlockIndex AIV 子核下标 0 或 1。
     * @param[in,out] rowDeltaOp delta 计算原语。
     */
    __aicore__ inline void ComputeWideRowDelta(const __gm__ BlockSparseAttentionGradTilingData* tilingData,
                                               const WideWorkState& work, uint32_t subBlockIndex,
                                               EpilogueFAGSfmg& rowDeltaOp) const
    {
        for (uint32_t queryLocal = 0; queryLocal < work.queryCount; ++queryLocal) {
            const WideQInfo& query = work.queries[queryLocal];
            const WideSubBlockRows rows = WideSplitRows(query.queryRows, subBlockIndex);
            // rows.executeRows == 0 是契约分支：该分片被 prefix 裁空，或全部行归另一子核。
            if (rows.executeRows != 0) {
                rowDeltaOp(query.queryOffset / tilingData->headDim + subBlockIndex * rows.firstHalfRows,
                           rows.executeRows);
            }
        }
    }

    /**
     * @brief 计算一个 packet 的 P 和 dS。
     * @param[in] params workspace 地址集合。
     * @param[in] ws workspace 分区布局。
     * @param[in] tilingData tile 与 head 尺寸。
     * @param[in] currentWork 当前物理任务坐标。
     * @param[in] work 当前 work item 的 Q 分片和扫描状态。
     * @param[in] packet 已由 AIC/AIV 共同构造的 packet。
     * @param[in] subBlockIndex 本 AIV 子核行范围。
     * @param[in,out] softmaxOp wide softmax 原语。
     * @note ready/done 与 raw-stage 生命周期由 AIV 主流程维护；本方法在无有效行时直接返回。
     * @note 流程框架（[Step 4,5]）：阶段A 有效边扁平化（按 activeQ 顺序展开本子核要算的 tile，
     *       并分配元数据槽；一条都没有时直接返回，但调用方仍须发布 done）→
     *       阶段B 首片参数与预取 → 阶段C 双缓冲流水：预取下一条（换 Q 时失效该 Q 的 LSE/D 缓存）
     *       → 计算当前 stage → 切换。
     * @note 元数据槽 = 参与计算的 Q 在本 packet 内的序号（& 3），必须与 softmax 侧
     *       WIDE_Q_METADATA_SLOTS 一致（当前均为 4，且 WIDE_MAX_Q_TASKS = 2 ≤ 4）。
     */
    __aicore__ inline void RunWideVectorPacket(const Params& params, const WideWorkspaceLayout& ws,
                                               const WidePacketInfo& packet, const WideTileInfoCache& tiles,
                                               const WideWorkState& work, uint32_t subBlockIndex,
                                               EpilogueFAGOp& softmaxOp) const
    {
        uint16_t validEdges[WIDE_MAX_EDGES];
        uint8_t validMetadataSlots[WIDE_MAX_EDGES];
        const uint32_t validEdgeCount =
            FlattenValidWideEdges(packet, work, subBlockIndex, validEdges, validMetadataSlots);
        if (validEdgeCount == 0) {
            return;
        }

        softmaxOp.ResetWideQMetadata();
        SfmParams tileParams[2];
        uint32_t inputStageIndex = 0;
        const uint32_t firstEdge = validEdges[0];
        const uint32_t firstMetadataSlot = validMetadataSlots[0];
        const TaskInfo& firstTile = tiles.slots[firstEdge];
        FillWideSfmParams(tileParams[0], params, firstTile, ws, subBlockIndex, firstMetadataSlot);
        softmaxOp.InvalidateWideQMetadataSlot(firstMetadataSlot);
        softmaxOp.PrefetchWide(tileParams[0], 0);
        for (uint32_t edgeIndex = 0; edgeIndex < validEdgeCount; ++edgeIndex) {
            if (edgeIndex + 1 < validEdgeCount) {
                const uint32_t nextInputStageIndex = 1 - inputStageIndex;
                const uint32_t nextEdge = validEdges[edgeIndex + 1];
                const uint32_t currentEdge = validEdges[edgeIndex];
                const uint32_t nextQuery = WideEdgeQ(packet.edgeCode[nextEdge]);
                const uint32_t currentQuery = WideEdgeQ(packet.edgeCode[currentEdge]);
                const uint32_t nextMetadataSlot = validMetadataSlots[edgeIndex + 1];
                const TaskInfo& nextTile = tiles.slots[nextEdge];
                FillWideSfmParams(tileParams[nextInputStageIndex], params, nextTile, ws, subBlockIndex,
                                  nextMetadataSlot);
                if (nextQuery != currentQuery) {
                    softmaxOp.InvalidateWideQMetadataSlot(nextMetadataSlot);
                }
                softmaxOp.PrefetchWide(tileParams[nextInputStageIndex], nextInputStageIndex);
            }
            softmaxOp.ComputeWide(tileParams[inputStageIndex], inputStageIndex);
            inputStageIndex = 1 - inputStageIndex;
        }
    }

    // 把 packet 的 Q 优先索引展开成本子核要处理的 tile 序列，并按 activeQ 顺序分配元数据槽。
    // 返回有效边数；返回 0 表示本子核在该 packet 上没有行要算（调用方仍须发布 done 令牌）。
    // 契约：被 prefix 裁空的 Q 不占用槽位，故同一 packet 内同一 Q 的边共享同一个元数据槽；
    //       边的产出顺序（qOrder）决定 tile 的计算顺序，不改变 workspace 槽位。
    __aicore__ inline uint32_t FlattenValidWideEdges(const WidePacketInfo& packet, const WideWorkState& work,
                                                     uint32_t subBlockIndex, uint16_t* validEdges,
                                                     uint8_t* validMetadataSlots) const
    {
        uint32_t validEdgeCount = 0;
        uint32_t activeQueryCount = 0;
        for (uint32_t activeQuery = 0; activeQuery < packet.activeQCount; ++activeQuery) {
            const WideQDesc& queryDesc = packet.activeQ[activeQuery];
            const WideQInfo& query = work.queries[queryDesc.qLocal];
            const WideSubBlockRows rows = WideSplitRows(query.queryRows, subBlockIndex);
            if (rows.executeRows == 0) {
                continue;
            }
            const uint8_t metadataSlot = static_cast<uint8_t>(WideMetadataSlot(activeQueryCount));
            ++activeQueryCount;
            const uint32_t edgeEnd = queryDesc.edgeBegin + queryDesc.edgeCount;
            for (uint32_t queryEdge = queryDesc.edgeBegin; queryEdge < edgeEnd; ++queryEdge) {
                validEdges[validEdgeCount] = packet.qOrder[queryEdge];
                validMetadataSlots[validEdgeCount] = metadataSlot;
                ++validEdgeCount;
            }
        }
        return validEdgeCount;
    }

    // 算子 kernel 与 Params 的引用（由 kernel.h 在构造时注入）；resource 为 wide 自己的 L1/L0 资源视图。
    Kernel& kernel_;
    // 只读引用，生命周期由调用方保证。
    const Params& params_;
    // wide 的架构资源；三个 MMAD 对象共享它但各自划分槽位（见 ProcessWideAic）。
    NpuArch::Arch::Resource<ArchTag> resource;
};
} // namespace BSA::WideDetail
#endif
