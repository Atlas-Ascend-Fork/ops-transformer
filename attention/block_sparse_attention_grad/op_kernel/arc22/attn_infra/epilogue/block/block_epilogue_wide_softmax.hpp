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
 * \file block_epliogue_wide_softmax.h
 * \brief Block Epliogue Wide Softmax Kernel Implementation
 */

#ifndef BSAG_EPILOGUE_WIDE_SOFTMAX_HPP
#define BSAG_EPILOGUE_WIDE_SOFTMAX_HPP

#include "../../arch/bsag_resource.hpp"
#include "../bsag_epilogue_dispatch_policy.hpp"
#include "kernel_operator.h"

// ============================================================================
// wide 专用 softmax epilogue：由 S/dP 得到 P/dS，供 dQ/dKV 使用。
//
// 计算式
//   P  = exp(scale * S - LSE)
//   dS = P * (dP - D)        D = rowsum(dO * O)
// 其中 LSE 与 D 由外部（EpilogueFAGSfmg / 前置阶段）算好并写入 GM，本类只做搬运与逐行广播。
//
// 与 legacy softmax 的关系
//   * 本文件独立于 legacy 的 softmax epilogue：自持 UB 双 stage 与自己的事件编号，
//     不改变 legacy 的缓冲布局与事件计数（I2 阶段的硬性约束）；
//   * 只被 block_sparse_attention_grad_kernel_wide.h 使用。
//
// UB 布局（全部为编译期常量偏移，见构造函数）
//   [0, 64K)        S 的两个 stage（各 33 KiB 预留）
//   [66K, 130K)     dP 的两个 stage
//   [132K, ...)     P32 scratch（两个 stage 共享）
//   [164K, 168K)    LSE 广播缓冲（两 stage）
//   [168K, 169K)    LSE 行缓冲（两 stage，同时作为元数据槽 0/1 的 LSE 槽）
//   [169K, 170K)    D 行缓冲（两 stage，同时作为元数据槽 0/1 的 D 槽）
//   [174K, ...)     UB 尾部的额外两个 LSE 槽 + 两个 D 槽
//
// 双 stage 软件流水
//   PrefetchWide(stage) 搬入 S/dP 与元数据，ComputeWide(stage) 产出 P/dS；
//   调用方（ProcessWideAiv）以"预取下一条、计算当前条"的方式交替两个 stage。
//
// 事件一览（本类内部，均为同核事件，不与 MMAD 的槽位事件冲突）
//   V->MTE2  6/7 : stage 0/1 的输入缓冲可覆盖（BeginWidePipeline 预置，EndWidePipeline 回收）
//   MTE3->V  4/5 : LSE/D 元数据缓冲可覆盖
//   MTE3->MTE2 0/1: stage 0/1 的别名输出已写回（Begin/EndWideRawStageReuse 成对）
//   V->MTE3  4/5 : P16/dS16 计算结果就绪（ComputeWideNz 内 set/wait 立即串行化）
// ============================================================================
namespace NpuArch::Epilogue::Block {
using namespace AscendC;

template <typename InputDType>
class WideSoftmax {
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    // 一片 tile 的全部输入输出与尺寸；由 kernel_wide.h 的 FillWideSfmParams 聚合初始化。
    // 字段顺序即聚合初始化顺序，增删字段必须同步修改 FillWideSfmParams。
    struct Params {
        GM_ADDR rawScore;           // FP32、NZ 格式的 S（Q*K^T）tile 起点，位于 workspace 的 FP32 平面。
        GM_ADDR rawProbabilityGrad; // FP32、NZ 格式的 dP（dO*V^T）tile 起点，位于 sOutSize 之后。
        GM_ADDR probability; // 低精度、NZ 格式的 P（softmax(S)）输出起点，位于 workspace 的低精度平面。
        GM_ADDR scoreGrad; // 低精度、NZ 格式的 dS 输出起点，与 P 同平面相邻放置。
        uint32_t queryRows; // 当前 AIV 子核负责的有效 Q 行数（调用方保证非零，空子核不会进到这里）。
        uint32_t columnCount; // 当前 tile 的物理 KV 列数（对齐前的原始列数，决定 C1 slab 数量）。
        uint32_t validColumns; // prefix 裁剪后的有效 KV 列数；小于物理列数时需显式清零无效列。
        // batch 与 Q head 下标，用于定位 LSE/D 的行地址。
        uint32_t batchIndex;
        uint32_t queryHeadIndex;
        uint32_t queryStart; // 当前子核首行的 s1 位置（= tile.queryStart + subBlockIdx * firstHalfRows）。
        uint32_t metadataSlot; // 当前 Q 使用的 LSE/D 元数据槽，取值 0..3（由 AIV 侧按参与计算的 Q 轮转分配）。
        uint32_t matrixRows; // 两个 AIV 子核合并后的完整 tile 行数（子核行数之和），决定 NZ slab 跨度。
        uint32_t rowOffset; // 当前子核在完整 tile 中的起始行（子核 0 为 0，子核 1 为 ceil(rows/2)）。
    };

    // 从 Kernel 借用的 UB 资源；本类所有缓冲都在构造时从 ubBuf 划分。
    NpuArch::Arch::Resource<ArchTag> resource;

    // —— 编译期常量 ——
    // 输入双缓冲 stage 数；元数据槽固定 4 个，与 kernel_wide.h 的 metadataSlot 轮转一致。
    constexpr static uint64_t WIDE_INPUT_STAGES = 2;
    constexpr static uint64_t WIDE_Q_METADATA_SLOTS = 4;
    // BRCB 广播的基础列数（一行 LSE 广播成 8 列，供 SubBrcb 复用）。
    constexpr static uint64_t BRCB_BASE_NUM = 8;
    // 单次 repeat 处理的字节数（256B = 64 个 FP32）。
    constexpr static uint64_t REAPTE_BYTE = 256;

    // 一个 32B block 内可容纳的 FP32 个数。
    constexpr static uint64_t BLOCK_FP32_NUM = 8;
    // NZ 布局的 C0 元素数（固定 16）。
    constexpr static uint64_t BLOCK_16_NUM = 16;
    // 单个 AIV 子核的最大处理行数（一个 AIC 由两个子核分担，共 128 行）。
    constexpr static uint64_t baseM = 64;

    // raw stage 间距保留 33 KiB；64x128 FP32 NZ 最多使用 32 KiB。
    // S 消费完成后，复用为两个互不重叠的 16 KiB P/dS 输出。
    constexpr static uint32_t WIDE_RAW_STAGE_BYTES = 33 * 1024;
    constexpr static uint32_t WIDE_S_STAGE_BASE = 0;
    constexpr static uint32_t WIDE_DP_STAGE_BASE = 66 * 1024;
    // P32 scratch 起点：两个 stage 共享，Cast 出 P16 后立即被复用为 dS。
    constexpr static uint32_t WIDE_NZ_SCRATCH_OFFSET = 132 * 1024;
    // 元数据槽 0/1 复用 stage 内的 LSE/D 缓冲（位于 174 KiB 以下），
    // 槽 2/3 使用 UB 尾部的独立空间，从而支持 4 个 Q 交叉复用。
    constexpr static uint32_t WIDE_EXTRA_LSE_OFFSET = 174 * 1024;
    // 尾部 D 槽紧跟尾部 LSE 槽（每个槽 64*8*4 = 2 KiB，两组共 4 KiB）。
    constexpr static uint32_t WIDE_EXTRA_D_OFFSET = WIDE_EXTRA_LSE_OFFSET + 2 * baseM * BRCB_BASE_NUM * sizeof(float);

    uint64_t row = 0; // 本子核负责的 Q 行数（子核内行数，非 tile 总行数）。
    uint64_t col = 0; // 当前 tile 的物理 KV 列数。
    // KV 维按 32B(8 个 FP32) 对齐的列数；与 16 元素对齐一起决定 alignCol。
    uint64_t align32Col = 0;
    uint64_t align16Col = 0;
    uint64_t alignCol = 0;
    // —— 当前 tile 坐标（由 SetWideTileState 从 Params 刷新） ——
    uint64_t curCoreBatch = 0;
    uint64_t curCoreN1Idx = 0; // q_n
    uint64_t curT1Idx = 0;     // q_t
    uint64_t curCoreS1Idx = 0; // q_s
    // LSE 的行 stride 参数（BeginWidePipeline 时取一次，之后不变）。
    uint64_t maxQSeqlen = 0;
    uint64_t n1 = 0; // Q head 数（numHeads），LSE 寻址用。
    // 单个 P16/dS16 块的字节数，用于在同一 stage 内切出相邻的 P16 与 dS16。
    uint64_t p16BaseBufLen = 0;
    // softmax 缩放系数（由 Host 下发），在 Muls 中作用于 S。
    float scaleValue = 0.0f;
    // prefix 裁剪后的有效 KV 列数；< alignCol 时须显式清零无效列。
    uint64_t validCol = 0;
    // 两个子核合并的完整 tile 行数，决定 NZ slab 的行跨度。
    uint64_t rawMatrixRows = 0;
    // 本子核在完整 tile 中的起始行（写回 GM 时的行偏移）。
    uint64_t rawRowOffset = 0;
    // 位 i 表示元数据槽 i 的 LSE/D 已装载，避免同一个 Q 被重复搬运。
    uint32_t wideQMetadataMask = 0;

    GlobalTensor<float> sGm;                 // S 的 GM 视图，(N, s1, s2)，每次 Prefetch 重新绑定。
    GlobalTensor<float> softmaxLseGm;        // LSE 的 GM 视图，(N, s1, 1)，每行一个值。
    GlobalTensor<float> dpGm;                // dP 的 GM 视图，(N, s1, s2)，每次 Prefetch 重新绑定。
    GlobalTensor<InputDType> pWorkspaceGm;   // P 的输出视图（低精度），(N, s1, s2)。
    GlobalTensor<float> softGradworkspaceGm; // D = rowsum(dO*O) 的 GM 视图，(N, s1, 8)，8 列供 BRCB。
    GlobalTensor<InputDType> dsWorkspaceGm;  // dS 的输出视图（低精度），(N, s1, s2)。

    // —— UB 缓冲（除 P32 scratch 外均按 stage 各一份） ——
    // sTensor/dpFp32Tensor：FP32 原始输入（S/dP），双 stage。
    LocalTensor<float> sTensor[WIDE_INPUT_STAGES];
    // LSE 行缓冲（同时是元数据槽 0/1 的 LSE 槽）。
    LocalTensor<float> lseTensor[WIDE_INPUT_STAGES];
    // LSE 经 BRCB 广播后的结果，供 SubBrcb 以 src1BlkStride=0 复用。
    LocalTensor<float> lseBrocTensor[WIDE_INPUT_STAGES];
    // P/dS 的 FP32 scratch（两 stage 指向同一块 WIDE_NZ_SCRATCH_OFFSET，见 PrepareWideNz）。
    LocalTensor<float> pFp32Tensor[WIDE_INPUT_STAGES];
    // P 的低精度输出缓冲（复用本 stage 的 S 空间，S 消费完后才写入）。
    LocalTensor<InputDType> p16Tensor[WIDE_INPUT_STAGES];
    // dP 的 FP32 输入缓冲，双 stage。
    LocalTensor<float> dpFp32Tensor[WIDE_INPUT_STAGES];
    // D 的 FP32 缓冲，双 stage，并兼作元数据槽 0/1 的 D 槽。
    LocalTensor<float> softmaxGradTensor[WIDE_INPUT_STAGES];
    // —— 跨 tile 复用的 Q 元数据槽（4 个）——
    // D 槽：槽 0/1 复用 softmaxGradTensor，槽 2/3 使用 UB 尾部。
    LocalTensor<float> wideDCache[WIDE_Q_METADATA_SLOTS];
    // LSE 槽：槽 0/1 复用 lseTensor，槽 2/3 使用 WIDE_EXTRA_LSE_OFFSET。
    LocalTensor<float> wideLseCache[WIDE_Q_METADATA_SLOTS];
    // dS 的低精度输出缓冲（与本 stage 的 P16 相邻，互不重叠）。
    LocalTensor<InputDType> ds16Tensor[WIDE_INPUT_STAGES];

    /**
     * @brief 按编译期偏移划分 UB：两个 stage 的 S/dP/LSE/广播/输出缓冲，以及 4 个元数据槽。
     * @note 只做地址划分，不发起搬运，也不设置事件（事件在 BeginWidePipeline 预置）。
     * @note 元数据槽 0/1 直接复用 stage 缓冲以节省 UB，槽 2/3 使用尾部独立空间。
     */
    __aicore__ inline WideSoftmax()
    {
        // 分核：一个 core 最大 128 * 128，一个子核（vec）64 * 128；各缓冲按此上界分配。
        // 33 KiB 的 raw stage 足以容纳 64x128 的 FP32 NZ（实际 <= 32 KiB）。
        uint64_t sBufferLen = WIDE_RAW_STAGE_BYTES;
        uint64_t lBufferLen = baseM * sizeof(float);
        uint64_t lBrobBufferLen = BRCB_BASE_NUM * baseM * sizeof(float);
        uint64_t dpBufLen = WIDE_RAW_STAGE_BYTES;
        uint64_t p16BufLen = baseM * 128 * sizeof(InputDType);
        uint64_t dBufLen = BRCB_BASE_NUM * baseM * sizeof(float);
        p16BaseBufLen = p16BufLen;

        // UB 尾部固定分区：164K 广播 LSE、168K LSE 行、169K D 行。
        constexpr uint64_t LSE_BROADCAST_BASE = 164 * 1024;
        constexpr uint64_t LSE_BASE = 168 * 1024;
        constexpr uint64_t SOFTMAX_GRAD_BASE = 169 * 1024;

        for (uint64_t i = 0; i < WIDE_INPUT_STAGES; i++) {
            // 每个 stage 一组缓冲：S/dP 各两份；P16/dS16 复用本 stage 的 S 空间；
            // 缓冲区能否被覆盖由 MTE3->MTE2 的 EVENT_ID0/1 保护（见 PrefetchWide）。
            sTensor[i] = resource.ubBuf.template GetBufferByByte<float>(WIDE_S_STAGE_BASE + sBufferLen * i);
            lseBrocTensor[i] = resource.ubBuf.template GetBufferByByte<float>(LSE_BROADCAST_BASE + lBrobBufferLen * i);
            lseTensor[i] = resource.ubBuf.template GetBufferByByte<float>(LSE_BASE + lBufferLen * i);
            dpFp32Tensor[i] = resource.ubBuf.template GetBufferByByte<float>(WIDE_DP_STAGE_BASE + dpBufLen * i);
            softmaxGradTensor[i] = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_GRAD_BASE + dBufLen * i);
        }
        // 槽 0/1 直接指向 stage 缓冲，省下两块 UB。
        wideDCache[0] = softmaxGradTensor[0];
        wideDCache[1] = softmaxGradTensor[1];
        wideLseCache[0] = lseTensor[0];
        wideLseCache[1] = lseTensor[1];
        // 四个 Q 元数据槽可跨 tile 复用 LSE/D。
        constexpr uint64_t metadataBytes = baseM * BRCB_BASE_NUM * sizeof(float);
        // 槽 2/3 使用 UB 尾部的独立空间，避免与 stage 缓冲争用。
        wideLseCache[2] = resource.ubBuf.template GetBufferByByte<float>(WIDE_EXTRA_LSE_OFFSET);
        wideLseCache[3] = resource.ubBuf.template GetBufferByByte<float>(WIDE_EXTRA_LSE_OFFSET + metadataBytes);
        wideDCache[2] = resource.ubBuf.template GetBufferByByte<float>(WIDE_EXTRA_D_OFFSET);
        wideDCache[3] = resource.ubBuf.template GetBufferByByte<float>(WIDE_EXTRA_D_OFFSET + metadataBytes);
    }

    /**
     * @brief 建立 wide softmax 的流水：绑定 GM、读取核内不变量、预置复用令牌。
     * @param softmaxLse_ LSE 的 GM 地址。
     * @param softGradworkspace_ D = rowsum(dO*O) 的 GM 地址。
     * @param tilingData_ Host tiling 地址（读取 maxQSeqlen/numHeads/scaleValue）。
     * @note 预置的四对令牌必须在 EndWidePipeline 成对回收，否则核退出时事件未清零。
     */
    __aicore__ inline void BeginWidePipeline(GM_ADDR softmaxLse_, GM_ADDR softGradworkspace_, GM_ADDR tilingData_)
    {
        // 核内不变量只读取一次，后续 stage 只更新 tile 坐标。
        __gm__ BlockSparseAttentionGradTilingData* tilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData*>(tilingData_);
        maxQSeqlen = tilingData->maxQSeqlen;
        n1 = tilingData->numHeads;
        scaleValue = tilingData->scaleValue;
        softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse_);
        softGradworkspaceGm.SetGlobalBuffer((__gm__ float*)softGradworkspace_);
        // 每个方向分别初始化两个复用令牌，并在 EndWidePipeline 成对回收。
        // 6/7 与 PrefetchWide/ComputeWide 的 MTE2<->V 双向握手共用同一 event id。
        // 4/5 另在 ComputeWideNz 中作为 P16/dS16 写回前的 V->MTE3 栅栏（set/wait 紧邻）复用。
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    }

    /**
     * @brief 复位元数据装载位图（每个 packet 开始时调用）。
     * @note 复位后由调用方对实际用到的槽调用 InvalidateWideQMetadataSlot，
     *       使 PrefetchWide 重新搬运这些 Q 的 LSE/D。
     */
    __aicore__ inline void ResetWideQMetadata()
    {
        wideQMetadataMask = 0;
    }

    /**
     * @brief 使一个元数据槽失效，强制下一次 PrefetchWide 重新装载该 Q 的 LSE/D。
     * @param slot 逻辑槽号（内部按 WIDE_Q_METADATA_SLOTS-1 取模到物理槽）。
     */
    __aicore__ inline void InvalidateWideQMetadataSlot(uint32_t slot)
    {
        // 新 Q 只失效自己的槽；PrefetchWide 在 V->MTE2 屏障后补载元数据。
        const uint32_t qSlot = slot & (WIDE_Q_METADATA_SLOTS - 1U);
        wideQMetadataMask &= ~(1U << qSlot);
    }

    /**
     * @brief 进入 raw stage 复用区间：声明 stage 0/1 的别名输出已写回 GM。
     * @note 与 EndWideRawStageReuse 成对（EVENT_ID0/1），由 ProcessWideAiv 按 work item 调用。
     */
    __aicore__ inline void BeginWideRawStageReuse()
    {
        // D 写回屏障后，MTE3->MTE2 的 0/1 归 raw stage 复用；P/dS 写完后交还令牌。
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }

    /**
     * @brief 退出 raw stage 复用区间：回收 stage 0/1 的令牌。
     * @note 回收点同时是一次 V->MTE2 复用屏障：确保本区间内最后一次 MTE3 写回已完成。
     */
    __aicore__ inline void EndWideRawStageReuse()
    {
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }

    /**
     * @brief 流水收尾：回收 BeginWidePipeline 预置的四对令牌。
     * @note 逐个 wait：若前面某处漏 set 会在这里死等，属于有意的"事件计数自检"。
     */
    __aicore__ inline void EndWidePipeline()
    {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    }

    /**
     * @brief 把 NZ 布局的 FP32 tile 从 GM 搬到 UB。
     * @param dst UB 目标（本子核对应的紧凑区起点）。
     * @param src GM 源（S 或 dP 的 tile 起点）。
     * @note 源的行 stride 用"矩阵行对齐值 - 本子核行数"计算，目标在 UB 中紧凑排列故 dstStride=0。
     */
    __aicore__ inline void CopyNzRawToUb(LocalTensor<float> dst, GlobalTensor<float> src) const
    {
        constexpr uint32_t C0_SIZE = 16;
        constexpr uint32_t FP32_PER_BLOCK = 8;
        const uint32_t matrixRowsAlign = (rawMatrixRows + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(alignCol / C0_SIZE);
        copyParams.blockLen = static_cast<uint16_t>(row * C0_SIZE / FP32_PER_BLOCK);
        // 源按对齐后的矩阵行数留空洞（srcStride），目标在 UB 中紧凑排列（dstStride=0）。
        copyParams.srcStride = static_cast<uint16_t>((matrixRowsAlign - row) * C0_SIZE / FP32_PER_BLOCK);
        copyParams.dstStride = 0;
        DataCopy(dst, src[rawRowOffset * C0_SIZE], copyParams);
    }

    /**
     * @brief 为指定 stage 绑定 P32/P16/dS16 的 UB 地址。
     * @param stage 输入 stage（0/1）。
     * @note P32 与 stage 无关（共享 scratch）；P16/dS16 复用本 stage 的 S 空间，
     *       因此只能在 S 被消费完之后绑定使用。
     */
    __aicore__ inline void PrepareWideNz(uint32_t stage)
    {
        // P32 使用共享 scratch；S 消费完后复用其 stage 存放 P16/dS16。
        // MTE3->MTE2 令牌保证下一次预取不会覆盖尚未写回的输出。
        pFp32Tensor[stage] = resource.ubBuf.template GetBufferByByte<float>(WIDE_NZ_SCRATCH_OFFSET);
        p16Tensor[stage] =
            resource.ubBuf.template GetBufferByByte<InputDType>(WIDE_S_STAGE_BASE + WIDE_RAW_STAGE_BYTES * stage);
        ds16Tensor[stage] = resource.ubBuf.template GetBufferByByte<InputDType>(
            WIDE_S_STAGE_BASE + WIDE_RAW_STAGE_BYTES * stage + p16BaseBufLen);
    }

    /**
     * @brief 预取一个 tile：搬运 S/dP 原始数据，并按需装载该 Q 的 LSE/D 元数据。
     * @param params tile 参数（含 S/dP 地址与行/列信息）。
     * @param stage 输入 stage（0/1），决定缓冲与事件编号。
     * @note 两次 wait 的顺序是正确性关键：先等本 stage 上一轮 Vector 计算结束（V->MTE2），
     *       再等别名输出写完（MTE3->MTE2），之后才允许覆盖 S/dP 缓冲。
     * @note 元数据按 Q 序号惰性装载：位图已置位则跳过，避免重复搬运 LSE/D。
     */
    __aicore__ inline void PrefetchWide(Params const& params, uint32_t stage)
    {
        SetWideTileState(params);
        sGm.SetGlobalBuffer((__gm__ float*)params.rawScore);
        dpGm.SetGlobalBuffer((__gm__ float*)params.rawProbabilityGrad);
        const auto eventId = stage == 0 ? EVENT_ID6 : EVENT_ID7;
        wait_flag(PIPE_V, PIPE_MTE2, eventId);
        const auto outputEventId = stage == 0 ? EVENT_ID0 : EVENT_ID1;
        // 等待别名输出写完，再恢复 raw 输入描述符。
        wait_flag(PIPE_MTE3, PIPE_MTE2, outputEventId);
        sTensor[stage] =
            resource.ubBuf.template GetBufferByByte<float>(WIDE_S_STAGE_BASE + WIDE_RAW_STAGE_BYTES * stage);
        dpFp32Tensor[stage] =
            resource.ubBuf.template GetBufferByByte<float>(WIDE_DP_STAGE_BASE + WIDE_RAW_STAGE_BYTES * stage);
        CopyNzRawToUb(sTensor[stage], sGm);
        CopyNzRawToUb(dpFp32Tensor[stage], dpGm);
        // 用 Q 序号（而不仅是物理槽号）判重：物理槽只有 4 个，Q 数量可能更多。
        const uint32_t qSlot = static_cast<uint32_t>(curT1Idx);
        const uint32_t metadataSlot = qSlot & (WIDE_Q_METADATA_SLOTS - 1U);
        if (((wideQMetadataMask >> qSlot) & 1U) == 0) {
            LseCopy(softmaxLseGm, wideLseCache[metadataSlot], row, curCoreS1Idx);
            CopyDIn(softGradworkspaceGm, wideDCache[metadataSlot], row, curCoreS1Idx);
            wideQMetadataMask |= 1U << qSlot;
        }
        set_flag(PIPE_MTE2, PIPE_V, eventId);
    }

    /**
     * @brief 消费指定 stage 的 S/dP，产出 P/dS 并写回 GM。
     * @param params tile 参数（提供 P/dS 的目标地址）。
     * @param stage 输入 stage（0/1）。
     * @note 先等本 stage 的搬运完成（MTE2->V），计算结束后把令牌交还给 PrefetchWide。
     */
    __aicore__ inline void ComputeWide(Params const& params, uint32_t stage)
    {
        SetWideTileState(params);
        pWorkspaceGm.SetGlobalBuffer((__gm__ InputDType*)params.probability);
        dsWorkspaceGm.SetGlobalBuffer((__gm__ InputDType*)params.scoreGrad);
        const auto eventId = stage == 0 ? EVENT_ID6 : EVENT_ID7;
        wait_flag(PIPE_MTE2, PIPE_V, eventId);
        PrepareWideNz(stage);
        ComputeWideNz(stage);
        set_flag(PIPE_V, PIPE_MTE2, eventId);
    }

    /**
     * @brief 把低精度 NZ tile 从 UB 写回 GM。
     * @param dst GM 目标（P 或 dS）。
     * @param src UB 源。
     * @param storeRows 本子核实际写回的行数（不含 Q 尾块 padding）。
     * @note C0=16 且低精度，故一行 C0 恰好是一个 32B DMA block，blockLen 直接用行数。
     */
    __aicore__ inline void CopyNzLowpToGm(GlobalTensor<InputDType> dst, LocalTensor<InputDType> src,
                                          uint32_t storeRows) const
    {
        constexpr uint32_t C0_SIZE = 16;
        const uint32_t matrixRowsAlign = (rawMatrixRows + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(alignCol / C0_SIZE);
        // FP16/BF16 的一行 C0 正好是一个 32B DMA block。
        copyParams.blockLen = static_cast<uint16_t>(storeRows);
        copyParams.srcStride = 0;
        copyParams.dstStride = static_cast<uint16_t>(matrixRowsAlign - storeRows);
        DataCopy(dst[rawRowOffset * C0_SIZE], src, copyParams);
    }

    /**
     * @brief 把 Q 尾块 padding 行清零，避免下游读到 UB 里的历史数据。
     * @param dst GM 目标（P 或 dS）。
     * @param zero 已清零的源张量（提供零值）。
     * @param qPadding padding 行数；0 表示无需清理，函数立即返回。
     * @note 只有负责 tile 最后一行的子核会传入非零 qPadding。
     */
    __aicore__ inline void ClearNzLowpQPadding(GlobalTensor<InputDType> dst, LocalTensor<InputDType> zero,
                                               uint32_t qPadding) const
    {
        constexpr uint32_t C0_SIZE = 16;
        // 满高 tile 无 padding，直接返回，省掉整轮 DataCopy。
        if (qPadding == 0) {
            return;
        }
        const uint32_t matrixRowsAlign = (rawMatrixRows + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        // FP16/BF16 的一行 C0 正好是一个 32B DMA block。
        copyParams.blockLen = static_cast<uint16_t>(qPadding);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        const uint32_t paddingRow = static_cast<uint32_t>(rawRowOffset + row);
        const uint32_t c1Count = static_cast<uint32_t>(alignCol / C0_SIZE);
        for (uint32_t c1 = 0; c1 < c1Count; ++c1) {
            const uint64_t destinationOffset =
                static_cast<uint64_t>(c1) * matrixRowsAlign * C0_SIZE + static_cast<uint64_t>(paddingRow) * C0_SIZE;
            DataCopy(dst[destinationOffset], zero, copyParams);
        }
    }

    /**
     * @brief 把 tile 内"物理存在但 prefix 无效"的列填充为指定值。
     * @param tensor 待填充的 FP32 张量（S 或 P/dS）。
     * @param value 填充值：S 用 -FLT_MAX（使 exp 下溢为 0），P/dS 用 0。
     * @note 只处理从 validCol 所在 slab 开始的尾部 slab；首个 slab 用掩码保留前 validCol 个 lane。
     */
    __aicore__ inline void FillNzInvalidColumns(LocalTensor<float> tensor, float value) const
    {
        constexpr uint32_t C0_SIZE = 16;
        const uint32_t firstSlab = static_cast<uint32_t>(validCol / C0_SIZE);
        const uint32_t slabCount = static_cast<uint32_t>(alignCol / C0_SIZE);
        for (uint32_t slab = firstSlab; slab < slabCount; ++slab) {
            const uint32_t validLanes = slab == firstSlab ? validCol % C0_SIZE : 0;
            uint64_t mask[2] = {(0xffffULL << validLanes) & 0xffffULL, 0};
            Duplicate(tensor[slab * row * C0_SIZE], value, mask, static_cast<uint8_t>(row), 1,
                      C0_SIZE / BLOCK_FP32_NUM);
        }
        // Duplicate 修改了向量 mask，必须恢复全 mask，否则会污染后续指令。
        SetVectorMask<int8_t>(~0ULL, ~0ULL);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * @brief 计算 P = exp(scale*S - LSE) 与 dS = P * (dP - D)，并以低精度写回。
     * @param stage 输入 stage（0/1），决定缓冲、事件与别名关系。
     * @note 布局约定：S/dP 以 NZ（C0=16）在 UB 中紧凑存放；P32 原地复用为 dS；
     *       P16 与 dS16 分别写回 workspace 低精度平面中 P 与 dS 的位置。
     * @note 顺序：先算 P（含无效列清零）-> Cast P16 -> 写回 P -> 复用 P32 算 dS -> 写回 dS。
     */
    __aicore__ inline void ComputeWideNz(uint32_t stage)
    {
        constexpr uint32_t C0_SIZE = 16;
        constexpr float NEGATIVE_MAX = -3.402823466e+38F;
        const uint32_t metadataSlot = static_cast<uint32_t>(curT1Idx) & (WIDE_Q_METADATA_SLOTS - 1U);
        LocalTensor<float> currentLse = wideLseCache[metadataSlot];
        LocalTensor<float> currentLseBroadcast = lseBrocTensor[stage];
        LocalTensor<float> currentD = wideDCache[metadataSlot];
        LocalTensor<float> rawS = sTensor[stage];
        LocalTensor<float> rawDp = dpFp32Tensor[stage];
        LocalTensor<float> p32 = pFp32Tensor[stage];
        LocalTensor<InputDType> p16 = p16Tensor[stage];
        LocalTensor<InputDType> ds16 = ds16Tensor[stage];

        const uint32_t qRound = (rawMatrixRows + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
        // 仅负责 Q tile 最后一行的子核清零 GM 对齐尾部，避免扩展容量为 64 行的 UB slab。
        const uint32_t qPadding = rawRowOffset + row == rawMatrixRows ? qRound - rawMatrixRows : 0;
        const uint32_t storeRows = static_cast<uint32_t>(row);
        const uint32_t c1Count = static_cast<uint32_t>(alignCol / C0_SIZE);
        const uint32_t inputSlabElements = static_cast<uint32_t>(row) * C0_SIZE;
        const uint32_t outputSlabElements = storeRows * C0_SIZE;
        const uint32_t storeElements = storeRows * static_cast<uint32_t>(alignCol);
        const bool hasInvalidColumns = validCol < alignCol;

        const uint8_t repeatTimes = static_cast<uint8_t>(CeilDiv(row, BRCB_BASE_NUM));
        // 每行的 LSE 广播成 8 列，使 SubBrcb 可以用 src1BlkStride=0 复用同一个值。
        Brcb(currentLseBroadcast, currentLse, repeatTimes, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();

        // S *= scale；NZ slab 在 UB 中紧凑排列，一次 Vector 指令即可缩放全部有效元素。
        Muls(rawS, rawS, static_cast<float>(scaleValue), row * alignCol);
        AscendC::PipeBarrier<PIPE_V>();

        if (hasInvalidColumns) {
            // 无效列置 -FLT_MAX，使随后的 Exp 下溢为 0，不参与归约。
            FillNzInvalidColumns(rawS, NEGATIVE_MAX);
        }

        for (uint32_t c1 = 0; c1 < c1Count; ++c1) {
            const uint32_t inputOffset = c1 * inputSlabElements;
            const uint32_t outputOffset = c1 * outputSlabElements;
            // P32 = S - LSE，逐 C1 slab 处理（广播值随行推进）。
            SubBrcb(p32[outputOffset], rawS[inputOffset], currentLseBroadcast, row, C0_SIZE);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // Exp 不依赖行元数据，可合并处理连续的全部 C1 slab（storeElements = 行数 * alignCol）。
        Exp(p32, p32, storeElements);
        AscendC::PipeBarrier<PIPE_V>();

        // 显式清零无效列，避免依赖 Exp 下溢或 scratch 的历史内容。
        if (hasInvalidColumns) {
            FillNzInvalidColumns(p32, 0.0F);
        }
        // 先把 P 转成低精度（P32 还要复用来算 dS），四舍五入与 legacy 保持一致。
        Cast(p16, p32, AscendC::RoundMode::CAST_ROUND, storeElements);
        // 满高 tile 无 padding，使用起点描述符，避免构造越界的一尾地址。
        const uint32_t paddingZeroOffset = qPadding == 0 ? 0 : storeElements;
        LocalTensor<InputDType> paddingZero = p16[paddingZeroOffset];
        if (qPadding != 0) {
            Duplicate(paddingZero, static_cast<InputDType>(0), qPadding * C0_SIZE);
        }

        const auto eventId = stage == 0 ? EVENT_ID5 : EVENT_ID4;
        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);
        // V->MTE3 事件（set/wait 相邻）：保证 p16 写完再 DMA 出去，P 与 dS 各用一次。
        CopyNzLowpToGm(pWorkspaceGm, p16, storeRows);

        for (uint32_t c1 = 0; c1 < c1Count; ++c1) {
            const uint32_t inputOffset = c1 * inputSlabElements;
            SubBrcb(rawDp[inputOffset], rawDp[inputOffset], currentD, row, C0_SIZE);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // dS = P32 * (dP - D)：P32 就地复用为 dS FP32；P16 地址独立，故写回可与 Vector 计算重叠。
        Mul(p32, p32, rawDp, storeElements);
        AscendC::PipeBarrier<PIPE_V>();
        if (hasInvalidColumns) {
            FillNzInvalidColumns(p32, 0.0F);
        }
        Cast(ds16, p32, AscendC::RoundMode::CAST_ROUND, storeElements);

        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);
        CopyNzLowpToGm(dsWorkspaceGm, ds16, storeRows);
        if (qPadding != 0) {
            ClearNzLowpQPadding(pWorkspaceGm, paddingZero, qPadding);
            ClearNzLowpQPadding(dsWorkspaceGm, paddingZero, qPadding);
        }
        const auto outputEventId = stage == 0 ? EVENT_ID0 : EVENT_ID1;
        // 收尾令牌：告知 PrefetchWide 本 stage 的输出（别名 S/dP 空间）已可覆盖。
        set_flag(PIPE_MTE3, PIPE_MTE2, outputEventId);
    }

    /**
     * @brief 把 Params 的 tile 坐标与尺寸刷到成员变量，并预计算三种对齐列数。
     * @param params 本 tile 的参数（见 Params）。
     * @note 每个 PrefetchWide / ComputeWide 开头都会调用，保证两条流水读到同一份 tile 状态。
     */
    __aicore__ inline void SetWideTileState(Params const& params)
    {
        col = params.columnCount;
        align32Col = (col + BLOCK_FP32_NUM - 1) / BLOCK_FP32_NUM * BLOCK_FP32_NUM;
        align16Col = (col + BLOCK_16_NUM - 1) / BLOCK_16_NUM * BLOCK_16_NUM;
        // NZ 的 C0=16：行内 slab 必须同时满足 32B 与 16 元素对齐，取二者中合适的那个。
        alignCol = (align32Col % BLOCK_16_NUM != 0) ? align16Col : align32Col;
        row = params.queryRows;
        curCoreBatch = params.batchIndex;
        curCoreN1Idx = params.queryHeadIndex;
        curCoreS1Idx = params.queryStart;
        curT1Idx = params.metadataSlot;
        validCol = params.validColumns;
        rawMatrixRows = params.matrixRows;
        rawRowOffset = params.rowOffset;
    }

    /**
     * @brief 取一行 LSE：按 (batch, Q head, s1) 定位，元素数 = 本子核行数。
     * @param LseGm LSE 的 GM 视图。
     * @param lse UB 目标。
     * @param count 本子核行数。
     * @param curS1Idx 本子核首行的 s1 位置。
     * @note 用 DataCopyPad 以支持非 32B 对齐的行数。
     */
    __aicore__ inline void LseCopy(GlobalTensor<float>& LseGm, LocalTensor<float>& lse, uint64_t count,
                                   uint64_t curS1Idx)
    {
        const uint64_t startOffset = curCoreBatch * (n1 * maxQSeqlen) + curCoreN1Idx * maxQSeqlen + curS1Idx;
        DataCopyPad(
            lse, LseGm[startOffset],
            {static_cast<uint16_t>(1), static_cast<uint32_t>(count * sizeof(float)), static_cast<uint32_t>(0), 0, 0},
            {false, 0, 0, 0});
    }

    /**
     * @brief 逐 C0 slab 的行广播减法：out[i][j] = in0[i][j] - rowConst[i]。
     * @param ubOut 输出（可与 ubIn0 相同，实现原位相减）。
     * @param ubIn0 被减数（NZ，每行连续 alignCol 个元素）。
     * @param ubIn1 行常量（BRCB 广播后每行 8 列）。
     * @param row 行数（作为 repeatTimes）。
     * @param col 单个 slab 的列数（C0=16）。
     * @note src1BlkStride=0 + src1RepStride=1 是"每 repeat 前进一行常量"的关键。
     */
    __aicore__ inline void SubBrcb(LocalTensor<float> const& ubOut, LocalTensor<float> const& ubIn0,
                                   LocalTensor<float> const& ubIn1, uint64_t row, uint64_t col)
    {
        // 每次 repeat 处理 64 个 FP32 元素，repeat 沿 Q 行推进。
        uint32_t countEachRepeat = REAPTE_BYTE / sizeof(float);
        uint32_t colLoop = (col + countEachRepeat - 1) / countEachRepeat;
        uint32_t remain = col % countEachRepeat;
        uint64_t mask = countEachRepeat;
        uint8_t repeatTimes = row;
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        // src1 的块 stride 为 0：同一个广播值在整个 slab 上复用。
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = col / BLOCK_FP32_NUM;
        repeatParams.src0RepStride = col / BLOCK_FP32_NUM;
        repeatParams.src1RepStride = 1;

        for (uint32_t i = 0; i < colLoop; i++) {
            if (i == colLoop - 1 && remain != 0) {
                mask = remain;
            }
            AscendC::Sub(ubOut[i * countEachRepeat], ubIn0[i * countEachRepeat], ubIn1[0], mask, repeatTimes,
                         repeatParams);
        }
    }

    /**
     * @brief 取一行 D：按 (batch, Q head, s1) 定位，每行 BRCB_BASE_NUM 列。
     * @param d D 的 GM 视图。
     * @param dLocal UB 目标。
     * @param count 本子核行数。
     * @param curS1Idx 本子核首行的 s1 位置。
     * @note stride 用的是 maxQSeqlen（序列上界），与 LseCopy 保持一致。
     */
    __aicore__ inline void CopyDIn(GlobalTensor<float>& d, LocalTensor<float>& dLocal, uint64_t count,
                                   uint64_t curS1Idx)
    {
        const uint64_t startOffset = curCoreBatch * (n1 * maxQSeqlen * BRCB_BASE_NUM) +
                                     curCoreN1Idx * maxQSeqlen * BRCB_BASE_NUM + curS1Idx * BRCB_BASE_NUM;
        DataCopyPad(dLocal, d[startOffset],
                    {static_cast<uint16_t>(count), static_cast<uint32_t>(BRCB_BASE_NUM * sizeof(float)), 0, 0, 0},
                    {false, 0, 0, 0});
    }
};
} // namespace NpuArch::Epilogue::Block
#endif
