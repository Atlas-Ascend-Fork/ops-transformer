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
 * \file block_mmad_bsag_wide.hpp
 * \brief Block MAD BSAG WIDE Kernel Implementation
 */

#ifndef GEMM_BLOCK_MMAD_BSAG_WIDE_HPP
#define GEMM_BLOCK_MMAD_BSAG_WIDE_HPP

// ============================================================================
// wide 专用 MMAD 封装：在 legacy BlockMmad 之上补齐"缓存布局"与"zN 源累加"两类能力。
//
// 设计约束（I2 阶段，不得改动 legacy 行为）：
//   * 模板参数 LegacyMmad 提供全部 tile 形状、layout 与 tile 级 copy 类型，
//     本类只新增接口，不修改 legacy 的资源布局与 tile 语义；
//   * WideCacheMode 的三种取值对应三段互不相同的 L1 分区，分区表见构造函数内注释；
//   * L0A/L0B/L0C 沿用 legacy 的 [0, STAGES) 槽位；所有"按槽位索引"的事件编号都加上
//     构造参数 pingpongFlagOffset，以便与同核 Cube1 使用的 0/1 事件隔离。
//
// 事件与槽位的对应关系是本文件最容易出错的地方，修改任何 WaitFlag/SetFlag 前请先确认
// 它保护的是哪个 buffer（L1 复用 / L0 操作数 / L0C 累加器）。
// ============================================================================

#include "block_mmad_bsag2.hpp"

namespace NpuArch::Gemm::Block {
// L1 缓存模式：决定构造函数如何为 l1A/l1B 分配物理槽位。
// 三种模式的 L1 分区互不相同，选错会覆盖其它阶段仍在使用的 L1 内容。
enum class WideCacheMode : uint32_t {
    STREAMED,      // Cube1：A 双缓冲 + K/V 各一槽（PreloadB / WithCachedBNz 路径）。
    RESIDENT_K,    // dQ：8 个常驻 K 槽；A 槽用于 P/dS 流（PreloadAccumANz + AccumulatePreloadedA）。
    RESIDENT_QUERY // dKV：常驻 Q/dO 对；A 操作数来自外部 L1，本对象只做 zN -> L0A 转置加载。
};
// wide 专用缓存与 zN 累加接口。本类自持 l1/l0 张量与事件，因此可与 legacy BlockMmad 实例并存。
/**
 * @brief legacy BlockMmad 的 wide 扩展包装：增加 L1 常驻缓存、zN 源预载与转置累加接口。
 * @tparam LegacyMmad 被包装的 legacy BlockMmad（提供 tile 形状、元素/布局与 copy 器）。
 * @note 本类不持有 Cube1 使用的 Streamed 语义；三种缓存模式各自独立布局 L1。
 */
template <class LegacyMmad>
class WideBlockMmad {
public:
    // —— 从 Legacy 转发的类型：tile 形状、元素/布局、tile 级拷贝器 ——
    using L1TileShape_ = typename LegacyMmad::L1TileShape;
    using L0TileShape_ = typename LegacyMmad::L0TileShape;
    using AType_ = Gemm::GemmType<typename LegacyMmad::ElementA, typename LegacyMmad::LayoutA>;
    using BType_ = Gemm::GemmType<typename LegacyMmad::ElementB, typename LegacyMmad::LayoutB>;
    using CType_ = Gemm::GemmType<typename LegacyMmad::ElementC, typename LegacyMmad::LayoutC>;
    using TileMmad_ = typename LegacyMmad::TileMmad;
    using DispatchPolicy = MmadAtlasA2SBSAG2;
    using ArchTag = typename DispatchPolicy::ArchTag;

    // 对外沿用 Legacy 的命名，便于 Kernel 侧按同一套名字取用。
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename LegacyMmad::CopyGmToL1A;
    using CopyGmToL1B = typename LegacyMmad::CopyGmToL1B;
    using CopyL1ToL0A = typename LegacyMmad::CopyL1ToL0A;
    using CopyL1ToL0B = typename LegacyMmad::CopyL1ToL0B;
    using CopyL0CToGm = typename LegacyMmad::CopyL0CToGm;
    // NZ 写回专用拷贝器：Fixpipe FP32 NZ 的 C0 固定 16 个元素，
    // 不能用按 32B 推导 C0=8 的通用布局（见 WithCachedBNz 内的 layoutNz 构造）。
    using CopyL0CToGmNz = Gemm::Tile::CopyL0CToGm<
        ArchTag, typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator,
        Gemm::GemmType<ElementC, layout::zN>>;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    // —— 尺寸与容量常量 ——
    // STAGES：legacy 的 ping-pong 深度；L0A/L0B/L0C 均按此分槽。
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    // L1 张量数组长度：常驻模式下最多 8 个 B 槽（RESIDENT_K 的 8 个 K 槽，或 RESIDENT_QUERY 的 dO+Q 槽）。
    static constexpr uint32_t EXTENDED_CACHE_SLOTS = 8;
    // STREAMED 模式的 K/V 缓存槽数（当前只用 2 个，其余留作扩展）。
    static constexpr uint32_t EXTENDED_B_SLOTS = 4;

    // 单个 L1 tile 的字节数，用于推导各分区的绝对偏移（见构造函数）。
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);

    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;

    // L0 单槽字节数：把 L0 容量按 ping-pong 均分。
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;

    static constexpr uint32_t BLOCK_SIZE = 16;

    /**
     * @brief 绑定资源，并按缓存模式划分 L1/L0 槽位。
     * @param resource Kernel 持有的 L1/L0 buffer 资源。
     * @param l1BufAddrStart 本对象 l1A 的起始字节偏移（常驻模式下由调用方给高 L1 基址）。
     * @param pingpongFlagOffset 槽位型事件编号基址，用于与同核其它 MMAD 对象隔离。
     * @param cacheMode 缓存模式：STREAMED 供 Cube1，RESIDENT_K 供 dQ，RESIDENT_QUERY 供 dKV。
     * @note 本函数只做槽位划分，不发起任何数据搬运。
     */
    __aicore__ inline WideBlockMmad(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0,
                                    uint32_t pingpongFlagOffset = 0, WideCacheMode cacheMode = WideCacheMode::STREAMED)
    {
        // 事件编号基址：所有按槽位索引的 MTE1_MTE2 事件都加这个偏移。
        // 紧随其后的循环把 L0A/L0B/L0C 各按 STAGES 均分成 ping-pong 槽（与缓存模式无关）。
        PINGPONG_FLAG_OFFSET = pingpongFlagOffset;
        for (uint32_t i = 0; i < STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
        // 常驻模式：l1A 从调用方指定的基址起按 stage 分配；l1B 使用固定绝对槽位（见下）。
        if (cacheMode == WideCacheMode::RESIDENT_K || cacheMode == WideCacheMode::RESIDENT_QUERY) {
            for (uint32_t i = 0; i < STAGES; ++i) {
                l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1A_SIZE * i);
            }
            if (cacheMode == WideCacheMode::RESIDENT_K) {
                // P/dS 占 [0,64) KiB（l1BufAddrStart=0 起两个 L1A_SIZE）；
                // 八个 K 缓存槽紧随其后：L1A_SIZE*2 + L1B_SIZE*i => [64,320) KiB。
                for (uint32_t i = 0; i < EXTENDED_CACHE_SLOTS; ++i) {
                    l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(L1A_SIZE * 2 + L1B_SIZE * i);
                }
            } else {
                // dO 槽 = L1B_SIZE*(10..12) => [320,416) KiB；Q 槽 = L1B_SIZE*(13..15) => [416,512) KiB。
                // 两半各 3 槽，而 Host 当前最多下发 WIDE_MAX_Q_TASKS(=2) 个 Q task，余 1 槽留作扩展。
                // 注意这些槽位是绝对偏移，不叠加 l1BufAddrStart；RESIDENT_QUERY 下本对象的 l1A
                // 不被使用（dKV 的 A 操作数由 AccumulateTransposeFromExternalZNL1 直接从外部 L1 装入 L0A）。
                for (uint32_t i = 0; i < 3; ++i) {
                    l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(L1B_SIZE * (10 + i));
                    l1BTensor[i + 4] = resource.l1Buf.template GetBufferByByte<ElementB>(L1B_SIZE * (13 + i));
                }
            }
        } else {
            for (uint32_t i = 0; i < STAGES; ++i) {
                l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1A_SIZE * i);
            }
            // STREAMED（Cube1）：K/V 缓存紧跟 l1A 之后，共 EXTENDED_B_SLOTS 个槽。
            for (uint32_t i = 0; i < EXTENDED_B_SLOTS; ++i) {
                l1BTensor[i] =
                    resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1A_SIZE * 2 + L1B_SIZE * i);
            }
        }
    }

    /**
     * @brief 把一片 B（Cube1 的 K 或 V）搬入 l1B 的指定槽。
     * @param gB GM 上的 B 起点。
     * @param layoutB B 的 GM 布局（用 actualShape.k()/n() 取 tile）。
     * @param actualShape 本次实际 K/N 尺寸。
     * @param cacheSlot l1B 槽号，同时是 MTE1_MTE2 事件号（加 PINGPONG_FLAG_OFFSET）。
     * @param holdForConsumer true 表示不释放槽令牌，由 WithCachedBNz 的 releaseB 在最后一条边后释放。
     * @note MTE2->MTE1 的 0 号事件用 set/wait 立即串行化，保证拷贝完成后再被 L1->L0 读取。
     */
    __aicore__ inline void PreloadB(AscendC::GlobalTensor<ElementB> gB, LayoutB layoutB, GemmCoord actualShape,
                                    uint32_t cacheSlot, bool holdForConsumer = false)
    {
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(cacheSlot + PINGPONG_FLAG_OFFSET);
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
        copyGmToL1B(l1BTensor[cacheSlot], gB, layoutBInL1, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        if (!holdForConsumer) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(cacheSlot + PINGPONG_FLAG_OFFSET);
        }
    }

    /**
     * @brief 只发起搬运、不做同步的 B 预载，供一次性批量预载后统一 FinishFullPreload。
     * @param gB GM 上的 B 起点。
     * @param layoutB B 的 GM 布局。
     * @param actualShape 实际尺寸（仅 k()/n() 参与 tile 布局）。
     * @param cacheSlot l1B 槽号（常驻模式下即 K 列号或 dO/Q 槽号）。
     * @note 无事件保护，调用方必须在消费前调用 FinishFullPreload 或等价的同步。
     */
    __aicore__ inline void PreloadFullB(AscendC::GlobalTensor<ElementB> gB, LayoutB layoutB, GemmCoord actualShape,
                                        uint32_t cacheSlot)
    {
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
        copyGmToL1B(l1BTensor[cacheSlot], gB, layoutBInL1, layoutTileB);
    }

    /**
     * @brief 为一批 PreloadFullB 补齐 MTE2 -> MTE1 同步。
     * @note 使用 0 号 MTE2_MTE1 事件，set/wait 相邻即完成一次全批次的搬运完成确认。
     */
    __aicore__ inline void FinishFullPreload()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    /**
     * @brief 取常驻 dO/Q 缓存槽的 L1 张量，用作 MMAD 的 A 操作数。
     * @param slot 槽号：dO 用 queryLocalIndex，Q 用 queryLocalIndex + WIDE_QUERY_CACHE_OFFSET。
     */
    __aicore__ inline AscendC::LocalTensor<ElementB> GetCachedB(uint32_t slot)
    {
        return l1BTensor[slot];
    }

    /**
     * @brief 取得"可以覆盖常驻 Q/dO 槽"的许可（等待上一 work item 的 MTE1 读取结束）。
     * @note 与 ReleaseFullPreloadReuse 成对；由 WideExecutor 在每个 work item 的常驻预载前后调用。
     */
    __aicore__ inline void AcquireFullPreloadReuse()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(PINGPONG_FLAG_OFFSET);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(PINGPONG_FLAG_OFFSET + 1);
    }

    /**
     * @brief 归还常驻 Q/dO 槽的使用权，允许下一 work item 覆盖。
     * @note 与 AcquireFullPreloadReuse 成对，使用同一组（+0/+1）MTE1_MTE2 令牌。
     */
    __aicore__ inline void ReleaseFullPreloadReuse()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(PINGPONG_FLAG_OFFSET);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(PINGPONG_FLAG_OFFSET + 1);
    }

    /**
     * @brief Cube1 主操作：以常驻 A（Q 或 dO）与缓存 B（K 或 V）做一次 MMAD，并把 FP32 结果按 NZ 写回 GM。
     * @param cacheSlot l1B 槽号：0=K（算 S=Q*K^T），1=V（算 dP=dO*V^T）。
     * @param gC 结果（S 或 dP）在 workspace 的 FP32 NZ tile 起点。
     * @param actualShape M/N/K 的三段尺寸（m=Q 行数，n=KV 行数，k=headDim）。
     * @param pingpongFlag 既是 L0 槽号也是 M 事件号；函数结束时翻转，供下一条边使用。
     * @param residentA 常驻 A 的 L1 张量（Q 或 dO）。
     * @param releaseB 是否为该 B 槽的最后一条边；true 则在读完 L1 后释放槽令牌。
     * @note L0C 的 FIX_M 事件按槽号保护，避免上一轮 Fixpipe 未完成就覆盖累加器。
     */
    __aicore__ inline void WithCachedBNz(uint32_t cacheSlot, AscendC::GlobalTensor<ElementC> gC, GemmCoord actualShape,
                                         uint32_t& pingpongFlag, AscendC::LocalTensor<ElementA> residentA,
                                         bool releaseB = false)
    {
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);

        const uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        const uint32_t kRound = RoundUp<L1BAlignHelper::K_ALIGNED>(actualShape.k());

        // 等上一轮 MMAD 释放该 L0 槽，才能重新装配 L0A/L0B。
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);
        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kRound, nRound);
        copyL1ToL0B(l0BTensor[pingpongFlag], l1BTensor[cacheSlot], layoutBInL0, layoutBInL1);
        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kRound);
        copyL1ToL0A(l0ATensor[pingpongFlag], residentA, layoutAInL0, layoutAInL1);
        // 该 B 槽最后一条边读完，归还 MTE1_MTE2 令牌，允许后续 PreloadB 覆盖。
        if (releaseB) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(cacheSlot + PINGPONG_FLAG_OFFSET);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        const uint32_t scratchCSlot = pingpongFlag;
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(scratchCSlot);
        tileMmad(l0CTensor[scratchCSlot], l0ATensor[pingpongFlag], l0BTensor[pingpongFlag], mRound, nRound,
                 actualShape.k(), true, 3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);

        const auto blockShape = MakeCoord(actualShape.m(), actualShape.n());
        const auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
        // Fixpipe FP32 NZ 的 C0 固定为 16 个元素，不能使用按 32B 推导 C0=8 的通用布局。
        // 对应的 dstStride 以 32B 为单位，等于 mRound*2。
        // 手工构造 Fixpipe 的 NZ 布局：C0 固定 16，dstStride 以 32B 为单位（等于 mRound*2）。
        const auto layoutNz =
            layout::zN(actualShape.m(), actualShape.n(), 16, mRound / 16, 16, RoundUp<16>(actualShape.n()) / 16, 16,
                       256, 1, static_cast<int64_t>(mRound) * 16);
        copyL0CToGmNz(gC, l0CTensor[scratchCSlot], layoutNz, layoutInL0C, 3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(scratchCSlot);
        pingpongFlag = 1 - pingpongFlag;
    }

    /**
     * @brief 把一片 zN 格式的 A（P / dS / dS 的 FP32 源）预载进 l1A 的指定槽。
     * @param gA GM 上的 zN 源起点（workspace 低精度平面内的 tile）。
     * @param actualShape 源 tile 的 M/K 尺寸。
     * @param slot l1A 槽号（0/1 构成 ping-pong）。
     * @note 只发出 MTE2->MTE1 的 set，等待方是 AccumulatePreloadedA 或
     *       FinishAccumPreloadForExternal；MTE1_MTE2 令牌用于防止覆盖未消费的源。
     */
    __aicore__ inline void PreloadAccumANz(AscendC::GlobalTensor<ElementA> gA, GemmCoord actualShape, uint32_t slot)
    {
        using NzType = Gemm::GemmType<ElementA, layout::zN>;
        Gemm::Tile::CopyGmToL1<ArchTag, NzType> copyGmNzToL1;
        const auto sourceLayout = layout::zN::template MakeLayout<ElementA>(actualShape.m(), actualShape.k());
        const auto destinationLayout = layout::zN::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot + PINGPONG_FLAG_OFFSET);
        copyGmNzToL1(l1ATensor[slot], gA, destinationLayout, sourceLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
    }

    /**
     * @brief 等待指定槽的 zN 预载完成，并把该槽交还给外部消费者使用。
     * @param slot l1A 槽号。
     * @note "External" 指 A 操作数不是由本对象自己搬进 L0A，而是由
     *       AccumulateTransposeFromExternalZNL1 直接从这块 L1 做转置装载。
     */
    __aicore__ inline void FinishAccumPreloadForExternal(uint32_t slot)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot + PINGPONG_FLAG_OFFSET);
    }

    /**
     * @brief 暴露 l1A 槽，使 dKV 阶段可以把 P/dS 的 zN 张量交给另一个 MMAD 对象消费。
     * @param slot l1A 槽号。
     */
    __aicore__ inline AscendC::LocalTensor<ElementA> GetAccumL1A(uint32_t slot) const
    {
        return l1ATensor[slot];
    }

    /**
     * @brief 以外部 L1 的 zN 张量为 A，与缓存 B 做转置累加 MMAD（P^T*dO 得 dV，dS^T*Q 得 dK）。
     * @param externalA 外部 L1 的 zN A 张量（通常来自 blockMmad2 的 l1A 槽）。
     * @param externalReuseEvent 归还该外部 L1 槽所用的事件号。
     * @param cacheBSlot l1B 缓存槽号（dO 槽算 dV，Q 槽算 dK）。
     * @param layoutB B 的 GM 布局（仅用于取 tile）。
     * @param actualShape m=KV 行数，n=headDim，k=Q 行数。
     * @param accumulatorSlot L0C/L0A 槽号（0 或 1）。
     * @param initC true 表示首条边，L0C 由 MMAD 直接写入而非累加。
     * @param finalC true 表示末条边，MMAD 置 unitFlag=3 触发 Fixpipe。
     * @note 本函数不读写本对象的 l1ATensor：A 走 externalA -> L0A 的直接转置装载。
     *       K 行数等于满 tile 行数时可合并为一次 repeat，否则按 C0 分形逐块搬。
     */
    __aicore__ inline void AccumulateTransposeFromExternalZNL1(AscendC::LocalTensor<ElementA> externalA,
                                                               uint32_t externalReuseEvent, uint32_t cacheBSlot,
                                                               LayoutB layoutB, GemmCoord actualShape,
                                                               uint32_t accumulatorSlot, bool initC = true,
                                                               bool finalC = true)
    {
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        const uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        const uint32_t kRound = RoundUp<L1AAlignHelper::K_ALIGNED>(actualShape.k());

        // 消费 zN->L0A 读取令牌，防止下一次 GM 预载提前覆盖 dS。
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(externalReuseEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(accumulatorSlot);

        // dQ 的 L1 zN 行跨度固定为 128；dK 的 zZ 按实际 Q 行数对齐并紧凑存放。
        // 逐 KV C0 分形转置，跳过 Q 尾块未使用的 L1 行分形。
        const uint32_t qC0 = kRound / C0_NUM_PER_FRACTAL;
        const uint32_t sourceFractalStride = L1TileShape::M * C0_NUM_PER_FRACTAL;
        const uint32_t destinationFractalStride = kRound * C0_NUM_PER_FRACTAL;
        const uint32_t mC0 = mRound / C0_NUM_PER_FRACTAL;
        if (kRound == L1TileShape::M) {
            // 满 Q tile 的源/目的分形跨度相同，可合并为一次重复转置加载。
            AscendC::LoadData2dParams transposeParams{0, static_cast<uint8_t>(qC0 * mC0), 1, 0, 0, true, 0};
            AscendC::LoadData(l0ATensor[accumulatorSlot], externalA, transposeParams);
        } else {
            for (uint32_t mFractal = 0; mFractal < mC0; ++mFractal) {
                AscendC::LoadData2dParams transposeParams{0, static_cast<uint8_t>(qC0), 1, 0, 0, true, 0};
                AscendC::LoadData(l0ATensor[accumulatorSlot][mFractal * destinationFractalStride],
                                  externalA[mFractal * sourceFractalStride], transposeParams);
            }
        }

        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kRound, nRound);
        copyL1ToL0B(l0BTensor[accumulatorSlot], l1BTensor[cacheBSlot], layoutBInL0, layoutBInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(externalReuseEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

        if (initC) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(accumulatorSlot);
        }
        tileMmad(l0CTensor[accumulatorSlot], l0ATensor[accumulatorSlot], l0BTensor[accumulatorSlot], mRound, nRound,
                 actualShape.k(), initC, finalC ? 3 : 2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(accumulatorSlot);
    }

    /**
     * @brief dQ 主操作：以预载好的 zN A（dS 分片）与常驻 K 槽做累加 MMAD。
     * @param slot l1A 槽号，同时是该槽 MTE1_MTE2 事件的槽位（加 PINGPONG_FLAG_OFFSET）。
     * @param cacheBSlot 常驻 K 槽号（等于 packet 内的 kvLocal）。
     * @param layoutA/layoutB 仅用于取 tile 布局。
     * @param actualShape m=Q 行数，n=headDim，k=KV 行数。
     * @param initC true 表示该 Q 在本 packet 的首条边（L0C 首写）。
     * @param finalC true 表示该 Q 在本 packet 的末条边（触发 Fixpipe）。
     * @param accumulatorSlot L0C 槽号，由调用方按 qLocal 奇偶给出，使两个 Q task 的累加器互不干扰。
     */
    __aicore__ inline void AccumulatePreloadedA(uint32_t slot, uint32_t cacheBSlot, LayoutA layoutA, LayoutB layoutB,
                                                GemmCoord actualShape, bool initC, bool finalC,
                                                uint32_t accumulatorSlot)
    {
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        const uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        const uint32_t kRound = RoundUp<L1BAlignHelper::K_ALIGNED>(actualShape.k());

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(slot);
        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kRound);
        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kRound, nRound);
        copyL1ToL0A(l0ATensor[slot], l1ATensor[slot], layoutAInL0, layoutAInL1);
        copyL1ToL0B(l0BTensor[slot], l1BTensor[cacheBSlot], layoutBInL0, layoutBInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot + PINGPONG_FLAG_OFFSET);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        if (initC) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(accumulatorSlot);
        }
        tileMmad(l0CTensor[accumulatorSlot], l0ATensor[slot], l0BTensor[slot], mRound, nRound, actualShape.k(), initC,
                 finalC ? 3 : 2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(slot);
    }

    /**
     * @brief 把 L0C 累加器写回 GM。
     * @param gC 目标（dQ / dK / dV）地址。
     * @param layoutC 目标布局（dQ 与 dKV 的 shape 方向不同）。
     * @param actualShape m/n 决定写回块形状。
     * @param atomicWrite true 走 SetAtomicAdd：同一 Q 跨 packet 或同一 KV 被多个 packet 分摊时做累加。
     * @param accumulatorSlot L0C 槽号。
     * @note 结束时 set FIX_M，表示该 L0C 槽已可被下一轮 MMAD 覆盖。
     */
    __aicore__ inline void FlushAccumulator(AscendC::GlobalTensor<ElementC> gC, LayoutC layoutC, GemmCoord actualShape,
                                            bool atomicWrite, uint32_t accumulatorSlot = 0)
    {
        auto blockShape = MakeCoord(actualShape.m(), actualShape.n());
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
        if (atomicWrite) {
            AscendC::SetAtomicAdd<ElementC>();
            copyL0CToGm(gC, l0CTensor[accumulatorSlot], layoutC, layoutInL0C, 3);
            AscendC::SetAtomicNone();
        } else {
            copyL0CToGm(gC, l0CTensor[accumulatorSlot], layoutC, layoutInL0C, 3);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(accumulatorSlot);
    }

protected:
    // L1 张量：l1A 用于 A 操作数 / P/dS 源；l1B 用于 K/V 或常驻 dO/Q。
    AscendC::LocalTensor<ElementA> l1ATensor[EXTENDED_CACHE_SLOTS];
    AscendC::LocalTensor<ElementB> l1BTensor[EXTENDED_CACHE_SLOTS];

    // L0 张量：按 STAGES 分槽，累加族用槽号直接索引，Cube1 用 pingpongFlag 索引。
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];

    // tile 级 MMAD 与各类搬运器，均从 Legacy 继承其实现。
    TileMmad tileMmad;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
    CopyL0CToGmNz copyL0CToGmNz;

    // 槽位型事件编号基址（构造参数保存）；同一核内多个对象必须取不同值。
    uint32_t PINGPONG_FLAG_OFFSET = 0;
};
} // namespace NpuArch::Gemm::Block
#endif
