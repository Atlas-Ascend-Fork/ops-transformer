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
 * \file block_sparse_attention_grad_kernel_common.h
 * \brief Block Sparse Attention Grad Kernel Common Implementation
 */

#ifndef BSAG_KERNEL_COMMON_H
#define BSAG_KERNEL_COMMON_H

// wide 专用常量、Q 位图类型与容量断言。
namespace BSA::WideDetail {
// 调度模式编码：0=legacy，2=wide（1=grouped 为本仓保留值，I2 阶段不可达）。
// Kernel 侧仅把该值与 TilingData 的 scheduleMode 比较，作为 wide 路径的唯一谓词。
constexpr uint32_t SCHEDULE_WIDE = 2;
// 单个 work item 聚合的物理 Q task 上限；wide 常驻布局按 2 规划 L1/UB，
// 并与 host 侧常量 WIDE_COMPACT_Q_TASKS 必须相等。
// 同时决定 edgeCode 中 qLocal 的位宽，以及 WideExecutor 内
// qInfos / qLimits / maskRowOffsets / maskWindows 的数组长度。
constexpr uint32_t WIDE_MAX_Q_TASKS = 2; // D=128 常驻 wide 的物理 Q task 容量。
// 单个 packet 的稀疏连接（边）容量：限制 edgeCode / qOrder / activeQ /
// validEdges 的数组长度，也等于一个 stage 在 workspace 中预留的 tile 槽数。
constexpr uint32_t WIDE_MAX_EDGES = 8; // 单 packet 的稀疏连接容量。
// 单个 packet 内唯一 KV 基础块的上限：等于 dQ 阶段常驻 K 缓存的 L1 槽位数，
// 也是 WidePacketInfo::kv 的数组长度。
constexpr uint32_t WIDE_MAX_KV_COLUMNS = 8; // dQ 使用的 K 驻留槽位数。
// 常驻缓存中 Q 槽相对 dO 槽的偏移：dO 占槽 [0,4)，Q 占槽 [4,8)。
// 取 dO 用 queryLocalIndex，取 Q 用 queryLocalIndex + 本偏移。
constexpr uint32_t WIDE_QUERY_CACHE_OFFSET = 4; // Q 槽相对 dO 槽的偏移。
// 单条边（一个 128x128 分片）在 workspace 中占用的元素数，用于槽位寻址：
// tile.workspaceOffset = packet.workspaceBase + edgeIndex * WIDE_TILE_ELEMENTS。
constexpr uint64_t WIDE_TILE_ELEMENTS = 128 * 128;
// —— 跨核令牌编号 ——
// AIC 写回 S/dP 后 Set CUBE2VEC_*，AIV 消费前 Wait；
// AIV 写回 P/dS 后 Set VEC2CUBE_*，AIC 覆盖该 stage 前 Wait。
// 双 stage 各一对，构成 packet 级软件流水的反压；编号不得与其它核间事件冲突。
constexpr uint32_t WIDE_CUBE2VEC_STAGE0 = 5;
// stage 1 的 Cube -> Vector 完成令牌。
constexpr uint32_t WIDE_CUBE2VEC_STAGE1 = 6;
// stage 0 的 Vector -> Cube 完成令牌：P/dS 已写回 workspace，该 stage 槽可被覆盖。
constexpr uint32_t WIDE_VEC2CUBE_STAGE0 = 7;
// stage 1 的 Vector -> Cube 完成令牌。
constexpr uint32_t WIDE_VEC2CUBE_STAGE1 = 8;
// AIC 全部 work item 结束后发出的收尾令牌：通知 POST 阶段可以开始，
// 避免 POST 提前读取尚未写完的 dQ/dK/dV。
constexpr uint32_t WIDE_CUBE2POST = 9;
// 128 KiB：L1 分区的基本步长。
// blockMmad2 取 l1BufAddrStart=0（P/dS 与 K），
// blockMmad3 取 WIDE_L1_OFFSET*7/2 = 448 KiB，即高 L1 的 Q/dO 常驻区起点。
constexpr uint64_t WIDE_L1_OFFSET = 131072;
// blockMmad2 的 ping-pong 事件编号基址，避开 Cube1 占用的 0/1。
constexpr uint32_t WIDE_PINGPONG_OFFSET_2 = 2;
// blockMmad3 的 ping-pong 事件编号基址，避开 Cube1 与 blockMmad2。
constexpr uint32_t WIDE_PINGPONG_OFFSET_4 = 4;
// Q task 位图：第 i 位为 1 表示 qLocal=i "已处理/已就绪"。
// 用途一：mask 窗口缓存的 loadedMaskWindows（按 Q 行惰性加载）；
// 用途二：dQ 的 qSeenMask —— 同一 Q 跨多个 packet 时，首写用普通写、后续必须原子累加。
using WideQMask = uint64_t;
// —— 容量自检 ——
// 1) Q 位图宽度必须覆盖全部 Q task；
// 2) qLocal 用 1 字节编码（edgeCode 低 8 位），故 Q 下标必须 < 256；
// 3) 边下标用 1 字节编码（WideQDesc 的 edgeBegin/edgeCount），故边数必须 < 256；
// 4) 唯一 KV 列数不得超过边容量，也不得超过上层 K 缓存槽位数（8）。
static_assert(WIDE_MAX_Q_TASKS <= sizeof(WideQMask) * 8, "wide Q bitmap capacity exceeded");
static_assert(WIDE_MAX_Q_TASKS < 256, "wide Q index exceeds edge encoding");
static_assert(WIDE_MAX_EDGES < 256, "wide edge index exceeds descriptor encoding");
static_assert(WIDE_MAX_KV_COLUMNS <= WIDE_MAX_EDGES && WIDE_MAX_KV_COLUMNS <= 8,
              "wide packet exceeds K cache capacity");
} // namespace BSA::WideDetail
#endif
