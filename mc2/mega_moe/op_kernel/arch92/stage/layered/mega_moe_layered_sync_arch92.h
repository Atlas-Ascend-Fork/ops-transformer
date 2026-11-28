/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYERED_SYNC_ARCH92_H
#define MEGA_MOE_LAYERED_SYNC_ARCH92_H

namespace MegaMoeImpl {

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GetChannelOwnerBlock(uint32_t rank) const
{
    // 通道归属在逻辑 AIC 域内计算，不使用物理 AIV 编号。
    // Dispatch 两级通信、Combine 写入和最终 Drain 共用此映射，且只有归属 block 的 AIV 可访问通道。
    return rank % blockNum_;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::IsChannelOwner(uint32_t rank) const
{
    if constexpr (g_coreType == AIC) {
        return false;
    }
    return blockIdx_ == GetChannelOwnerBlock(rank);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LockOwnedRemoteChannels()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // Wave 内 Dispatch 一级写、Dispatch 二级读和 Combine 写始终使用同一套固定归属核。
    // 在 Wave 开始前一次性取得全部归属通道，整个 Wave 内不重复 Lock/Unlock。
    for (uint32_t rank = 0; rank < worldSize_; ++rank) {
        if (rank == rankId_ || !IsChannelOwner(rank)) {
            continue;
        }
        comm_.Lock(comm_.GetChannel(rank));
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DrainAndUnlockOwnedRemoteChannels()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // Drain 全部归属远端 rank，不能只处理最后一个专家的活跃 rank。
    // 否则当最后一个专家无 token 时，早期专家的 WriteNbi 可能遗留到流水结束之后。
    // Drain 后 Unlock 发布通道 counter 并释放所有权，允许后续同步阶段换核使用同一通道。
    for (uint32_t rank = 0; rank < worldSize_; ++rank) {
        if (rank == rankId_ || !IsChannelOwner(rank)) {
            continue;
        }
        Channel channel = comm_.GetChannel(rank);
        BatchHandle batchHandle = comm_.MakeBatchHandle(channel, GetRankWinAddrWithOffset(rank, 0));
        // 同一通道可能同时残留 Dispatch 一级和 Combine 的已提交批次，统一按批量句柄等待完成。
        comm_.Drain(batchHandle);
        comm_.Unlock(channel);
    }
}

/* 全部 AIV 每次同步均推进本核轮次，保持参与范围改变时的通知值一致。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline int32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AdvanceRankSyncRound()
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    auto* syncCount =
        reinterpret_cast<__gm__ int32_t*>(peermem.syncCount.get_ptr(commBuffer) + aivCoreIdx_ * PEERMEM_SYNC_SLOT_SIZE);
    const int32_t syncRound = GmRead(syncCount) + 1;
    GmWrite(syncCount, syncRound);
    GmDataBarrier(); // 通信引擎读取本核通知槽前，确保旁路 GM 写已完成。
    return syncRound;
}

/* 常规入口：推进本核轮次，再执行本核负责的跨卡通信。 */
template <TemplateMegaMoeLayeredTypeClass>
template <bool ReuseChannelLock>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CrossRankSync(const AivJobContext& syncJob)
{
    const auto syncRound = AdvanceRankSyncRound();
    CrossRankSync<ReuseChannelLock>(syncJob, syncRound);
}

/* 按 syncJob 分担跨卡通知和等待；本卡会合及参与核选择由调用处编排。 */
template <TemplateMegaMoeLayeredTypeClass>
template <bool ReuseChannelLock>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CrossRankSync(const AivJobContext& syncJob,
                                                                                     int32_t syncRound)
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    auto* syncRank = reinterpret_cast<__gm__ int32_t*>(peermem.rankSync.get_ptr(commBuffer));
    auto* syncCount =
        reinterpret_cast<__gm__ int32_t*>(peermem.syncCount.get_ptr(commBuffer) + aivCoreIdx_ * PEERMEM_SYNC_SLOT_SIZE);
    // 先向本 AIV 负责的所有 peer 发出通知，再进入等待。256P 场景下可同时维持多个
    // channel 在途，避免原实现逐 peer“写-等-Drain”造成的串行握手。
    for (uint32_t rankIndex = syncJob.jobIndex; rankIndex < worldSize_; rankIndex += syncJob.totalJobs) {
        if (rankIndex == rankId_) {
            continue;
        }
        Channel channel = comm_.GetChannel(rankIndex);
        if constexpr (!ReuseChannelLock) {
            // 同步任务的通道归属可能与 Wave 不同，先由当前通信核取得锁。
            comm_.Lock(channel);
        }
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(g_winRankAddr_[rankIndex]) + rankId_ * 16;
        comm_.WriteNbi(channel, (GM_ADDR)syncRemoteAddr, (GM_ADDR)syncCount, static_cast<int64_t>(sizeof(int32_t)));
    }
    for (uint32_t rankIndex = syncJob.jobIndex; rankIndex < worldSize_; rankIndex += syncJob.totalJobs) {
        if (rankIndex == rankId_) {
            continue;
        }
        Channel channel = comm_.GetChannel(rankIndex);
        auto syncCheck = syncRank + rankIndex * 16;
        WaitUntilGmFlagEquals(syncCheck, syncRound);
        // 等待完成后释放通道锁：复用调用方已持有的锁时由本函数统一释放，否则释放本函数取得的锁。
        comm_.Drain(channel);
        comm_.Unlock(channel);
    }
}

/* mask 发送核交还通道锁，供首次跨卡同步重新分配 owner。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DrainAndUnlockMaskChannels(
    const AivJobContext& job)
{
    for (uint32_t rankIndex = job.jobIndex; rankIndex < worldSize_; rankIndex += job.totalJobs) {
        if (rankIndex != rankId_) {
            Channel channel = comm_.GetChannel(rankIndex);
            comm_.Drain(channel);  // 等待该通道的 mask 写请求完成。
            comm_.Unlock(channel); // 交还通道，供同步核重新持锁。
        }
    }
}

} // namespace MegaMoeImpl
#endif // MEGA_MOE_LAYERED_SYNC_ARCH92_H
