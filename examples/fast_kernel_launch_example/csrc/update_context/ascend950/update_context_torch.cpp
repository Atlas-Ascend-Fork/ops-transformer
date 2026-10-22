/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file update_context_torch.cpp
 * \brief A5 (ascend950) updateContext：产出内核直接读取的 HcclCombinOpParam 上下文（A3 式直接传值）。
 *
 * A3 example 的做法：host 填充上下文结构 → 作为 mc2Context GM_ADDR 直接传入内核，内核类直接读取，
 * 不走 GetHcclContext。A5 内核读的是 HcclCombinOpParam（moe_distribute_comm_ctx.h），
 * 故此处按该布局产出上下文，字段（winSize/windowsIn/rankId/rankDim）来自 HCCL 通信资源。
 */

#include "update_context_torch.h"

#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <utility>

#include "hccl/hccl.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_channel.h"
#include "hccl/hccl_rank_graph.h"

namespace ascend_ops {
namespace Update_Context {

extern "C" HcclResult HcomGetCommHandleByGroup(const char* groupName, HcclComm* comm);

namespace {

constexpr uint8_t COMM_ENGINE_AIV = 4;
constexpr uint32_t HCCL_MTE_MIN_RANK_NUM = 2;
constexpr uint32_t HCCL_CONTEXT_TAG_MAX_LEN = 255;
constexpr uint32_t HCCL_MTE_MAX_RANK_NUM = 64;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint32_t EP_RANK_OFFSET_STEP = 1024;

// 对齐内核 A5 直接读取的 HcclCombinOpParam（moe_distribute_comm_ctx.h）
struct A5CombinOpParam {
    uint64_t workSpace = 0;                          // 0
    uint64_t workSpaceSize = 0;                      // 8
    uint32_t rankId = 0;                             // 16
    uint32_t rankDim = 0;                            // 20
    uint64_t winSize = 0;                            // 24
    uint64_t windowsIn[HCCL_MTE_MAX_RANK_NUM] = {};  // 32 MTE 数据区
    uint64_t windowsOut[HCCL_MTE_MAX_RANK_NUM] = {}; // 544 MTE 状态区
    uint64_t xnAddr = 0;                             // ccu
    uint64_t ckeAddr = 0;                            // ccu
    uint64_t msAddr = 0;                             // ccu
    uint64_t msSize = 0;                             // ccu
};

class A5ChannelContextBuilder {
public:
    void Build(const std::string& group, int64_t worldSize, int64_t& cclBufferSize, at::Tensor& contextTensor)
    {
        HcclComm hcclHandle = nullptr;
        HcclResult ret = static_cast<HcclResult>(HcomGetCommHandleByGroup(group.c_str(), &hcclHandle));
        TORCH_CHECK(ret == HCCL_SUCCESS, "Get HCCL handle failed, group: ", group, ", ret: ", ret);

        CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        GetCommProtocol(hcclHandle, protocol);

        uint32_t rankId = GetRankId(hcclHandle);
        epRankSize_ = GetRankSize(hcclHandle);

        // 建立 HCCL channel（对齐 mc2 GetHcclCommChannel）：为每个 peer 获取 channel 句柄
        std::vector<ChannelHandle> channels(epRankSize_ - 1);
        InitHcclChannel(hcclHandle, epRankSize_, rankId, protocol, channels);
        HcclResult acqRet = HcclChannelAcquire(hcclHandle, CommEngine::COMM_ENGINE_AIV, channelDescs_.data(),
                                               epRankSize_ - 1, channels.data());
        TORCH_CHECK(acqRet == HCCL_SUCCESS, "Acquire HCCL channel failed, ret: ", acqRet);

        A5CombinOpParam ctx;
        ctx.rankId = rankId;
        ctx.rankDim = epRankSize_;
        ctx.winSize = GetHcclBufferSize(hcclHandle);
        ctx.workSpaceSize = ctx.winSize;
        for (uint32_t i = 0; i < epRankSize_; ++i) {
            void* tempBuffer = nullptr;
            uint64_t bufSize = 0;
            if (i == rankId) {
                HcclResult r = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &bufSize);
                TORCH_CHECK(r == HCCL_SUCCESS && tempBuffer != nullptr, "Get self HCCL buffer failed, ret: ", r);
                hcclBuffSize_ = bufSize;
            } else {
                uint32_t idx = (i < rankId) ? i : (i - 1);
                HcclResult r = HcclChannelGetHcclBuffer(hcclHandle, channels[idx], &tempBuffer, &bufSize);
                TORCH_CHECK(r == HCCL_SUCCESS && tempBuffer != nullptr, "Get peer HCCL buffer failed, ret: ", r);
            }
            ctx.windowsIn[i] = reinterpret_cast<uint64_t>(tempBuffer);
            ctx.windowsOut[i] = reinterpret_cast<uint64_t>(tempBuffer);
        }
        cclBufferSize = static_cast<int64_t>(ctx.winSize);

        CopyContextToTensor(ctx, contextTensor);
    }

private:
    void InitHcclChannel(const HcclComm& commHandle, uint32_t rankDim, uint32_t srcRankId, const CommProtocol& protocol,
                         std::vector<ChannelHandle>& channels)
    {
        channelDescs_.resize(rankDim - 1);
        HcclResult hcclRet = HcclChannelDescInit(channelDescs_.data(), static_cast<uint32_t>(channelDescs_.size()));
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HCCL channel init failed, ret: ", hcclRet);

        uint32_t netLayerNum = 0;
        uint32_t* netLayerList = nullptr;
        HcclResult ret = HcclRankGraphGetLayers(commHandle, &netLayerList, &netLayerNum);
        TORCH_CHECK(ret == HCCL_SUCCESS && netLayerList != nullptr && netLayerNum > 0, "Get HCCL layers failed");

        for (uint32_t dstRank = 0; dstRank < rankDim; ++dstRank) {
            if (dstRank == srcRankId) {
                continue;
            }
            uint32_t channelId = (dstRank > srcRankId) ? (dstRank - 1) : dstRank;
            uint32_t linkLayer = netLayerList[HCCL_COMM_LAYERS_UB_MEM];
            CommLink* linksList = nullptr;
            uint32_t netLinkNum = 0;
            HcclResult lr = HcclRankGraphGetLinks(commHandle, linkLayer, srcRankId, dstRank, &linksList, &netLinkNum);
            TORCH_CHECK(lr == HCCL_SUCCESS && linksList != nullptr && netLinkNum > 0,
                        "Get HCCL link failed, src: ", srcRankId, ", dst: ", dstRank);
            uint32_t index = 0;
            for (; index < netLinkNum; ++index) {
                if (linksList[index].linkAttr.linkProtocol == protocol) {
                    break;
                }
            }
            TORCH_CHECK(index < netLinkNum, "No link with protocol ", static_cast<int>(protocol));
            channelDescs_[channelId].channelProtocol = protocol;
            channelDescs_[channelId].remoteRank = dstRank;
            channelDescs_[channelId].localEndpoint = linksList[index].srcEndpointDesc;
            channelDescs_[channelId].remoteEndpoint = linksList[index].dstEndpointDesc;
        }
        (void)channels;
    }
    void GetCommProtocol(const HcclComm& commHandle, CommProtocol& protocol)
    {
        // 单机/本节点场景默认 UB_MEM 协议（多机跨超节点拓扑解析暂不适用）
        (void)commHandle;
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    }

    uint32_t GetRankId(const HcclComm& commHandle) const
    {
        uint32_t rankId = 0;
        HcclResult hcclRet = HcclGetRankId(commHandle, &rankId);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank ID failed, ret: ", hcclRet);
        return rankId;
    }

    uint32_t GetRankSize(const HcclComm& commHandle) const
    {
        uint32_t rankSize = 0;
        HcclResult hcclRet = HcclGetRankSize(commHandle, &rankSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank size failed, ret: ", hcclRet);
        TORCH_CHECK(rankSize >= HCCL_MTE_MIN_RANK_NUM && rankSize <= HCCL_MTE_MAX_RANK_NUM,
                    "Invalid HCCL rank size (expect [", HCCL_MTE_MIN_RANK_NUM, ", ", HCCL_MTE_MAX_RANK_NUM,
                    "]): ", rankSize);
        return rankSize;
    }

    uint64_t GetHcclBufferSize(const HcclComm& commHandle) const
    {
        void* tempBuffer = nullptr;
        uint64_t bufSize = 0;
        HcclResult hcclRet = HcclGetHcclBuffer(commHandle, &tempBuffer, &bufSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS && tempBuffer != nullptr, "Get HCCL Buffer Size failed, ret: ", hcclRet);
        return bufSize;
    }

    static void CopyContextToTensor(const A5CombinOpParam& context, at::Tensor& tensor)
    {
        at::Tensor hostContext =
            at::from_blob(const_cast<A5CombinOpParam*>(&context),
                          {static_cast<int64_t>(sizeof(A5CombinOpParam) / sizeof(int32_t))}, at::kInt);
        tensor.copy_(hostContext);
    }

    uint32_t epRankSize_ = 0;
    uint64_t hcclBuffSize_ = 0;
    std::vector<HcclChannelDesc> channelDescs_;
};

} // namespace

// 进程级静态缓存：updateContext 在测试脚本中每个 rank 进程仅对单一 EP group 构造一次，
// 单线程调用、无多 group 并发场景，故不引入锁；同 group 重复调用直接复用已构建的上下文，
// 避免重复 Acquire HCCL channel 与重建 HcclCombinOpParam。若未来支持多 group 并发，需改为
// 按 group 键控的缓存并加锁。
static bool isInit = false;
static std::string last_group_ep_str = "";
static at::Tensor output;
int64_t bufferSize = 0;

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("updateContext(Tensor x, str group_ep, int ep_world_size) -> (Tensor, int)");
}

std::tuple<at::Tensor, int64_t> update_context(const at::Tensor& x, c10::string_view group_ep, int64_t ep_world_size)
{
    bool isSameGroup = false;
    std::string new_group_ep_str = std::string(group_ep);
    isSameGroup = (new_group_ep_str == last_group_ep_str);
    if (isInit && isSameGroup) {
        return std::make_tuple(output, bufferSize);
    }

    at::Tensor context = at::empty({static_cast<int64_t>(sizeof(A5CombinOpParam) / sizeof(int32_t))},
                                   at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1));
    int64_t cclBufferSize = 0;
    A5ChannelContextBuilder builder;
    builder.Build(new_group_ep_str, ep_world_size, cclBufferSize, context);

    last_group_ep_str = new_group_ep_str;
    bufferSize = cclBufferSize;
    output = context;
    isInit = true;
    return std::make_tuple(output, bufferSize);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("updateContext", update_context);
}
} // namespace Update_Context
} // namespace ascend_ops
