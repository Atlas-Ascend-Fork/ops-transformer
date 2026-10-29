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
 * \file mc2_context.cpp
 * \brief MC2 Context management implementation
 */

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <dlfcn.h>
#include "acl/acl_rt.h"
#include "mc2_context.h"
#include "opdev/platform.h"
#include "op_host/util/op_const_def.h"
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

namespace {
constexpr uint64_t KOPY_DEFAULT_CTX_OFFSET = 0;
constexpr uint64_t MAX_CONTEXT_TAG_SIZE = 255;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint32_t EP_RANK_OFFSET_STEP = 1024;
constexpr uint64_t DEFAULT_HCCL_BUFFSIZE_MB = 200;
constexpr uint64_t REGISTERED_WINDOW_BYTES_PER_MB = 2UL * 1024UL * 1024UL;
constexpr const char* REGISTERED_WINDOWS_TAG = "mc2_registered_windows_v3";
constexpr const char* UB_WINDOW_TAG = "mc2_ub_window_v3";
constexpr const char* ROCE_WINDOW_TAG = "mc2_roce_window_v3";

constexpr uint32_t GROUP_RANK_NUM = 4U;

bool MatchesInstanceEndpoint(const EndpointDesc& linkEndpoint, const EndpointDesc& instanceEndpoint)
{
    if (linkEndpoint.protocol != instanceEndpoint.protocol ||
        linkEndpoint.loc.locType != instanceEndpoint.loc.locType ||
        linkEndpoint.commAddr.type != instanceEndpoint.commAddr.type) {
        return false;
    }
    // 实例查询不填设备位置ID，仅比较有效地址字段
    const auto& lhs = linkEndpoint.commAddr;
    const auto& rhs = instanceEndpoint.commAddr;
    switch (lhs.type) {
        case COMM_ADDR_TYPE_EID:
            return std::memcmp(lhs.eid, rhs.eid, sizeof(lhs.eid)) == 0;
        case COMM_ADDR_TYPE_IP_V4:
            return std::memcmp(&lhs.addr, &rhs.addr, sizeof(lhs.addr)) == 0;
        case COMM_ADDR_TYPE_IP_V6:
            return std::memcmp(&lhs.addr6, &rhs.addr6, sizeof(lhs.addr6)) == 0;
        case COMM_ADDR_TYPE_ID:
            return lhs.id == rhs.id;
        default:
            return false;
    }
}

} // namespace

namespace Mc2Aclnn {
struct Mc2Context::TopologyInstance {
    uint32_t layer;
    CommTopo type;
    std::vector<uint32_t> ranks;
    std::vector<EndpointDesc> endpoints;
};

const std::string Mc2Context::GetLibPath()
{
    const char* ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Ascend home path doesn't exist.");
        return "";
    }
#if defined(__x86_64__)
    std::string hcclPathPostfix = "/x86_64-linux/lib64/libhccl_fwk.so";
#elif defined(__aarch64__)
    std::string hcclPathPostfix = "/aarch64-linux/lib64/libhccl_fwk.so";
#endif
    std::string fullPath = std::string(ascendPath) + hcclPathPostfix;
    OP_LOGI("Loading lib in path %s.", fullPath.c_str());
    return fullPath;
}

template <typename T>
T Mc2Context::GetHcclLibFunc(void* handle, const std::string& funcName)
{
    T func = reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
    if (func == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Load func=%s error=%s in lib hccl failed.", funcName.c_str(), dlerror());
    }
    return func;
}

Mc2Context::Mc2Context()
{
    OP_LOGI("Init Mc2Context Success!");
}

Mc2Context::~Mc2Context()
{
    if (hcclLibHandle_ != nullptr) {
        dlclose(hcclLibHandle_);
        hcclLibHandle_ = nullptr;
    }
}

aclnnStatus Mc2Context::OpenHcclLibrary()
{
    OP_LOGD("Start to load HCCL library and symbols");
    const std::string libPath = GetLibPath();
    if (libPath.empty()) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Failed to get HCCL library path");
        return ACLNN_ERR_INNER;
    }

    hcclLibHandle_ = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (hcclLibHandle_ == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "dlopen HCCL library failed: %s, error: %s", libPath.c_str(), dlerror());
        return ACLNN_ERR_INNER;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::LoadProtocolSymbols(CommProtocol protocol)
{
    if (protocol == CommProtocol::COMM_PROTOCOL_ROCE) {
        HcclCommMemReg = GetHcclLibFunc<decltype(HcclCommMemReg)>(hcclLibHandle_, "HcclCommMemReg");
        HcclChannelGetRemoteMems =
            GetHcclLibFunc<decltype(HcclChannelGetRemoteMems)>(hcclLibHandle_, "HcclChannelGetRemoteMems");
        CHECK_RET(HcclCommMemReg != nullptr && HcclChannelGetRemoteMems != nullptr, ACLNN_ERR_INNER);
    } else {
        HcclRankGraphGetRankSizeByLayer = GetHcclLibFunc<decltype(HcclRankGraphGetRankSizeByLayer)>(
            hcclLibHandle_, "HcclRankGraphGetRankSizeByLayer");
        HcclChannelGetHcclBuffer =
            GetHcclLibFunc<decltype(HcclChannelGetHcclBuffer)>(hcclLibHandle_, "HcclChannelGetHcclBuffer");
        CHECK_RET(HcclRankGraphGetRankSizeByLayer != nullptr && HcclChannelGetHcclBuffer != nullptr, ACLNN_ERR_INNER);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::LoadHcclSymbols(CommProtocol protocol)
{
    aclnnStatus ret = OpenHcclLibrary();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    struct SymbolInfo {
        void** ptr;
        const char* name;
    };

    SymbolInfo symbols[] = {
        {reinterpret_cast<void**>(&HcomGetCommHandleByGroup), "HcomGetCommHandleByGroup"},
        {reinterpret_cast<void**>(&HcclRankGraphGetLinks), "HcclRankGraphGetLinks"},
        {reinterpret_cast<void**>(&HcclRankGraphGetLayers), "HcclRankGraphGetLayers"},
        {reinterpret_cast<void**>(&HcclChannelAcquire), "HcclChannelAcquire"},
        {reinterpret_cast<void**>(&HcclGetHcclBuffer), "HcclGetHcclBuffer"},
        {reinterpret_cast<void**>(&HcclEngineCtxCreate), "HcclEngineCtxCreate"},
        {reinterpret_cast<void**>(&HcclEngineCtxGet), "HcclEngineCtxGet"},
        {reinterpret_cast<void**>(&HcclEngineCtxCopy), "HcclEngineCtxCopy"},
        {reinterpret_cast<void**>(&HcclGetRankId), "HcclGetRankId"},
        {reinterpret_cast<void**>(&HcclGetRankSize), "HcclGetRankSize"},
        {reinterpret_cast<void**>(&HcclRankGraphGetRanksByLayer), "HcclRankGraphGetRanksByLayer"},
    };

    for (auto& sym : symbols) {
        *(sym.ptr) = GetHcclLibFunc<void*>(hcclLibHandle_, sym.name);
        if (*(sym.ptr) == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Failed to load %s symbol", sym.name);
            dlclose(hcclLibHandle_);
            hcclLibHandle_ = nullptr;
            return ACLNN_ERR_INNER;
        }
    }
    OP_LOGD("All HCCL symbols loaded successfully");
    return LoadProtocolSymbols(protocol);
}

aclnnStatus Mc2Context::LoadTopologySymbols()
{
    struct SymbolInfo {
        void** ptr;
        const char* name;
    };
    SymbolInfo topologySymbols[] = {
        {reinterpret_cast<void**>(&HcclRankGraphGetTopoTypeByLayer), "HcclRankGraphGetTopoTypeByLayer"},
        {reinterpret_cast<void**>(&HcclRankGraphGetTopoInstsByLayer), "HcclRankGraphGetTopoInstsByLayer"},
        {reinterpret_cast<void**>(&HcclRankGraphGetTopoType), "HcclRankGraphGetTopoType"},
        {reinterpret_cast<void**>(&HcclRankGraphGetRanksByTopoInst), "HcclRankGraphGetRanksByTopoInst"},
        {reinterpret_cast<void**>(&HcclRankGraphGetEndpointNum), "HcclRankGraphGetEndpointNum"},
        {reinterpret_cast<void**>(&HcclRankGraphGetEndpointDesc), "HcclRankGraphGetEndpointDesc"},
    };
    for (auto& sym : topologySymbols) {
        *(sym.ptr) = GetHcclLibFunc<void*>(hcclLibHandle_, sym.name);
        if (*(sym.ptr) == nullptr) {
            return ACLNN_ERR_INNER;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetCommHandle(const char* groupEp, HcclComm& hcclHandle)
{
    OP_LOGI("Start to get HCCL communication handle, groupEp: %s", groupEp);
    auto ret = HcomGetCommHandleByGroup(groupEp, &hcclHandle);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL handle failed, groupEp: %s", groupEp);
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Get HCCL communication handle success hcclHandle is: %p", hcclHandle);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclCommLink(const HcclComm& hcclHandle, uint32_t netLayerId, uint32_t srcRankId,
                                        uint32_t dstRankId, const CommProtocol& protocol, CommLink*& links)
{
    OP_LOGD("Start to get HCCL communication link");
    CommLink* linksList = nullptr;
    uint32_t netLinkNum = 0;
    auto hcclRet = HcclRankGraphGetLinks(hcclHandle, netLayerId, srcRankId, dstRankId, &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL Communication link failed");
        return ACLNN_ERR_INNER;
    }
    if (linksList == nullptr || netLinkNum == 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                                "The Net Link Is nullptr. srcRankId is %u, dstRankId is %u, layerId is %u", srcRankId,
                                dstRankId, netLayerId);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Rank Links Success Links Num is: %u", netLinkNum);
    uint32_t index = 0;
    for (; index < netLinkNum; ++index) {
        if (linksList[index].linkAttr.linkProtocol == protocol) {
            links = &linksList[index];
            break;
        }
    }
    if (index == netLinkNum) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                                "No matching communication protocol found in HCCL links, "
                                "protocol: %d, srcRankId: %u, dstRankId: %u",
                                protocol, srcRankId, dstRankId);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL communication link success, protocol: %d, srcRankId: %u, dstRankId: %u", protocol, srcRankId,
            dstRankId);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetNetLayers(const HcclComm& hcclHandle, uint32_t*& netLayerList, uint32_t& netLayerNum)
{
    auto hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS || netLayerList == nullptr || netLayerNum == 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL layers failed or returned an empty layer list");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Get HCCL layers success, netLayerNum is: %u", netLayerNum);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::InitRegisteredChannel(const HcclComm& hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                              const CommProtocol& protocol, uint32_t channelsPerPeer,
                                              std::vector<HcclChannelDesc>& channelDesc)
{
    HcclResult hcclRet = HcclChannelDescInit(channelDesc.data(), static_cast<uint32_t>(channelDesc.size()));
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "HCCL channel init failed");
        return ACLNN_ERR_INNER;
    }
    uint32_t channelId = 0;
    for (uint32_t dstRank = 0; dstRank < rankDim; ++dstRank) {
        std::unordered_map<uint32_t, RankLinkInfo>::const_iterator it = rankLinkMap_.find(dstRank);
        if (dstRank == srcRankId || it == rankLinkMap_.end() || it->second.protocol != protocol) {
            continue;
        }
        const RankLinkInfo& linkInfo = it->second;
        CommLink* links = nullptr;
        aclnnStatus ret = GetHcclCommLink(hcclHandle, linkInfo.layer, srcRankId, dstRank, linkInfo.protocol, links);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        if (channelId + channelsPerPeer > channelDesc.size()) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Too many channel descriptors for protocol %d", protocol);
            return ACLNN_ERR_INNER;
        }
        HcclChannelDesc& desc = channelDesc[channelId];
        desc.channelProtocol = linkInfo.protocol;
        desc.remoteRank = dstRank;
        desc.localEndpoint = links->srcEndpointDesc;
        desc.remoteEndpoint = links->dstEndpointDesc;
        desc.memHandles = protocol == CommProtocol::COMM_PROTOCOL_UB_MEM ? &registeredWindows_->ub.handle :
                                                                           &registeredWindows_->roce.handle;
        desc.memHandleNum = 1;
        // 同一对端连续占用 channelsPerPeer 个通道。
        for (uint32_t i = 1; i < channelsPerPeer; ++i) {
            channelDesc[channelId + i] = desc;
        }
        channelId += channelsPerPeer;
    }
    CHECK_RET(channelId == channelDesc.size(), ACLNN_ERR_INNER);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetRegisteredCommChannels(const HcclComm& hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                                  const CommEngine& engine, const CommProtocol& protocol,
                                                  std::vector<ChannelHandle>& channels)
{
    OP_LOGD("Start to get HCCL communication channel, protocol: %d", protocol);
    // RoCE 每个对端要两条通道：token 通道与专家计数通道分离，避免两类核并发操作同一通道的SQ/CQ。
    const uint32_t channelsPerPeer = protocol == CommProtocol::COMM_PROTOCOL_ROCE ? ROCE_CHANNEL_PER_SERVER : 1U;
    const uint32_t channelNum =
        (protocol == CommProtocol::COMM_PROTOCOL_UB_MEM ? rankSizePerServer_ - 1 : rankDim / rankSizePerServer_ - 1) *
        channelsPerPeer;
    if (channelNum > HCCL_MAX_RANK_SIZE) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Channel count %u exceeds context handle capacity %u", channelNum,
                                HCCL_MAX_RANK_SIZE);
        return ACLNN_ERR_PARAM_INVALID;
    }
    channels.resize(channelNum);
    if (channelNum == 0) {
        return ACLNN_SUCCESS;
    }
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    aclnnStatus ret = InitRegisteredChannel(hcclHandle, rankDim, srcRankId, protocol, channelsPerPeer, channelDesc);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    HcclResult hcclRet = HcclChannelAcquire(hcclHandle, engine, channelDesc.data(), channelNum, channels.data());
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Acquire HCCL channel failed, protocol: %d", protocol);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::InitRegisteredBuffer(const HcclComm& hcclHandle, const char* tag, RegisteredBuffer& buffer)
{
    if (buffer.handle != nullptr) {
        CHECK_RET(buffer.addr != nullptr && buffer.size == hcclBuffSize_, ACLNN_ERR_INNER);
        return ACLNN_SUCCESS;
    }
    // Engine contexts keep the allocation alive until the communication domain is destroyed.
    if (buffer.addr == nullptr) {
        HcclResult ret = HcclEngineCtxGet(hcclHandle, tag, CommEngine::COMM_ENGINE_AIV, &buffer.addr, &buffer.size);
        if (ret == HCCL_E_NOT_FOUND) {
            buffer.size = hcclBuffSize_;
            ret = HcclEngineCtxCreate(hcclHandle, tag, CommEngine::COMM_ENGINE_AIV, buffer.size, &buffer.addr);
        }
        CHECK_RET(ret == HCCL_SUCCESS, ACLNN_ERR_INNER);
    }
    CHECK_RET(buffer.addr != nullptr && buffer.size == hcclBuffSize_, ACLNN_ERR_INNER);
    // Only unregistered buffers are cleared; cached windows contain live sequence state.
    CHECK_RET(aclrtMemset(buffer.addr, buffer.size, 0, buffer.size) == ACL_SUCCESS, ACLNN_ERR_INNER);
    CommMem mem{};
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = buffer.addr;
    mem.size = buffer.size;
    HcclMemHandle handle = nullptr;
    HcclResult ret = HcclCommMemReg(hcclHandle, tag, &mem, &handle);
    CHECK_RET(ret == HCCL_SUCCESS && handle != nullptr, ACLNN_ERR_INNER);
    buffer.handle = handle;
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetConfiguredWindowSize(uint64_t& windowSize) const
{
    windowSize = 0;
    uint64_t bufferSizeMb = DEFAULT_HCCL_BUFFSIZE_MB;
    const char* env = std::getenv("HCCL_BUFFSIZE");
    if (env != nullptr) {
        if (*env == '\0' || std::strspn(env, "0123456789") != std::strlen(env)) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "HCCL_BUFFSIZE must be a positive integer in MB");
            return ACLNN_ERR_PARAM_INVALID;
        }
        errno = 0;
        bufferSizeMb = std::strtoull(env, nullptr, 10);
        if (errno == ERANGE) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "HCCL_BUFFSIZE exceeds the supported integer range");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // GetMaxWindowSize in A5 tiling stores the configured MB value in uint16_t.
    const uint64_t maxBufferSizeMb = std::numeric_limits<uint16_t>::max();
    if (bufferSizeMb == 0 || bufferSizeMb > maxBufferSizeMb) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "HCCL_BUFFSIZE is out of range: %lu MB", bufferSizeMb);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Match the test script and A5 tiling: the configured MB value describes half of each window.
    windowSize = bufferSizeMb * REGISTERED_WINDOW_BYTES_PER_MB;
    OP_LOGI("HCCL_BUFFSIZE: %lu MB; each registered UB/RoCE window: %lu bytes", bufferSizeMb, windowSize);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckRegisteredWindowSize(const HcclComm& hcclHandle, uint64_t& hcclBuffSize)
{
    hcclBuffSize = 0;
    uint64_t expectedSize = 0;
    aclnnStatus ret = GetConfiguredWindowSize(expectedSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    void* defaultBuffer = nullptr;
    uint64_t reportedSize = 0;
    HcclResult hcclRet = HcclGetHcclBuffer(hcclHandle, &defaultBuffer, &reportedSize);
    if (hcclRet != HCCL_SUCCESS || defaultBuffer == nullptr || reportedSize == 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL buffer size failed, ret: %d, size: %lu", hcclRet, reportedSize);
        return ACLNN_ERR_INNER;
    }
    if (reportedSize != expectedSize) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "HCCL buffer size %lu differs from 2 * HCCL_BUFFSIZE: %lu bytes",
                                reportedSize, expectedSize);
        return ACLNN_ERR_INNER;
    }
    // The API supplies the size; its default buffer address is not one of our registered windows.
    hcclBuffSize = reportedSize;
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::InitRegisteredWindows(const HcclComm& hcclHandle)
{
    aclnnStatus ret = CheckRegisteredWindowSize(hcclHandle, hcclBuffSize_);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    void* cache = nullptr;
    uint64_t cacheSize = 0;
    HcclResult hcclRet =
        HcclEngineCtxGet(hcclHandle, REGISTERED_WINDOWS_TAG, CommEngine::COMM_ENGINE_CPU, &cache, &cacheSize);
    if (hcclRet == HCCL_E_NOT_FOUND) {
        cacheSize = sizeof(RegisteredWindows);
        hcclRet =
            HcclEngineCtxCreate(hcclHandle, REGISTERED_WINDOWS_TAG, CommEngine::COMM_ENGINE_CPU, cacheSize, &cache);
        if (hcclRet == HCCL_SUCCESS && cache != nullptr) {
            *static_cast<RegisteredWindows*>(cache) = {};
        }
    }
    CHECK_RET(hcclRet == HCCL_SUCCESS && cache != nullptr && cacheSize == sizeof(RegisteredWindows), ACLNN_ERR_INNER);
    registeredWindows_ = static_cast<RegisteredWindows*>(cache);
    ret = InitRegisteredBuffer(hcclHandle, UB_WINDOW_TAG, registeredWindows_->ub);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = InitRegisteredBuffer(hcclHandle, ROCE_WINDOW_TAG, registeredWindows_->roce);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(registeredWindows_->ub.addr != registeredWindows_->roce.addr, ACLNN_ERR_INNER);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetRegisteredRemoteBuffer(const HcclComm& hcclHandle, ChannelHandle channel, const char* tag,
                                                  void*& buffer, uint64_t& bufferSize)
{
    uint32_t memNum = 0;
    CommMem* remoteMems = nullptr;
    char** memTags = nullptr;
    HcclResult ret = HcclChannelGetRemoteMems(hcclHandle, channel, &memNum, &remoteMems, &memTags);
    CHECK_RET(ret == HCCL_SUCCESS && remoteMems != nullptr && memTags != nullptr, ACLNN_ERR_INNER);
    for (uint32_t i = 0; i < memNum; ++i) {
        if (memTags[i] != nullptr && std::strcmp(memTags[i], tag) == 0) {
            // Match the registered tag as in Mega context; remote type metadata is not a DEVICE guarantee.
            buffer = remoteMems[i].addr;
            bufferSize = remoteMems[i].size;
            OP_LOGI("Registered remote buffer tag: %s, reported type: %d, size: %lu", tag,
                    static_cast<int>(remoteMems[i].type), bufferSize);
            return ACLNN_SUCCESS;
        }
    }
    OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Registered remote buffer not found, tag: %s", tag);
    return ACLNN_ERR_INNER;
}

aclnnStatus Mc2Context::CheckHcclBuffer(uint32_t srcRankId, uint32_t peer, const void* buffer,
                                        uint64_t bufferSize) const
{
    const uint64_t addr = reinterpret_cast<uint64_t>(buffer);
    if (peer >= epRankSize_ || buffer == nullptr || bufferSize != hcclBuffSize_ ||
        addr > std::numeric_limits<uint64_t>::max() - bufferSize) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Invalid registered buffer, src: %u, dst: %u", srcRankId, peer);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::SetHcclBufferAddress(uint32_t srcRankId, uint32_t peer, void* buffer, uint64_t bufferSize,
                                             uint64_t& bufferAddr)
{
    aclnnStatus ret = CheckHcclBuffer(srcRankId, peer, buffer, bufferSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    const uint64_t offset = static_cast<uint64_t>(peer) * EP_RANK_OFFSET_STEP;
    if (bufferSize <= offset) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "HCCL buffer size %lu is too small for rank %u", bufferSize, peer);
        return ACLNN_ERR_INNER;
    }
    bufferAddr = reinterpret_cast<uint64_t>(buffer) + offset;
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::SetRoceBufferAddress(uint32_t srcRankId, uint32_t peer, void* buffer, uint64_t bufferSize,
                                             uint64_t& bufferAddr)
{
    aclnnStatus ret = CheckHcclBuffer(srcRankId, peer, buffer, bufferSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    bufferAddr = reinterpret_cast<uint64_t>(buffer);
    roceBufferSize_ = bufferSize;
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclUbMemResource(const HcclComm& hcclHandle, const CommEngine& engine,
                                             Mc2MoeContext* mc2ContextStruct)
{
    const uint32_t rankId = mc2ContextStruct->epRankId;
    std::vector<ChannelHandle> channels;
    aclnnStatus ret = GetRegisteredCommChannels(hcclHandle, epRankSize_, rankId, engine,
                                                CommProtocol::COMM_PROTOCOL_UB_MEM, channels);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    uint32_t channelId = 0;
    const uint32_t firstRank = rankId / rankSizePerServer_ * rankSizePerServer_;
    for (uint32_t peer = firstRank; peer < firstRank + rankSizePerServer_; ++peer) {
        void* buffer = nullptr;
        uint64_t bufferSize = 0;
        if (peer == rankId) {
            // Match Mega: local DEVICE memory retains its address; peers use the IPC-imported address.
            buffer = registeredWindows_->ub.addr;
            bufferSize = registeredWindows_->ub.size;
        } else {
            ret = GetRegisteredRemoteBuffer(hcclHandle, channels[channelId++], UB_WINDOW_TAG, buffer, bufferSize);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        }
        ret = SetHcclBufferAddress(rankId, peer, buffer, bufferSize, mc2ContextStruct->epHcclBuffer_[peer]);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclRoceResource(const HcclComm& hcclHandle, const CommEngine& engine,
                                            Mc2MoeContext* mc2ContextStruct)
{
    const uint32_t rankId = mc2ContextStruct->epRankId;
    const uint32_t localServer = rankId / rankSizePerServer_;
    const uint32_t localIndex = rankId % rankSizePerServer_;
    std::vector<ChannelHandle> channels;
    aclnnStatus ret =
        GetRegisteredCommChannels(hcclHandle, epRankSize_, rankId, engine, CommProtocol::COMM_PROTOCOL_ROCE, channels);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    uint32_t channelId = 0;
    for (uint32_t server = 0; server < epRankSize_ / rankSizePerServer_; ++server) {
        if (server == localServer) {
            continue;
        }
        const uint32_t peer = server * rankSizePerServer_ + localIndex;
        const uint32_t firstChannel = channelId * ROCE_CHANNEL_PER_SERVER;
        void* buffer = nullptr;
        uint64_t bufferSize = 0;
        ret = GetRegisteredRemoteBuffer(hcclHandle, channels[firstChannel], ROCE_WINDOW_TAG, buffer, bufferSize);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = SetRoceBufferAddress(rankId, peer, buffer, bufferSize, mc2ContextStruct->roceHcclBuffer_[server]);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        // Channels follow server order with the local server omitted; 每个对端连续占 ROCE_CHANNEL_PER_SERVER 个。
        for (uint32_t i = 0; i < ROCE_CHANNEL_PER_SERVER; ++i) {
            mc2ContextStruct->hcommHandle_[firstChannel + i] = channels[firstChannel + i];
        }
        ++channelId;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetRegisteredCommResource(const HcclComm& hcclHandle, const CommEngine& engine,
                                                  Mc2MoeContext* mc2ContextStruct)
{
    mc2ContextStruct->rankSizePerServer = rankSizePerServer_;
    aclnnStatus ret = InitRegisteredWindows(hcclHandle);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = GetHcclUbMemResource(hcclHandle, engine, mc2ContextStruct);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    const uint32_t rankId = mc2ContextStruct->epRankId;
    // 两个独立分配分别注册；RoCE 地址不叠加 UB 窗的 rank 偏移。
    ret = SetRoceBufferAddress(rankId, rankId, registeredWindows_->roce.addr, registeredWindows_->roce.size,
                               mc2ContextStruct->roceHcclBuffer_[rankId / rankSizePerServer_]);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (rankSizePerServer_ != epRankSize_) {
        ret = GetHcclRoceResource(hcclHandle, engine, mc2ContextStruct);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    // 各端窗口已校验为相同容量。
    mc2ContextStruct->roceBufferSize = roceBufferSize_;
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CreateContextCache(const HcclComm& hcclHandle, const std::string& tag, void*& ctx,
                                           void**& readyContext)
{
    void* state = nullptr;
    uint64_t stateSize = 0;
    HcclResult ret = HcclEngineCtxGet(hcclHandle, tag.c_str(), CommEngine::COMM_ENGINE_CPU, &state, &stateSize);
    if (ret == HCCL_E_NOT_FOUND) {
        stateSize = sizeof(void*);
        ret = HcclEngineCtxCreate(hcclHandle, tag.c_str(), CommEngine::COMM_ENGINE_CPU, stateSize, &state);
        CHECK_RET(ret == HCCL_SUCCESS && state != nullptr, ACLNN_ERR_INNER);
        *static_cast<void**>(state) = nullptr;
    }
    CHECK_RET(ret == HCCL_SUCCESS && state != nullptr && stateSize == sizeof(void*), ACLNN_ERR_INNER);
    readyContext = static_cast<void**>(state);
    ret = HcclEngineCtxCreate(hcclHandle, tag.c_str(), CommEngine::COMM_ENGINE_AIV, sizeof(Mc2MoeContext), &ctx);
    CHECK_RET(ret == HCCL_SUCCESS && ctx != nullptr, ACLNN_ERR_INNER);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetRankId(const HcclComm& hcclHandle, uint32_t& rankId) const
{
    HcclResult hcclRet = HcclGetRankId(hcclHandle, &rankId);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank ID failed, ret: %d", hcclRet);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetRankSize(const HcclComm& hcclHandle, uint32_t& rankSize) const
{
    HcclResult hcclRet = HcclGetRankSize(hcclHandle, &rankSize);
    if (hcclRet != HCCL_SUCCESS || rankSize == 0 || rankSize > HCCL_MAX_RANK_SIZE) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get valid HCCL rank size failed, ret: %d, rank size: %u", hcclRet,
                                rankSize);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetRankSizePerServer(const HcclComm& hcclHandle, uint32_t netLayers)
{
    auto hcclRet = HcclRankGraphGetRankSizeByLayer(hcclHandle, netLayers, &rankSizePerServer_);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL rank size per server failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Get HCCL rank size per server success, rankSizePerServer_ is: %u", rankSizePerServer_);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::InitHcclChannel(const HcclComm& hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                        const CommProtocol& protocol, std::vector<HcclChannelDesc>& channelDesc)
{
    uint32_t channelNum = channelDesc.size();
    auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "HCCL channel init failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("HCCL channel init success");

    uint32_t netLayerNum = 0;
    uint32_t layerId = 0;
    uint32_t* netLayerList = nullptr;
    auto ret = GetNetLayers(hcclHandle, netLayerList, netLayerNum);
    if (ret != ACLNN_SUCCESS || netLayerNum == 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL net layers failed netLayerNum is: %u", netLayerNum);
        return ret;
    }

    for (uint32_t i = 0; i < rankDim; ++i) {
        if (i == srcRankId) {
            continue;
        }
        uint32_t dstRank = i;
        uint32_t channelId = (i > srcRankId) ? (i - 1) : i;
        CommLink* links = nullptr;
        layerId = netLayerNum == 1 ?
                      netLayerList[HCCL_COMM_LAYERS_UB_MEM] :
                      layerMap[dstRank]; // 如果只有一层通信，直接使用该层；如果有多层通信，使用之前记录的通信层
        ret = GetHcclCommLink(hcclHandle, layerId, srcRankId, dstRank, protocol, links);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        channelDesc[channelId].channelProtocol = protocol;
        channelDesc[channelId].remoteRank = dstRank;
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclCommChannel(const HcclComm& hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                           const CommProtocol& protocol, const CommEngine& engine,
                                           std::vector<ChannelHandle>& channels, Mc2MoeContext* mc2ContextStruct)
{
    OP_LOGD("Start to get HCCL communication channel");
    uint32_t channelNum = rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    channels.resize(channelNum);

    auto ret = InitHcclChannel(hcclHandle, rankDim, srcRankId, protocol, channelDesc);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    // COMM_PROTOCOL_UBC_CTP协议 使用 mc2ContextStruct->hcommHandle_
    auto useChannel = mc2ContextStruct->hcommHandle_;
    if (CommProtocol::COMM_PROTOCOL_UBC_CTP != protocol) {
        useChannel = channels.data();
    }
    ret = HcclChannelAcquire(hcclHandle, engine, channelDesc.data(), channelNum, useChannel);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Acquire HCCL channel failed");
        return ret;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclCommResource(const HcclComm& hcclHandle, const CommEngine& engine,
                                            const CommProtocol& protocol, Mc2MoeContext* mc2ContextStruct)
{
    OP_LOGD("Start to get HCCL communication resource");
    uint32_t rankId = mc2ContextStruct->epRankId;
    std::vector<ChannelHandle> channels;

    aclnnStatus ret = GetHcclCommChannel(hcclHandle, epRankSize_, rankId, protocol, engine, channels, mc2ContextStruct);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    // 将rankSizePerServer_写入通信资源结构体中，后续DPU直驱需要使用
    mc2ContextStruct->rankSizePerServer = rankSizePerServer_;
    OP_LOGD("Get HCCL communication channel success, channel num is: %zu", channels.size());

    for (uint32_t i = 0; i < epRankSize_; ++i) {
        void* tempBuffer = nullptr;
        uint64_t bufSize = 0;
        HcclResult hcclRet;

        if (i == rankId) {
            hcclRet = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &hcclBuffSize_);
        } else {
            uint32_t idx = (i < rankId) ? i : (i - 1);
            auto channelHandle = channels[idx];
            if (CommProtocol::COMM_PROTOCOL_UBC_CTP == protocol) {
                channelHandle = mc2ContextStruct->hcommHandle_[idx];
            }
            hcclRet = HcclChannelGetHcclBuffer(hcclHandle, channelHandle, &tempBuffer, &bufSize);
        }

        if (hcclRet != HCCL_SUCCESS || tempBuffer == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL buffer failed, src: %u, dst: %u", rankId, i);
            return ACLNN_ERR_INNER;
        }

        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_3510) {
            mc2ContextStruct->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer) + i * EP_RANK_OFFSET_STEP;
        } else {
            mc2ContextStruct->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer);
        }
    }
    OP_LOGD("Get HCCL CommResource success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckGroupLinks(const HcclComm& hcclHandle, const TopologyInstance& instance, uint32_t rank,
                                        std::vector<bool>& mesh, std::vector<bool>& clos) const
{
    const bool isMesh = instance.type == COMM_TOPO_1DMESH;
    if (isMesh) {
        if (instance.ranks.size() != GROUP_RANK_NUM) {
            return ACLNN_SUCCESS;
        }
        const uint32_t groupStart = rank / GROUP_RANK_NUM * GROUP_RANK_NUM;
        for (uint32_t i = 0; i < GROUP_RANK_NUM; ++i) {
            if (std::find(instance.ranks.begin(), instance.ranks.end(), groupStart + i) == instance.ranks.end()) {
                return ACLNN_SUCCESS;
            }
        }
    }
    auto& matchedLinks = isMesh ? mesh : clos;
    for (uint32_t peer : instance.ranks) {
        const bool isSameGroup = peer / GROUP_RANK_NUM == rank / GROUP_RANK_NUM;
        if (peer == rank || (!isMesh && isSameGroup)) {
            continue;
        }
        CommLink* links = nullptr;
        uint32_t count = 0U;
        HcclResult hcclRet = HcclRankGraphGetLinks(hcclHandle, instance.layer, rank, peer, &links, &count);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGI("Get topology links failed, rank=%u, peer=%u, layer=%u, ret=%d", rank, peer, instance.layer,
                    hcclRet);
            return ACLNN_ERR_INNER;
        }
        for (uint32_t i = 0U; i < count; ++i) {
            if (links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UB_MEM) {
                continue;
            }
            const bool endpointMatched =
                instance.endpoints.empty() ||
                std::any_of(instance.endpoints.begin(), instance.endpoints.end(), [&](const EndpointDesc& endpoint) {
                    return MatchesInstanceEndpoint(links[i].srcEndpointDesc, endpoint);
                });
            if (!endpointMatched) {
                continue;
            }
            matchedLinks[peer] = true;
            break;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetTopologyEndpoints(const HcclComm& hcclHandle, uint32_t layer, uint32_t instance,
                                             std::vector<EndpointDesc>& endpoints) const
{
    uint32_t count = 0U;
    HcclResult hcclRet = HcclRankGraphGetEndpointNum(hcclHandle, layer, instance, &count);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGI("Get topology endpoint count failed, ret=%d", hcclRet);
        return ACLNN_ERR_INNER;
    }
    endpoints.resize(count);
    if (count == 0U) {
        return ACLNN_SUCCESS;
    }
    hcclRet = HcclRankGraphGetEndpointDesc(hcclHandle, layer, instance, &count, endpoints.data());
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGI("Get topology endpoints failed, ret=%d", hcclRet);
        return ACLNN_ERR_INNER;
    }
    endpoints.resize(count);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckGroupInstances(const HcclComm& hcclHandle, uint32_t layer, uint32_t rank,
                                            std::vector<bool>& mesh, std::vector<bool>& clos) const
{
    uint32_t* list = nullptr;
    uint32_t count = 0U;
    HcclResult hcclRet = HcclRankGraphGetTopoInstsByLayer(hcclHandle, layer, &list, &count);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGI("Get topology instances failed, ret=%d", hcclRet);
        return ACLNN_ERR_INNER;
    }
    const std::vector<uint32_t> instances(list, list + count);
    for (uint32_t instance : instances) {
        CommTopo type = COMM_TOPO_RESERVED;
        hcclRet = HcclRankGraphGetTopoType(hcclHandle, layer, instance, &type);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGI("Get topology instance type failed, ret=%d", hcclRet);
            return ACLNN_ERR_INNER;
        }
        OP_LOGD("Group send topology: rank=%u, layer=%u, instance=%u, type=%d", rank, layer, instance, type);
        if (type != COMM_TOPO_1DMESH && type != COMM_TOPO_CLOS) {
            continue;
        }
        hcclRet = HcclRankGraphGetRanksByTopoInst(hcclHandle, layer, instance, &list, &count);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGI("Get topology instance ranks failed, ret=%d", hcclRet);
            return ACLNN_ERR_INNER;
        }
        TopologyInstance topology{layer, type, std::vector<uint32_t>(list, list + count), {}};
        auto ret = GetTopologyEndpoints(hcclHandle, layer, instance, topology.endpoints);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        // CUSTOM层按实例端点筛选，避免混用Mesh和Clos链路
        if (topology.endpoints.empty()) {
            continue;
        }
        ret = CheckGroupLinks(hcclHandle, topology, rank, mesh, clos);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckGroupLayer(const HcclComm& hcclHandle, uint32_t layer, uint32_t rank,
                                        std::vector<bool>& mesh, std::vector<bool>& clos) const
{
    uint32_t* list = nullptr;
    uint32_t count = 0U;
    CommTopo type = COMM_TOPO_RESERVED;
    HcclResult hcclRet = HcclRankGraphGetTopoTypeByLayer(hcclHandle, layer, &type);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGI("Get topology layer type failed, ret=%d", hcclRet);
        return ACLNN_ERR_INNER;
    }
    aclnnStatus ret = ACLNN_SUCCESS;
    OP_LOGD("Group send topology: rank=%u, layer=%u, type=%d", rank, layer, type);
    if (type == COMM_TOPO_CUSTOM) {
        ret = CheckGroupInstances(hcclHandle, layer, rank, mesh, clos);
    } else if (type == COMM_TOPO_1DMESH || type == COMM_TOPO_CLOS) {
        hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layer, &list, &count);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGI("Get topology layer ranks failed, ret=%d", hcclRet);
            return ACLNN_ERR_INNER;
        }
        TopologyInstance topology{layer, type, std::vector<uint32_t>(list, list + count), {}};
        ret = CheckGroupLinks(hcclHandle, topology, rank, mesh, clos);
    }
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckGroupSendEnabled(const HcclComm& hcclHandle, uint32_t rankSize, uint32_t rank,
                                              bool& enableGroupSend) const
{
    if (rankSize <= GROUP_RANK_NUM || rankSize % GROUP_RANK_NUM != 0U || rank >= rankSize) {
        OP_LOGI("Group send disabled: unsupported rank layout, rank=%u, rankSize=%u, groupSize=%u", rank, rankSize,
                GROUP_RANK_NUM);
        return ACLNN_ERR_INNER;
    }
    uint32_t* list = nullptr;
    uint32_t count = 0U;
    HcclResult hcclRet = HcclRankGraphGetLayers(hcclHandle, &list, &count);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGI("Get topology layers failed, ret=%d", hcclRet);
        return ACLNN_ERR_INNER;
    }
    // 库内列表可能被后续查询覆盖
    const std::vector<uint32_t> layers(list, list + count);
    std::vector<bool> mesh(rankSize, false);
    std::vector<bool> clos(rankSize, false);
    for (uint32_t layer : layers) {
        aclnnStatus ret = CheckGroupLayer(hcclHandle, layer, rank, mesh, clos);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    for (uint32_t peer = 0U; peer < rankSize; ++peer) {
        const bool isSameGroup = peer / GROUP_RANK_NUM == rank / GROUP_RANK_NUM;
        const bool hasRequiredLink = isSameGroup ? mesh[peer] : clos[peer];
        if (peer != rank && !hasRequiredLink) {
            OP_LOGI("Group send disabled: missing UB_MEM link, rank=%u, peer=%u, sameGroup=%d, mesh=%d, clos=%d", rank,
                    peer, isSameGroup, static_cast<bool>(mesh[peer]), static_cast<bool>(clos[peer]));
            return ACLNN_ERR_INNER;
        }
    }
    enableGroupSend = true;
    return ACLNN_SUCCESS;
}

void Mc2Context::InitGroupSend(const HcclComm& hcclHandle, CommProtocol protocol, Mc2MoeContext* mc2ContextStruct)
{
    mc2ContextStruct->channelsPerRank = 0;
    mc2ContextStruct->enableGroupSend = false;
#if __has_include("version/hcomm_version.h")
    if (protocol == COMM_PROTOCOL_UB_MEM) {
        aclnnStatus ret = LoadTopologySymbols();
        if (ret == ACLNN_SUCCESS) {
            ret = CheckGroupSendEnabled(hcclHandle, epRankSize_, mc2ContextStruct->epRankId,
                                        mc2ContextStruct->enableGroupSend);
        }
        OP_LOGI("Group send topology check completed, ret=%d, enabled=%d", ret, mc2ContextStruct->enableGroupSend);
    }
#endif
}

aclnnStatus Mc2Context::CreatMc2Context(const HcclComm& hcclHandle, const std::string& mc2ContextTag,
                                        const CommEngine& engine, CommProtocol protocol,
                                        Mc2MoeContext* mc2ContextStruct, void*& ctx, uint64_t& hcclBuffSize)
{
    OP_LOGD("Start to create HCCL context");
    auto ret = GetRankId(hcclHandle, mc2ContextStruct->epRankId);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = GetRankSize(hcclHandle, epRankSize_);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(mc2ContextStruct->epRankId < epRankSize_, ACLNN_ERR_INNER);
    ret = protocol == CommProtocol::COMM_PROTOCOL_ROCE ?
              GetRegisteredCommResource(hcclHandle, engine, mc2ContextStruct) :
              GetHcclCommResource(hcclHandle, engine, protocol, mc2ContextStruct);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    InitGroupSend(hcclHandle, protocol, mc2ContextStruct);
    const uint64_t ctxSize = sizeof(Mc2MoeContext);
    void** readyContext = nullptr;
    ret = CreateContextCache(hcclHandle, mc2ContextTag, ctx, readyContext);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto hcclRet = HcclEngineCtxCopy(hcclHandle, engine, mc2ContextTag.c_str(), mc2ContextStruct, ctxSize,
                                     KOPY_DEFAULT_CTX_OFFSET);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                                "Copy context failed; exit and recreate the communication domain before retrying");
        return ACLNN_ERR_INNER;
    }
    *readyContext = ctx;
    hcclBuffSize = hcclBuffSize_;
    OP_LOGD("Copy context from host to device success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CreatMc2ContextTensor(void* ctx, aclTensor*& mc2Context)
{
    OP_LOGD("Start to create Mc2Context Tensor");
    if (ctx == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Create Mc2Context Tensor failed ctx is nullptr.");
        return ACLNN_ERR_INNER;
    }

    uint64_t mc2ContextLength = sizeof(Mc2MoeContext);
    int64_t shape[1] = {static_cast<int64_t>(mc2ContextLength / sizeof(uint32_t))};
    int64_t strides[1] = {1};

    mc2Context = aclCreateTensor(shape, 1, ACL_INT32, strides, 0, ACL_FORMAT_ND, shape, 1, ctx);
    if (mc2Context == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Create Mc2Context Tensor failed.");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("CreatMc2ContextTensor Success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetLayerRanks(const HcclComm& hcclHandle, uint32_t layerId, LayerRanks& layer) const
{
    uint32_t rankNum = 0;
    uint32_t* rankList = nullptr;
    HcclResult hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layerId, &rankList, &rankNum);
    if (hcclRet != HCCL_SUCCESS || rankList == nullptr || rankNum == 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank IDs by layer %u failed or returned an empty rank list",
                                layerId);
        return ACLNN_ERR_INNER;
    }
    layer = {layerId, std::vector<uint32_t>(rankList, rankList + rankNum)};
    return ACLNN_SUCCESS;
}

bool Mc2Context::HasLinkWithProtocol(const CommLink* linksList, uint32_t netLinkNum, const CommProtocol& protocol)
{
    for (uint32_t linkIdx = 0; linkIdx < netLinkNum; ++linkIdx) {
        if (linksList[linkIdx].linkAttr.linkProtocol == protocol) {
            return true;
        }
    }
    return false;
}

bool Mc2Context::SupportsProtocol(const HcclComm& hcclHandle, uint32_t layerId, uint32_t srcRankId, uint32_t dstRankId,
                                  const CommProtocol& protocol) const
{
    CommLink* linksList = nullptr;
    uint32_t netLinkNum = 0;
    HcclResult hcclRet = HcclRankGraphGetLinks(hcclHandle, layerId, srcRankId, dstRankId, &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS || linksList == nullptr || netLinkNum == 0) {
        OP_LOGW("Get HCCL links failed when checking protocol support, srcRankId: %u, dstRankId: %u, "
                "layerId: %u, ret: %d, netLinkNum: %u",
                srcRankId, dstRankId, layerId, hcclRet, netLinkNum);
        return false;
    }
    return HasLinkWithProtocol(linksList, netLinkNum, protocol);
}

bool Mc2Context::SupportsDomainProtocolWithAllRanks(const HcclComm& hcclHandle, const LayerRanks& layer,
                                                    const CommProtocol& domainProtocol, uint32_t srcRankId) const
{
    for (uint32_t dstRank : layer.ranks) {
        if (dstRank >= epRankSize_) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Invalid rank %u in layer %u", dstRank, layer.layer);
            return false;
        }
        if (dstRank == srcRankId || rankLinkMap_.count(dstRank) > 0) {
            continue;
        }
        if (!SupportsProtocol(hcclHandle, layer.layer, srcRankId, dstRank, domainProtocol)) {
            return false;
        }
    }
    return true;
}

void Mc2Context::RecordDomainLayerRanks(const CommProtocol& protocol, const LayerRanks& layer, uint32_t srcRankId)
{
    for (uint32_t dstRank : layer.ranks) {
        if (dstRank != srcRankId && rankLinkMap_.count(dstRank) == 0) {
            rankLinkMap_[dstRank] = {protocol, layer.layer};
        }
    }
}

bool Mc2Context::FindUbDomain(const HcclComm& hcclHandle, const uint32_t* layerList, uint32_t layerNum,
                              const CommProtocol& domainProtocol, uint32_t srcRankId, LayerRanks& ubDomain)
{
    bool hasDomainLayer = false;
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        LayerRanks layer;
        aclnnStatus ret = GetLayerRanks(hcclHandle, layerList[layerIndex], layer);
        CHECK_RET(ret == ACLNN_SUCCESS, false);
        if (!SupportsDomainProtocolWithAllRanks(hcclHandle, layer, domainProtocol, srcRankId)) {
            break;
        }
        RecordDomainLayerRanks(domainProtocol, layer, srcRankId);
        ubDomain = std::move(layer);
        hasDomainLayer = true;
    }
    return hasDomainLayer;
}

bool Mc2Context::CheckCrossUbDomainProtocols(const HcclComm& hcclHandle, const uint32_t* layerList, uint32_t layerNum,
                                             uint32_t srcRankId)
{
    const uint32_t localServer = srcRankId / rankSizePerServer_;
    const uint32_t localIndex = srcRankId % rankSizePerServer_;
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        LayerRanks layer;
        aclnnStatus ret = GetLayerRanks(hcclHandle, layerList[layerIndex], layer);
        CHECK_RET(ret == ACLNN_SUCCESS, false);
        for (uint32_t dstRank : layer.ranks) {
            if (dstRank >= epRankSize_) {
                OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Invalid rank %u in layer %u", dstRank, layer.layer);
                return false;
            }
            if (dstRank / rankSizePerServer_ == localServer || dstRank % rankSizePerServer_ != localIndex ||
                rankLinkMap_.count(dstRank) > 0) {
                continue;
            }
            if (SupportsProtocol(hcclHandle, layer.layer, srcRankId, dstRank, CommProtocol::COMM_PROTOCOL_ROCE)) {
                rankLinkMap_[dstRank] = {CommProtocol::COMM_PROTOCOL_ROCE, layer.layer};
                OP_LOGI("Record ROCE link: src %u, dst %u, layer %u", srcRankId, dstRank, layer.layer);
            }
        }
    }
    for (uint32_t server = 0; server < epRankSize_ / rankSizePerServer_; ++server) {
        const uint32_t peer = server * rankSizePerServer_ + localIndex;
        if (server != localServer && rankLinkMap_.count(peer) == 0) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Missing ROCE link: src %u, dst %u", srcRankId, peer);
            return false;
        }
    }
    return true;
}

aclnnStatus Mc2Context::ValidateRankLinkMap(uint32_t rankSize, uint32_t srcRankId) const
{
    if (srcRankId >= rankSize) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Invalid local rank %u for rank size %u", srcRankId, rankSize);
        return ACLNN_ERR_INNER;
    }
    for (const std::pair<const uint32_t, RankLinkInfo>& linkEntry : rankLinkMap_) {
        const uint32_t peer = linkEntry.first;
        const bool local = peer / rankSizePerServer_ == srcRankId / rankSizePerServer_;
        const CommProtocol protocol = local ? CommProtocol::COMM_PROTOCOL_UB_MEM : CommProtocol::COMM_PROTOCOL_ROCE;
        if (peer >= rankSize || peer == srcRankId || linkEntry.second.protocol != protocol ||
            (!local && peer % rankSizePerServer_ != srcRankId % rankSizePerServer_)) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Invalid topology rank %u for local rank %u", peer, srcRankId);
            return ACLNN_ERR_INNER;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckRoceProtocolSupport(const HcclComm& hcclHandle, const uint32_t* layerList,
                                                 uint32_t layerNum, const CommProtocol& protocol)
{
    uint32_t srcRankId = 0;
    aclnnStatus ret = GetRankId(hcclHandle, srcRankId);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = GetRankSize(hcclHandle, epRankSize_);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(srcRankId < epRankSize_, ACLNN_ERR_INNER);
    rankLinkMap_.clear();
    rankSizePerServer_ = 0;
    // Keep the layer IDs stable while HCCL reuses query buffers inside the helpers.
    const std::vector<uint32_t> layerIds(layerList, layerList + layerNum);
    LayerRanks ubDomain;
    if (!FindUbDomain(hcclHandle, layerIds.data(), layerNum, protocol, srcRankId, ubDomain)) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Failed to determine UB domain for rank %u, protocol: %d", srcRankId,
                                protocol);
        return ACLNN_ERR_INNER;
    }
    rankSizePerServer_ = static_cast<uint32_t>(ubDomain.ranks.size());
    if (rankSizePerServer_ == 0 || rankSizePerServer_ > epRankSize_ || epRankSize_ % rankSizePerServer_ != 0 ||
        rankLinkMap_.size() != rankSizePerServer_ - 1) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Invalid UB domain size %u for rank size %u", rankSizePerServer_,
                                epRankSize_);
        return ACLNN_ERR_INNER;
    }
    ret = ValidateRankLinkMap(epRankSize_, srcRankId);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    OP_LOGI("Layer %u is current rank's UB domain, rankSizePerServer_: %u", ubDomain.layer, rankSizePerServer_);
    if (rankSizePerServer_ == epRankSize_) {
        return ACLNN_SUCCESS;
    }
    if (!CheckCrossUbDomainProtocols(hcclHandle, layerIds.data(), layerNum, srcRankId)) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Missing required cross-domain ROCE links for rank %u", srcRankId);
        return ACLNN_ERR_INNER;
    }
    ret = ValidateRankLinkMap(epRankSize_, srcRankId);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    const uint32_t peerNum = rankSizePerServer_ - 1 + epRankSize_ / rankSizePerServer_ - 1;
    if (rankLinkMap_.size() != peerNum) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Incomplete topology for rank %u, recorded %zu of %u peers", srcRankId,
                                rankLinkMap_.size(), peerNum);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::BuildRankLinkMap(const HcclComm& hcclHandle)
{
    OP_LOGD("Start to build HCCL rank link map");
    uint32_t layerNum = 0;
    uint32_t* layerList = nullptr;
    aclnnStatus ret = GetNetLayers(hcclHandle, layerList, layerNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return CheckRoceProtocolSupport(hcclHandle, layerList, layerNum, CommProtocol::COMM_PROTOCOL_UB_MEM);
}

aclnnStatus Mc2Context::ValidateContextTag(const std::string& mc2ContextTag)
{
    if (mc2ContextTag.size() > MAX_CONTEXT_TAG_SIZE) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Mc2ContextTag is too long, max size is %lu, but current size is %zu",
                                MAX_CONTEXT_TAG_SIZE, mc2ContextTag.size());
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclBufferSize(const HcclComm& hcclHandle, uint64_t& hcclBuffSize)
{
    void* tempBuffer = nullptr;
    auto hcclRet = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &hcclBuffSize);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL Buffer Size failed");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckContextCache(const HcclComm& hcclHandle, const std::string& mc2ContextTag,
                                          const CommEngine& engine, CommProtocol protocol, void*& ctx,
                                          uint64_t& hcclBuffSize)
{
    uint64_t ctxSize = 0;
    auto hcclRet = HcclEngineCtxGet(hcclHandle, mc2ContextTag.c_str(), engine, &ctx, &ctxSize);
    if (hcclRet == HCCL_E_NOT_FOUND) {
        ctx = nullptr;
        hcclBuffSize = 0;
        OP_LOGI("Context cache not found, need to create");
        return ACLNN_SUCCESS;
    }
    if (hcclRet != HCCL_SUCCESS || ctx == nullptr || ctxSize != sizeof(Mc2MoeContext)) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Invalid cached context, size: %lu", ctxSize);
        return ACLNN_ERR_INNER;
    }
    void* state = nullptr;
    uint64_t stateSize = 0;
    hcclRet = HcclEngineCtxGet(hcclHandle, mc2ContextTag.c_str(), CommEngine::COMM_ENGINE_CPU, &state, &stateSize);
    CHECK_RET(hcclRet == HCCL_SUCCESS && state != nullptr && stateSize == sizeof(void*), ACLNN_ERR_INNER);
    if (*static_cast<void**>(state) != ctx) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Context copy did not complete; recreate the communication domain");
        return ACLNN_ERR_INNER;
    }
    if (protocol != CommProtocol::COMM_PROTOCOL_ROCE) {
        return GetHcclBufferSize(hcclHandle, hcclBuffSize);
    }
    auto aclnnRet = CheckRegisteredWindowSize(hcclHandle, hcclBuffSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    void* cache = nullptr;
    uint64_t cacheSize = 0;
    hcclRet = HcclEngineCtxGet(hcclHandle, REGISTERED_WINDOWS_TAG, CommEngine::COMM_ENGINE_CPU, &cache, &cacheSize);
    CHECK_RET(hcclRet == HCCL_SUCCESS && cache != nullptr && cacheSize == sizeof(RegisteredWindows), ACLNN_ERR_INNER);
    const RegisteredWindows* windows = static_cast<const RegisteredWindows*>(cache);
    if (windows->ub.addr == nullptr || windows->roce.addr == nullptr || windows->ub.handle == nullptr ||
        windows->roce.handle == nullptr || windows->ub.size != hcclBuffSize || windows->roce.size != hcclBuffSize) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                                "Registered window cache differs from HCCL_BUFFSIZE; recreate the domain");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckLinks(uint32_t& netLinkNum, CommLink* linksList)
{
    bool isFoundUbMemProtocol = false;
    for (uint32_t j = 0; j < netLinkNum; ++j) {
        if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
            isFoundUbMemProtocol = true;
            break;
        }
    }
    return isFoundUbMemProtocol ? ACLNN_SUCCESS : ACLNN_ERR_INNER;
}

aclnnStatus Mc2Context::CheckProtocolSupport(const HcclComm& hcclHandle, uint32_t*& layerList, uint32_t& layerNum)
{
    uint32_t srcRankId = 0;
    uint32_t netLinkNum = 0;
    uint32_t rankNumInLayer = 0;
    uint32_t* rankIdLists = nullptr;
    CommLink* linksList = nullptr;

    aclnnStatus ret = GetRankId(hcclHandle, srcRankId);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    HcclResult hcclRet;
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        OP_LOGD("CheckProtocolSupport Check layer %u", layerList[layerIndex]);
        hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank IDs by layer failed");
            return ACLNN_ERR_INNER;
        }
        for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
            if (rankIdLists[rankId] == srcRankId ||
                layerMap.find(rankIdLists[rankId]) != layerMap.end()) { // 本卡或者已经校验过的卡跳过
                continue;
            }
            hcclRet = HcclRankGraphGetLinks(hcclHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId],
                                            &linksList, &netLinkNum);
            if (hcclRet != HCCL_SUCCESS) {
                OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL links failed");
                return ACLNN_ERR_INNER;
            }
            if (netLinkNum == 0) {
                OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                                        "No available HCCL links found, srcRankID %u, dstRankID %u layer is %u",
                                        srcRankId, rankIdLists[rankId], layerList[layerIndex]);
                return ACLNN_ERR_INNER;
            }
            if (CheckLinks(netLinkNum, linksList) != ACLNN_SUCCESS) {
                OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                                        "No HCCL links support UB_MEM srcRankID %u, dstRankID %u layer is %u",
                                        srcRankId, rankIdLists[rankId], layerList[layerIndex]);
                return ACLNN_ERR_INNER;
            }
            layerMap[rankIdLists[rankId]] = layerList[layerIndex];
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetCommProtocol(const HcclComm& hcclHandle, CommProtocol& protocol)
{
    OP_LOGD("Start to get HCCL communication protocol");
    uint32_t layerNum = 0;
    uint32_t* layerList = nullptr;
    auto ret = HcclRankGraphGetLayers(hcclHandle, &layerList, &layerNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL layers failed");
        return ACLNN_ERR_INNER;
    }

    if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
        OP_LOGI("HCCL communication layerNum is %u,so set protocol to UB_MEM", layerNum);
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        return ACLNN_SUCCESS;
    }

    OP_LOGD("start CheckProtocolSupport, layerNum is %u", layerNum);

    auto aclnnRet = CheckProtocolSupport(hcclHandle, layerList, layerNum);
    if (aclnnRet != ACLNN_SUCCESS) {
        return aclnnRet;
    }

    OP_LOGD("CheckProtocolSupport success!");
    protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::PrepareContextTopology(const HcclComm& hcclHandle, CommProtocol& protocol)
{
    if (protocol == CommProtocol::COMM_PROTOCOL_ROCE) {
        return BuildRankLinkMap(hcclHandle);
    }
    uint32_t* layers = nullptr;
    uint32_t layerNum = 0;
    aclnnStatus ret = GetNetLayers(hcclHandle, layers, layerNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = GetRankSizePerServer(hcclHandle, layers[GET_LOCAL_SERVER_RANK_SIZE_LAYER]);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return protocol == CommProtocol::COMM_PROTOCOL_UBC_CTP ? ACLNN_SUCCESS : GetCommProtocol(hcclHandle, protocol);
}

aclnnStatus Mc2Context::GetMc2ContextTensor(const char* groupEp, const char* opName, uint64_t& hcclBuffSize,
                                            aclTensor*& mc2Context, CommProtocol protocol, uint32_t* rankSizePerServer)
{
    hcclBuffSize = 0;
    mc2Context = nullptr;
    CHECK_RET(groupEp != nullptr && opName != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    Mc2Context instance;
    auto ret = instance.LoadHcclSymbols(protocol);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    HcclComm hcclHandle;
    ret = instance.GetCommHandle(groupEp, hcclHandle);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = instance.PrepareContextTopology(hcclHandle, protocol);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    const std::string tag = std::string(groupEp) + opName + "_context_v5_" + std::to_string(static_cast<int>(protocol));
    ret = instance.ValidateContextTag(tag);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    const CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    void* ctx = nullptr;
    ret = instance.CheckContextCache(hcclHandle, tag, engine, protocol, ctx, hcclBuffSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (ctx == nullptr) {
        Mc2MoeContext context{};
        ret = instance.CreatMc2Context(hcclHandle, tag, engine, protocol, &context, ctx, hcclBuffSize);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    if (rankSizePerServer != nullptr) {
        *rankSizePerServer = instance.rankSizePerServer_;
    }
    return instance.CreatMc2ContextTensor(ctx, mc2Context);
}

aclnnStatus Mc2Context::GetMc2RankSize(const char* groupEp, uint32_t& rankSize)
{
    OP_LOGI("Start to get Mc2RankSize Tensor");
    Mc2Context instance;

    auto aclnnRet = instance.LoadHcclSymbols(CommProtocol::COMM_PROTOCOL_UB_MEM);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    HcclComm hcclHandle;
    aclnnRet = instance.GetCommHandle(groupEp, hcclHandle);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    auto hcclRet = instance.HcclGetRankSize(hcclHandle, &rankSize);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank size failed");
        return ACLNN_ERR_INNER;
    }

    OP_LOGI("Get rank size success, rankSize is: %u", rankSize);
    return ACLNN_SUCCESS;
}

template void* Mc2Context::GetHcclLibFunc<void*>(void* handle, const std::string& funcName);

} // namespace Mc2Aclnn
#endif
