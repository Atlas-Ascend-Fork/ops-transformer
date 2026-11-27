# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise production layer selection with local rank graphs and threaded collectives.

Requires a C++17 compiler; Torch, CANN and NPU hardware are not required.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "torch_extension/csrc/moe_shared_jetty_context.cpp"
)


def definition(source, marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if marker.startswith("struct "):
        end += 1
    return source[start:end]


PRELUDE = r"""
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
using HcclResult = int;
using EndpointAttrBwCoeff = uint32_t;
using EndpointAttrDieId = uint32_t;
constexpr int HCCL_SUCCESS = 0;
constexpr int HCCL_E_INTERNAL = 3;
constexpr HcclResult MOCK_QUERY_FAILURE_RET = 7;
constexpr uint32_t GLOBAL_LANE_ROOT_RANK = 0;
constexpr uint32_t MIN_TEST_RANKS = 2;
constexpr uint32_t MAX_TEST_RANKS = 8;
constexpr uint32_t MESH_INSTANCE_ID = 10;
constexpr uint32_t CLOS_INSTANCE_ID = 20;
constexpr uint32_t FALLBACK_CLOS_INSTANCE_ID = 30;
constexpr size_t CLOS_TAG_BYTE = 2;
constexpr size_t INACTIVE_ADDRESS_BYTE = 20;
constexpr unsigned char UNSPECIFIED_BYTE = 42;
uint32_t g_worldSize = 4;
template<class... Args> void Fail(Args... args) {
    std::ostringstream message;
    (message << ... << args);
    throw std::runtime_error(message.str());
}
#define TORCH_CHECK(c, ...) do { if (!(c)) { Fail(__VA_ARGS__); } } while (false)
#define ASCEND_LOGI(...) ((void)0)
template<class... Args> void LogWarning(const char* format, Args... args) {
    char message[512];
    std::snprintf(message, sizeof(message), format, args...);
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
    std::cerr << message << '\n';
}
#define ASCEND_LOGW(...) LogWarning(__VA_ARGS__)
enum class CommProtocol { COMM_PROTOCOL_UBC_CTP = 4, COMM_PROTOCOL_UB_RTP = 9 };
enum CommTopo { COMM_TOPO_RESERVED, COMM_TOPO_CLOS, COMM_TOPO_1DMESH, COMM_TOPO_CUSTOM };
enum { COMM_ADDR_TYPE_EID, COMM_ADDR_TYPE_IP_V4, COMM_ADDR_TYPE_IP_V6, COMM_ADDR_TYPE_ID };
enum EndpointAttr { ENDPOINT_ATTR_BW_COEFF, ENDPOINT_ATTR_DIE_ID };
struct EndpointDesc {
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    struct {
        int type = COMM_ADDR_TYPE_EID;
        union {
            unsigned char raws[36]{};
            unsigned char eid[16];
            uint32_t addr;
            unsigned char addr6[16];
            uint32_t id;
        };
    } commAddr;
    struct {
        int locType = 0;
        unsigned char raws[8]{};
    } loc;
};
struct CommLink {
    EndpointDesc srcEndpointDesc;
    EndpointDesc dstEndpointDesc;
    struct { CommProtocol linkProtocol; } linkAttr;
};
struct HcclChannelDesc {
    CommProtocol channelProtocol;
    uint32_t remoteRank;
    EndpointDesc localEndpoint;
    EndpointDesc remoteEndpoint;
};
HcclResult HcclChannelDescInit(HcclChannelDesc*, uint32_t) { return 0; }
thread_local uint32_t g_myRank;
std::string g_scenario;
uint32_t g_failingRank = 1;
uint32_t ParseTestUint(const char* argument, const char* name, uint32_t minimum, uint32_t maximum) {
    const std::string text(argument);
    if (text.empty() || !std::all_of(text.begin(), text.end(), [](unsigned char c) {
            return c >= '0' && c <= '9';
        })) {
        throw std::invalid_argument(std::string(name) + " must be an unsigned decimal integer");
    }
    const auto value = std::stoull(text);
    if (value > std::numeric_limits<uint32_t>::max() || value < minimum || value > maximum) {
        throw std::out_of_range(std::string(name) + " must be in [" + std::to_string(minimum) + ", " +
                                std::to_string(maximum) + "]");
    }
    return static_cast<uint32_t>(value);
}
bool IsCustom() {
    return g_scenario.find("custom_") == 0 || g_scenario.find("instance_") == 0 ||
           g_scenario.find("endpoint_") == 0;
}
bool QueryFails(const char* operation, uint32_t layer) {
    return g_myRank == g_failingRank && (g_scenario == std::string(operation) + "_failure_all" ||
                                       (layer == 0 && g_scenario == std::string(operation) + "_failure"));
}
EndpointDesc MakeEndpoint(uint32_t layer, uint32_t rank) {
    EndpointDesc endpoint;
    endpoint.commAddr.raws[0] = static_cast<unsigned char>(layer);
    endpoint.commAddr.raws[1] = static_cast<unsigned char>(rank);
    return endpoint;
}
HcclResult HcclRankGraphGetLayersFunc(int, uint32_t** layers, uint32_t* count) {
    if (QueryFails("layers", 0)) {
        return MOCK_QUERY_FAILURE_RET;
    }
    thread_local std::vector<uint32_t> ids;
    ids = g_myRank % 2 == 0 ? std::vector<uint32_t>{0, 1} : std::vector<uint32_t>{1, 0};
    if (g_scenario == "different_layers" && g_myRank == 1) {
        ids = {1};
    }
    if (g_scenario == "empty_peer" && g_myRank == 1) {
        ids.clear();
    }
    if (g_scenario == "empty_root" && g_myRank == 0) {
        ids.clear();
    }
    *layers = ids.data();
    *count = ids.size();
    if (QueryFails("layers_null", 0)) {
        *layers = nullptr;
    }
    return 0;
}
HcclResult HcclRankGraphGetRanksByLayerFunc(int, uint32_t layer, uint32_t** ranks, uint32_t* count) {
    if (QueryFails("layer_ranks", layer) || QueryFails("protocol_ranks", layer)) {
        return MOCK_QUERY_FAILURE_RET;
    }
    thread_local std::vector<uint32_t> ids;
    ids.clear();
    for (uint32_t rank = 0; rank < g_worldSize; ++rank) {
        ids.push_back(rank);
    }
    if (g_scenario == "partial_coverage" && g_myRank == 1 && layer == 0) {
        ids.pop_back();
    }
    if (g_scenario == "custom_partial_layer") {
        ids.pop_back();
    }
    if (QueryFails("layer_ranks_invalid", layer) || QueryFails("protocol_ranks_invalid", layer)) {
        ids.back() = g_worldSize;
    }
    *ranks = ids.data();
    *count = ids.size();
    if (QueryFails("layer_ranks_null", layer) || QueryFails("protocol_ranks_null", layer)) {
        *ranks = nullptr;
    }
    return 0;
}
HcclResult HcclRankGraphGetLinksFunc(int, uint32_t layer, uint32_t src, uint32_t dst,
                                     CommLink** links, uint32_t* count) {
    if (QueryFails("links", layer) || QueryFails("protocol_links", layer)) {
        return MOCK_QUERY_FAILURE_RET;
    }
    TORCH_CHECK(src == g_myRank, "queried a remote rank's outgoing links");
    thread_local std::vector<CommLink> values;
    values.clear();
    const bool missing = src == 1 && dst == 0 &&
        ((g_scenario == "missing_link" && layer == 0) || g_scenario == "all_missing");
    // The UBG fallback scenario must have no usable UBC path.
    if (!missing && g_scenario != "ubg") {
        auto remote = MakeEndpoint(layer, dst);
        if (g_scenario == "reciprocal_mismatch" && layer == 0 && src == 1 && dst == 2) {
            remote.commAddr.raws[CLOS_TAG_BYTE] = 1;
        }
        values.push_back({MakeEndpoint(layer, src), remote, {CommProtocol::COMM_PROTOCOL_UBC_CTP}});
        if (IsCustom()) {
            // Mesh is first and fully connected. A CUSTOM layer must filter it out by endpoint membership.
            auto localClos = MakeEndpoint(layer, src);
            auto remoteClos = MakeEndpoint(layer, dst);
            localClos.commAddr.raws[CLOS_TAG_BYTE] = 1;
            remoteClos.commAddr.raws[CLOS_TAG_BYTE] = 1;
            if (!(g_scenario == "custom_missing_link" && src == 1 && dst == 0)) {
                values.push_back({localClos, remoteClos, {CommProtocol::COMM_PROTOCOL_UBC_CTP}});
            }
        }
    }
    if ((g_scenario == "ubg" || g_scenario == "ubg_with_clos") && layer == 1 && src == 1 && dst == 0) {
        values.push_back({MakeEndpoint(layer, src), MakeEndpoint(layer, dst), {CommProtocol::COMM_PROTOCOL_UB_RTP}});
    }
    *links = values.data();
    *count = values.size();
    if (QueryFails("links_null", layer) || QueryFails("protocol_links_null", layer)) {
        *links = nullptr;
        *count = 1;
    }
    return 0;
}
HcclResult HcclRankGraphGetTopoTypeByLayerFunc(int, uint32_t layer, CommTopo* type) {
    if (QueryFails("topology", layer)) {
        return MOCK_QUERY_FAILURE_RET;
    }
    *type = COMM_TOPO_CLOS;
    if (g_scenario.find("protocol_") == 0) {
        *type = COMM_TOPO_1DMESH;
        return HCCL_SUCCESS;
    }
    if (g_scenario == "mesh_only" || (g_scenario == "mesh_before_clos" && layer == 0) ||
        (g_scenario == "different_topology" && layer == 0 && g_myRank == 1)) {
        *type = COMM_TOPO_1DMESH;
    }
    if (IsCustom()) {
        const bool failureScenario = g_scenario.find("_failure") != std::string::npos;
        *type = layer == 0 || failureScenario ? COMM_TOPO_CUSTOM : COMM_TOPO_1DMESH;
    }
    return HCCL_SUCCESS;
}
HcclResult HcclRankGraphGetTopoInstsByLayerFunc(int, uint32_t layer, uint32_t** ids, uint32_t* count) {
    if (QueryFails("instance_list", layer)) {
        return MOCK_QUERY_FAILURE_RET;
    }
    thread_local std::vector<uint32_t> instances;
    instances = {MESH_INSTANCE_ID, CLOS_INSTANCE_ID};
    if (g_scenario == "custom_partial_instance_then_full") {
        instances.push_back(FALLBACK_CLOS_INSTANCE_ID);
    }
    if (g_scenario == "custom_empty_instances") {
        instances.clear();
    }
    *ids = instances.data();
    *count = instances.size();
    if (g_scenario == "instance_list_null_failure" && g_myRank == 1 && layer == 0) {
        *ids = nullptr;
    }
    return HCCL_SUCCESS;
}
HcclResult HcclRankGraphGetTopoTypeFunc(int, uint32_t layer, uint32_t instance, CommTopo* type) {
    if (QueryFails("instance_type", layer)) {
        return MOCK_QUERY_FAILURE_RET;
    }
    const bool isClos = instance == CLOS_INSTANCE_ID || instance == FALLBACK_CLOS_INSTANCE_ID;
    *type = isClos && g_scenario != "custom_mesh_only" ? COMM_TOPO_CLOS : COMM_TOPO_1DMESH;
    return HCCL_SUCCESS;
}
HcclResult HcclRankGraphGetRanksByTopoInstFunc(int, uint32_t layer, uint32_t instance,
                                               uint32_t** ranks, uint32_t* count) {
    if (QueryFails("instance_ranks", layer)) {
        return MOCK_QUERY_FAILURE_RET;
    }
    thread_local std::vector<uint32_t> ids;
    ids.clear();
    for (uint32_t rank = 0; rank < g_worldSize; ++rank) {
        ids.push_back(rank);
    }
    if (g_myRank == 1 && instance == CLOS_INSTANCE_ID &&
        (g_scenario == "custom_partial_instance" || g_scenario == "custom_partial_instance_then_full")) {
        ids.pop_back();
    }
    if (g_scenario == "custom_duplicate_instance_ranks" && g_myRank == 1) {
        ids.back() = 0;
    }
    if (g_scenario == "instance_ranks_invalid_failure" && g_myRank == g_failingRank && layer == 0) {
        ids.back() = g_worldSize;
    }
    *ranks = ids.data();
    *count = ids.size();
    if (g_scenario == "instance_ranks_null_failure" && g_myRank == g_failingRank && layer == 0) {
        *ranks = nullptr;
    }
    return HCCL_SUCCESS;
}
HcclResult HcclRankGraphGetEndpointNumFunc(int, uint32_t layer, uint32_t, uint32_t* count) {
    if (QueryFails("endpoint_count", layer)) {
        return MOCK_QUERY_FAILURE_RET;
    }
    *count = g_scenario == "custom_missing_clos" && g_myRank == 1 ? 0 : 1;
    return HCCL_SUCCESS;
}
HcclResult HcclRankGraphGetEndpointDescFunc(int, uint32_t layer, uint32_t, uint32_t* count,
                                            EndpointDesc* values) {
    if (QueryFails("endpoint_desc", layer)) {
        return MOCK_QUERY_FAILURE_RET;
    }
    TORCH_CHECK(*count == 1, "endpoint capacity mismatch");
    values[0] = MakeEndpoint(layer, g_myRank);
    values[0].commAddr.raws[CLOS_TAG_BYTE] = 1;
    // Instance descriptors may omit location details and leave inactive union bytes unspecified.
    values[0].loc.raws[0] = UNSPECIFIED_BYTE;
    values[0].commAddr.raws[INACTIVE_ADDRESS_BYTE] = UNSPECIFIED_BYTE;
    if (g_scenario == "endpoint_desc_oversize_failure" && g_myRank == 1 && layer == 0) {
        ++*count;
    }
    return HCCL_SUCCESS;
}
HcclResult HcclRankGraphGetEndpointInfoFunc(int, uint32_t rank, const EndpointDesc* endpoint,
                                            EndpointAttr attr, uint32_t, void* value) {
    TORCH_CHECK(rank == g_myRank && endpoint->commAddr.raws[1] == rank, "queried remote attributes");
    *static_cast<uint32_t*>(value) = attr == ENDPOINT_ATTR_BW_COEFF ? 1 : 0;
    return 0;
}
class Collective {
    std::mutex mutex;
    std::condition_variable cv;
    uint32_t arrived = 0;
    uint32_t generation = 0;
    std::vector<std::vector<char>> input;
    std::vector<char> output;
public:
    template<class T> std::vector<T> Gather(const std::vector<T>& data) {
        std::unique_lock<std::mutex> guard(mutex);
        TORCH_CHECK(!data.empty(), "empty collective");
        const auto previous = generation;
        if (arrived == 0) {
            input.resize(g_worldSize);
        }
        const auto bytes = data.size() * sizeof(T);
        input[g_myRank].resize(bytes);
        std::memcpy(input[g_myRank].data(), data.data(), bytes);
        if (++arrived == g_worldSize) {
            output.clear();
            for (const auto& part : input) {
                TORCH_CHECK(part.size() == bytes, "collective payload sizes differ");
                output.insert(output.end(), part.begin(), part.end());
            }
            arrived = 0;
            ++generation;
            cv.notify_all();
        } else {
            TORCH_CHECK(cv.wait_for(guard, std::chrono::seconds(2), [&] { return generation != previous; }),
                        "collective order mismatch or timeout");
        }
        std::vector<T> result(output.size() / sizeof(T));
        std::memcpy(result.data(), output.data(), output.size());
        return result;
    }
};
Collective g_collective;
class Harness {
public:
"""

MEMBERS = r"""
    int comm_ = 0;
    uint32_t rankSize_ = g_worldSize;
    struct {
        uint32_t epRankId = g_myRank;
        uint32_t sharedJettyNum = 32;
    } context_;
    std::vector<SharedLinkGroup> sharedLinkGroups_;
    template<class T> std::vector<T> AllGatherMetadata(const std::vector<T>& data) {
        return g_collective.Gather(data);
    }
"""

EPILOGUE = r"""
    static void CheckAddressPair(const EndpointDesc& original, const EndpointDesc& different) {
        TORCH_CHECK(MatchesInstanceEndpoint(original, original), "identical address rejected");
        auto instance = original;
        std::memset(instance.loc.raws, UNSPECIFIED_BYTE, sizeof(instance.loc.raws));
        instance.commAddr.raws[INACTIVE_ADDRESS_BYTE] = UNSPECIFIED_BYTE;
        TORCH_CHECK(MatchesInstanceEndpoint(original, instance), "unspecified bytes affected matching");
        TORCH_CHECK(MatchesInstanceEndpoint(instance, original), "matching was asymmetric");
        TORCH_CHECK(!MatchesInstanceEndpoint(original, different), "different address accepted");
        TORCH_CHECK(!MatchesInstanceEndpoint(different, original), "mismatch was asymmetric");
    }

    static void CheckEndpointMatching() {
        EndpointDesc eid;
        eid.commAddr.eid[0] = 1;
        auto differentEid = eid;
        differentEid.commAddr.eid[sizeof(differentEid.commAddr.eid) - 1] = 1;
        CheckAddressPair(eid, differentEid);

        EndpointDesc ipv4;
        ipv4.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        ipv4.commAddr.addr = 1;
        auto differentIpv4 = ipv4;
        differentIpv4.commAddr.addr = 2;
        CheckAddressPair(ipv4, differentIpv4);

        EndpointDesc ipv6;
        ipv6.commAddr.type = COMM_ADDR_TYPE_IP_V6;
        ipv6.commAddr.addr6[0] = 1;
        auto differentIpv6 = ipv6;
        differentIpv6.commAddr.addr6[sizeof(differentIpv6.commAddr.addr6) - 1] = 1;
        CheckAddressPair(ipv6, differentIpv6);

        EndpointDesc id;
        id.commAddr.type = COMM_ADDR_TYPE_ID;
        id.commAddr.id = 1;
        auto differentId = id;
        differentId.commAddr.id = 2;
        CheckAddressPair(id, differentId);

        auto differentProtocol = eid;
        differentProtocol.protocol = CommProtocol::COMM_PROTOCOL_UB_RTP;
        TORCH_CHECK(!MatchesInstanceEndpoint(eid, differentProtocol), "protocol mismatch accepted");
        auto differentLocationType = eid;
        differentLocationType.loc.locType = eid.loc.locType + 1;
        TORCH_CHECK(!MatchesInstanceEndpoint(eid, differentLocationType), "location type mismatch accepted");
        auto differentAddressType = eid;
        differentAddressType.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        TORCH_CHECK(!MatchesInstanceEndpoint(eid, differentAddressType), "address type mismatch accepted");
        auto unsupported = eid;
        constexpr int unsupportedAddressType = -1;
        unsupported.commAddr.type = unsupportedAddressType;
        TORCH_CHECK(!MatchesInstanceEndpoint(unsupported, unsupported), "unsupported address type accepted");
    }
};
int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: selection scenario [world_size [failing_rank]]\n";
        return 2;
    }
    try {
        g_scenario = argv[1];
        TORCH_CHECK(!g_scenario.empty(), "scenario must not be empty");
        if (argc > 2) {
            g_worldSize = ParseTestUint(argv[2], "world_size", MIN_TEST_RANKS, MAX_TEST_RANKS);
        }
        if (argc > 3) {
            g_failingRank = ParseTestUint(argv[3], "failing_rank", 0, g_worldSize - 1);
        }
    } catch (const std::exception& error) {
        std::cerr << "Invalid test arguments: " << error.what() << '\n';
        return 2;
    }
    if (g_scenario == "endpoint_matching") {
        Harness::CheckEndpointMatching();
        std::cout << "PASS " << g_scenario << '\n';
        return 0;
    }
    std::vector<std::string> errors(g_worldSize);
    std::vector<int> selected(g_worldSize, -1);
    std::vector<std::thread> threads;
    for (uint32_t rank = 0; rank < g_worldSize; ++rank) {
        threads.emplace_back([&, rank] {
            g_myRank = rank;
            try {
                Harness host;
                host.SelectSharedLayer();
                TORCH_CHECK(host.sharedLinkGroups_.size() == 1, "unexpected lane count");
                selected[rank] = host.sharedLinkGroups_[0].channelDescs[0].localEndpoint.commAddr.raws[0];
                if (IsCustom()) {
                    for (const auto& desc : host.sharedLinkGroups_[0].channelDescs) {
                        TORCH_CHECK(desc.localEndpoint.commAddr.raws[CLOS_TAG_BYTE] == 1 &&
                                    desc.remoteEndpoint.commAddr.raws[CLOS_TAG_BYTE] == 1, "fullmesh endpoint selected");
                    }
                }
            } catch (const std::exception& error) {
                errors[rank] = error.what();
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    const bool rejected = g_scenario == "all_missing" || g_scenario == "ubg" ||
                          g_scenario == "empty_root" || g_scenario == "empty_peer" || g_scenario == "mesh_only" ||
                          g_scenario == "custom_mesh_only" || g_scenario == "custom_missing_clos" ||
                          g_scenario == "custom_missing_link" || g_scenario == "custom_empty_instances" ||
                          g_scenario == "custom_partial_instance" || g_scenario == "custom_duplicate_instance_ranks" ||
                          g_scenario.find("_failure") != std::string::npos;
    const bool firstLayer = g_scenario == "different_order" || g_scenario == "custom_mix" || g_scenario == "ubg_with_clos" ||
                            g_scenario == "custom_partial_layer" || g_scenario == "custom_partial_instance_then_full";
    const int expected = firstLayer ? 0 : 1;
    for (uint32_t rank = 0; rank < g_worldSize; ++rank) {
        if (rejected) {
            TORCH_CHECK(selected[rank] == -1 && !errors[rank].empty(), "ranks did not reject together");
            TORCH_CHECK(errors[rank] == errors[0], "different errors across ranks");
            TORCH_CHECK(errors[rank].find("timeout") == std::string::npos, errors[rank]);
            if (g_scenario.find("_failure") != std::string::npos) {
                TORCH_CHECK(errors[rank].find("rank=" + std::to_string(g_failingRank)) != std::string::npos,
                            errors[rank]);
                const bool contractViolation = g_scenario.find("_null_") != std::string::npos ||
                                               g_scenario.find("_invalid_") != std::string::npos ||
                                               g_scenario.find("_oversize_") != std::string::npos;
                const auto expectedRet = contractViolation ? HCCL_E_INTERNAL : MOCK_QUERY_FAILURE_RET;
                TORCH_CHECK(errors[rank].find("ret=" + std::to_string(expectedRet)) != std::string::npos,
                            errors[rank]);
                TORCH_CHECK(errors[rank].find("HcclRankGraphGet") != std::string::npos, errors[rank]);
                TORCH_CHECK(errors[rank].find("layer=0") != std::string::npos, errors[rank]);
                if (IsCustom() && g_scenario.find("instance_list") != 0) {
                    const auto instance = g_scenario.find("instance_type") == 0 ? MESH_INSTANCE_ID : CLOS_INSTANCE_ID;
                    TORCH_CHECK(errors[rank].find("instance=" + std::to_string(instance)) != std::string::npos,
                                errors[rank]);
                }
                if (g_scenario.find("links") == 0 || g_scenario.find("protocol_links") == 0) {
                    const auto peer = g_failingRank == 0 ? 1 : 0;
                    TORCH_CHECK(errors[rank].find("peer=" + std::to_string(peer)) != std::string::npos, errors[rank]);
                }
                std::cerr << errors[rank] << '\n';
            }
            if (g_scenario == "ubg") {
                TORCH_CHECK(errors[rank].find("does not support UBG") != std::string::npos, errors[rank]);
            }
        } else {
            TORCH_CHECK(errors[rank].empty(), "rank=", rank, " ", errors[rank]);
            TORCH_CHECK(selected[rank] == expected, "ranks selected different layers");
        }
    }
    std::cout << "PASS " << g_scenario << '\n';
    return 0;
}
"""


class MoeSharedJettyLayerSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = SOURCE.read_text(encoding="utf-8")
        query_failure_text_max = next(
            line
            for line in source.splitlines()
            if line.startswith("constexpr size_t QUERY_FAILURE_TEXT_MAX =")
        )
        structs = [
            definition(source, "struct " + name + " {")
            for name in (
                "SharedLinkGroup",
                "QueryFailureMetadata",
                "GlobalLane",
                "RootLaneMetadata",
                "SelectedEndpointMetadata",
                "LayerCandidateMetadata",
                "UnsupportedProtocolMetadata",
            )
        ]
        methods = []
        for marker in (
            "static void CheckHccl(",
            "static HcclResult RecordQueryFailure(",
            "static void CheckQueryFailure(",
            "static bool IsSameEndpoint(",
            "static bool IsEndpointLess(",
            "HcclResult GetProtocolLinks(",
            "bool TryGetLocalEndpointInfo(",
            "SelectedEndpointMetadata SelectLocalEndpoint(",
            "SharedLinkGroup MakeSharedLinkGroup(",
            "HcclResult GetRankCoverage(",
            "static bool MatchesInstanceEndpoint(",
            "HcclResult GetClosEndpoints(",
            "bool SelectSharedLinks(",
            "void CheckSupportedProtocols(",
            "void SelectSharedLayer(",
        ):
            method = definition(source, marker)
            if marker.startswith("bool TryGet"):
                method = "template<typename T>\n" + method
            methods.append(method)
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            raise unittest.SkipTest(
                "C++17 compiler required for shared layer regression"
            )
        cls.directory = tempfile.TemporaryDirectory(prefix="shared_jetty_layers_")
        cls.addClassCleanup(cls.directory.cleanup)
        path = Path(cls.directory.name) / "selection.cpp"
        path.write_text(
            PRELUDE.replace(
                "class Harness {", query_failure_text_max + "\nclass Harness {"
            )
            + "\n".join(structs)
            + MEMBERS
            + "\n".join(methods)
            + EPILOGUE,
            encoding="utf-8",
        )
        cls.executable = Path(cls.directory.name) / "selection.exe"
        result = subprocess.run(
            [compiler, "-std=c++17", "-pthread", str(path), "-o", str(cls.executable)],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def _check_scenario(self, scenario, world_size=4, failing_rank=1):
        with self.subTest(
            world_size=world_size, scenario=scenario, failing_rank=failing_rank
        ):
            result = subprocess.run(
                [str(self.executable), scenario, str(world_size), str(failing_rank)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            if "_failure" in scenario:
                query = scenario.split("_failure")[0]
                api = {
                    "layers": "HcclRankGraphGetLayers",
                    "layers_null": "HcclRankGraphGetLayers",
                    "links": "HcclRankGraphGetLinks",
                    "links_null": "HcclRankGraphGetLinks",
                    "protocol_links": "HcclRankGraphGetLinks",
                    "protocol_links_null": "HcclRankGraphGetLinks",
                    "protocol_ranks": "HcclRankGraphGetRanksByLayer",
                    "protocol_ranks_null": "HcclRankGraphGetRanksByLayer",
                    "protocol_ranks_invalid": "HcclRankGraphGetRanksByLayer",
                    "layer_ranks": "HcclRankGraphGetRanksByLayer",
                    "layer_ranks_null": "HcclRankGraphGetRanksByLayer",
                    "layer_ranks_invalid": "HcclRankGraphGetRanksByLayer",
                    "topology": "HcclRankGraphGetTopoTypeByLayer",
                    "instance_list": "HcclRankGraphGetTopoInstsByLayer",
                    "instance_list_null": "HcclRankGraphGetTopoInstsByLayer",
                    "instance_type": "HcclRankGraphGetTopoType",
                    "instance_ranks": "HcclRankGraphGetRanksByTopoInst",
                    "instance_ranks_null": "HcclRankGraphGetRanksByTopoInst",
                    "instance_ranks_invalid": "HcclRankGraphGetRanksByTopoInst",
                    "endpoint_count": "HcclRankGraphGetEndpointNum",
                    "endpoint_desc": "HcclRankGraphGetEndpointDesc",
                    "endpoint_desc_oversize": "HcclRankGraphGetEndpointDesc",
                }[query]
                self.assertIn(f"{api}: ", result.stderr)
                if "_null" in query:
                    self.assertIn("Null ", result.stderr)
                if "_invalid" in query:
                    self.assertIn("Invalid rank", result.stderr)
            if scenario.startswith("instance_ranks_"):
                self.assertIn(
                    f"rank={failing_rank}, layer 0, instance 20", result.stderr
                )
            if scenario == "instance_ranks_null_failure":
                self.assertIn("Null rank list", result.stderr)
            if scenario == "instance_ranks_invalid_failure":
                self.assertIn(f"Invalid rank {world_size}", result.stderr)
            if scenario == "instance_list_null_failure":
                self.assertIn(
                    "Null topology instance list, rank=1, layer=0, count=2",
                    result.stderr,
                )
            if scenario == "endpoint_desc_oversize_failure":
                self.assertIn(
                    "Endpoint count exceeds capacity, rank=1, layer=0, instance=20, count=2, capacity=1",
                    result.stderr,
                )

    def test_invalid_arguments(self):
        cases = [[], [""], ["different_order", "4", "1", "extra"]]
        invalid_world_sizes = (
            "0",
            "1",
            "9",
            "4294967295",
            "4294967296",
            "4294967300",
            "18446744073709551616",
            "-1",
            "+4",
            " 4",
            "4x",
            "4.0",
            "",
        )
        cases.extend(["different_order", value] for value in invalid_world_sizes)
        invalid_ranks = (
            "4",
            "4294967295",
            "4294967296",
            "18446744073709551616",
            "-1",
            "+1",
            " 1",
            "1x",
            "1.0",
            "",
        )
        cases.extend(["different_order", "4", value] for value in invalid_ranks)
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [str(self.executable), *arguments],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertNotIn("PASS", result.stdout)
                self.assertTrue(result.stderr)

    def test_endpoint_matching(self):
        self._check_scenario("endpoint_matching")

    def test_argument_boundaries(self):
        self._check_scenario("different_order", world_size=2, failing_rank=0)
        self._check_scenario("different_order", world_size=8, failing_rank=7)

    def test_collective_layer_selection(self):
        for scenario in (
            "different_order",
            "different_layers",
            "partial_coverage",
            "missing_link",
            "reciprocal_mismatch",
            "all_missing",
            "empty_root",
            "empty_peer",
            "ubg",
        ):
            self._check_scenario(scenario)

    def test_clos_topology_filter(self):
        for world_size in (2, 4, 8):
            for scenario in (
                "mesh_only",
                "mesh_before_clos",
                "different_topology",
                "custom_mix",
                "custom_partial_layer",
                "custom_partial_instance",
                "custom_partial_instance_then_full",
                "custom_duplicate_instance_ranks",
                "custom_mesh_only",
                "custom_empty_instances",
                "custom_missing_clos",
                "custom_missing_link",
                "ubg_with_clos",
            ):
                self._check_scenario(scenario, world_size)

    def test_single_rank_query_failure(self):
        for world_size in (2, 4, 8):
            for failing_rank in (0, 1):
                for query in (
                    "layers",
                    "layers_null",
                    "links",
                    "links_null",
                    "protocol_links",
                    "protocol_links_null",
                    "protocol_ranks",
                    "protocol_ranks_null",
                    "protocol_ranks_invalid",
                    "layer_ranks_null",
                    "layer_ranks_invalid",
                    "topology",
                    "layer_ranks",
                    "instance_list",
                    "instance_type",
                    "instance_ranks",
                    "endpoint_count",
                    "endpoint_desc",
                ):
                    for suffix in ("_failure", "_failure_all"):
                        self._check_scenario(query + suffix, world_size, failing_rank)
            for scenario in (
                "instance_list_null_failure",
                "instance_ranks_null_failure",
                "instance_ranks_invalid_failure",
                "endpoint_desc_oversize_failure",
            ):
                self._check_scenario(scenario, world_size)


if __name__ == "__main__":
    unittest.main()
