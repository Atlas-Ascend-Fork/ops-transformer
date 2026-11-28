# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""检查源码与 CPU 上的接口约定，不包含 CANN 编译、Hcomm 执行或 NPU 精度验证。"""

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH35 = ROOT / "op_kernel/arch35/mega_moe_layered.h"
ARCH92 = ROOT / "op_kernel/arch92/mega_moe_layered_arch92.h"
ARCH35_WAVE = ARCH35.with_name("mega_moe_layered_wave.h")
ARCH92_WAVE = ARCH92.with_name("mega_moe_layered_wave_arch92.h")


def extract(source, marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def body(source, marker):
    result = extract(source, marker)
    return result[result.index("{") :]


PREFIX = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#define __aicore__
#define __gm__
#define __ubuf__
#undef assert
#define assert(x) ((x)?void(0):(std::fprintf(stderr,"assertion %d: %s\n",__LINE__,#x),std::exit(1)))
"""

WAVE_MOCK = r"""
enum { AIC=0,AIV=1 };
constexpr int g_coreType=CORE;
enum class AddrUpdateMode { GMM1,GMM2 };
enum class AxWMode { A8W4 };
struct MoeQuantConfig { static constexpr AxWMode AXW_MODE=AxWMode::A8W4; };
struct ExpertTokenPosition { uint32_t expertIdx=0,tokenIndexInExpert=0; };
struct ExpertTokenRange { ExpertTokenPosition begin,end; };
struct ExpertLoopState {};
struct GMMAddrInfo {};
using Event=std::tuple<char,uint32_t,uint32_t,uint32_t>;
struct WaveBase {
    uint32_t moeExpertPerRank_,mGroupsPerWave_,gmmWaveCount_,startBlockIdx_=7,subBlockIdx_=1;
    std::vector<Event> events;
    WaveBase(uint32_t waves,uint32_t groups):moeExpertPerRank_(waves*groups),mGroupsPerWave_(groups),gmmWaveCount_(waves) {}
    ExpertTokenRange PlanTokenWave(ExpertTokenPosition begin,uint32_t groups) {
        return {begin,{std::min(begin.expertIdx+groups,moeExpertPerRank_),0}};
    }
    void ReceiveDispatchTokenRange(ExpertTokenRange range) {
        events.emplace_back('D',range.begin.expertIdx,range.end.expertIdx,startBlockIdx_);
    }
    template<AddrUpdateMode M,typename... Legacy> void ProcessGmmTokenRange(ExpertTokenRange range,
            ExpertLoopState&,GMMAddrInfo&,Legacy&&...) {
        events.emplace_back(M==AddrUpdateMode::GMM1?'1':'2',range.begin.expertIdx,range.end.expertIdx,startBlockIdx_);
        startBlockIdx_=(startBlockIdx_+(range.end.expertIdx-range.begin.expertIdx)*(M==AddrUpdateMode::GMM1?3:2))%64;
    }
    void ProcessCombineTokenRange(ExpertLoopState,ExpertTokenRange range) {
        assert(g_coreType==AIV);
        events.emplace_back('C',range.begin.expertIdx,range.end.expertIdx,startBlockIdx_);
    }
};
"""

NATIVE_MOCK = r"""
namespace AscendC { constexpr int AIC=0,AIV=1; }
using namespace AscendC;
constexpr int g_coreType=CORE;
constexpr int M_VALUE=0,N_VALUE=1;
template<int I,typename T> auto Get(const T& t) { return std::get<I>(t); }
namespace asc::te {
enum class location { gm };
template<location,typename T> T* make_mem_ptr(T* p) { return p; }
struct Tensor {};
template<typename P,typename L> Tensor make_tensor(P,L) { return {}; }
}
int gmMmad=0,gmAct=0,prefetchAct=0,live=0;
const void* lastContext=nullptr;
struct Sync { int waits=0,acks=0; void WaitForGmm1(){++waits;} void NotifyGmm1(){++acks;} };
using GM_ADDR=uint8_t*;
struct Params {};
struct GMMAddrInfo {
    GM_ADDR aGlobal=nullptr,bGlobal=nullptr,aScaleGlobal=nullptr,bScaleGlobal=nullptr,gmm1OutGlobal=nullptr,metaInfoGlobal=nullptr;
    Sync* gmm1ActivationSync;
};
struct KernelConfig {
    using ElementAType=uint8_t;using ElementBType=uint8_t;using ElementMxScaleAType=uint8_t;using ElementMxScaleBType=uint8_t;
    static constexpr bool IS_WEIGHT_NZ=true;
};
struct Config { using KernelConfig=::KernelConfig; bool topkWeightsPrefetch=false; uint32_t blockNum=1; };
struct Layouts { int a=0,b=0,scaleA=0,scaleB=0,c=0; };
struct Scheduler {
    auto GetBlockCoord(uint32_t t) { return std::make_tuple(int64_t(0),int64_t(t)*256); }
    template<typename T> auto GetBlockShape(T) { return std::make_tuple(int64_t(256),int64_t(256)); }
};
template<typename T> struct BlockMmadContext {
    BlockMmadContext() { ++live; }
    ~BlockMmadContext() { --live; }
};
template<typename S,typename A,typename B,typename SA,typename SB,typename C,typename M> struct Gmm1WorkSet {
    S& scheduler;A& gmA;B& gmB;SA& gmScaleA;SB& gmScaleB;C& gmC;M& metaInfoGm;
};
template<bool,typename K,typename... T> void SetWaveWeightL2CacheHint(T&&...) {}
template<typename C,bool S,typename W>
void Gmm1AicMmadToGmGeneric(W&,const Params&,const GMMAddrInfo&,const Config&,uint32_t,uint32_t,uint32_t,C* context) {
    assert(context&&live==1);
    lastContext=context;
    ++gmMmad;
}
template<typename L,typename C,bool W,typename... T> void Gmm1Aiv0PrefetchEpilogueGeneric(T&&...) { ++prefetchAct; }
template<typename C,typename L,bool W,typename... T> void Gmm1AivReadGmEpilogueTileGeneric(T&&...) { ++gmAct; }
"""


class Arch92UrmaMainlineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a5 = ARCH35.read_text(encoding="utf-8")
        cls.a6 = ARCH92.read_text(encoding="utf-8")
        cls.a5_wave = ARCH35_WAVE.read_text(encoding="utf-8")
        cls.a6_wave = ARCH92_WAVE.read_text(encoding="utf-8")

    def cpp(self, source, defines=()):
        compiler = (
            shlex.split(os.environ["CXX"])
            if "CXX" in os.environ
            else [shutil.which("clang++") or shutil.which("g++")]
        )
        if not compiler[0]:
            self.skipTest("No host compiler; not CANN validation")
        with tempfile.TemporaryDirectory(prefix="a6-urma-mainline-") as directory:
            path, binary = Path(directory) / "test.cpp", Path(directory) / "test.exe"
            path.write_text(PREFIX + source, encoding="utf-8")
            result = subprocess.run(
                compiler + ["-std=c++17", *defines, str(path), "-o", str(binary)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run(
                [str(binary)], capture_output=True, text=True, timeout=120
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_layered_stage_uses_architecture_local_includes_without_soc_macros(self):
        shared_stage = ROOT / "op_kernel/arch35_92/stage/layered"
        for shared_name in ("dispatch", "combine", "combine_relay", "unpermute"):
            shared = shared_stage / f"mega_moe_layered_{shared_name}.h"
            self.assertNotIn(
                "MEGA_MOE_ARCH92_DEVICE", shared.read_text(encoding="utf-8")
            )
        for arch, entry in ((35, ARCH35), (92, ARCH92)):
            source = entry.read_text(encoding="utf-8")
            self.assertIn(
                '#include "../arch35_92/stage/layered/mega_moe_layered_dispatch.h"',
                source,
            )
            self.assertIn(
                '#include "../arch35_92/stage/layered/mega_moe_layered_unpermute.h"',
                source,
            )
            for stage_name in ("unpermute", "gmm", "sync"):
                relative = f"stage/layered/mega_moe_layered_{stage_name}_arch{arch}.h"
                self.assertIn(f'#include "{relative}"', source)
                self.assertNotIn(
                    "MEGA_MOE_ARCH92_DEVICE",
                    (entry.parent / relative).read_text(encoding="utf-8"),
                )

    def test_host_entry_initializes_urma_layout_features_before_planning(self):
        path = ROOT / "op_host/op_tiling/arch92"
        tiling = (path / "mega_moe_tiling.cpp").read_text(encoding="utf-8")
        prefix = body(tiling, "static ge::graphStatus SetAdaptiveBufferConfigs(")
        # 运行真实 Host 路径直到 GM 规划完成；后续 UB 计算依赖 CANN。
        prefix = prefix.split("tilingData->dispatchTableUbBaseAddr", 1)[0]
        helpers = "\n".join(
            extract(tiling, marker)
            for marker in (
                "static uint64_t CalcCombineSyncSlotCountPerExpert(",
                "static void SetCclLayoutParams(",
            )
        )
        layout = ROOT / "op_kernel/arch92/common/mega_moe_layout_arch92.h"
        source = (
            f'\n#include "{layout.as_posix()}"\n'
            + r"""
using namespace MegaMoeImpl;
constexpr int COMBINE_NO_QUANT=0;
constexpr uint32_t COMBINE_TOKEN_GROUP_SIZE=256U;
namespace ops {
template<typename T> constexpr T CeilDiv(T value,T divisor) { return (value+divisor-1)/divisor; }
}
namespace ge { enum graphStatus {GRAPH_SUCCESS,GRAPH_FAILED}; }
namespace gert {
struct TilingContext { const void* GetInputDesc(int) const { return nullptr; } };
}
struct MegaMoeConfig { int topkWeightsIndex=0,xIndex=1; };
struct MegaMoeTilingData {
    CclComputeLayoutParams compute{};
    CclCommLayoutParams comm{};
    uint32_t rankNumPerServer=8,blockAivNum=64;
    uint8_t moeGmmMode=GMM_MODE_A8W8_ND,sharedGmmMode=GMM_MODE_A8W8_ND,actMode=0;
    int64_t combineQuantMode=0;
    bool isTopkWeightsPrefetch=false;
};
#define OP_TILING_CHECK(condition, log, action) do { if (condition) { action; } } while (false)
"""
            + helpers
            + r"""
static ge::graphStatus PlanLayout(const gert::TilingContext* context,const MegaMoeConfig& config,
                                 MegaMoeTilingData* tilingData,const char* nodeName)
"""
            + prefix
            + r"""
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}
int main() {
    gert::TilingContext context;
    for(uint32_t rankNumPerServer:{4U,8U})
    for(auto activation:{MegaMoeActMode::SWIGLU,MegaMoeActMode::GELU})
    for(uint8_t mode:{GMM_MODE_A8W8_ND,GMM_MODE_A4W4_NZ}) for(bool serverCombine:{false,true}) {
        for(bool combineQuant:{false,true}) for(bool prefetch:{false,true}) {
            if(serverCombine&&combineQuant) continue;
            MegaMoeTilingData tiling;
            tiling.rankNumPerServer=rankNumPerServer;
            auto& p=tiling.compute;
            p.topoType=TOPO_TYPE_URMA;p.epWorldSize=8;p.moeExpertPerRank=8;p.bs=128;
            p.numMaxTokensPerRank=256;p.h=7168;p.hiddenDim=4096;p.topK=6;p.aicNum=64;
            p.maxOutputSize=256*8*6;
            p.dedupMode=serverCombine?DEDUP_MODE_SERVER_COMBINE:0;
            p.activationOutputDivisor=0;
            p.combineQuant=!combineQuant;p.topkWeightsPrefetch=!prefetch;
            tiling.moeGmmMode=tiling.sharedGmmMode=mode;
            tiling.actMode=static_cast<uint8_t>(activation);
            tiling.combineQuantMode=combineQuant?4:0;
            tiling.isTopkWeightsPrefetch=prefetch;
            assert(PlanLayout(&context,{},&tiling,"MegaMoe")==ge::GRAPH_SUCCESS);
            auto divisor=activation==MegaMoeActMode::GELU?1U:2U;
            assert(p.activationOutputDivisor==divisor);
            assert(p.combineQuant==combineQuant&&tiling.comm.isQuantCombine==combineQuant);
            assert(p.topkWeightsPrefetch==prefetch&&tiling.comm.topkWeightsPrefetch==(prefetch||serverCombine));
            assert(tiling.comm.serverCombineReduce==serverCombine&&p.combineSyncSlotCountPerExpert==64);
            assert(tiling.comm.elemsPerByte==(IsA4W4GmmMode(mode)?2U:1U));
            auto layout=LayoutArch92::MakeWorkspaceLayout(p,mode,mode,false,true);
            assert(layout.routed.gmm2Input.data.sizeBytes==p.maxOutputSize*p.hiddenDim/divisor);
            assert(p.serverNum==p.epWorldSize/rankNumPerServer);
            auto completedBytes=serverCombine?MemoryLayoutDetail::CeilAlign(
                p.serverNum*MemoryLayoutDetail::INT_CACHELINE*SIZE_INT_32,MemoryLayoutDetail::ALIGN_512):0;
            assert(layout.urma.serverCombine.completedCounts.sizeBytes==completedBytes);
            assert(layout.reset.serverCombine.sizeBytes==completedBytes);
            if(serverCombine) assert(layout.reset.serverCombine.offsetBytes==layout.urma.serverCombine.completedCounts.offsetBytes);
        }
    }
}
"""
        )
        self.cpp(source)

    def test_server_combine_keeps_wave_order_and_reuses_the_single_aiv(self):
        process = body(self.a6, "::ProcessWave(")
        self.assertLess(
            process.index("wave.ProcessMoeExpertWave("),
            process.index("ProcessCombineRelay()"),
        )
        self.assertLess(
            process.index("ProcessCombineRelay()"),
            process.index("FinishCombineAndSync()"),
        )
        self.assertLess(
            process.index("FinishCombineAndSync()"), process.index("Unpermute()")
        )
        for entry in (self.a5, self.a6):
            self.assertNotIn("UnpermuteServerCombine()", body(entry, "::ProcessWave("))
            self.assertNotIn("UnpermuteBuffInit()", body(entry, "::ProcessWave("))
        for prepare in ("::PrepareWithSharedExpert(", "::PrepareWithoutSharedExpert()"):
            code = body(self.a6, prepare)
            self.assertLess(
                code.index("PrepareDispatchRelayQueues("), code.index("CrossRankSync<")
            )
        relay = body(self.a6, "::ProcessCombineRelay()")
        self.assertEqual(
            re.findall(
                r"(AllVectorBarrier|BuildServerCombineLists|ReduceRelayContributions|SendServerCombineResults)\(\)",
                relay,
            ),
            [
                "AllVectorBarrier",
                "BuildServerCombineLists",
                "ReduceRelayContributions",
                "SendServerCombineResults",
            ],
        )

    def test_layered_server_combine_waits_for_target_rank_count(self):
        relay = (
            ROOT / "op_kernel/arch35_92/stage/layered/mega_moe_layered_combine_relay.h"
        ).read_text(encoding="utf-8")
        reduce = body(relay, "::ReduceRelayContributionsInRange(")
        self.assertLess(
            reduce.index("ReduceRelayContributionsForBatch("),
            reduce.index("AtomicAdd("),
        )
        self.assertIn(
            "AtomicAdd(completedCountPtr, static_cast<int32_t>(ownedRange.count))",
            reduce,
        )
        common = ROOT / "op_kernel/arch35_92/common"
        utils = (common / "mega_moe_utils.h").read_text(encoding="utf-8")
        sync = (common / "mega_moe_gmm_epilogue_sync.h").read_text(encoding="utf-8")
        wait_helpers = "\n".join(
            (
                extract(utils, "__aicore__ inline void BackoffBeforeGmPollRetry()"),
                extract(
                    utils,
                    "template <typename FlagType>\n__aicore__ inline FlagType WaitUntilGmFlagIsNonZero(",
                ),
                extract(sync, "__aicore__ inline void WaitUntilGmFlagEquals("),
            )
        )
        source = (
            r"""
constexpr uint32_t ALIGN_512=512U,INT_CACHELINE=16U;
constexpr int64_t GM_FLAG_POLL_BACKOFF_CYCLES=200;
int64_t GetDeviceCycle() {
    static int64_t cycle=0;
    return cycle+=GM_FLAG_POLL_BACKOFF_CYCLES;
}
enum class TPosition {VECIN};
template<typename T> struct LocalTensor {
    LocalTensor(TPosition,uint32_t,uint32_t) {}
};
std::vector<const int32_t*> reads;
int32_t GmRead(const int32_t* p) {
    if (reads.size()>=8U) throw std::runtime_error("still waiting");
    reads.push_back(p);
    return *p;
}
"""
            + wait_helpers
            + r"""
constexpr int64_t CalcServerCombineTokenListStrideBytes(int64_t capacity) {
    return (capacity+1)*sizeof(int32_t);
}
struct Region { uint32_t offset; uint8_t* get_ptr(uint8_t* p) const { return p+offset; } };
struct Case {
    static constexpr uint32_t Servers=4,Capacity=5,ListStride=Capacity+1,CounterOffset=32;
    uint32_t serverNum_=Servers,rankNumPerServer_=2,rankIdInServer_=1,rankId_=7,worldSize_=8;
    int sends=0;
    alignas(64) int32_t storage[CounterOffset+Servers*INT_CACHELINE]{};
    struct {struct {uint32_t numMaxTokensPerRank=Capacity;} compute;} tiling;
    struct {
        struct {
            struct {struct {Region tokenIds,completedCounts;} serverCombine;} urma;
        } layout;
        uint8_t* base;
    } work{{{{{0},{CounterOffset*sizeof(int32_t)}}}},reinterpret_cast<uint8_t*>(storage)};
    struct {decltype(work)& workspace;decltype(tiling)* tilingData;} params_{work,&tiling};
    int32_t& Header(uint32_t server) {return storage[server*ListStride];}
    int32_t& Completed(uint32_t server) {return storage[CounterOffset+server*INT_CACHELINE];}
    bool IsChannelOwner(uint32_t rank) const {return rank%3U==1U;}
    void SendRelayResultsToTargetRank(uint32_t server,uint32_t rank,uint32_t count,uint32_t batch,
                                      uint32_t* ids,const LocalTensor<uint32_t>&) {
        assert(server==0&&rank==1&&count==5&&batch==256);
        assert(ids==reinterpret_cast<uint32_t*>(&Header(0)+1));
        ++sends;
    }
    void Run()
"""
            + body(relay, "::SendServerCombineResults()")
            + r"""
};
int main(){
    Case c;
    c.Header(0)=6;
    c.Completed(0)=3;
    for (int32_t count : {3,4,6}) {
        c.Completed(0)=count;
        reads.clear();
        bool blocked=false;
        try { c.Run(); }
        catch (const std::runtime_error&) { blocked=true; }
        assert(blocked&&c.sends==0&&reads.front()==&c.Header(0));
        for (auto it=reads.begin()+1;it!=reads.end();++it) assert(*it==&c.Completed(0));
    }
    // 只等目标 rank 1 的完成计数，其他 Server 的表头及计数可以都未就绪。
    c.Completed(0)=5;
    reads.clear();
    c.Run();
    assert(c.sends==1);
    assert((reads==std::vector<const int32_t*>{&c.Header(0),&c.Completed(0)}));

    // 已发布的空列表不等归约计数；未发布的列表仍会等待。
    c.Header(0)=1;
    reads.clear();
    c.Run();
    assert(c.sends==1&&(reads==std::vector<const int32_t*>{&c.Header(0)}));
    c.Header(0)=0;
    reads.clear();
    bool blocked=false;
    try { c.Run(); }
    catch (const std::runtime_error&) { blocked=true; }
    assert(blocked&&c.sends==1);
    for (auto* p:reads) assert(p==&c.Header(0));

    // 单域中继也是源 rank，不读取完成计数，不做第二跳。
    c.serverNum_=1;c.rankId_=1;c.worldSize_=2;
    reads.clear();
    c.Run();
    assert(c.sends==1&&reads.empty());
}
"""
        )
        self.cpp(source)

    def test_layered_server_combine_bulk_bitmaps_and_relay_counts(self):
        dispatch = (
            ROOT / "op_kernel/arch35_92/stage/layered/mega_moe_layered_dispatch.h"
        ).read_text(encoding="utf-8")
        source = (
            r"""
#include <cstring>
using GM_ADDR=uint8_t*;
constexpr uint32_t ALIGN_32=32;
namespace AscendC { constexpr uint32_t GetVecLen(){return 256;} }
namespace Ops { namespace Base {
template<typename T> T CeilAlign(T n,T a){return (n+a-1)/a*a;}
} }
template<typename T> T GmRead(T* p){return *p;}
template<typename T,typename V> void GmWrite(T* p,V v){*p=static_cast<T>(v);}
int32_t AtomicAdd(int32_t* p,int32_t delta){int32_t old=*p;*p+=delta;return old;}
int barriers=0;
void GmDataBarrier(){++barriers;}
template<typename T> struct LocalTensor {
    T* data;
    T GetValue(uint32_t i) const {return data[i];}
    LocalTensor operator[](uint32_t i) const {return {data+i};}
    uintptr_t GetPhyAddr() const {return reinterpret_cast<uintptr_t>(data);}
    template<typename U> LocalTensor<U> ReinterpretCast() const {return {reinterpret_cast<U*>(data)};}
};
struct ConfusionTransposeTiling {uint32_t param0,param1,param2;};
enum class TransposeType {TRANSPOSE_ND2ND_021};
template<typename T> void Transpose(const LocalTensor<T>& dst,const LocalTensor<T>& src,
    LocalTensor<uint8_t>,TransposeType,ConfusionTransposeTiling t){
    assert(t.param0==1);
    for(uint32_t i=0;i<t.param1;++i)for(uint32_t j=0;j<t.param2;++j)
        dst.data[j*t.param1+i]=src.data[i*t.param2+j];
}
struct CopyRows {uint16_t rows;uint32_t rowBytes,gmPitchBytes,ubPitchBytes;};
void CopyUbToGm(uint32_t* dst,uint32_t* src,CopyRows rows){
    assert(rows.rowBytes==4&&rows.gmPitchBytes==8&&rows.ubPitchBytes==4);
    for(uint32_t i=0;i<rows.rows;++i){
        std::memcpy(reinterpret_cast<uint8_t*>(dst)+i*rows.gmPitchBytes,
                    reinterpret_cast<uint8_t*>(src)+i*rows.ubPitchBytes,rows.rowBytes);
    }
}
struct WorkRange {uint32_t start,count;};
struct Region {
    uint32_t offset;
    GM_ADDR get_ptr(GM_ADDR p) const {return p+offset;}
};
struct Case {
    static constexpr uint32_t Capacity=8,Servers=3,Stride=Capacity+1,DISPATCH_RELAY_QUEUE_ENTRY_BYTES=32;
    uint32_t m_=6,serverId_=1;
    bool serverDedup=true;
    alignas(8) uint64_t window[2*Stride*Servers]{};
    alignas(32) uint8_t queues[Servers*(ALIGN_32+Capacity*DISPATCH_RELAY_QUEUE_ENTRY_BYTES)]{};
    struct {GM_ADDR base;struct {Region sourceReady,dispatchFlag;} layout;} comm{
        reinterpret_cast<GM_ADDR>(window),{Region{0},Region{Stride*Servers*sizeof(uint64_t)}}};
    struct {GM_ADDR base;struct {struct {Region relaySendQueue;} urma;} layout;} work{queues,{{{0}}}};
    struct {struct {uint32_t numMaxTokensPerRank=Capacity;} compute;} tiling;
    struct {decltype(comm)& communication;decltype(work)& workspace;decltype(tiling)* tilingData;}
        params_{comm,work,&tiling};
    bool IsCombineServerDedupEnabled() const {return serverDedup;}
    uint64_t RelayFlagOffset(uint32_t server,uint32_t token) const {return (server*Stride+token)*sizeof(uint64_t);}
    uint64_t DispatchRelayQueueServerOffset(uint32_t server) const {
        return server*(ALIGN_32+m_*DISPATCH_RELAY_QUEUE_ENTRY_BYTES);
    }
    uint32_t TransposeBitmaps(const LocalTensor<uint32_t>& masks,const LocalTensor<uint32_t>& transposedMasks,
                              uint32_t tokenCount)
"""
            + body(dispatch, "::TransposeDispatchRelayBitmaps(")
            + r"""
    void Publish(const LocalTensor<uint32_t>& masks,const WorkRange& tokenRange,const WorkRange& serverRange,
                 uint32_t rowElements)
"""
            + body(dispatch, "::PublishDispatchRelayBitmaps(")
            + r"""
    void EnqueueDispatchRelayToken(uint32_t targetServer,uint32_t tokenIdx)
"""
            + body(dispatch, "::EnqueueDispatchRelayToken(")
            + r"""
    void EnqueueCandidates(const LocalTensor<uint32_t>& masks,const WorkRange& tokenRange,const WorkRange& serverRange)
"""
            + body(dispatch, "::EnqueueDispatchRelayCandidates(")
            + r"""
    void PublishLocalDispatchRelayCount()
"""
            + body(dispatch, "::PublishLocalDispatchRelayCount()")
            + r"""
    uint64_t DispatchBulkReadyFlagOffset(uint32_t sourceServer) const
"""
            + body(dispatch, "::DispatchBulkReadyFlagOffset(")
            + r"""
    GM_ADDR PrepareDispatchReadySource(uint32_t targetServer)
"""
            + body(dispatch, "::PrepareDispatchReadySource(")
            + r"""
    uint32_t QueueToken(uint32_t server,uint32_t slot) const {
        return *reinterpret_cast<const uint32_t*>(queues+DispatchRelayQueueServerOffset(server)+
                                                  ALIGN_32+slot*DISPATCH_RELAY_QUEUE_ENTRY_BYTES);
    }
};
int main() {
    Case c;
    constexpr uint32_t width=64;
    uint32_t masks[2][8*width]{},transposed[8*width]{};
    masks[0][width+1]=0x80000000U;
    masks[0][2*width]=3U;
    masks[1][1]=5U;
    masks[1][2*width]=16U;
    c.window[Case::Stride+1]=uint64_t{0x1234}<<32;
    for(uint32_t batch=0;batch<2;++batch){
        const uint32_t rowElements=c.TransposeBitmaps({masks[batch]},{transposed},3);
        assert(rowElements==8);
        c.Publish({transposed},{batch*3,3},{0,3},rowElements);
        c.EnqueueCandidates({masks[batch]},{batch*3,3},{0,3});
    }
    assert(c.QueueToken(0,0)==2&&c.QueueToken(0,1)==5);
    assert(c.QueueToken(1,0)==1&&c.QueueToken(1,1)==3);
    assert(c.window[Case::Stride+1]==((uint64_t{0x1234}<<32)|0x80000000U));
    assert(c.window[4*Case::Stride+1]==0x80000000U);
    assert(c.window[4*Case::Stride+3]==5U);
    for(uint32_t server:{0U,2U}) for(uint32_t token=0;token<Case::Stride;++token)
        assert(c.window[(Case::Servers+server)*Case::Stride+token]==0U);
    for(uint32_t server=0;server<Case::Servers;++server) {
        assert(c.window[server*Case::Stride+6]==0U);
        assert(c.window[server*Case::Stride+Case::Capacity]==0U);
    }
    c.PublishLocalDispatchRelayCount();
    assert(c.window[Case::Stride+Case::Capacity]==3U);
    assert(c.window[4*Case::Stride+Case::Capacity]==3U);
    auto* ready=c.PrepareDispatchReadySource(0);
    assert(ready==reinterpret_cast<GM_ADDR>(&c.window[Case::Capacity]));
    assert(*reinterpret_cast<uint64_t*>(ready)==3U);
    ready=c.PrepareDispatchReadySource(2);
    assert(ready==reinterpret_cast<GM_ADDR>(&c.window[2*Case::Stride+Case::Capacity]));
    assert(*reinterpret_cast<uint64_t*>(ready)==1U);
    c.serverDedup=false;
    ready=c.PrepareDispatchReadySource(2);
    assert(ready==c.queues+c.DispatchRelayQueueServerOffset(2));
    assert(*reinterpret_cast<uint64_t*>(ready)==1U&&barriers==3);
}
"""
        )
        self.cpp(source)

    def test_no_phased_pipeline_or_vector_prologue(self):
        for token in [
            "RunAllMoeGmm",
            "LayeredGmm1GmSequence",
            "TryWait",
            "LayeredActivationEpilogueArch92",
            "CombineTokenGroupArch92",
            "GetSubBlockIdx",
            "subBlockIdx_",
            "RunGmm1A8W4",
            "RunGmm2A8W4",
        ]:
            self.assertNotIn(token, self.a6 + self.a6_wave)
        self.assertIn("Gmm1ActivationSync sharedGmm1ActivationSync(0U", self.a6)
        self.assertIn("Gmm1ActivationSync gmm1ActivationSync(0U", self.a6)
        process = body(self.a6, "::ProcessWave(")
        self.assertLess(
            process.index("sharedGmm1ActivationSync.EndSync()"),
            process.index("ProcessSharedExpertGmm2()"),
        )
        prepare = body(self.a6, "::PrepareWithSharedExpert(")
        self.assertLess(
            prepare.index("SyncIntraArrive<PIPE_MTE3>(INPUT_PREPARE_DONE_FLAG)"),
            prepare.index("ComputeExpertMasks"),
        )
        self.assertLess(
            prepare.rindex("ProcessSharedExpertGmm1"),
            prepare.index("CrossRankSync<false>"),
        )

    def test_real_wave_traces_match_arch35_a8w4(self):
        source = WAVE_MOCK
        for name, text in [("Legacy", self.a5_wave), ("Native", self.a6_wave)]:
            source += "struct " + name + ":WaveBase { using WaveBase::WaveBase;\n"
            source += "void AdjustStartBlockIdxForNextWave(const ExpertTokenRange& wave,uint32_t waveStartBlockIdx,uint32_t gmm1EndBlockIdx)"
            source += body(text, "::AdjustStartBlockIdxForNextWave(")
            source += "void Run(ExpertLoopState& gmm1State,ExpertLoopState& gmm2State,GMMAddrInfo& gmm1AddrInfo,GMMAddrInfo& gmm2AddrInfo,int& vecSetSyncCom,ExpertTokenRange currentWave)"
            source += body(text, "::ProcessMoeExpertWaveLoop(") + "};\n"
        source += r"""
int main() {
    for(uint32_t waves:{0,1,2,3,4,7,16,32,65}) for(uint32_t groups:{1,2,4}) {
        Legacy a(waves,groups);Native b(waves,groups);
        ExpertLoopState s1,s2;GMMAddrInfo i1,i2;int unused=0;
        a.Run(s1,s2,i1,i2,unused,a.PlanTokenWave({},groups));
        b.Run(s1,s2,i1,i2,unused,b.PlanTokenWave({},groups));
        assert(a.events==b.events && a.startBlockIdx_==b.startBlockIdx_);
        int dispatches=0,calculations=0;bool sawCompute=false;
        for(const auto& e:b.events) {
            if(std::get<0>(e)=='D') { assert(!sawCompute);++dispatches; }
            if(std::get<0>(e)=='1') { sawCompute=true;++calculations; }
        }
        assert(calculations==int(waves));
        if constexpr(g_coreType==AIV) {
            assert(dispatches==int(waves==0?0:(waves==1?1:2)));
            Legacy aiv0(waves,groups);
            aiv0.subBlockIdx_=0;
            aiv0.Run(s1,s2,i1,i2,unused,aiv0.PlanTokenWave({},groups));
            for (const auto& event : aiv0.events) {
                assert(std::get<0>(event)!='D' && std::get<0>(event)!='C');
            }
        }
    }
}
"""
        for core in (0, 1):
            self.cpp(source, [f"-DCORE={core}"])

    def test_real_ready_ack_retains_backpressure_and_event_targets(self):
        sync = (
            ROOT / "op_kernel/arch35_92/common/mega_moe_gmm_epilogue_sync.h"
        ).read_text(encoding="utf-8")
        source = (
            r"""
namespace AscendC { constexpr int AIC=0,AIV=1; }
constexpr int g_coreType=CORE;
constexpr uint32_t GMM_MAX_PENDING_TILES=15;
constexpr uint16_t GMM_AIV_FLAG_OFFSET=16;
enum {PIPE_S,PIPE_FIX,PIPE_MTE2,PIPE_MTE3};
std::map<int,int> credits;
template<int P> void SyncIntraArrive(uint16_t id) { ++credits[id]; }
template<int P> void SyncIntraWait(uint16_t id) { if(credits[id]==0) throw std::runtime_error("blocked"); --credits[id]; }
"""
            + extract(sync, "enum class GmmEventPair")
            + ";\n"
            + extract(sync, "class Gmm1ActivationSync")
            + ";\n"
        )
        source += r"""
int main() {
    Gmm1ActivationSync routed(0,GmmEventPair::MOE_GMM1),shared(0,GmmEventPair::SHARED_GMM1);
    if constexpr(g_coreType==AscendC::AIC) {
        for(int i=0;i<15;++i) { routed.WaitForActivation();routed.NotifyActivation(); }
        bool blocked=false;try{routed.WaitForActivation();}catch(const std::runtime_error&){blocked=true;}
        assert(blocked && credits[8]==15 && credits[24]==0);
        --credits[8];++credits[9];routed.WaitForActivation();routed.NotifyActivation();
        credits[8]=0;credits[9]+=15;routed.EndSync();assert(credits[9]==0);
        shared.WaitForActivation();shared.NotifyActivation();assert(credits[12]==1 && credits[28]==0);
        credits[12]=0;++credits[13];shared.EndSync();
        Gmm1ActivationSync legacy(1);legacy.WaitForActivation();legacy.NotifyActivation();
        assert(credits[24]==1);credits[24]=0;++credits[25];legacy.EndSync();
    } else {
        ++credits[8];routed.WaitForGmm1();routed.NotifyGmm1();assert(credits[8]==0 && credits[9]==1);
        ++credits[12];shared.WaitForGmm1();shared.NotifyGmm1();assert(credits[12]==0 && credits[13]==1);
    }
}
"""
        for core in (0, 1):
            self.cpp(source, [f"-DCORE={core}"])

    def test_shared_and_routed_gmm1_use_gm_with_scoped_or_persistent_context(self):
        stage = (
            ROOT / "op_kernel/arch92/stage/mega_moe_gmm1_activation_arch92.h"
        ).read_text(encoding="utf-8")
        source = (
            NATIVE_MOCK
            + r"""
template<bool UsePersistentMmad,typename BlockMmad,typename ElementC,typename MakeLayoutC,bool IsShared=false,
         bool IsWaveFlagGrained=false,typename Scheduler,typename Config,typename ActivationQuantOp,typename LayoutBundle>
"""
            + extract(stage, "__aicore__ inline void Gmm1ExecGeneric(")
        )
        source += r"""
template<bool Persistent,bool Shared> void run(bool prefetch,BlockMmadContext<int>* context=nullptr) {
    Scheduler scheduler;Params p;Sync sync;GMMAddrInfo info{};info.gmm1ActivationSync=&sync;
    Config config{prefetch,1};Layouts layouts;int epi=0;
    Gmm1ExecGeneric<Persistent,int,uint16_t,int,Shared,!Shared>(scheduler,p,info,config,0,2,epi,0,0,context,false,layouts);
    assert(live==(context?1:0));
    if constexpr(Persistent&&g_coreType==AIC) assert(lastContext==context);
    if(!prefetch&&g_coreType==AIV) assert(sync.waits==2&&sync.acks==2);
}
int main() {
    run<false,true>(false);run<false,false>(false);run<false,false>(true);
    if constexpr(g_coreType==AIC) {
        BlockMmadContext<int> context;
        run<true,true>(false,&context);
        assert(gmMmad==4);
    } else {
        run<true,true>(false);
        assert(gmAct==6&&prefetchAct==1);
    }
}
"""
        for core in (0, 1):
            self.cpp(source, [f"-DCORE={core}"])

    def test_native_w4_resolver_keeps_legacy_default(self):
        check = (
            ROOT / "op_host/op_tiling/arch35_92/mega_moe_tiling_check.h"
        ).read_text(encoding="utf-8")
        layout = ROOT / "op_kernel/arch35_92/common/mega_moe_gm_layout.h"
        source = (
            '\n#include "'
            + layout.as_posix()
            + '"\nusing namespace MegaMoeImpl;\n'
            + r"""
namespace ge {
enum DataType {DT_BF16,DT_FLOAT8_E5M2,DT_FLOAT8_E4M3FN,DT_FLOAT4_E2M1};
enum Format {FORMAT_ND=2,FORMAT_FRACTAL_NZ=29,FORMAT_FRACTAL_NZ_C0_32=51};
}
"""
            + extract(check, "static uint8_t ResolveExpertGmmMode(")
        )
        source += r"""
int main() {
    using namespace ge;
    assert(ResolveExpertGmmMode(DT_FLOAT8_E4M3FN,DT_FLOAT4_E2M1,FORMAT_FRACTAL_NZ_C0_32,FORMAT_FRACTAL_NZ_C0_32)==GMM_MODE_A8W4_NZ);
    assert(ResolveExpertGmmMode(DT_FLOAT8_E4M3FN,DT_FLOAT4_E2M1,FORMAT_FRACTAL_NZ,FORMAT_FRACTAL_NZ)==255);
    assert(ResolveExpertGmmMode(DT_FLOAT8_E4M3FN,DT_FLOAT4_E2M1,FORMAT_FRACTAL_NZ,FORMAT_FRACTAL_NZ,true)==GMM_MODE_A8W4_NZ);
    assert(ResolveExpertGmmMode(DT_FLOAT4_E2M1,DT_FLOAT4_E2M1,FORMAT_ND,FORMAT_FRACTAL_NZ,true)==GMM_MODE_A4W4_ND);
    assert(ResolveExpertGmmMode(DT_FLOAT4_E2M1,DT_FLOAT4_E2M1,FORMAT_FRACTAL_NZ,FORMAT_FRACTAL_NZ,true)==GMM_MODE_A4W4_NZ);
    assert(ResolveExpertGmmMode(DT_FLOAT8_E5M2,DT_FLOAT4_E2M1,FORMAT_FRACTAL_NZ,FORMAT_FRACTAL_NZ,true)==255);
    assert(ResolveExpertGmmMode(DT_FLOAT8_E4M3FN,DT_FLOAT4_E2M1,FORMAT_FRACTAL_NZ_C0_32,FORMAT_FRACTAL_NZ_C0_32,true)==255);
    assert(ResolveExpertGmmMode(DT_FLOAT8_E5M2,DT_FLOAT8_E5M2,FORMAT_ND,FORMAT_ND)==GMM_MODE_A8W8_ND);
}
"""
        self.cpp(source)

    def test_controller_wrappers_bind_offsets_and_native_contexts(self):
        epilogue = (
            ROOT
            / "op_kernel/arch35_92/blaze/epilogue/mega_moe_block_epilogue_activation_mx_quant.h"
        ).read_text(encoding="utf-8")
        block_coord = re.search(r"using BlockCoord = Coord<[^;]+>;", epilogue)
        self.assertIsNotNone(block_coord)
        gmm = (ARCH92.parent / "stage/layered/mega_moe_layered_gmm_arch92.h").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("UpdateGlobalAddr(", body(gmm, "::UpdateGmm1GlobalBuffer("))
        source = (
            r"""
enum {AIC=0,AIV=1,FORMAT_ND=2};constexpr int g_coreType=CORE;
constexpr uint32_t GMM1_TILE_M=256,MXFP_DIVISOR_SIZE=64,MXFP_MULTI_BASE_SIZE=2;
constexpr int N_VALUE=1;
template<int I,typename T>auto Get(const T& t){return std::get<I>(t);}
namespace Ops {namespace Base {
template<typename T,typename U>auto CeilDiv(T a,U b){return (a+b-1)/b;}
}}
struct Fp4{};struct Fp8{};struct Bf16{};using bfloat16_t=Bf16;
struct ExpertLoopState{uint64_t globalTokenStartIndex=768;std::tuple<int64_t,int64_t,int64_t> problemShape{512,2048,7168};};
struct GMMAddrInfo{};
struct StridedAConfig{uint32_t row,scale;};struct BlockJobContext{uint32_t blockIdx,blockNum;};
template<typename... T>using Coord=std::tuple<T...>;
struct Epilogue{
    EPILOGUE_BLOCK_COORD
    BlockCoord offset{};
    void UpdateGlobalAddr(const BlockCoord& v){offset=v;}
};
struct Tiling{bool isTopkWeightsPrefetch=true;int combineQuantMode=3;};
struct Params{Tiling* tilingData;};
int calls1=0,calls2=0;
template<bool Persistent,typename A,typename Out,typename B,typename C,typename SA,typename SB,bool Nz,uint32_t Tile,bool Shared,bool Wave,typename E,typename... Layout>
void RunGmm1Generic(bool prefetch,E&,const Params&,const std::tuple<int64_t,int64_t,int64_t>&,const GMMAddrInfo&,
                   uint32_t&,BlockJobContext job,uint32_t rows,uint32_t expert,uint32_t divisor,
                   void* pointer,bool bypass,const Layout&... layout){
    static_assert(!Persistent&&Wave==!Shared&&Tile==256&&std::is_same_v<Out,Fp8>);
    assert(!pointer);
    assert(prefetch==!Shared&&job.blockIdx==3&&job.blockNum==64&&expert==2&&divisor==2&&!bypass);
    assert(rows==uint32_t(Shared?768:1280));
    if constexpr(Shared){static_assert(sizeof...(Layout)==1);auto stride=std::get<0>(std::tie(layout...));
        assert(stride.row==uint32_t(8192*(std::is_same_v<A,Fp4>?2:1))&&stride.scale==8192);}
    else static_assert(sizeof...(Layout)==0);
    ++calls1;
}
template<bool Persistent,typename A,typename B,typename C,typename SA,typename SB,bool Nz,bool Layered,uint32_t Tile,bool Shared,bool Wave,bool Notify>
void RunGmm2Generic(int quant,bool prefetch,const std::tuple<int64_t,int64_t,int64_t>&,const GMMAddrInfo&,
                   uint32_t&,BlockJobContext job,void* pointer,bool bypass,uint32_t row,void* sync,uint32_t divisor,uint32_t count){
    static_assert(!Persistent&&std::is_same_v<A,Fp8>&&Layered&&Wave==!Shared&&!Notify&&Tile==256);
    assert(!pointer);
    assert(quant==3&&prefetch&&job.blockIdx==3&&job.blockNum==64&&!bypass&&!sync&&divisor==2&&count==1792);
    assert(row==uint32_t(Shared?0:512));++calls2;
}
template<typename Q,typename W,int F1,int F2>struct Controller {
    using QuantOutType=Q;using Weight1Type=W;using QuantScaleOutType=uint8_t;using ActivationQuantOutType=Fp8;
    static constexpr int Weight1Format=F1,Weight2Format=F2,C_ELEMS_PER_BYTE=1;
    static constexpr uint32_t A_ELEMS_PER_BYTE=std::is_same_v<Q,Fp4>?2:1;
    uint32_t blockIdx_=3,blockNum_=64,startBlockIdx_=0,activationOutputDivisor_=2,relayRecordBytes_=8192;
    Epilogue sharedEpilogueOp_,epilogueOp_;Tiling tiling;Params params_{&tiling};
    template<bool IsShared,typename Epilogue,typename... LayoutArgs>
    void RunGmm1WithEpilogue(Epilogue& epilogue,const GMMAddrInfo& info,const ExpertLoopState& state,uint32_t expertIdx,
                           uint32_t rowOffsetInExpert,const LayoutArgs&... layoutArgs)
"""
            + body(self.a6, "::RunGmm1WithEpilogue(")
            + r"""
    template<bool IsShared>void GroupMatmulWithActivationQuant(const GMMAddrInfo& gmmAddrInfo,const ExpertLoopState& state,
                                uint32_t expertIdx,uint32_t rowOffsetInExpert)
"""
            + body(self.a6, "::GroupMatmulWithActivationQuant(")
            + r"""
    template<bool IsShared>void GroupMatmulWithCombine(const GMMAddrInfo& info,const ExpertLoopState& state,
                                                      uint32_t expertTokenCount,uint32_t rowOffsetInExpert)
"""
            + body(self.a6, "::GroupMatmulWithCombine(")
            + r"""
};
template<typename Q,typename W,int F1,int F2>void run(){
    Controller<Q,W,F1,F2> c;ExpertLoopState state;GMMAddrInfo info;
    c.template GroupMatmulWithActivationQuant<true>(info,state,2,0);
    c.template GroupMatmulWithActivationQuant<false>(info,state,2,512);
    if constexpr(g_coreType==AIV){
        assert((c.sharedEpilogueOp_.offset==Epilogue::BlockCoord{768*1024,768*32}));
        assert((c.epilogueOp_.offset==Epilogue::BlockCoord{1280*1024,1280*32}));
    }
    c.template GroupMatmulWithCombine<true>(info,state,1792,0);
    c.template GroupMatmulWithCombine<false>(info,state,1792,512);
}
int main(){run<Fp4,Fp4,2,29>();run<Fp8,Fp4,29,29>();run<Fp8,Fp8,2,2>();assert(calls1==6&&calls2==6);}
"""
        )
        source = "#include <type_traits>\n" + source.replace(
            "EPILOGUE_BLOCK_COORD", block_coord.group()
        )
        for core in (0, 1):
            self.cpp(source, [f"-DCORE={core}"])

    def test_native_weight_rows_are_aligned(self):
        definitions = (ROOT / "op_host/mega_moe_def.cpp").read_text(encoding="utf-8")
        self.assertIn(
            "DefineArch92WeightSignatures(aicore_config_arch92);", definitions
        )
        source = (
            r"""
namespace ge { enum DataType {DT_FLOAT8_E5M2,DT_FLOAT8_E4M3FN,DT_FLOAT4_E2M1};
enum Format {FORMAT_ND,FORMAT_FRACTAL_NZ,FORMAT_FRACTAL_NZ_C0_32}; }
constexpr int DYNAMIC=0;
struct Row {std::vector<ge::DataType> types;std::vector<ge::Format> formats;};
struct OpAICoreConfig {
    std::map<std::string,Row> rows;
    struct Proxy {
        Row& row;Proxy& ParamType(int){return *this;}
        Proxy& DataType(const std::vector<ge::DataType>& list){row.types=list;return *this;}
        Proxy& Format(const std::vector<ge::Format>& list){row.formats=list;return *this;}
        Proxy& IgnoreContiguous(){return *this;}
    };
    Proxy Input(const char* name){return {rows[name]};}
};
"""
            + extract(definitions, "static void DefineArch92WeightSignatures(")
            + r"""
int main() {
    OpAICoreConfig aicore_config_arch92;
    DefineArch92WeightSignatures(aicore_config_arch92);
    auto& rows=aicore_config_arch92.rows;
    for(const auto& pair:rows) {
        assert(pair.second.types.size()==pair.second.formats.size());
        assert(pair.second.types.size()==rows["weight1"].types.size());
        assert(pair.second.types.size()==25);
    }
    for(const auto& pair:{std::make_pair("weight1","weight2"),std::make_pair("shared_weight1","shared_weight2")}) {
        const auto& w1=rows[pair.first];
        const auto& w2=rows[pair.second];
        int fp4Nd=0,fp4Nz=0;
        for(size_t i=0;i<w1.types.size();++i) {
            assert(w1.types[i]==w2.types[i]);
            if(w1.types[i]!=ge::DT_FLOAT4_E2M1) continue;
            assert(w2.formats[i]==ge::FORMAT_FRACTAL_NZ);
            fp4Nd+=w1.formats[i]==ge::FORMAT_ND;
            fp4Nz+=w1.formats[i]==ge::FORMAT_FRACTAL_NZ;
        }
        assert(fp4Nd==5&&fp4Nz==5);
    }
}
"""
        )
        self.cpp(source)
        self.cpp(source, ["-DENABLE_FORMAT_NZ_C0_32"])

    def test_layered_mask_mutexes_are_balanced(self):
        dispatch = (
            ROOT / "op_kernel/arch35_92/stage/layered/mega_moe_layered_dispatch.h"
        ).read_text(encoding="utf-8")
        source = (
            r"""
enum {PIPE_V,PIPE_MTE2,PIPE_MTE3,PIPE_S,SYNC_EVENT_ID1=1,SYNC_EVENT_ID3=3};
constexpr uint32_t ALIGN_32=32,DOUBLE_BUFFER=2;
std::map<std::tuple<int,int,int>,int> events;
template<int P,int C> void PipeNotify(uint16_t id) { assert(id<8);++events[{P,C,id}]; }
template<int P,int C> void PipeWait(uint16_t id) { auto& c=events[{P,C,id}];assert(c>0);--c; }
template<int P,int C,int I> void PipeFence() { PipeNotify<P,C>(I);PipeWait<P,C>(I); }
std::map<uint8_t,int> locks;
void asc_lock(int pipe,uint8_t id){assert(id<2&&locks.emplace(id,pipe).second);}
void asc_unlock(int pipe,uint8_t id){assert(locks.at(id)==pipe);locks.erase(id);}
namespace Ops { namespace Base {
template<typename T,typename U> auto CeilDiv(T a,U b){return (a+b-1)/b;}
template<typename T,typename U> auto CeilAlign(T a,U b){return CeilDiv(a,b)*b;}
} }
enum class TPosition {VECCALC};
template<typename T> struct LocalTensor { LocalTensor(TPosition,uint64_t,uint32_t){} };
template<typename T> void Duplicate(LocalTensor<T>,T,uint32_t){}
template<typename... T> void CopyGmToUb(T&&...){}
template<typename... T> void CopyUbToGm(T&&...){}
int calculations=0,countCopies=0;
void ComputeExpertMaskVF(int32_t*,uint8_t*,int32_t*,uint32_t,uint32_t){++calculations;}
struct WorkRange {uint32_t start,count;};
struct Case {
    uint32_t totalRounds_,roundSendTotalNum_=512,roundCompareCount_=512,roundMaskSlotSize_=64,maskAlignSize_;
    uint64_t sendTotalNum_;
    int32_t counts[128]{},ids[512]{},input[4096]{};
    uint8_t masks[2][64]{},gm[16*512]{};
    int32_t *sendMaskCounts_=counts,*topkIdsTensor_=ids;
    uint8_t* sendMaskTensor_[2]{masks[0],masks[1]};
    struct {uint8_t* expertIdxGmAddr;} params_{reinterpret_cast<uint8_t*>(input)};
    Case(uint32_t rounds):totalRounds_(rounds),maskAlignSize_(rounds*64),sendTotalNum_(rounds*512){}
    uint8_t* GetSendMaskSlot(uint32_t expert){return gm+expert*512;}
    void CopyExpertMaskCounts(WorkRange){++countCopies;}
    void Run(const WorkRange& expertRange)
"""
            + body(dispatch, "::ComputeExpertMasks(")
            + r"""
};
int main(){
    for(uint32_t rounds:{0,1,3,7})for(uint32_t experts:{1,2,5}){
        Case c(rounds);events.clear();calculations=countCopies=0;
        c.Run({3,experts});c.Run({3,experts});
        assert(calculations==int(rounds*experts*2)&&countCopies==2&&locks.empty());
        for(auto e:events)assert(e.second==0);
    }
}
"""
        )
        self.cpp(source)

    def test_unpermute_selects_path_and_balances_sync(self):
        for arch, entry in ((35, ARCH35), (92, ARCH92)):
            with self.subTest(arch=arch):
                stage = (
                    entry.parent
                    / f"stage/layered/mega_moe_layered_unpermute_arch{arch}.h"
                )
                self.check_unpermute(stage.read_text(encoding="utf-8"))

    def check_unpermute(self, stage):
        source = (
            r"""
using bfloat16_t=uint16_t;
using GM_ADDR=uint8_t*;
using TEventID=uint16_t;
enum {PIPE_V,PIPE_MTE2,PIPE_MTE3,PIPE_S,SYNC_EVENT_ID3=3};
std::map<std::tuple<int,int,int>,int> events;
std::map<uint8_t,int> locks;
template<int P,int C> void PipeNotify(uint16_t id){assert(id<8);++events[{P,C,id}];}
template<int P,int C> void PipeWait(uint16_t id){auto& c=events[{P,C,id}];assert(c>0);--c;}
template<int P,int C,int I> void PipeFence(){PipeNotify<P,C>(I);PipeWait<P,C>(I);}
void asc_lock(int pipe,uint8_t id){assert(locks.emplace(id,pipe).second);}
void asc_unlock(int pipe,uint8_t id){assert(locks.at(id)==pipe);locks.erase(id);}
namespace Std{template<typename T,typename U>using IsSame=std::is_same<T,U>;}
enum class TPosition{VECCALC};enum class RoundMode{CAST_NONE,CAST_RINT};
template<typename T>struct LocalTensor{LocalTensor(TPosition,uint64_t,uint32_t){}};
template<typename T,typename U>void Cast(T,U,RoundMode,uint32_t){}
int outputCopies=0,weightCopies=0;
template<typename... T>void CopyGmToUb(T&&...){++weightCopies;}
template<typename... T>void CopyUbToGm(T&&...){++outputCopies;}
struct WorkRange{uint32_t start,count;};struct Job{uint32_t jobIndex,totalJobs;};
WorkRange GetBalancedWorkRange(uint32_t n,Job j){
    uint32_t base=n/j.totalJobs,extra=n%j.totalJobs;
    return {j.jobIndex*base+std::min(j.jobIndex,extra),base+uint32_t(j.jobIndex<extra)};
}
struct MegaMoeUnpermuteBufferConfig{int tokensPerBatch=3,inputBufferCount=2;uint32_t bf16SlotElementCount=64,fp32SlotElementCount=64;};
template<typename W>struct Case {
    using TopkWeightsType=W;
    uint32_t m_,k_=64,topK_=3,sharedExpertNum_=2,aivCoreIdx_,blockAivNum_;
    struct Tiling{bool isTopkWeightsPrefetch=false;} tiling;
    std::vector<W> weights;
    std::vector<bfloat16_t> output;
    bfloat16_t data[64]{};float accum[64]{},weightUb[512]{};
    bfloat16_t* dataResTensor_=data;float *dataResFp32Tensor_=accum,*topKWeightsTensor_=weightUb;
    uint32_t topKWeightsTempAddr_=64;
    MegaMoeUnpermuteBufferConfig cfg;
    bool serverDedup=false;
    int bufferInits=0,serverUnpermutes=0;
    struct Buffer{template<typename T>T* get_ptr(GM_ADDR p){return reinterpret_cast<T*>(p);}};
    struct { Tiling* tilingData;GM_ADDR probsGmAddr,y2GmAddr;
        struct {struct {Buffer combineSend;} layout;GM_ADDR base;} communication;
    } params_;
    Case(uint32_t n,uint32_t jobs,uint32_t index,bool prefetch):m_(n),aivCoreIdx_(index),blockAivNum_(jobs),
        weights(std::max(n,1U)*3),output(std::max(n,1U)*64){
        tiling.isTopkWeightsPrefetch=prefetch;
        params_={&tiling,reinterpret_cast<GM_ADDR>(weights.data()),reinterpret_cast<GM_ADDR>(output.data()),{{},reinterpret_cast<GM_ADDR>(output.data())}};
    }
    void input(uint32_t& index,const MegaMoeUnpermuteBufferConfig& cfg){
        PipeWait<PIPE_V,PIPE_MTE2>(index);PipeNotify<PIPE_MTE2,PIPE_V>(index);PipeWait<PIPE_MTE2,PIPE_V>(index);
        PipeNotify<PIPE_V,PIPE_MTE2>(index);index=(index+1)%cfg.inputBufferCount;
    }
    void AccumulateUnpermuteExperts(bfloat16_t*,int32_t,int32_t,const MegaMoeUnpermuteBufferConfig& cfg,uint32_t& index){
        for(uint32_t i=0;i<topK_;++i)input(index,cfg);
    }
    void UnpermuteSharedExpert(int32_t,uint32_t&,const MegaMoeUnpermuteBufferConfig& cfg,uint32_t& index){
        for(uint32_t i=0;i<sharedExpertNum_;++i)input(index,cfg);
    }
    bool IsCombineServerDedupEnabled() const { return serverDedup; }
    MegaMoeUnpermuteBufferConfig UnpermuteBuffInit(){++bufferInits;return cfg;}
    void UnpermuteServerCombine(){++serverUnpermutes;}
    void UnpermuteTokenChunk(const WorkRange& tokenRange,const MegaMoeUnpermuteBufferConfig& bufferConfig,
                            uint32_t& inputBufferIdx,uint32_t& readySharedGroup)
"""
            + body(stage, "::UnpermuteTokenChunk(")
            + r"""
    void Unpermute()
"""
            + body(stage, "::Unpermute(")
            + r"""
};
template<typename W>void run(){
    for(uint32_t n:{0,1,7,32,97})for(uint32_t jobs:{1,64})for(uint32_t slots:{2,3,6})for(bool prefetch:{false,true}){
        Case<W> c(n,jobs,jobs==64?63:0,prefetch);c.cfg.inputBufferCount=slots;
        auto count=GetBalancedWorkRange(n,{c.aivCoreIdx_,jobs}).count;
        events.clear();outputCopies=weightCopies=0;c.Unpermute();c.Unpermute();
        assert(c.bufferInits==2&&c.serverUnpermutes==0&&locks.empty());
        assert(outputCopies==int(count*2));
        assert(weightCopies==int(prefetch?0:((count+2)/3)*2));
        for(auto e:events)assert(e.second==0);
        events.clear();outputCopies=weightCopies=0;c.serverDedup=true;c.Unpermute();
        assert(c.bufferInits==2&&c.serverUnpermutes==1);
        assert(events.empty()&&locks.empty()&&outputCopies==0&&weightCopies==0);
    }
}
int main(){run<float>();run<bfloat16_t>();}
"""
        )
        self.cpp(source)


if __name__ == "__main__":
    unittest.main()
