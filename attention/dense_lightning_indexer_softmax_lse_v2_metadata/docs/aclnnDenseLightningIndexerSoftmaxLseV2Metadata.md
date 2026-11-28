# aclnnDenseLightningIndexerSoftmaxLseV2Metadata

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 算子功能：该算子为AI CPU算子，是aclnnDenseLightningIndexerSoftmaxLseV2算子的前置算子。根据aclnnDenseLightningIndexerSoftmaxLseV2算子的输入shape、layout、mask和压缩比例信息，计算并输出分核切分metadata。输出结果可作为aclnnDenseLightningIndexerSoftmaxLseV2算子的metadataOptional输入，减少主算子tiling阶段对host array的访问。

  **该算子不建议单独使用，建议与aclnnDenseLightningIndexerSoftmaxLseV2算子配合使用，形成完整的工作流。**
    1. 接收主算子的shape信息，包括batchSize、maxSeqLenQ、maxSeqLenK、numHeadsQ、numHeadsK、headDim、layout和mask信息。
    2. 根据layoutQ和可选输入seqUsedQOptional计算参与运算的query token总数（totalNum），采用strided方式将任务均衡切分到可用AIC核上。
    3. 输出metadata后，后续作为aclnnDenseLightningIndexerSoftmaxLseV2算子的metadataOptional输入使用。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize"获取workspace大小，再调用"aclnnDenseLightningIndexerSoftmaxLseV2Metadata"执行计算。

``` cpp
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize(
    const aclTensor   *cuSeqLensQOptional,
    const aclTensor   *cuSeqLensKOptional,
    const aclTensor   *seqUsedQOptional,
    const aclTensor   *seqUsedKOptional,
    const aclTensor   *cmpResidualKOptional,
    int64_t            batchSize,
    int64_t            maxSeqLenQ,
    int64_t            maxSeqLenK,
    int64_t            numHeadsQ,
    int64_t            numHeadsK,
    int64_t            headDim,
    char              *layoutQ,
    char              *layoutK,
    int64_t            maskMode,
    int64_t            cmpRatio,
    const aclTensor   *metadata,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

``` cpp
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseV2Metadata(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

## aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 350px">
  <col style="width: 150px">
  <col style="width: 70px">
  <col style="width: 70px">
  <col style="width: 190px">
  <col style="width: 80px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>cuSeqLensQOptional（const aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中query的累积sequence length。<br>TND场景下必传，并可通过该入参shape推导batch。<br>第一个值固定为0。</td>
      <td>TND场景下必须传入非空Tensor。BSND场景下支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqLensKOptional（const aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中key的累积sequence length。<br>TND场景下必传。<br>第一个值固定为0。</td>
      <td>TND场景下必须传入非空Tensor。BSND场景下支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>seqUsedQOptional（const aclTensor*）</td>
      <td>输入</td>
      <td>表示每个batch段实际参与运算的query长度。</td>
      <td>非空时，totalNum为所有元素之和。shape为(0,)的一维空Tensor按未传处理，此时TND场景取cuSeqLensQOptional的最后一个元素作为totalNum。传入零维标量返回161002。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B,)或(0,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>seqUsedKOptional（const aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中key实际参与运算的sequence length。</td>
      <td>支持空Tensor。当该入参存在时，seqused_k的值不大于各batch的实际seqlen_k。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional（const aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中key的sequence length与cmpRatio相关的残差。</td>
      <td>支持空Tensor。当maskMode=3且cmpRatio>1时必须传入，且每个值必须小于cmpRatio。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batchSize（int64_t）</td>
      <td>输入</td>
      <td>表示一个batch中的序列条数B。</td>
      <td>支持非负数。TND场景可填0，并通过cuSeqLensQOptional推导。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqLenQ（int64_t）</td>
      <td>输入</td>
      <td>表示query的最大sequence length。</td>
      <td>支持非负数。BSND场景必须为正数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqLenK（int64_t）</td>
      <td>输入</td>
      <td>表示key的最大sequence length。</td>
      <td>支持非负数。BSND场景必须为正数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsQ（int64_t）</td>
      <td>输入</td>
      <td>表示query的head个数。</td>
      <td>必须为正数，取值范围[1, 128]，并且能被numHeadsK整除。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK（int64_t）</td>
      <td>输入</td>
      <td>表示key的head个数。</td>
      <td>必须为正数。当前仅支持1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim（int64_t）</td>
      <td>输入</td>
      <td>表示q/k的head dimension。</td>
      <td>必须为正数。当前仅支持128。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ（char*）</td>
      <td>输入</td>
      <td>表示query侧的排列格式。</td>
      <td>支持BSND/TND，传空指针时为BSND。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutK（char*）</td>
      <td>输入</td>
      <td>表示key侧的排列格式。</td>
      <td>支持BSND/TND，传空指针时为BSND。必须与layoutQ保持一致。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode（int64_t）</td>
      <td>输入</td>
      <td>表示sparse mask模式。</td>
      <td>0: No mask<br>3: Causal</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio（int64_t）</td>
      <td>输入</td>
      <td>表示key的压缩率。</td>
      <td>取值范围[1, 128]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata（const aclTensor*）</td>
      <td>输出</td>
      <td>表示负载均衡结果输出。</td>
      <td>输出结果作为aclnnDenseLightningIndexerSoftmaxLseV2的metadataOptional输入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(L,)。L的计算见<a href="#Metadata">Metadata输出布局</a>。</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>当前实现返回0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>metadata、workspaceSize、executor存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>metadata的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>numHeadsQ、numHeadsK、headDim不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>layoutQ与layoutK不一致，或不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>batchSize、maxSeqLenQ、maxSeqLenK不满足取值约束。</td>
    </tr>
    <tr>
      <td>maskMode不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>cmpRatio不在支持的范围之内。</td>
    </tr>
    </tbody></table>

## aclnnDenseLightningIndexerSoftmaxLseV2Metadata

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
    </colgroup>
    <thead>
        <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        </tr></thead>
    <tbody>
        <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
        </tr>
    </tbody>
    </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

  - aclnnDenseLightningIndexerSoftmaxLseV2Metadata为确定性实现，确定性计算配置不会改变其输出规则。

  - BSND场景：
    - 必传batchSize、maxSeqLenQ和maxSeqLenK参数，以获取shape信息。
  - TND场景：
    - 必传cuSeqLensQOptional和cuSeqLensKOptional参数，以获取正确shape信息。
    - 当batchSize为0时，通过cuSeqLensQOptional的shape推导batch。

<details>
<summary><a id="Mask"></a>Mask</summary>
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 165px">
        <col style="width: 625px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>maskMode</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>无mask。</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</td>
            <td>支持</td>
        </tr>
        </tbody>
    </table>
</details>

<details>
<summary><a id="特殊约束"></a>特殊约束</summary>

  - Batch取值规则
    - batchSize大于0时，B等于batchSize。TND场景下，cuSeqLensQOptional和cuSeqLensKOptional各有B + 1个元素。
    - TND场景下，batchSize为0时，B等于cuSeqLensQOptional的元素个数减1。batchSize小于0为非法输入。
    - BSND场景下，batchSize应传正数。
  - Seqlen取值规则
    - TND场景下，通过cuSeqLensQOptional和cuSeqLensKOptional计算每个batch的实际q/k长度。
    - BSND场景下，通过maxSeqLenQ和maxSeqLenK获取q/k长度。
  - layout约束
    - layoutQ必须为BSND或TND。
    - layoutK支持BSND和TND，必须与layoutQ保持一致。
  - head约束
    - numHeadsQ取值范围为[1, 128]，必须能被numHeadsK整除。
    - numHeadsK当前仅支持1。
    - headDim当前仅支持128。
  - 负载均衡约束
    - cmpRatio取值范围为[1, 128]。
    - maskMode当前仅支持0和3。
  - cmp_residual_k约束
    - 当maskMode=3且cmpRatio>1时，必须传入cmpResidualKOptional。
    - cmpResidualKOptional的每个值必须小于cmpRatio。
  - seqused约束
    - seqUsedQOptional的每个值不大于各batch的实际seqlen_q。
    - seqUsedKOptional的每个值不大于各batch的实际seqlen_k。

</details>

<details>
<summary><a id="Metadata"></a>Metadata输出布局</summary>

  调用方需申请shape为`(L,)`的一维INT32 Tensor。L表示INT32元素个数。本仓库版本按以下规则计算L。

  1. 根据调用参数计算上限`U`。TND下，`maxSeqLenQ`为正数时取该值，否则取`65536`。BSND下，`batchSize`和`maxSeqLenQ`均为正数时取两者的乘积，否则取`0`。
  2. 当`U = 0`时，`L = 64`。当`U > 0`时，`L = 136 + 8 * min(U, 65536)`。例如，TND下`maxSeqLenQ = 32`时，即使各batch实际使用的query token数都是0，也需要申请392个INT32元素。

  将这个Tensor直接传给主算子。下表说明前四个元素的含义。

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 820px">
    </colgroup>
    <thead>
        <tr>
        <th>字段</th>
        <th>index</th>
        <th>说明</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>forecore_num</td>
        <td>0</td>
        <td>前置核数。核编号从0开始，编号小于forecore_num的核为前置核。</td>
        </tr>
        <tr>
        <td>tail_core_num</td>
        <td>1</td>
        <td>尾核数。从核编号forecore_num开始，连续tail_core_num个核为尾核。</td>
        </tr>
        <tr>
        <td>b_s1_per_core</td>
        <td>2</td>
        <td>前置核每核的任务数。totalNum大于0时为ceil(totalNum / aicCoreNum)。零任务时该字段为1，无计算核使用它。</td>
        </tr>
        <tr>
        <td>b_s1_per_tail_core</td>
        <td>3</td>
        <td>尾核每核的任务数。没有尾核参与时为0。</td>
        </tr>
    </tbody>
  </table>

  每个参与运算的query token计为一个任务，N表示可用AIC核数aicCoreNum。totalNum为0时，前四个字段为[0, 0, 1, 0]，所有核均无任务。0 &lt; totalNum &lt; N时，前totalNum个核各处理1个任务，其余核空闲。totalNum ≥ N时，令r为totalNum除以N的余数。若r为0，每个核处理totalNum / N个任务。若r大于0，前r个核各处理floor(totalNum / N) + 1个任务，其余核各处理floor(totalNum / N)个任务。

</details>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参见[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

``` cpp
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse_v2_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

struct DenseLISoftmaxLseV2MetaData {
    int32_t forecore_num;
    int32_t tail_core_num;
    int32_t b_s1_per_core;
    int32_t b_s1_per_tail_core;
};

struct ScopeGuard
{
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)),
        m_isDismissed(false) {}
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    ~ScopeGuard()
    {
        if (!m_isDismissed) {
            m_exitFunc();
        }
    }

    void Dismiss()
    {
        m_isDismissed = true;
    }

    std::function<void()> m_exitFunc;
    bool m_isDismissed;
};

struct Tensor {
    void *hostAddr { nullptr };
    void *deviceAddr { nullptr };
    aclTensor *data { nullptr };
};

struct ArgScenario {
    bool hasCuSeq { true };
};

struct ArgContext {
    Tensor cuSeqLensQOptional {};
    Tensor cuSeqLensKOptional {};
    Tensor seqUsedQOptional {};
    Tensor seqUsedKOptional {};
    Tensor cmpResidualKOptional {};
    Tensor metadata {};
    int64_t batchSize { 0 };
    int64_t maxSeqLenQ { 0 };
    int64_t maxSeqLenK { 0 };
    int64_t numHeadsQ { 8 };
    int64_t numHeadsK { 1 };
    int64_t headDim { 128 };
    char *layoutQ { nullptr };
    char *layoutK { nullptr };
    int64_t maskMode { 3 };
    int64_t cmpRatio { 4 };
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclnnStatus Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclInit failed. ERROR: %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSetDevice failed. ERROR: %d", ret);
    ret = aclrtCreateStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtCreateStream failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

aclnnStatus CreateTensor(aclDataType dataType, const std::vector<int64_t> &shape, Tensor &tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);

    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);

    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void SetInt32TensorData(Tensor &tensor, const std::vector<int32_t> &hostData)
{
    auto size = hostData.size() * sizeof(int32_t);
    memcpy(tensor.hostAddr, hostData.data(), size);
    aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
}

void DestroyTensor(Tensor &tensor)
{
    if (tensor.data != nullptr) {
        aclDestroyTensor(tensor.data);
        tensor.data = nullptr;
    }
    if (tensor.deviceAddr != nullptr) {
        aclrtFree(tensor.deviceAddr);
        tensor.deviceAddr = nullptr;
    }
    if (tensor.hostAddr != nullptr) {
        aclrtFreeHost(tensor.hostAddr);
        tensor.hostAddr = nullptr;
    }
}

void DestroyArgs(ArgContext &context)
{
    DestroyTensor(context.metadata);
    DestroyTensor(context.cuSeqLensQOptional);
    DestroyTensor(context.cuSeqLensKOptional);
    DestroyTensor(context.seqUsedQOptional);
    DestroyTensor(context.seqUsedKOptional);
    DestroyTensor(context.cmpResidualKOptional);

    if (context.layoutQ != nullptr) {
        free(context.layoutQ);
        context.layoutQ = nullptr;
    }
    if (context.layoutK != nullptr) {
        free(context.layoutK);
        context.layoutK = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;

    int64_t batchSize = 1;
    context.maxSeqLenQ = 16;
    context.maxSeqLenK = 4;
    context.layoutQ = (char *)malloc(sizeof(char) * 16);
    context.layoutK = (char *)malloc(sizeof(char) * 16);
    strcpy(context.layoutQ, scenario.hasCuSeq ? "TND" : "BSND");
    strcpy(context.layoutK, scenario.hasCuSeq ? "TND" : "BSND");

    // In this example batchSize is 1 and maxSeqLenQ is 16, so both layouts require 264 INT32 elements.
    const int64_t itemBound = scenario.hasCuSeq ? context.maxSeqLenQ : batchSize * context.maxSeqLenQ;
    const int64_t metadataLength = 136 + 8 * itemBound;
    ret = CreateTensor(aclDataType::ACL_INT32, { metadataLength }, context.metadata);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create metadata failed. Error: %d", ret);

    if (scenario.hasCuSeq) {
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqLensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqLensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqLensKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqLensKOptional failed. Error: %d", ret);
        SetInt32TensorData(context.cuSeqLensQOptional, { 0, static_cast<int32_t>(context.maxSeqLenQ) });
        SetInt32TensorData(context.cuSeqLensKOptional, { 0, static_cast<int32_t>(context.maxSeqLenK) });
        context.batchSize = 0;
    } else {
        context.batchSize = batchSize;
    }

    // cmp_residual_k is required when maskMode=3 and cmpRatio>1
    if (context.maskMode == 3 && context.cmpRatio > 1) {
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize }, context.cmpResidualKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cmpResidualKOptional failed. Error: %d", ret);
        SetInt32TensorData(context.cmpResidualKOptional, std::vector<int32_t>(batchSize, 0));
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

void PrintMetadata(const DenseLISoftmaxLseV2MetaData &metadata)
{
    printf("forecore_num      : %d\n", metadata.forecore_num);
    printf("tail_core_num     : %d\n", metadata.tail_core_num);
    printf("b_s1_per_core     : %d\n", metadata.b_s1_per_core);
    printf("b_s1_per_tail_core: %d\n", metadata.b_s1_per_tail_core);
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    ArgScenario scenario {};
    scenario.hasCuSeq = true;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize(
        context.cuSeqLensQOptional.data, context.cuSeqLensKOptional.data, context.seqUsedQOptional.data,
        context.seqUsedKOptional.data, context.cmpResidualKOptional.data, context.batchSize, context.maxSeqLenQ,
        context.maxSeqLenK, context.numHeadsQ, context.numHeadsK, context.headDim, context.layoutQ,
        context.layoutK, context.maskMode, context.cmpRatio, context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize failed. ERROR: %d", ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });

    ret = aclnnDenseLightningIndexerSoftmaxLseV2Metadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnDenseLightningIndexerSoftmaxLseV2Metadata failed. ERROR: %d", ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d", ret);

    DenseLISoftmaxLseV2MetaData result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    PrintMetadata(result);
    printf("pass\n");

    return 0;
}
```

## 问题定位说明

- 关于AI CPU算子Kernel常见执行问题或异常错误，问题定位方法请参考《故障处理》中“[故障案例集>算子执行问题>AI CPU算子Kernel执行报错](https://www.hiascend.com/document/detail/zh/canncommercial/latest/maintenref/troubleshooting/troubleshooting_0151.html)”。
