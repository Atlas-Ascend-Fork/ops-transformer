# aclnnEngramGate

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

- 接口功能：执行Engram残差门控前向计算，根据残差特征、Key和融合权重计算门控值，并将共享Value写入残差流。

- 计算公式：参见[EngramGate](../README.md#功能说明)。`dotOutOptional`保存未归一化加权点积`raw_dot`。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnEngramGateGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEngramGate”接口执行计算。

```cpp
aclnnStatus aclnnEngramGateGetWorkspaceSize(
  const aclTensor *hiddenStates,
  const aclTensor *kv,
  const aclTensor *weightFused,
  double          eps,
  double          clampValue,
  bool            saveForBackward,
  const aclTensor *outputOut,
  const aclTensor *dotOutOptional,
  const aclTensor *gateScoreOutOptional,
  const aclTensor *rstdXOutOptional,
  const aclTensor *rstdKOutOptional,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnEngramGate(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnEngramGateGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <td>hiddenStates（const aclTensor*）</td>
      <td>输入</td>
      <td>表示残差特征。</td>
      <td><ul><li>不能传入空指针，支持空Tensor。</li><li>T大于等于0，H固定为4，D必须为64的正整数倍。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>(T, H, D)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kv（const aclTensor*）</td>
      <td>输入</td>
      <td>表示Key和共享Value的融合输入。</td>
      <td><ul><li>不能传入空指针，支持空Tensor。</li><li>每个Token依次存放H组Key和1组共享Value。</li><li>数据类型必须与hiddenStates一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>(T, (H+1)×D)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightFused（const aclTensor*）</td>
      <td>输入</td>
      <td>表示已融合权重。</td>
      <td><ul><li>不能传入空指针。</li><li>shape必须为(H, D)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(H, D)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>eps（double）</td>
      <td>输入</td>
      <td>表示RMS计算中的防除零参数。</td>
      <td><ul><li>必须显式传入有限正数（不接受0）。</li><li>GE IR与PyTorch接口的默认值为1e-6。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clampValue（double）</td>
      <td>输入</td>
      <td>表示带符号开方前的幅值下限。</td>
      <td><ul><li>必须显式传入有限正数。</li><li>GE IR与PyTorch接口的默认值为1e-6。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>saveForBackward（bool）</td>
      <td>输入</td>
      <td>表示是否输出反向计算所需的4个中间结果。</td>
      <td><ul><li>必须显式传入。</li><li>GE IR与PyTorch接口的默认值为true。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputOut（const aclTensor*）</td>
      <td>输出</td>
      <td>表示门控残差输出。</td>
      <td><ul><li>不能传入空指针。</li><li>shape和数据类型必须与hiddenStates一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>(T, H, D)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dotOutOptional（const aclTensor*）</td>
      <td>可选输出</td>
      <td>表示未归一化加权点积raw_dot。</td>
      <td><ul><li>saveForBackward为true时必须提供。</li><li>saveForBackward为false时可传nullptr或shape为(0)的空Tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(T, H)或(0)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gateScoreOutOptional（const aclTensor*）</td>
      <td>可选输出</td>
      <td>表示门控值。</td>
      <td><ul><li>传入规则与dotOutOptional一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(T, H)或(0)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>rstdXOutOptional（const aclTensor*）</td>
      <td>可选输出</td>
      <td>表示hiddenStates沿D轴计算得到的RMS倒数。</td>
      <td><ul><li>传入规则与dotOutOptional一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(T, H)或(0)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>rstdKOutOptional（const aclTensor*）</td>
      <td>可选输出</td>
      <td>表示Key沿D轴计算得到的RMS倒数。</td>
      <td><ul><li>传入规则与dotOutOptional一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(T, H)或(0)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
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
      <td>传入的hiddenStates、kv、weightFused、outputOut或executor是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>必选输入或输出的数据类型、数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>输入或输出的shape不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnEngramGate

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnEngramGateGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- hiddenStates必须为3维Tensor，kv和weightFused必须为2维Tensor。
- T大于等于0，H固定为4，D必须为64的正整数倍。T为0时支持空Tensor；H不等于4或D不满足约束时返回错误。
- hiddenStates、kv和outputOut的数据类型必须一致，只支持FLOAT16和BFLOAT16；weightFused和4个中间结果输出的数据类型必须为FLOAT32。
- 3个输入均支持非连续Tensor，aclnn框架执行前会自动转为连续Tensor。
- outputOut及4个可选输出不支持非连续Tensor，调用者必须传入连续Tensor。
- 不支持输入与输出使用同一块Device内存，不支持原地更新。
- 确定性计算：aclnnEngramGate默认确定性实现。
- NaN和Inf按照浮点计算规则传播；输入乘法或归约发生溢出时，不保证输出为有限值。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)，完整代码参见 [test_aclnn_engram_gate.cpp](../examples/test_aclnn_engram_gate.cpp)。

```cpp
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_engram_gate.h"

namespace {
void Check(int32_t status)
{
    if (status != 0) {
        throw std::runtime_error("ACL/ACLNN failure: " + std::to_string(status));
    }
}

struct Resources {
    aclrtStream stream = nullptr;
    std::vector<void*> buffers;
    std::vector<aclTensor*> tensors;

    ~Resources()
    {
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        for (auto tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        for (auto buffer : buffers) {
            aclrtFree(buffer);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        aclrtResetDevice(0);
        aclFinalize();
    }

    void* Allocate(size_t bytes)
    {
        void* ptr = nullptr;
        Check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        buffers.push_back(ptr);
        return ptr;
    }

    aclTensor* Tensor(std::vector<int64_t> shape, aclDataType dtype, void* ptr)
    {
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size() - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }
        auto tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                      shape.data(), shape.size(), ptr);
        if (tensor == nullptr) {
            throw std::runtime_error("aclCreateTensor failed");
        }
        tensors.push_back(tensor);
        return tensor;
    }
};
} // namespace

int main()
{
    try {
        constexpr int64_t t = 2;
        constexpr int64_t h = 4;
        constexpr int64_t d = 64;
        Check(aclInit(nullptr));
        Resources resources;
        Check(aclrtSetDevice(0));
        Check(aclrtCreateStream(&resources.stream));

        std::vector<uint16_t> x(t * h * d, 0x3c00);
        std::vector<uint16_t> kv(t * (h + 1) * d, 0x3c00);
        std::vector<float> weight(h * d, 1.0F);
        auto xDevice = resources.Allocate(x.size() * sizeof(uint16_t));
        auto kvDevice = resources.Allocate(kv.size() * sizeof(uint16_t));
        auto weightDevice = resources.Allocate(weight.size() * sizeof(float));
        Check(aclrtMemcpy(xDevice, x.size() * sizeof(uint16_t), x.data(), x.size() * sizeof(uint16_t),
                          ACL_MEMCPY_HOST_TO_DEVICE));
        Check(aclrtMemcpy(kvDevice, kv.size() * sizeof(uint16_t), kv.data(), kv.size() * sizeof(uint16_t),
                          ACL_MEMCPY_HOST_TO_DEVICE));
        Check(aclrtMemcpy(weightDevice, weight.size() * sizeof(float), weight.data(), weight.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));

        auto hiddenStates = resources.Tensor({t, h, d}, ACL_FLOAT16, xDevice);
        auto kvTensor = resources.Tensor({t, (h + 1) * d}, ACL_FLOAT16, kvDevice);
        auto weightFused = resources.Tensor({h, d}, ACL_FLOAT, weightDevice);
        auto outputOut = resources.Tensor({t, h, d}, ACL_FLOAT16, resources.Allocate(t * h * d * sizeof(uint16_t)));

        std::vector<aclTensor*> stats;
        std::vector<void*> statBuffers;
        for (int i = 0; i < 4; ++i) {
            statBuffers.push_back(resources.Allocate(t * h * sizeof(float)));
            stats.push_back(resources.Tensor({t, h}, ACL_FLOAT, statBuffers.back()));
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        Check(aclnnEngramGateGetWorkspaceSize(hiddenStates, kvTensor, weightFused, 1e-6, 1e-6, true, outputOut,
                                              stats[0], stats[1], stats[2], stats[3], &workspaceSize, &executor));
        void* workspace = workspaceSize == 0 ? nullptr : resources.Allocate(workspaceSize);
        Check(aclnnEngramGate(workspace, workspaceSize, executor, resources.stream));
        Check(aclrtSynchronizeStream(resources.stream));

        std::vector<float> dot(t * h);
        Check(aclrtMemcpy(dot.data(), dot.size() * sizeof(float), statBuffers[0], dot.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));
        for (auto value : dot) {
            if (!std::isfinite(value) || std::fabs(value - static_cast<float>(d)) > 1e-5F) {
                throw std::runtime_error("dot result mismatch");
            }
        }
        std::cout << "EngramGate PASS" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
```
