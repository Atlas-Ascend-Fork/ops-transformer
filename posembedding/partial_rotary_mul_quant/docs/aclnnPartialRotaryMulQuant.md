# aclnnPartialRotaryMulQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    x    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    x    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    x    |
| <term>Atlas 训练系列产品</term>                              |    x     |

## 功能说明

`aclnnPartialRotaryMulQuant` 对输入 `x` 最后一维的 `[start, end)` 区间执行部分旋转位置编码（Partial RoPE），再将完整结果乘以可选的标量 `scale`，量化为 HiFloat8 输出。

计算使用 FLOAT32 中间结果。非旋转前缀 `[0, start)` 不参与 RoPE，但仍参与缩放和量化。输出 `yOut` 的 shape 与 `x` 完全一致，每个元素占用 1 字节；`yScaleOut` 为 shape 为 `[0]` 的空 FLOAT32 张量，不生成分组缩放因子。

`partialSliceOptional` 不传时按 `[0, 0]` 处理。`start == end` 表示空旋转区间，此时跳过 RoPE，输出为完整输入的缩放和量化结果 `Cast_HiFloat8(scale * float32(x))`。cos/sin 不参与计算，无须满足广播关系或末维长度约束，但仍须提供同 shape、同输入类型的四维 Tensor。

### 计算公式

记 `D = x.shape[3]`，`partialSliceOptional = [start, end]`，`L = end - start`。非空旋转区间要求 `end = D`。以下公式适用于 `L > 0`，计算先将 `x`、`cos`、`sin` 转为 FLOAT32，并记：

$$
v = x[..., start:end],\quad c = cos,\quad s = sin
$$

`cos`、`sin` 按下文的广播规则扩展至 `v` 的形状。`concat` 表示沿最后一维拼接，`*` 表示逐元素乘法。

**（1）`rotaryMode = 0`：half。** 将旋转区间分成连续的两半：

$$
a = v[..., :L/2],\quad b = v[..., L/2:]
$$

$$
r = concat(-b, a),\quad z = v * c + r * s
$$

**（2）`rotaryMode = 1`：interleave。** 对相邻的偶数、奇数位置组成的元素对执行旋转：

$$
r[..., 2j] = -v[..., 2j+1],\quad r[..., 2j+1] = v[..., 2j]
$$

$$
z = v * c + r * s,\quad 0 \le j < L/2
$$

**（3）`rotaryMode = 2`：quarter。** 将旋转区间分成连续的四段 `a`、`b`、`c_0`、`d`，每段长度为 `L/4`，分别对前两段、后两段执行 half 旋转：

$$
v = concat(a, b, c_0, d)
$$

$$
r = concat(-b, a, -d, c_0),\quad z = v * c + r * s
$$

**（4）`rotaryMode = 3`：deepseek_interleave（interleave-half）。** 将输入的偶数、奇数位置分别组成两半；`cos`、`sin` 仍按连续两半使用，输出按两半拼接：

$$
a = v[..., 0::2],\quad b = v[..., 1::2]
$$

$$
u = concat(a, b),\quad r = concat(-b, a),\quad z = u * c + r * s
$$

最后，将非旋转前缀与旋转结果拼接，再进行整体缩放和 HiFloat8 转换：

$$
t = concat(x[..., :start], z)
$$

$$
\alpha = \begin{cases}
1, & scaleOptional = nullptr \\
scaleOptional[0], & \text{传入单元素 scaleOptional}
\end{cases}
$$

$$
yOut = Cast_{HiFloat8}(\alpha * t, roundModeOptional)
$$

- `roundModeOptional = "round"`：使用 `CAST_ROUND` 转换。
- `roundModeOptional = "hybrid"`：使用 `CAST_HYBRID` 混合舍入；其中包含随机舍入行为，相同输入的多次执行结果可能不同。
- `yScaleOut` 始终为空 FLOAT32 张量。

## 函数原型

算子使用[两段式接口](../../../docs/zh/context/two_phase_api.md)：先调用 `aclnnPartialRotaryMulQuantGetWorkspaceSize` 查询 workspace 大小并获取执行器，再调用 `aclnnPartialRotaryMulQuant` 执行计算。

以下参数名与生成的 `aclnnop/aclnn_partial_rotary_mul_quant.h` 一致；其中 `scaleOptional`、`roundModeOptional`、`partialSliceOptional` 分别对应算子定义中的 `scale`、`round_mode`、`partial_slice`。

```cpp
aclnnStatus aclnnPartialRotaryMulQuantGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* cos,
    const aclTensor* sin,
    const aclTensor* scaleOptional,
    int64_t yDtype,
    int64_t quantMode,
    char* roundModeOptional,
    int64_t axis,
    int64_t rotaryMode,
    const aclIntArray* partialSliceOptional,
    const aclTensor* yOut,
    const aclTensor* yScaleOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```cpp
aclnnStatus aclnnPartialRotaryMulQuant(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## aclnnPartialRotaryMulQuantGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述与使用说明 | 数据类型 | 数据格式 | 维度或 shape | 非连续 Tensor |
| --- | --- | --- | --- | --- | --- | --- |
| x | 输入 | 待执行部分 RoPE 和量化的张量。所有维度须大于 0。 | FLOAT16、BFLOAT16、FLOAT32 | ND | 4 维 | 支持 |
| cos | 输入 | 旋转区间的余弦系数。数据类型须与 `x` 一致，shape 满足广播约束，末维长度为 `L`。 | FLOAT16、BFLOAT16、FLOAT32 | ND | 4 维 | 支持 |
| sin | 输入 | 旋转区间的正弦系数。数据类型须与 `x` 一致，shape 须与 `cos` 完全一致。 | FLOAT16、BFLOAT16、FLOAT32 | ND | 4 维 | 支持 |
| scaleOptional | 输入 | 可选的单元素缩放张量；传 `nullptr` 时缩放系数为 1。对包含非旋转前缀的完整结果相乘。 | FLOAT32 | ND | 元素总数必须为 1，推荐 `[1]` | 支持 |
| yDtype | 输入 | 输出类型标识，ascend950 仅支持 `0`（HiFloat8）。 | INT64 | - | - | - |
| quantMode | 输入 | 保留参数，HiFloat8 路径忽略该参数，建议传 `0`。 | INT64 | - | - | - |
| roundModeOptional | 输入 | 支持 `"round"`、`"hybrid"`。传 `nullptr` 时使用默认值 `"round"`。 | char* | - | - | - |
| axis | 输入 | 保留参数，HiFloat8 路径忽略该参数；旋转和量化始终作用于最后一维，建议传 `-1`。 | INT64 | - | - | - |
| rotaryMode | 输入 | 支持 `0`（half）、`1`（interleave）、`2`（quarter）、`3`（deepseek_interleave）。 | INT64 | - | - | - |
| partialSliceOptional | 输入 | 指定最后一维的旋转区间 `[start, end)`。传 `nullptr` 时默认 `[0, 0]`，仅执行整体缩放和量化。 | aclIntArray，元素为 INT64 | - | 非空指针时长度为 2 | - |
| yOut | 输出 | HiFloat8 量化结果，shape 与 `x` 完全一致。须创建 `ACL_HIFLOAT8` 类型的输出描述，按每元素 1 字节分配 Device 内存。 | HIFLOAT8 | ND | 4 维 | 不支持 |
| yScaleOut | 输出 | 空输出，须提供非空的 `aclTensor` 描述；其 shape 为 `[0]`，不需要分配数据内存。 | FLOAT32 | ND | 1 维，`[0]` | 不适用 |
| workspaceSize | 输出 | 返回 Device 侧需要申请的 workspace 字节数。 | uint64_t* | - | - | - |
| executor | 输出 | 返回算子执行器，供第二段接口调用。 | aclOpExecutor** | - | - | - |

输入非连续 Tensor 由接口进行连续化处理。`x`、`cos`、`sin`、可选 `scaleOptional` 和输出的数据内存应位于同一 Device 上。

### 返回值说明

返回 `aclnnStatus`，详见[aclnn 返回码](../../../docs/zh/context/aclnn_return_code.md)。常见错误如下，具体以接口返回值及运行日志为准。

| 返回值 | 错误码 | 常见原因 |
| --- | --- | --- |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必需的输入、输出 Tensor 描述或返回参数指针为空。`scaleOptional` 允许为空。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 输入、输出数据类型或参数组合不满足接口要求。 |
| ACLNN_ERR_INNER_TILING_ERROR | 561002 | tiling 检查失败，例如 shape、广播关系、旋转模式、切片、舍入模式或 scale 不满足约束。 |

## aclnnPartialRotaryMulQuant

### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| workspace | 输入 | Device 侧 workspace 地址；大小为 `workspaceSize`。当 `workspaceSize` 为 0 时可传 `nullptr`。 |
| workspaceSize | 输入 | 第一段接口返回的 workspace 字节数。 |
| executor | 输入 | 第一段接口返回的执行器。 |
| stream | 输入 | ACL stream，用于异步调度算子执行。读取输出或释放相关 Device 内存前，应同步 stream。 |

### 返回值说明

返回 `aclnnStatus`，详见[aclnn 返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

### 维度和数据类型约束

- `x`、`cos`、`sin`、`yOut` 均为 4 维张量。`x`、`yOut` 所有维度均大于 0；旋转区间非空时，cos/sin 所有维度也须大于 0。
- `x`、`cos`、`sin` 数据类型完全一致，支持 FLOAT16、BFLOAT16、FLOAT32，不进行隐式类型提升。
- `x.shape` 与 `yOut.shape` 完全一致，`cos.shape` 与 `sin.shape` 完全一致。
- `yOut` 数据类型必须为 HIFLOAT8；ACLNN 描述的最后一维仍为 `D`，存储大小为 `x` 元素总数乘以 1 字节。
- `yScaleOut` 数据类型必须为 FLOAT32，shape 必须为 `[0]`。不能以 `nullptr` 替代其 Tensor 描述。
- `scaleOptional` 非空时，必须为单元素 FLOAT32 张量，不支持按通道或按组的 scale。

### D 维和 Slice 约束

记 `D = x.shape[3]`，`L = end - start`：

- `0 < D <= 1024`。
- `partialSliceOptional` 若包含元素，其长度必须为 2，取值满足 `0 <= start <= end <= D`。
- `start == end` 时跳过 RoPE，对完整 `x` 缩放和量化，不要求 `D` 满足旋转模式的整除约束。
- 非空旋转区间要求 `end == D`；`rotaryMode` 为 `0`、`1`、`3` 时，`D` 和 `L` 均须被 2 整除，为 `2` 时均须被 4 整除。
- 非空旋转区间要求 `cos.shape[3] = sin.shape[3] = L`。

未提供切片信息时按 `[0, 0]` 处理，与属性默认值一致。ACLNN 调用应传 `nullptr` 或含两个元素的 `aclIntArray`；Python 封装会将 `None` 或空列表转换成 `[0, 0]`。若需对完整最后一维执行 RoPE，应显式传入 `[0, D]`。

例如，`D = 80` 时，half、interleave、deepseek_interleave 模式下 `[0, 80]`、`[16, 80]`、`[2, 80]` 均合法；quarter 模式下 `[0, 80]`、`[16, 80]` 合法，`[2, 80]` 因旋转区间长度不能被 4 整除而非法。`[0, 64]` 因 `end != D` 而非法。

### 广播和布局约束

以下广播约束仅适用于非空旋转区间。为避免混淆 BSND、BNSD、SBND 的轴含义，将 `x.shape` 的前三维统一记为 `(A, B, C)`，即 `x.shape = (A, B, C, D)`。`cos`、`sin` 可采用以下任一种形状：

| cos/sin shape | 广播方式 |
| --- | --- |
| `(A, B, C, L)` | 不广播 |
| `(1, 1, 1, L)` | 向前三维广播 |
| `(A, 1, 1, L)` | 向第 1、2 维广播 |
| `(A, B, 1, L)` | 向第 2 维广播 |
| `(1, B, 1, L)` | 向第 0、2 维广播 |
| `(1, 1, C, L)` | 向第 0、1 维广播 |
| `(A, 1, C, L)` | 向第 1 维广播 |
| `(1, B, C, L)` | 向第 0 维广播 |

上述维度索引从 0 开始。等价地，`cos`、`sin` 的前三维中，每一维只能等于 `x` 的对应维度或 1；最后一维必须等于 `L`，不能在最后一维广播。

### 属性约束

| 属性 | ascend950 取值约束 |
| --- | --- |
| yDtype | 必须为 `0`（HiFloat8）。 |
| rotaryMode | 支持 `0`、`1`、`2`、`3`，具体公式见功能说明。 |
| roundModeOptional | 支持 `"round"`、`"hybrid"`；不支持其他非空字符串。 |
| quantMode | HiFloat8 路径忽略该值，建议传 `0`。 |
| axis | HiFloat8 路径忽略该值，建议传 `-1`；不用于选择其他旋转轴。 |

## 调用示例

下面使用 FLOAT32 输入，`x.shape = (1, 2, 2, 80)`，`cos/sin.shape = (1, 1, 1, 64)`，`partialSliceOptional = [16, 80]`，展示不要求 64 对齐的 `D` 和切片起点。`cos = 1`、`sin = 0`、`scale = 0.5`，因此量化前的完整结果为 `0.5 * x`。

编译和执行流程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。使用安装的算子包提供的头文件和算子 API 库，运行前加载 CANN 及自定义算子环境。

```cpp
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_partial_rotary_mul_quant.h"

void Check(int32_t status, const char* operation)
{
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
    }
}

struct Resources {
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    aclIntArray* slice = nullptr;
    std::vector<aclTensor*> tensors;
    std::vector<void*> buffers;

    ~Resources()
    {
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        for (auto tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        if (slice != nullptr) {
            aclDestroyIntArray(slice);
        }
        for (auto buffer : buffers) {
            aclrtFree(buffer);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(0);
        }
        if (initialized) {
            aclFinalize();
        }
    }

    void* Allocate(size_t bytes)
    {
        if (bytes == 0) {
            return nullptr;
        }
        void* buffer = nullptr;
        Check(aclrtMalloc(&buffer, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
        buffers.push_back(buffer);
        return buffer;
    }

    aclTensor* CreateTensor(const std::vector<int64_t>& shape, aclDataType dtype,
                            size_t bytes, const void* host = nullptr)
    {
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size(); i > 1; --i) {
            strides[i - 2] = strides[i - 1] * shape[i - 1];
        }
        void* data = Allocate(bytes);
        if (host != nullptr && bytes != 0) {
            Check(aclrtMemcpy(data, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
        }
        auto tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                      ACL_FORMAT_ND, shape.data(), shape.size(), data);
        if (tensor == nullptr) {
            throw std::runtime_error("aclCreateTensor failed");
        }
        tensors.push_back(tensor);
        return tensor;
    }
};

int main()
{
    try {
        Resources resources;
        Check(aclInit(nullptr), "aclInit");
        resources.initialized = true;
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        resources.deviceSet = true;
        Check(aclrtCreateStream(&resources.stream), "aclrtCreateStream");

        const int64_t d = 80;
        const int64_t start = 16;
        const int64_t length = d - start;
        const size_t elements = 1 * 2 * 2 * d;
        std::vector<float> xHost(elements, 2.0f);
        std::vector<float> cosHost(length, 1.0f);
        std::vector<float> sinHost(length, 0.0f);
        float scaleHost = 0.5f;

        auto x = resources.CreateTensor({1, 2, 2, d}, ACL_FLOAT, elements * sizeof(float), xHost.data());
        auto cos = resources.CreateTensor({1, 1, 1, length}, ACL_FLOAT, length * sizeof(float), cosHost.data());
        auto sin = resources.CreateTensor({1, 1, 1, length}, ACL_FLOAT, length * sizeof(float), sinHost.data());
        auto scale = resources.CreateTensor({1}, ACL_FLOAT, sizeof(float), &scaleHost);
        auto y = resources.CreateTensor({1, 2, 2, d}, ACL_HIFLOAT8, elements);
        void* yDevice = resources.buffers.back();
        // 必须创建空 FLOAT32 Tensor 描述；数据地址允许为 nullptr。
        auto yScale = resources.CreateTensor({0}, ACL_FLOAT, 0);

        int64_t sliceValues[] = {start, d};
        resources.slice = aclCreateIntArray(sliceValues, 2);
        if (resources.slice == nullptr) {
            throw std::runtime_error("aclCreateIntArray failed");
        }
        char roundMode[] = "round";
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        Check(aclnnPartialRotaryMulQuantGetWorkspaceSize(
                  x, cos, sin, scale, 0, 0, roundMode, -1, 0, resources.slice,
                  y, yScale, &workspaceSize, &executor),
              "aclnnPartialRotaryMulQuantGetWorkspaceSize");

        void* workspace = resources.Allocate(workspaceSize);
        Check(aclnnPartialRotaryMulQuant(workspace, workspaceSize, executor, resources.stream),
              "aclnnPartialRotaryMulQuant");
        Check(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream");

        std::vector<uint8_t> yBytes(elements);
        Check(aclrtMemcpy(yBytes.data(), elements, yDevice, elements, ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy D2H");
        // 输出是 HiFloat8 编码，按字节打印；不能把 uint8_t 数值当作反量化后的浮点数。
        for (size_t i = 0; i < 8; ++i) {
            std::printf("y[%zu] HiFloat8 bits = 0x%02x\n", i, static_cast<unsigned int>(yBytes[i]));
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
```
