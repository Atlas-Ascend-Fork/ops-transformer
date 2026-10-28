# PartialRotaryMulQuant

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- **算子功能**：对输入张量`x`最后一维的`partial_slice = [start, end)`区间执行部分旋转位置编码，再将完整结果乘以可选标量`scale`，量化为HiFloat8输出`y`。非旋转前缀保留原值参与缩放和量化，`y_scale`为空FLOAT32张量。
- **计算公式**：

    记`D = x.shape[-1]`，`L = end - start`。对于非空旋转区间，要求`end = D`，并记：

    $$
    v = x[..., start:end]
    $$

    `cos`、`sin`广播至`v`的形状，`concat`表示沿最后一维拼接。

    half模式（`rotary_mode = 0`）：

    $$
    a = v[..., :L/2],\quad b = v[..., L/2:]
    $$

    $$
    z = v * cos + concat(-b, a) * sin
    $$

    interleave模式（`rotary_mode = 1`）：

    $$
    r[..., 2j] = -v[..., 2j+1],\quad r[..., 2j+1] = v[..., 2j]
    $$

    $$
    z = v * cos + r * sin,\quad 0 \le j < L/2
    $$

    quarter模式（`rotary_mode = 2`）：将`v`分为四个长度为`L/4`的连续分段`a`、`b`、`c`、`d`。

    $$
    z = v * cos + concat(-b, a, -d, c) * sin
    $$

    deepseek_interleave模式（`rotary_mode = 3`）：将偶数、奇数位置分别组成两半，输出按两半拼接。

    $$
    a = v[..., 0::2],\quad b = v[..., 1::2]
    $$

    $$
    z = concat(a, b) * cos + concat(-b, a) * sin
    $$

    拼接非旋转前缀与旋转结果：

    $$
    t = concat(x[..., :start], z)
    $$

    计算使用FLOAT32中间结果，并对完整结果缩放后转换：

    $$
    \alpha = \begin{cases}
    1, & \text{未传入scale} \\
    scale[0], & \text{传入单元素scale张量}
    \end{cases}
    $$

    $$
    y = Cast_{HiFloat8}(\alpha * t, round\_mode)
    $$

    此时`y_scale`为空FLOAT32张量，shape为`[0]`。当`start == end`时跳过旋转，令`t = float32(x)`，仍对完整输入执行缩放和量化。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1200px">
<colgroup>
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 650px">
  <col style="width: 250px">
  <col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>待执行部分旋转位置编码和量化的四维张量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>输入</td>
    <td>参与旋转计算的余弦位置编码张量，数据类型与x一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>输入</td>
    <td>参与旋转计算的正弦位置编码张量，shape及数据类型与cos一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scale</td>
    <td>输入</td>
    <td>可选的单元素缩放张量，对包含非旋转前缀的完整结果相乘，未传时为1。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>HiFloat8量化结果，shape与x一致。</td>
    <td>HIFLOAT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y_scale</td>
    <td>输出</td>
    <td>shape为[0]的空FLOAT32张量。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y_dtype</td>
    <td>属性</td>
    <td>输出量化类型，当前仅支持0，表示HiFloat8。默认值为0。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>quant_mode</td>
    <td>属性</td>
    <td>量化模式，默认值为0。保留用于接口兼容，当前忽略此参数。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>round_mode</td>
    <td>属性</td>
    <td>HiFloat8舍入模式，支持round和hybrid，空字符串按round处理。默认值为round。hybrid对应CAST_HYBRID混合舍入。</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>axis</td>
    <td>属性</td>
    <td>量化轴，默认值为-1。保留用于接口兼容，当前忽略此参数。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>rotary_mode</td>
    <td>属性</td>
    <td>旋转模式，0为half，1为interleave，2为quarter，3为deepseek_interleave。默认值为0。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>partial_slice</td>
    <td>属性</td>
    <td>最后一维的旋转区间[start, end)，默认值为[0, 0]。start等于end时跳过RoPE，仍执行缩放和量化。</td>
    <td>INT64数组</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## 约束说明

- 输入`x`、`cos`、`sin`及输出`y`均为四维ND张量，`x`、`cos`、`sin`的数据类型必须相同，`cos`和`sin`的shape必须相同。
- 记`x`的shape为`(B, S, N, D)`，`x`各维大小必须为正，且`0 < D <= 1024`。`y`的shape与`x`一致。
- `partial_slice`传入非空数组时必须包含两个整数，满足`0 <= start <= end <= D`。非空旋转区间要求`end = D`，且`cos`、`sin`最后一维大小等于`L = end - start`。
- 非空旋转区间要求`cos`、`sin`各维大小为正，前三维与`x`满足广播关系。
- ACLNN接口对非连续输入进行连续化处理。

- `y_dtype`必须为0，支持四种旋转模式。
- 非空旋转区间下，half、interleave和deepseek_interleave模式要求`D`和`L`均为2的倍数；quarter模式要求`D`和`L`均为4的倍数。不要求`D`或`start`按64对齐。
- 非空旋转区间下，`cos`、`sin`前三维的每一维必须为1或等于`x`对应维度。
- 未传入`partial_slice`或传入空数组时按`[0, 0]`处理。`start == end`时，跳过RoPE及旋转维度整除检查，`cos`、`sin`无须满足广播关系或末维长度约束，但仍须提供shape相同且数据类型与`x`一致的四维张量。
- `scale`非空时必须为单元素FLOAT32张量，不支持按通道或按组传入scale。
- `round_mode = "hybrid"`包含随机舍入行为，相同输入的多次执行结果可能不同。
- `y_scale`为shape为`[0]`的空FLOAT32张量，调用时仍需提供张量描述。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_partial_rotary_mul_quant](./examples/test_aclnn_partial_rotary_mul_quant.cpp) | 通过[aclnnPartialRotaryMulQuant](./docs/aclnnPartialRotaryMulQuant.md)接口调用PartialRotaryMulQuant算子。 |
| PyTorch API | [test_torch_partial_rotary_mul_quant](./examples/test_torch_partial_rotary_mul_quant.py) | 通过[cann_ops_transformer.partial_rotary_mul_quant](./torch_extension/partial_rotary_mul_quant.py)接口调用，返回HiFloat8编码的uint8张量及空FLOAT32张量。 |
| 图模式调用 | [算子IR](./op_graph/partial_rotary_mul_quant_proto.h) | 通过PartialRotaryMulQuant算子IR构图调用，y_dtype设为0。 |
