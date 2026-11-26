# engram_gate

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

执行Engram残差门控前向计算，根据残差特征、Key和融合权重计算门控值，并将共享Value写入残差流。计算公式参见[EngramGate](../README.md#功能说明)。

## 函数原型

```python
cann_ops_transformer.engram_gate(
    hidden_states,
    kv,
    weight_fused,
    eps=1e-6,
    clamp_value=1e-6,
    save_for_backward=True
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 140px">
<col style="width: 100px">
<col style="width: 100px">
<col style="width: 480px">
<col style="width: 140px">
<col style="width: 240px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>hidden_states</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>残差特征。Eager和TorchAir图模式均支持非连续Tensor，由框架执行前转为连续Tensor。</td>
        <td>float16、bfloat16</td>
        <td>(T, 4, D)</td>
    </tr>
    <tr>
        <td>kv</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>依次存放4组Key和1组共享Value。数据类型必须与hidden_states一致。Eager和TorchAir图模式均支持非连续Tensor，由框架执行前转为连续Tensor。</td>
        <td>float16、bfloat16</td>
        <td>(T, 5×D)</td>
    </tr>
    <tr>
        <td>weight_fused</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>融合权重。Eager和TorchAir图模式均支持非连续Tensor，由框架执行前转为连续Tensor。</td>
        <td>float32</td>
        <td>(4, D)</td>
    </tr>
    <tr>
        <td>eps</td>
        <td>float</td>
        <td>可选</td>
        <td>RMS计算中的防除零参数，必须为有限正数。默认值为1e-6。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>clamp_value</td>
        <td>float</td>
        <td>可选</td>
        <td>带符号开方前的幅值下限，必须为有限正数。默认值为1e-6。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>save_for_backward</td>
        <td>bool</td>
        <td>可选</td>
        <td>是否返回反向计算所需的4个中间结果。默认值为True。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody>
</table>

## 输出说明

返回包含5项的tuple：

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 140px">
<col style="width: 100px">
<col style="width: 100px">
<col style="width: 480px">
<col style="width: 140px">
<col style="width: 240px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>output</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>门控残差输出，与hidden_states的数据类型相同。</td>
        <td>float16、bfloat16</td>
        <td>(T, 4, D)</td>
    </tr>
    <tr>
        <td>dot</td>
        <td>Optional[Tensor]</td>
        <td>可选</td>
        <td>未归一化加权点积。save_for_backward为False时返回None。</td>
        <td>float32</td>
        <td>(T, 4)</td>
    </tr>
    <tr>
        <td>gate_score</td>
        <td>Optional[Tensor]</td>
        <td>可选</td>
        <td>门控值。save_for_backward为False时返回None。</td>
        <td>float32</td>
        <td>(T, 4)</td>
    </tr>
    <tr>
        <td>rstd_x</td>
        <td>Optional[Tensor]</td>
        <td>可选</td>
        <td>hidden_states沿D轴计算得到的RMS倒数。save_for_backward为False时返回None。</td>
        <td>float32</td>
        <td>(T, 4)</td>
    </tr>
    <tr>
        <td>rstd_k</td>
        <td>Optional[Tensor]</td>
        <td>可选</td>
        <td>Key沿D轴计算得到的RMS倒数。save_for_backward为False时返回None。</td>
        <td>float32</td>
        <td>(T, 4)</td>
    </tr>
    </tbody>
</table>

## 约束说明

- 该接口支持单算子模式和图模式调用。
- 参数约束：仅当T=0时，hidden_states和kv可为空Tensor；所有输入始终须满足对应的维度规格，weight_fused必须为非空Tensor。
- 规格约束：
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - T大于等于0，hidden_states仅支持3维格式，shape为(T, 4, D)。
    - kv仅支持2维格式，shape为(T, 5×D)。
    - weight_fused仅支持2维格式，shape为(4, D)。
    - D必须为64的正整数倍。
  <!-- end id7 -->
- hidden_states、kv和weight_fused必须位于同一NPU设备。
- Eager和TorchAir图模式均支持非连续输入，由框架执行前转为连续Tensor。
- 不支持输入与输出使用同一块Device内存，不支持原地更新。

## 确定性计算

默认支持确定性计算。

## 调用示例

```python
import torch
import torch_npu
from cann_ops_transformer.ops import engram_gate

torch_npu.npu.set_device(0)

t, h, d = 2, 4, 64
hidden_states = torch.randn((t, h, d), dtype=torch.float16, device="npu")
kv = torch.randn((t, (h + 1) * d), dtype=torch.float16, device="npu")
weight_fused = torch.randn((h, d), dtype=torch.float32, device="npu")

output, dot, gate_score, rstd_x, rstd_k = engram_gate(
    hidden_states,
    kv,
    weight_fused,
)
```
