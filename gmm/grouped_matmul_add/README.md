# GroupedMatmulAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：沿K轴对x和weight分组，将每组矩阵乘的结果累加到yRef的对应组。

  - 按transposeX处理输入后，第$i$组参与矩阵乘的形状为$[M,k_i]$和$[k_i,N]$。各组的$k_i$可以不同，输出均为$[M,N]$。
- 计算公式：

  $$
  yRef_i=x_i\times weight_i + y_i
  $$

## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">x</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的输入x。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">weight</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的weight。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupList</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">K轴各组大小的累积和或各组长度，由groupListType指定。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">y</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">累加前的矩阵，保存各组的初始值。</td>
      <td style="white-space: nowrap">FLOAT32；Atlas A2/A3 还支持与 x 同类型的 FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">transposeX</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">表示x矩阵是否转置。</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">transposeWeight</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">表示weight矩阵是否转置。</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupType</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">表示分组类型。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupListType</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">表示分组groupList格式。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">y</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">累加后的矩阵，每组结果等于该组的矩阵乘结果加上该组的初始值。</td>
      <td style="white-space: nowrap">与输入 y 相同</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
  </tbody>
</table>

在 ACLNN V1/V2 接口中，输入 `y` 和输出 `y` 对应同一个参数 `yRef`。调用前，`yRef` 保存初始值；调用后，计算结果写回 `yRef`。

## 约束说明

- x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32的最大值2147483647。
- x和weight必须同为FLOAT16或同为BFLOAT16。y支持FLOAT32；在Atlas A2/A3上，y还可与x同为FLOAT16或BFLOAT16，yRef与y类型相同。ACLNN接口要求yRef连续。16位y的视图偏移须为非负数且为16个元素的倍数，weight的N维须为16的倍数；matmul结果与y在FLOAT32中相加后转换为y的类型。
- 两个ACLNN接口的transposeX均支持true和false。在Ascend 950PR/950DT上，transposeX为false时，ACLNN将x整理为底层算子要求的[K, M]布局。该系列产品的y须为FLOAT32。transposeWeight仅支持false。
- Atlas A2/A3上，x和weight的最后一维须小于65536。使用x不转置或16位y时，M、K、N及分组数须在int32非负范围内，M × N × 分组数须在int64范围内。
- groupListType为0时，groupList为各组K长度的累积和；为1时，groupList为各组K长度。原始aclnnGroupedMatmulAdd接口使用累积和形式，V2接口支持两种形式。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_grouped_matmul_add](examples/test_aclnn_grouped_matmul_add.cpp) | 通过接口方式调用[GroupedMatmulAdd](docs/aclnnGroupedMatmulAdd.md)算子。 |
