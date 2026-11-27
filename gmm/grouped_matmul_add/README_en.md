# GroupedMatmulAdd

English | [简体中文](./README.md)

## Product Support

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: Splits x and weight along K, multiplies each pair of groups, and adds each result to the corresponding group of yRef in place.

  - For group $i$, the matrices have shapes $[M,k_i]$ and $[k_i,N]$ after applying transposeX. The group size $k_i$ may vary. Each result has shape $[M,N]$.
- Calculation formula:

  $$
  yRef_i=x_i\times weight_i + y_i
  $$

## Parameter Description

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter</th>
      <th style="white-space: nowrap">Input/Output/Attribute</th>
      <th style="white-space: nowrap">Description</th>
      <th style="white-space: nowrap">Data Type</th>
      <th style="white-space: nowrap">Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">x</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Input x in the formula.</td>
      <td style="white-space: nowrap">FLOAT16, BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">weight</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">weight in the formula.</td>
      <td style="white-space: nowrap">FLOAT16, BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupList</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Cumulative K-axis group sizes or individual group sizes, selected by groupListType.</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">y</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Matrix holding the initial values for each group before accumulation.</td>
      <td style="white-space: nowrap">FLOAT32. Atlas A2/A3 also support FLOAT16 and BFLOAT16 when the type matches x.</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">transposeX</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Specifies whether the x matrix is transposed.</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">transposeWeight</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Specifies whether the weight matrix is transposed.</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupType</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Indicates the grouping type.</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupListType</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Indicates the format of the grouping groupList.</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">y</td>
      <td style="white-space: nowrap">Output</td>
      <td style="white-space: nowrap">Matrix after accumulation. Each group’s result is the sum of its matrix product and its initial value.</td>
      <td style="white-space: nowrap">Same as input y</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
  </tbody>
</table>

In the V1 and V2 ACLNN interfaces, input `y` and output `y` correspond to the same parameter, `yRef`. It holds the initial values before the call and the results after the call.

## Constraint Description

- For each group of tensors in x and weight, the size of each dimension after 32-byte alignment must be less than the maximum int32 value 2147483647.
- x and weight must both be FLOAT16 or both BFLOAT16. y supports FLOAT32; on Atlas A2/A3, y may also have the same FLOAT16 or BFLOAT16 type as x, and yRef has the same type as y. ACLNN requires contiguous yRef. For a 16-bit y, its view offset must be a non-negative multiple of 16 elements, and weight's N dimension must be a multiple of 16. The matmul result and y are added in FLOAT32 and converted to y's type.
- Both ACLNN interfaces accept true and false for transposeX. On Ascend 950PR/950DT, ACLNN prepares the [K, M] layout required by the underlying operator when transposeX is false. These products require FLOAT32 y. transposeWeight must be false.
- On Atlas A2/A3, the last dimensions of x and weight must be less than 65536. When x is not transposed or y is 16-bit, M, K, N and the group count must fit in non-negative int32, and M × N × group count must fit in int64.
- When groupListType is 0, groupList contains cumulative K group sizes; when it is 1, groupList contains individual K group sizes. The original aclnnGroupedMatmulAdd API uses cumulative sizes, while V2 supports both formats.

## Interface Invocation

| Invocation Method | Sample | Description |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_grouped_matmul_add](examples/test_aclnn_grouped_matmul_add.cpp) | Calls the [GroupedMatmulAdd](docs/aclnnGroupedMatmulAdd_en.md) operator through the API. |
