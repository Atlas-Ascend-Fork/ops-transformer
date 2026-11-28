# mhc_post

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id5 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id5 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not Supported
<!-- end id6 -->

## Function

- Description:

    Implements the forward computation of the MHC Post component, used in the post-processing stage of multi-layer residual connections in Transformer models. This operator fuses the residual matrix transformation and the output state projection into a single computation, avoiding the extra overhead of multiple independent operator calls.

- Formula:

  - The core formula of the MHC Post operator is:

    $$x_{l+1} = (H_{l}^{res})^{T} \cdot x_{l} + h_{l}^{out} \cdot H_{t}^{post}$$

    Where:

    - $(H_{l}^{res})^{T} \cdot x_{l}$ represents the transposed matrix multiplication of the residual matrix on the input $x_{l}$. For the $i$th row of the output (corresponding to the $i$th head), the computation is:

      $$x_{l+1}[i] = \sum_{j=0}^{n-1} H_{l}^{res}[j, i] \cdot x_{l}[j]$$

      That is, the $H_{l}^{res}$ matrix is multiplied by $x_{l}$ in a transposed manner. $H_{l}^{res}[j, i]$ is a scalar, which is used to scalar-multiply the $j$th row of $x_{l}$ and accumulate it to the $i$th row of the output.

    - $h_{l}^{out} \cdot H_{t}^{post}$ represents the element-wise multiplication and broadcasting of the output state $h_{l}^{out}$ with the post-processing weight $H_{t}^{post}$. For the $i$th head:

      $$x_{l+1}[i] += H_{t}^{post}[i] \cdot h_{l}^{out}$$

      That is, $H_{t}^{post}[i]$ is a scalar, which is used to scalar-multiply the entire row of $h_{l}^{out}$ and add it to the $i$th row of the output.

  - The complete computation process is:

    $$x_{l+1}[i, :] = H_{t}^{post}[i] \cdot h_{l}^{out}[:] + \sum_{j=0}^{n-1} H_{l}^{res}[j, i] \cdot x_{l}[j, :]$$

    where $x_{l}$ represents the parameter `x`, $H_{l}^{res}$ represents the parameter `h_res`, $h_{l}^{out}$ represents the parameter `h_out`, $H_{t}^{post}$ represents the parameter `h_post`, and $x_{l+1}$ represents the output `y`.

  - When `h_res` is passed as None, the Res Mapping is skipped, and the formula degrades to a direct residual connection:

    $$x_{l+1} = x_{l} + h_{l}^{out} \cdot H_{t}^{post}$$

  >**Note:**
  >
  > The input supports two dimension formats: BSND (4D) and TND (3D). B (Batch) indicates the batch size, S (Seq-Length) indicates the sequence length, T indicates the cumulative sum of all batch sequence lengths ($T = B \times S$), n indicates the number of heads, and D indicates the hidden dimension size per head (headdim).

## Prototype

```python
cann_ops_transformer.mhc_post(x, h_res, h_out, h_post) -> Tensor
```

Where `h_res` is an optional input and can be passed as None (only supported in single operator mode for Ascend 950PR&950DT products; not supported in graph mode).

## Parameters

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>Parameter Name</th>
    <th>Parameter Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Shape</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Input token features of the current layer, corresponding to x<sub>l</sub> in the formula.</td>
        <td>bfloat16, float16</td>
        <td>
            <ul>
                <li>(B, S, n, D)</li>
                <li>(T, n, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h_res</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Residual connection matrix, corresponding to H<sub>l</sub><sup>res</sup> in the formula. When passed as None, the Res Mapping is skipped, and the formula degrades to a direct residual connection. Only Ascend 950PR&950DT products support passing None. Passing None for other product forms will cause an error.</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, n, n)</li>
                <li>(T, n, n)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h_out</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Output state of the previous layer, corresponding to H<sub>l</sub><sup>out</sup> in the formula.</td>
        <td>bfloat16, float16</td>
        <td>
            <ul>
                <li>(B, S, D)</li>
                <li>(T, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h_post</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Post-processing weight matrix, corresponding to H<sub>t</sub><sup>post</sup> in the formula.</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, n)</li>
                <li>(T, n)</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>

## Return Value Description

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>Parameter Name</th>
    <th>Parameter Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Shape</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>y</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>MHC Post computation output, corresponding to x<sub>l+1</sub> in the formula. Data type is the same as input `x`, and shape is the same as input `x`.</td>
        <td>bfloat16, float16</td>
        <td>
            <ul>
                <li>(B, S, n, D)</li>
                <li>(T, n, D)</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>

## Constraints

- This API supports training and inference scenarios.
- This API supports single operator mode and graph mode.
- Data type constraints:
  - The data types of `x` and `h_out` must be the same.
  - The data type of output `y` must be the same as `x`.

- `h_res` optional constraints:
  - When `h_res` is passed as None, the Res Mapping is skipped. Only Ascend 950PR&950DT products support this. For Atlas A2/A3 Series Products, `h_res` is a required parameter, and passing None will cause an error.
  - Passing None for `h_res` is only supported in single operator mode. In graph mode (torch.compile), `h_res` must be passed, and passing None will cause an error during the GE compilation phase.

- Dimension constraints:
  - The dimension of `h_res` must match the dimension format of `x`: (B, S, n, n) for 4D, and (T, n, n) for 3D.

- Shape consistency constraints:
  - In 4D (BSND) format:
    - The (B, S) dimensions of `h_res` must match the (B, S) dimensions of `x`. The last two dimensions of `h_res` are (n, n), where n matches the 3rd dimension of `x`.
    - The (B, S) dimensions of `h_out` must match the (B, S) dimensions of `x`. The D dimension of `h_out` must match the D dimension of `x`.
    - The (B, S) dimensions of `h_post` must match the (B, S) dimensions of `x`. The n dimension of `h_post` must match the n dimension of `x`.
  - In 3D (TND) format:
    - The T dimension of `h_res` must match the T dimension of `x`. The last two dimensions are (n, n).
    - The T dimension of `h_out` must match the T dimension of `x`. The D dimension must match the D dimension of `x`.
    - The T dimension of `h_post` must match the T dimension of `x`. The n dimension must match the n dimension of `x`.

- All dimension values of input Tensor shapes must be positive numbers (greater than 0).

## Deterministic Computation

Deterministic computation is supported by default.

## Example

- Single operator mode call:

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import mhc_post

  # Single-operator example: B is the batch size, S is the sequence length, n is the number of heads,
  # and D is the hidden size per head. The shape of output y is the same as that of x.
  B = 2
  S = 8
  n = 4
  D = 128

  x = torch.randn(B, S, n, D, dtype=torch.bfloat16).npu()
  h_res = torch.randn(B, S, n, n, dtype=torch.float32).npu()
  h_out = torch.randn(B, S, D, dtype=torch.bfloat16).npu()
  h_post = torch.randn(B, S, n, dtype=torch.float32).npu()

  y = mhc_post(x, h_res, h_out, h_post)
  print(f"output shape: {y.shape}")

  # When h_res is omitted (only supported by Ascend 950PR&950DT products, and only in single operator mode)
  y = mhc_post(x, None, h_out, h_post)
  print(f"output shape: {y.shape}")
  ```

- Graph mode call:

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import mhc_post

  torch_npu.npu.set_device(0)

  # Graph-mode example: compile and run the MhcPost forward computation with torch.compile.
  # The inputs and output shape are the same as those in the single-operator example.
  B = 2
  S = 8
  n = 4
  D = 128

  class MhcPostModel(torch.nn.Module):
      def forward(self, x, h_res, h_out, h_post):
          return mhc_post(x, h_res, h_out, h_post)

  model = MhcPostModel().npu()
  npu_backend = torchair.get_npu_backend()
  model = torch.compile(model, backend=npu_backend, dynamic=False)

  x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device="npu")
  h_res = torch.randn(B, S, n, n, dtype=torch.float32, device="npu")
  h_out = torch.randn(B, S, D, dtype=torch.bfloat16, device="npu")
  h_post = torch.randn(B, S, n, dtype=torch.float32, device="npu")

  y = model(x, h_res, h_out, h_post)
  ```
