# aclnnBSASelectBlockMask

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:20:41.442Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id5 -->
<!-- npu="910b" id6 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id6 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id2 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas training products</term>: Not supported
<!-- end id3 -->

## Function

- Description​: aclnnBSASelectBlockMask is the preceding operator of BSA (BlockSparseAttention). It dynamically generates blockSparseMask based on the content of Query and Key, changing the BSA call chain from "manually providing a mask" to "adaptively selecting the sparse mode based on Q/K content".

- Formula​:

  Let blockShape = [blockShapeX, blockShapeY], Sq be the maximum sequence length of query, and Skv be the maximum sequence length of key. The number of blocks after compression is:

  $$
  Xblocks = \lceil Sq / blockShapeX \rceil,\quad Yblocks = \lceil Skv / blockShapeY \rceil
  $$

  **Step1: Mean Pooling Compression**

  For each block, let L be the minimum of its block size, the remaining sequence length of the current batch from the start of the block, and its supplied actualBlockLen (use the block size when this input is null). When L > 0, the compressed vector is the mean of the first L tokens. When L = 0, the compressed vector is zero. A trailing block may therefore contain fewer tokens than the block size even when actualBlockLen is null.

  **Step2a: QK Matmul**

  $$
  score[b, n, x, y] = scale \cdot \sum_{d=0}^{D-1} q\_compressed[b, n, x, d] \cdot k\_compressed[b, n, y, d]
  $$

  **Step2b: Softmax**

  $$
  attn\_score[b, n, x, y] = softmax(score[b, n, x, :]) = \frac{\exp(score[b, n, x, y] - m_{final})}{l_{final}}
  $$

  **Step2c: Post Pooling Compression (Post-Softmax Mean Pooling)**

  When postBlockShape is not null, apply post mean pooling to attn_score to generate the coarse-grained pooled_score:

  $$
  postXBlocks = \lceil Xblocks / postBlockShapeX \rceil,\quad postYBlocks = \lceil Yblocks / postBlockShapeY \rceil
  $$

  $$
  pooled\_score[b, n, px, py] = \frac{1}{|R_{px,py}|} \sum_{x \in R_x(px)} \sum_{y \in R_y(py)} attn\_score[b, n, x, y]
  $$

  When postBlockShape is null, skip this step, and pooled_score = attn_score.

  **Step3: Block Selection**

  For each batch and head, let M be the number of blocks in the valid two-dimensional grid, after post pooling if enabled. For M > 1:

  - When sparsity <= 0.5, retain the max(1, round(sparsity * M)) blocks with the largest scores.
  - When sparsity > 0.5, discard the max(1, round((1 - sparsity) * M)) blocks with the smallest scores and retain the others.

  The rounding of positive values follows the kernel calculation of adding 0.5 and truncating. When M = 1, retain the block. The BNSD maximum grid must have more than one block. Use attn_score without post pooling and pooled_score otherwise. The set of retained positions is denoted by \(\mathcal{indices}\).

  **Step4: Generate BlockSparseMask**

  When postBlockShape is not null, the output is the **coarse-grained** mask directly (with shape [B, N, postXBlocks, postYBlocks], using the secondary pooling split semantics, without expansion):

  $$
  blockSparseMaskOut[b, n, px, py] =
  \begin{cases}
  1 & (px, py) \in \mathcal{indices} \\
  0 & (px, py) \notin \mathcal{indices}
  \end{cases}
  $$

  When postBlockShape is null, the output is the fine-grained mask (with shape [B, N, Xblocks, Yblocks]) and is generated element-wise directly:

  $$
  blockSparseMaskOut[b, n, x, y] =
  \begin{cases}
  1 & (b, n, x, y) \in \mathcal{indices} \\
  0 & (b, n, x, y) \notin \mathcal{indices}
  \end{cases}
  $$

- Data format:

  The data format of the query and key inputs to BSASelectBlockMask can be interpreted under multiple dimension layouts and is specified through qInputLayout and kvInputLayout. To facilitate understanding of the specific supported formats described later (such as BNSD and TND), the meaning of each dimension represented by the abbreviation letters in the format names is explained here first:

  - B: Indicates the batch size of the input samples (Batch).
  - T: Indicates the length of B and S concatenated along the axis and tightly packed (Total tokens).
  - S: Indicates the sequence length of the input samples (Seq-Length).
  - H: Indicates the size of the hidden layer (Head-Size).
  - N: Number of heads (Head-Num).
  - D: indicates the smallest unit size of the hidden layer and must satisfy D = H / N (Head-Dim)

- Currently supported layouts:

  - qInputLayout: "TND" "BNSD"
  - kvInputLayout: "TND" "BNSD"

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnBSASelectBlockMaskGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnBSASelectBlockMask" API to execute the computation.

```Cpp
aclnnStatus aclnnBSASelectBlockMaskGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclIntArray *blockShape,
  const aclIntArray *postBlockShape,
  const aclIntArray *actualSeqLengths,
  const aclIntArray *actualSeqLengthsKV,
  const aclIntArray *actualBlockLenQuery,
  const aclIntArray *actualBlockLenKey,
  char              *qInputLayout,
  char              *kvInputLayout,
  int64_t            numKeyValueHeads,
  double             scaleValue,
  double             sparsity,
  aclTensor         *blockSparseMaskOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnBSASelectBlockMask(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnBSASelectBlockMaskGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 140px">
  </colgroup>
  <thead>
    <tr>
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Instruction</th>
    <th>Data Type</th>
    <th>Data Format</th>
    <th>Dimension (shape)</th>
    <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
    <td>query (aclTensor*)</td>
    <td>Input</td>
    <td>The query matrix in the attention computation, which is the query in the formula.</td>
    <td><ul><li>Empty tensors are not supported.</li><li>Supported shapes:<ul><li>TND: [totalQTokens, headNum, headDim].</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li></ul></li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>key (aclTensor*)</td>
    <td>Input</td>
    <td>The key matrix in the attention computation, which is the key in the formula.</td>
    <td><ul><li>Empty tensors are not supported.</li><li>Supported shapes:<ul><li>TND: [totalKTokens, numKeyValueHeads, headDim].</li><li>BNSD: [batch, numKeyValueHeads, maxKSeqLength, headDim].</li></ul></li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
        <tr>
    <td>blockShape (aclIntArray*)</td>
    <td>Input</td>
    <td>Sparse block shape array, specifying the two-dimensional size (number of rows and columns) of each sparse block, which corresponds to blockShape in the formula.</td>
    <td><ul><li>Element requirements when this input is configured:<ul><li>Must contain at least two elements [blockShapeX, blockShapeY].</li><li>blockShapeX: block size in the Q direction, which must be a multiple of 8 and greater than 0.</li><li>blockShapeY: block size in the KV direction, which must be a multiple of 8 and greater than 0.</li></ul></li><li>If not configured (nullptr is passed), the operator uses blockShapeX = 128 and blockShapeY = 128 by default.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>postBlockShape (aclIntArray*)</td>
    <td>Input</td>
    <td>Secondary pooling block shape array, specifying the block group size of secondary pooling after Softmax (in blocks), that is, postBlockShape in the formula.</td>
    <td><ul><li>Optional input: When `[postBlockShapeX, postBlockShapeY]` is passed in, secondary pooling is enabled. TopK selection is performed on the coarse-grained pooled_score, and the output mask is coarse-grained [B, N, postXBlocks, postYBlocks] (secondary pooling split semantics: the selection result is output directly without expansion).</li><li>When nullptr is passed in, secondary pooling is disabled (for backward compatibility). TopK selection is performed directly on attn_score, and the output mask is fine-grained [B, N, Xblocks, Yblocks].</li><li>postBlockShapeX/postBlockShapeY must be a multiple of 8, indicating the number of block groups.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
        <tr>
    <td>actualSeqLengths (aclIntArray*)</td>
    <td>Input</td>
    <td>The actual sequence length of the query in each batch, that is, the actual valid value of Sq in the formula for each batch.<br>Describes the number of valid query tokens in each batch in a variable-length sequence scenario (that is, a scenario containing Padding data).</td>
    <td><ul><li>Variable-length sequence scenario (when qInputLayout is "TND"): This input must be configured.</li><li>Fixed-length/variable-length scenario (when qInputLayout is "BNSD"):<ul><li>If configured, the operator processes data based on the specified valid length and ignores Padding data, improving performance;</li><li>If not configured (nullptr is passed), the operator processes all data by default, using the S dimension in the query shape as the valid length.</li></ul></li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
        <tr>
    <td>actualSeqLengthsKV (aclIntArray*)</td>
    <td>Input</td>
    <td>Array of the actual sequence lengths of key, that is, the actual valid value of Skv in each batch in the formula.<br>Describes the number of actually valid key tokens in each batch in a variable-length sequence scenario (that is, a scenario containing Padding data).</td>
    <td><ul><li>Variable-length sequence scenario (when kvInputLayout is "TND"): This input must be configured.</li><li>>Fixed-length/variable-length scenario (when kvInputLayout is "BNSD"):<ul><li>If this item is configured, the operator processes data based on the specified valid length and ignores Padding data to improve performance;</li><li>If this item is not configured (set to nullptr), the operator uses the S dimension in the key shape as the valid length by default and processes all data.</li></ul></li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
        <tr>
    <td>actualBlockLenQuery (aclIntArray*)</td>
    <td>Input</td>
    <td>Actual valid seq length compressed within each query block, that is, actualBlockLenQuery in the formula.<br>Used in partial compression scenarios (for example, incomplete blocks at the end or compressing only valid tokens).</td>
    <td><ul><li>Optional input: <ul><li>BNSD scenario: the shape is [B, Xblocks].</li><li>TND scenario: the shape is [TotalBlockNum_Q] (the actual valid Xblocks of each batch are stacked; validXblocks = ceil(actualSeqQ / blockShapeX)).</li><li>The value range of each element is [0, blockShapeX].</li><li>When actualBlockLen = 0: q_compressed of the corresponding block is filled with a zero vector and the block is not guaranteed to be excluded from selection.</li><li>When actualBlockLen > 0: the first actualBlockLen tokens are averaged, clipped to the remaining sequence length of the current batch.</li></ul></li>
    <li>If not configured (nullptr is passed): the valid tokens in each query block are averaged, clipped to the remaining sequence length.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
        <tr>
    <td>actualBlockLenKey (aclIntArray*)</td>
    <td>Input</td>
    <td>Actual valid seq length compressed in each key block, that is, actualBlockLenKey in the formula.<br>Used for partial compression scenarios (for example, incomplete trailing blocks or compressing only valid tokens).</td>
    <td><ul><li>Optional input: <ul><li>BNSD scenario: the shape is [B, Yblocks].</li><li>TND scenario: the shape is [TotalBlockNum_K] (the actual valid Yblocks of each batch are stacked; validYblocks = ceil(actualSeqK / blockShapeY)).</li><li>The value range of each element is [0, blockShapeY].</li><li>When actualBlockLen = 0: k_compressed of the corresponding block is filled with a zero vector and the block is not guaranteed to be excluded from selection.</li><li>When actualBlockLen > 0: the first actualBlockLen tokens are averaged, clipped to the remaining sequence length of the current batch.</li></ul></li>
    <li>If not configured (nullptr is passed): the valid tokens in each key block are averaged, clipped to the remaining sequence length.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>qInputLayout (char*)</td>
    <td>Input</td>
    <td>Data format of the query. Indicates the specific layout of the input tensor in memory.</td>
    <td><ul>Currently, only "TND" and "BNSD" are supported. qInputLayout must be the same as kvInputLayout.</ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>kvInputLayout (char*)</td>
    <td>Input</td>
    <td>Data format of key. Indicates the specific layout of the input tensor in memory.</td>
    <td><ul>Only "TND" and "BNSD" are currently supported. qInputLayout and kvInputLayout must be the same.</ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numKeyValueHeads (int64_t)</td>
    <td>Input</td>
    <td>Number of attention heads of key, that is, the number of heads N of the key tensor in the formula.</td>
    <td><ul>Currently, only MHA is supported, and numKeyValueHeads must be equal to numHeads.</ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scaleValue (double)</td>
    <td>Input</td>
    <td>Scaling factor, that is, scale in the formula. Used to normalize attention scores.</td>
    <td><ul>Generally set to 1 / sqrt(D).</ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sparsity (double)</td>
    <td>Input</td>
    <td>Sparsity retention ratio, i.e., sparsity in the formula. Specifies the proportion of block positions to be retained in attn_score out of all block positions in the formula.</td>
    <td><ul>Value range (0.0, 1.0).</ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>blockSparseMaskOut (aclTensor*)</td>
    <td>Output</td>
    <td>Block sparse mask output, that is, blockSparseMaskOut in the formula. It denotes the sparse pattern adaptively generated based on the Q/K content, and can be directly used as the blockSparseMask input of the BSA operator.</td>
    <td> <ul><li>Empty tensors are not supported.</li><li>The shape varies with postBlockShape:<ul><li>If postBlockShape is not null: [B, N, postXBlocks, postYBlocks] (coarse granularity, secondary pooling split semantics), where postXBlocks = ceilDiv(Xblocks, postBlockShapeX), and postYBlocks = ceilDiv(Yblocks, postBlockShapeY).</li><li>If postBlockShape is null: [B, N, Xblocks, Yblocks] (fine granularity), where Xblocks = ceilDiv(maxQSeqlen, blockShapeX), and Yblocks = ceilDiv(maxKvSeqlen, blockShapeY).</li><li>A value of 1 indicates that the block participates in the attention computation, and a value of 0 indicates that it does not.</li></ul></li></ul></td>
    <td>INT8</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>workspaceSize (uint64_t*)</td>
    <td>Output</td>
    <td>Returns the size of the workspace to be applied for on the Device.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor (aclOpExecutor**)</td>
    <td>Output</td>
    <td>Returns the op executor, which contains the operator computation process.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1100px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="3">161001</td>
      <td>The input query is a null pointer.</td>
    </tr>
    <tr>
      <td>The input key is a null pointer.</td>
    </tr>
    <tr>
      <td>blockSparseMaskOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of query, key, or blockSparseMaskOut is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of query and key are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnBSASelectBlockMask

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device side, obtained by the first-phase API aclnnBSASelectBlockMaskGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>AscendCL stream used for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- When this API is used with PyTorch, ensure that the version of CANN-related packages matches that of PyTorch-related packages.
- actualSeqLengths is mandatory when qInputLayout is "TND"; actualSeqLengthsKV is mandatory when kvInputLayout is "TND".
- Based on the input layouts supported by the operator, the size of the head dimension in the query tensor shape is denoted as N1, and the size of the head dimension in the key tensor shape is denoted as N2. N1 = N2 must be satisfied (only MHA is supported).
- headDim = 128.
- blockShapeX and blockShapeY must be a multiple of 8.
- The maximum number of blocks in the selection grid must be greater than 1 for BNSD and greater than 0 for TND. It is Xblocks * Yblocks when postBlockShape is null, and postXBlocks * postYBlocks otherwise.
- The data types of query and key must be the same; only FLOAT16 and BFLOAT16 are supported.
- The data type of blockSparseMaskOut is INT8 (binary: 0 or 1).
- postBlockShape is an optional input. Passing nullptr disables secondary pooling (backward compatible). Passing `[postBlockShapeX, postBlockShapeY]` enables secondary pooling; postBlockShapeX/postBlockShapeY must be a multiple of 8, indicating the number of block groups.
- If actualBlockLenQuery / actualBlockLenKey is not null, the value range of each element is [0, blockShapeX] / [0, blockShapeY]; if it is null, complete compression is performed.
- Does not involve deterministic computation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_bsa_select_block_mask.h"

using namespace std;

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
     // Boilerplate: resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }

    auto size = GetShapeSize(shape) * sizeof(T);

    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n",
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }

    // Call aclrtMalloc to allocate device memory.
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n");
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return -1);
    return 0;
}

int main() {
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

   // 2. Construct inputs and outputs. (BNSD layout is used as an example)
    int32_t batch = 1;
    int32_t numHeads = 4;
    int32_t numKHeads = 4;      // Only MHA is supported: N == KN
    int32_t qSeqlen = 256 * 1024; // 256K
    int32_t kSeqlen = 256 * 1024; // 256K
    int32_t headDim = 128;
    int32_t blockShapeX = 128;
    int32_t blockShapeY = 128;

    // Calculate the number of blocks.
    int32_t ceilQ = (qSeqlen + blockShapeX - 1) / blockShapeX;
    int32_t ceilK = (kSeqlen + blockShapeY - 1) / blockShapeY;

    // Build the tensor shape.
    std::vector<int64_t> qShape = {batch, numHeads, qSeqlen, headDim};
    std::vector<int64_t> kShape = {batch, numKHeads, kSeqlen, headDim};
    std::vector<int64_t> maskShape = {batch, numHeads, ceilQ, ceilK};

    // Allocate and initialize Host data.
    int64_t qSize = GetShapeSize(qShape);
    int64_t kSize = GetShapeSize(kShape);

    std::vector<op::fp16_t> qData(qSize, 0.1f);
    std::vector<op::fp16_t> kData(kSize, 0.1f);

    // Initialize the mask output to 0.
    std::vector<int8_t> maskOutData(GetShapeSize(maskShape), 0);

    // Create all aclTensors.
    void *qAddr = nullptr, *kAddr = nullptr, *maskAddr = nullptr;
    aclTensor *qTensor = nullptr, *kTensor = nullptr, *maskTensor = nullptr;

    CreateAclTensor(qData, qShape, &qAddr, aclDataType::ACL_FLOAT16, &qTensor);
    CreateAclTensor(kData, kShape, &kAddr, aclDataType::ACL_FLOAT16, &kTensor);
    CreateAclTensor(maskOutData, maskShape, &maskAddr, aclDataType::ACL_INT8, &maskTensor);

    // Create the aclIntArray attribute parameters.
    std::vector<int64_t> blockShapeVec = {blockShapeX, blockShapeY};
    aclIntArray *blockShapeArr = aclCreateIntArray(blockShapeVec.data(), blockShapeVec.size());

    // Scalar and string parameter configuration.
    char queryLayoutBuffer[16] = "BNSD";
    char keyLayoutBuffer[16] = "BNSD";
    double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));
    double sparsity = 0.5;

     // 3. Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // Call the first-phase API of aclnnBSASelectBlockMask.
    LOG_PRINT("Calling aclnnBSASelectBlockMaskGetWorkspaceSize...\n");
    ret = aclnnBSASelectBlockMaskGetWorkspaceSize(
        qTensor,
        kTensor,
        blockShapeArr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        queryLayoutBuffer,
        keyLayoutBuffer,
        static_cast<int64_t>(numKHeads),
        scaleValue,
        sparsity,
        maskTensor,
        &workspaceSize,
        &executor
    );

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n"); return -1);
    LOG_PRINT("Workspace size required: %lu bytes\n", workspaceSize);

    // Request device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnBSASelectBlockMask.
    LOG_PRINT("Calling aclnnBSASelectBlockMask...\n");
    ret = aclnnBSASelectBlockMask(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBSASelectBlockMask failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value, and copy the result from the device memory to the host.
    int64_t maskSize = GetShapeSize(maskShape);
    ret = aclrtMemcpy(maskOutData.data(), maskSize * sizeof(int8_t), maskAddr, maskSize * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed.\n"); return ret);

    LOG_PRINT("Execution Success! BlockSparseMask output (first 20 elements):\n");
    int64_t displayCount = (maskSize < 20) ? maskSize : 20;
    for (int64_t i = 0; i < displayCount; i++) {
        LOG_PRINT("  mask index %ld: %u\n", i, static_cast<unsigned int>(maskOutData[i]));
    }

    // 6. Release aclTensor and aclIntArray.
    LOG_PRINT("Cleaning up resources...\n");
    aclDestroyTensor(qTensor);
    aclDestroyTensor(kTensor);
    aclDestroyTensor(maskTensor);

    aclDestroyIntArray(blockShapeArr);

    // 7. Release device resources.
    aclrtFree(qAddr);
    aclrtFree(kAddr);
    aclrtFree(maskAddr);
    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("BSASelectBlockMask Test completed successfully!\n");
    return 0;
}

```
