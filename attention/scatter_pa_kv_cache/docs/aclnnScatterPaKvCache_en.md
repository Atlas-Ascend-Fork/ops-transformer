# aclnnScatterPaKvCache

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:05:17.396Z -->

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
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Updates the key and value at specified positions in the KV cache.

- The input and output support the following scenarios:
  - Scenario 1:

    ```python
    key:[batch * seq_len, num_head, k_head_size]
    value:[batch * seq_len, num_head, v_head_size]
    keyCache:[num_blocks, num_head * k_head_size // last_dim_k, block_size, last_dim_k]/[num_blocks, num_head, k_head_size // last_dim_k, block_size, last_dim_k]
    valueCache:[num_blocks, num_head * v_head_size // last_dim_v, block_size, last_dim_v]/[num_blocks, num_head, v_head_size // last_dim_v, block_size, last_dim_v]
    slotMapping:[batch * seq_len]
    cacheMode:"PA_NZ"

    last_dim_k = 32 / sizeof(dtypeKey)
    last_dim_v = 32 / sizeof(dtypeValue)
    (k_head_size * sizeof(dtypeKey)) % 32 == 0
    (v_head_size * sizeof(dtypeValue)) % 32 == 0
    ```

  - Scenario 2:

    ```python
    key:[batch * seq_len, num_head, k_head_size]
    value:[batch * seq_len, num_head, v_head_size]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    valueCache:[num_blocks, block_size, num_head, v_head_size]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    scatterMode:"None"/"Nct"
    ```

    k_head_size and v_head_size can be different or the same.

  - Scenario 3:

    ```python
    key:[batch, seq_len, num_head, k_head_size]
    value:[batch, seq_len, num_head, v_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    valueCache:[num_blocks, block_size, 1, v_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch, num_head]
    seqLensOptional:[batch]
    compressSeqOffsetOptional:[batch * num_head]
    cacheMode:"Norm"
    ```

  - Scenario 4:

    ```python
    key:[num_tokens, num_head, k_head_size]
    value:[num_tokens, num_head, v_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    valueCache:[num_blocks, block_size, 1, v_head_size]
    slotMapping:[batch * num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    scatterMode:"Alibi"
    ```

  - Scenario 5:

    ```python
    key:[num_tokens, num_head, k_head_size]
    value:[num_tokens, num_head, v_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    valueCache:[num_blocks, block_size, 1, v_head_size]
    slotMapping:[batch * num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    compressSeqOffsetOptional:[batch * num_head]
    cacheMode:"Norm"
    scatterMode:"Rope"/"Omni"
    ```

  - Scenario 6:

    ```python
    key:[batch * seq_len, num_head, k_head_size]
    value:[]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    valueCache:[]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    scatterMode:"None"/"Nct"
    ```

  - Scenario 7:

    ```python
    key:[num_tokens, num_head, k_head_size]
    value:[num_tokens, num_head, v_head_size]
    keyCache:[num_blocks, num_head, block_size, k_head_size]
    valueCache:[num_blocks, num_head, block_size, v_head_size]
    slotMapping:[num_tokens]
    cacheMode:"Norm"
    scatterMode:"NHSD"
    ```

- The above scenarios are distinguished based on the constructed parameters. The first parameter construction corresponds to Scenario 1, the second to Scenario 2, the third to Scenario 3, the fourth to Scenario 4, the fifth to Scenario 5, the sixth to Scenario 6, and the seventh to Scenario 7. Scenarios 1, 2, 6, and 7 do not have the three optional parameters: compressLensOptional, seqLensOptional, and compressSeqOffsetOptional. Scenario 4 does not have the compressSeqOffsetOptional optional parameter.

<!-- npu="A3,910b" id7 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: only supports scenarios 1, 2, 4, 5, 6, and 7.
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>: only supports scenarios 1, 2, 3, 4, and 5.

<!-- end id8 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnScatterPaKvCacheGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that includes the operator computation process, and then call the "aclnnScatterPaKvCache" API to perform the computation.

```Cpp
aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(
  const aclTensor   *key,
  aclTensor         *keyCacheRef,
  const aclTensor   *slotMapping,
  const aclTensor   *value,
  aclTensor         *valueCacheRef,
  const aclTensor   *compressLensOptional,
  const aclTensor   *compressSeqOffsetOptional,
  const aclTensor   *seqLensOptional,
  char              *cacheModeOptional,
  char              *scatterModeOptional,
  const aclIntArray *stridesOptional,
  const aclIntArray *offsetsOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnScatterPaKvCache(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnScatterPaKvCacheGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1565px"><colgroup>
  <col style="width: 230px">
  <col style="width: 120px">
  <col style="width: 270px">
  <col style="width: 350px">
  <col style="width: 200px">
  <col style="width: 115px">
  <col style="width: 135px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">key (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Key values to be updated, that is, the keys of multiple tokens at the current step.</td>
      <td class="tg-0pky">Does not support null tensors.</td>
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E1M2, FLOAT4_E2M1</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">keyCacheRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Key cache to be updated, that is, the key cache of the current layer.</td>
      <td class="tg-0pky">Does not support null Tensor.<br>Only the ND memory layout format is supported when passing a null pointer or "Norm".<br>Only the FRACTAL_NZ memory layout is supported when passing "PA_NZ".</td>
      <td class="tg-0pky">Remain consistent with key</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">slotMapping (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Storage offset of each token key or value in the cache.</td>
      <td class="tg-0pky">Does not support null Tensor.<br></td>
      <td class="tg-0pky">INT32, INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">value (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Value to be updated, that is, the value of multiple tokens in the current step.</td>
      <td class="tg-0pky">Null tensors are not supported.<br>The shape of dimensions other than 0 must be consistent with that of key.</td>
      <td class="tg-0pky">Consistent with key</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0, 3, 4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">valueCacheRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Value cache to be updated, that is, the value cache of the current layer.</td>
      <td class="tg-0pky">Does not support null tensors.<br>Except for dimension 0, the shape must be consistent with keyCacheRef. When passing a null pointer or "Norm", only the ND memory layout format is supported.<br>When passing "PA_NZ", only the FRACTAL_NZ memory layout is supported.</td>
      <td class="tg-0pky">Remain consistent with key</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0, 4, 5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">compressLensOptional (aclTensor*)</td>
      <td class="tg-0pky">Optional input</td>
      <td class="tg-0pky">Compression amount.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">Remain consistent with slotMapping</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">compressSeqOffsetOptional (aclTensor*)</td>
      <td class="tg-0pky">Optional input</td>
      <td class="tg-0pky">Compression start point of each batch and each head.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">Remain consistent with slotMapping</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">seqLensOptional (aclTensor*)</td>
      <td class="tg-0pky">Optional input</td>
      <td class="tg-0pky">Actual seqLens of each batch.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">Remain consistent with slotMapping</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">cacheModeOptional(char*)</td>
      <td class="tg-0pky">input</td>
      <td class="tg-0pky">Indicates the memory layout format of keyCacheRef and valueCacheRef.</td>
      <td class="tg-0pky">When passing a null pointer or "Norm", only the ND memory layout format is supported.<br>When passing "PA_NZ", only the FRACTAL_NZ memory layout is supported.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">scatterModeOptional (char*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Indicates the state of the updated key and value.</td>
      <td class="tg-0lax">When passing a null pointer or "None", the updated key and value are in non-compression state and contiguous.<br>When passing "Alibi", the updated key and value are in compression state based on the Alibi structure.<br>When passing "Rope", the updated key and value are in compression state based on the Rope structure.<br>When passing "Omni", the updated key and value are in compression state based on the Omni structure.<br>When passing "Nct", the updated key and value are in non-compression state but non-contiguous.<br>When passing "NHSD", the format of keyCacheRef and valueCacheRef is [num_blocks, num_head, block_size, head_size].</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">stridesOptional (aclIntArray*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Stride of key and value in the non-contiguous state.</td>
      <td class="tg-0lax">Array length is 2. The value must be greater than 0. Takes effect only when scatterModeOptional is set to "Nct", and indicates strideK and strideV respectively.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">offsetsOptional (aclIntArray*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Offset of key and value in the non-contiguous state.</td>
      <td class="tg-0lax">The array length is 2. The value must be greater than or equal to 0. This parameter takes effect only when scatterModeOptional is "Nct". The two values indicate offsetK and offsetV respectively.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize (uint64_t*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the size of the Device-side-applied workspace.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor(aclOpExecutor**)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the op executor, which includes the operator calculation process.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    - The input key, keyCacheRef, value, and valueCacheRef only support the FLOAT16, BFLOAT16, and INT8 data types, and do not support FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E1M2, or FLOAT4_E2M1.

  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: supports all data types listed in the parameter table; among them, FLOAT4_E1M2 and FLOAT4_E2M1 are supported only in scenario 1 and scenario 2 when scatterMode is "None".

  <!-- end id10 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 302px">
  <col style="width: 119px">
  <col style="width: 731px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>key, keyCacheRef, slotMapping, value, and valueCacheRef are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of key, value, and slotMapping are not in the supported range.</td>
    </tr>
    <tr>
      <td>The dimension of key is not 3 or 4, or the dimension of value is not 0, 3, or 4.</td>
    </tr>
    <tr>
      <td>The data types of key, keyCacheRef, value, and valueCacheRef are inconsistent.</td>
    </tr>
    <tr>
      <td>The data types of slotMapping, compressLensOptional, compressSeqOffsetOptional, and seqLensOptional are inconsistent.</td>
    </tr>
  </tbody>
  </table>

## aclnnScatterPaKvCache

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory applied on the Device side, obtained by the first-phase API aclnnScatterPaKvCacheGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op. Includes the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnScatterPaKvCache defaults to a deterministic implementation.
- The data types of key, value, keyCacheRef, and valueCacheRef must be consistent.
- When cacheModeOptional is "PA_NZ", the data types of key, keyCacheRef and value, valueCacheRef can be inconsistent;
- The data types of slotMapping, compressLensOptional, compressSeqOffsetOptional, and seqLensOptional must be consistent (corresponding scenario 1);
- The value range of slotMapping is [0, num_blocks * block_size - 1], and the elements in slotMapping must be unique. Correctness is not guaranteed if duplicates exist.
- When both key and value are 3D, the shapes of the first two dimensions of key and value must be the same.
- When both key and value are 4D, the shapes of the first three dimensions of key and value must be the same, and the third dimension of keyCacheRef and valueCacheRef must be 1.
- When key and value are 4D, compressLensOptional and seqLensOptional are mandatory parameters. When key and value are 3D, compressLensOptional, compressSeqOffsetOptional, and seqLensOptional are optional parameters, but when scatterModeOptional is "Alibi", compressLensOptional and seqLensOptional are mandatory, and when scatterModeOptional is "Rope" or "Omni", compressLensOptional, compressSeqOffsetOptional, and seqLensOptional are all mandatory.
- When key and value are all 4-dimensional, slotMapping is 2-dimensional. The first dimension of slotMapping equals the first dimension of key, which is batch, and the second dimension of slotMapping equals the third dimension of key, which is num_head (corresponding to scenario 3);
- When key and value are all 4-dimensional, seqLensOptional is 1-dimensional, and the value of seqLensOptional equals the first dimension of key, which is batch (corresponding to scenario 3);
- When key and value are 3-dimensional and seqLensOptional exists, the sum of all values in seqLensOptional equals the first dimension of key, which is num_tokens (corresponding to scenarios 4 and 5);
- Each element in seqLensOptional and compressLensOptional must satisfy the formula: reduceSum(seqLensOptional[i] - compressLensOptional[i]) <= num_blocks * block_size (corresponding to scenarios 3, 4, and 5).
- When cacheModeOptional is "PA_NZ", the second-to-last dimension of keyCacheRef and valueCacheRef must be less than UINT16_MAX (corresponding to scenario 1).
- k_head_size and v_head_size must be 32-byte aligned (corresponding to scenario 7).
- num_head must be less than 4095 (corresponding to scenario 7).
- block_size *k_head_size and block_size* v_head_size must be less than UINT32_MAX (corresponding to scenario 7).
- (num_head *k_head_size + num_head* v_head_size) * sizeof(dtypeKey) must be less than 196608 (corresponding to scenario 7).

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950,A3,910b" id11 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, and <term>Ascend 950PR&950DT products</term>:

  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_scatter_pa_kv_cache.h"

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
  // Boilerplate for resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API.
    std::vector<int64_t> keyShape = {2, 2, 32};
    std::vector<int64_t> keyCacheShape = {1, 4, 32, 16};
    std::vector<int64_t> slotMappingShape = {2};
    std::vector<int64_t> valueShape = {2, 2, 32};
    std::vector<int64_t> valueCacheShape = {1, 4, 32, 16};

    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* slotMappingDeviceAddr = nullptr;
    void* keyCacheDeviceAddr = nullptr;
    void* valueCacheDeviceAddr = nullptr;
    void* compressLensDeviceAddr = nullptr;
    void* compressSeqOffsetDeviceAddr = nullptr;
    void* seqLensDeviceAddr = nullptr;

    aclTensor* key = nullptr;
    aclTensor* value = nullptr;
    aclTensor* slotMapping = nullptr;
    aclTensor* keyCache = nullptr;
    aclTensor* valueCache = nullptr;
    aclTensor* compressLens = nullptr;
    aclTensor* compressSeqOffset = nullptr;
    aclTensor* seqLens = nullptr;
    char * cacheMode = "PA_NZ";
    char * scatterMode = "None";

    std::vector<int16_t> hostKey(128, 0);
    std::vector<int16_t> hostValue(128, 0);
    std::vector<int32_t> hostSlotMapping = {0, 1};
    std::vector<int16_t> hostKeyCacheRef(2048, 1);
    std::vector<int16_t> hostValueCacheRef(2048, 1);
    std::vector<int64_t> hostStrides(2, 1);
    std::vector<int64_t> hostOffsets(2, 0);

    // Create the key aclTensor.
    ret = CreateAclTensor(hostKey, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &key, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the value aclTensor.
    ret = CreateAclTensor(hostValue, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &value, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the slotMapping aclTensor.
    ret = CreateAclTensor(hostSlotMapping, slotMappingShape, &slotMappingDeviceAddr, aclDataType::ACL_INT32, &slotMapping, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the keyCache aclTensor.
    ret = CreateAclTensor(hostKeyCacheRef, keyCacheShape, &keyCacheDeviceAddr, aclDataType::ACL_FLOAT16, &keyCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the valueCache aclTensor.
    ret = CreateAclTensor(hostValueCacheRef, valueCacheShape, &valueCacheDeviceAddr, aclDataType::ACL_FLOAT16, &valueCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclIntArray *strides = aclCreateIntArray(hostStrides.data(), 2);
    CHECK_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    aclIntArray *offsets = aclCreateIntArray(hostOffsets.data(), 2);
    CHECK_RET(offsets != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. Call the CANN operator library API. Replace Api with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnScatterPaKvCache.
    ret = aclnnScatterPaKvCacheGetWorkspaceSize(key, keyCache, slotMapping, value, valueCache, compressLens, compressSeqOffset, seqLens, cacheMode, scatterMode, strides, offsets, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaKvCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnScatterPaKvCache.
    ret = aclnnScatterPaKvCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaKvCache failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values and copy the results from device memory to host memory. Modify the code based on the definition of the specific API.
    auto keyCacheSize = GetShapeSize(keyCacheShape);
    std::vector<int16_t> keyCacheResult(keyCacheSize, 0);
    ret = aclrtMemcpy(keyCacheResult.data(), keyCacheResult.size() * sizeof(int16_t), keyCacheDeviceAddr,
                      keyCacheSize * sizeof(int16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy keyCache from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < keyCacheSize; i++) {
      LOG_PRINT("keyCache[%ld] is: %d\n", i, keyCacheResult[i]);
    }

    auto valueCacheSize = GetShapeSize(valueCacheShape);
    std::vector<int16_t> valueCacheResult(valueCacheSize, 0);
    ret = aclrtMemcpy(valueCacheResult.data(), valueCacheResult.size() * sizeof(int16_t), valueCacheDeviceAddr,
                      valueCacheSize * sizeof(int16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy valueCache from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < valueCacheSize; i++) {
      LOG_PRINT("valueCache[%ld] is: %d\n", i, valueCacheResult[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify according to the specific API definition.
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(keyCache);
    aclDestroyTensor(valueCache);
    aclDestroyTensor(compressLens);
    aclDestroyTensor(compressSeqOffset);
    aclDestroyTensor(seqLens);
    aclDestroyIntArray(strides);
    aclDestroyIntArray(offsets);
    // 7. Release device resources. Modify parameters according to the specific API definition.
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(slotMappingDeviceAddr);
    aclrtFree(keyCacheDeviceAddr);
    aclrtFree(valueCacheDeviceAddr);
     if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```

<!-- end id11 -->
