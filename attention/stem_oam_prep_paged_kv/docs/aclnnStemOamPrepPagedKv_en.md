# aclnnStemOamPrepPagedKv

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:38:00.683Z -->

## Applicable Products
<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: Stem OAM (Output-Aware Metric) is the pre-scoring module of the dynamic sparse attention mechanism for large model inference. It extracts K/V data from the Paged KV Cache, computes through K Processing (per-token K-scale × group sum + anti-diagonal flip) and V Processing (per-head vScale × L2 Norm → Log → Global Normalize → ReLU → Block Average), and outputs kFlat and vBias for Stem OAM score computation.

- Formula:

Let B = stemBlockSize, S = stemStride, D = 128 (the last dimension of kvCache), and λ = lambdaMag. K/V are processed per (b, h), with the sequence length padded to a multiple of B.

Stage 1: De-page + Cast FP32 + kScaleCache Gather

$$
K\_dense[b], V\_dense[b], kscale\_per\_row[b] = Cast\big(gather(kcache[kv\_indices[b]]),\ FP32\big),\ Cast\big(gather(vcache[kv\_indices[b]]),\ FP32\big),\ gather(kScaleCache[kv\_indices[b]])
$$

$$
k\_padded[b] = \lceil kv\_len[b] / B \rceil \times B, \quad num\_Kb[b] = k\_padded[b] / B, \quad R = B / S, \quad k\_down\_len[b] = k\_padded[b] / S
$$

$$
K\_scaled[b, h] = K\_dense[b, h, :] \times kscale\_per\_row[b, h, :] \quad (\text{per-token per-head, broadcast on dim})
$$

$$
K\_blocks[b,h] = reshape(K\_scaled[b,h],\ [num\_Kb[b], R, S, D]), \quad V\_rows[b,h] = reshape(V\_dense[b,h],\ [k\_down\_len[b], S, D])
$$

Stage 2: K Processing (Weighted Group Sum + Anti-diagonal Flip)

$$
K\_group\_sum[b,h,kb,g,:] = \sum_{r=0}^{R-1} K\_blocks[b,h,kb,r,g,:]
$$

$$
K\_group\_rev[b,h,kb,g',:] = K\_group\_sum[b,h,kb,\ S-1-g',:] \quad (\text{anti-diagonal flip})
$$

$$
kFlat[b,h,kb,:] = reshape(K\_group\_rev[b,h,kb,:,:],\ [kflat\_dim = S \times D]) \xrightarrow{Cast} BF16
$$

Stage 3: V Processing

**Step 3a: L2 Norm + Max Pool + Log**

$$
norms[b,h,idx,s] = \| V\_rows[b,h,idx,s,:] \times vScale[h] \|_2 =  \sqrt{\sum_d (V\_rows[b,h,idx,s,d] \times vScale[h])^2}
$$

$$
v\_norm\_down[b,h,idx] = \max_{s} norms[b,h,idx,s] \quad (\text{zero beyond } kv\_len)
$$

$$
log\_vals[b,h,idx] = \log(v\_norm\_down[b,h,idx] + \epsilon), \quad \epsilon = 10^{-6}
$$

**Step 3b: Global Normalize (μ/σ over ALL k_down_len values)**

$$
\mu[b,h] = \frac{1}{k\_down\_len} \sum_{idx=0}^{k\_down\_len-1} log\_vals[b,h,idx]
$$

$$
\sigma[b,h] = \sqrt{\frac{\sum(log\_vals - \mu)^2}{k\_down\_len - 1}} \quad (k\_down\_len > 1);\quad \sigma = 0 \quad (k\_down\_len = 1)
$$

**Step 3c: Normalize + ReLU + Block Average**

$$
normalized[b,h,idx] = \frac{log\_vals[b,h,idx] - \mu[b,h]}{\sigma[b,h] + \epsilon}
$$

$$
v\_final[b,h,idx] = \lambda \times ReLU(normalized[b,h,idx])
$$

$$
vBias[b,h,kb] = \frac{1}{R} \sum_{r=0}^{R-1} v\_final[b,h,\ kb \times R + r]
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnStemOamPrepPagedKvGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnStemOamPrepPagedKv" to execute the computation.

```Cpp
aclnnStatus aclnnStemOamPrepPagedKvGetWorkspaceSize(
  const aclTensor  *kCache,
  const aclTensor  *vCache,
  const aclTensor  *kvIndices,
  const aclIntArray  *kvSeqLens,
  const aclTensor  *kScaleCache,
  const aclTensor  *vScale,
  double            lambdaMag,
  char              *kvLayout,
  int64_t           stemBlockSize,
  int64_t           stemStride,
  const aclTensor  *kFlat,
  const aclTensor  *vBias,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnStemOamPrepPagedKv(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnStemOamPrepPagedKvGetWorkspaceSize

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
      <td class="tg-0pky">kCache (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Paged K cache.</td>
      <td class="tg-0pky">Empty tensors are not supported.<br>Two layouts specified by kvLayout. Currently only "BNBD" is supported:<br>"BBND" is [total_blocks, kvBlockSize, H_kv, D=128], <br>"BNBD" is [total_blocks, H_kv, kvBlockSize, D=128].<br>The first three dimensions can be non-contiguous. The last dimension must be contiguous. The maximum value of H_kv is 8.</td>
      <td class="tg-0pky">FLOAT8_E4M3FN</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">vCache (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Paged V cache.</td>
      <td class="tg-0pky">Empty tensors are not supported.<br>Layout must be consistent with kCache.</td>
      <td class="tg-0pky">Consistent with kCache</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">kvIndices (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">KV Block index array for each batch.</td>
      <td class="tg-0pky">Empty tensors are not supported.<br>shape:[batch, max_kv_blocks], where batch is up to 16 and max_kv_blocks is up to 2048.</td>
      <td class="tg-0pky">INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">kvSeqLens (aclIntArray*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Length of the KV sequence in each batch.</td>
      <td class="tg-0pky">Empty lists are not supported.<br>shape:[batch], maximum KV sequence length is 262144.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">kScaleCacheOptional (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Per-token per-head K scale.</td>
      <td class="tg-0pky">Required when the data type of kCache is FP8; optional for other data types (pass nullptr).<br>Two layouts, specified by kvLayout. Currently only "BNBD" is supported:<br>"BBND" is [total_blocks, kvBlockSize, H_kv, 1],<br>"BNBD" is [total_blocks, H_kv, kvBlockSize, 1].<br>The first three dimensions can be non-contiguous, and the last dimension must be contiguous.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">vScaleOptional (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Per-head V scale.</td>
      <td class="tg-0pky">Required when the data type of kCache is FP8; can be omitted for other data types (pass nullptr).<br>shape:[H_kv]</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">lambdaMag</td>
      <td class="tg-0pky">ATTR</td>
      <td class="tg-0pky">V bias multiplier</td>
      <td class="tg-0pky">Value range: [0,1].</td>
      <td class="tg-0pky">double</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">kvLayout</td>
      <td class="tg-0pky">ATTR</td>
      <td class="tg-0pky">KV Cache layout.</td>
      <td class="tg-0pky">Value range: "BBND", "BNBD". Currently only "BNBD" is supported.</td>
      <td class="tg-0pky">char*</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">stemBlockSize</td>
      <td class="tg-0lax">ATTR</td>
      <td class="tg-0lax">Stem block size</td>
      <td class="tg-0lax">Value range: %32==0, ≤256, and stemBlockSize must be a multiple of stemStride. The recommended value is 128.</td>
      <td class="tg-0lax">int64_t</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">stemStride</td>
      <td class="tg-0lax">ATTR</td>
      <td class="tg-0lax">Stride size</td>
      <td class="tg-0lax">Value range: %16==0, ≤64, ≤stemBlockSize; 16 is recommended.</td>
      <td class="tg-0lax">int64_t</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">kFlat (aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">K group sum + anti-diag flip result.</td>
      <td class="tg-0lax">Empty tensors are not supported.<br>The shape is [batch, H_kv, max_Kb, kflat_dim], where kflat_dim=stemStride*D, <br>max_Kb is calculated based on the kvSeqLens input.</td>
      <td class="tg-0lax">BFLOAT16</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">4</td>
      <td class="tg-0lax">×</td>
    </tr>
    <tr>
      <td class="tg-0lax">vBias (aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">V block bias result.</td>
      <td class="tg-0lax">Empty tensors are not supported.<br>The shape is [batch, H_kv, max_Kb], where max_Kb is computed based on the kvSeqLens input.</td>
      <td class="tg-0lax">FLOAT</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">3</td>
      <td class="tg-0lax">×</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize (uint64_t*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the size of the workspace to be requested on the Device.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor(aclOpExecutor**)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the op executor, which contains the operator computation flow.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

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
      <td>kCache, vCache, kFlat, or vBias is a null pointer.<br>kScaleCache or vScale is a null pointer when the data type of kCache is FP8.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>kCache dtype is not FLOAT8_E4M3FN.</td>
    </tr>
    <tr>
      <td>kScaleCache dtype is not FLOAT.</td>
    </tr>
    <tr>
      <td>vScale shape is not 1D [H_kv].</td>
    </tr>
    <tr>
      <td>kvLayout/stemBlockSize/stemStride does not meet the constraints.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>Exception when the API internally calls an NPU runtime interface.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>Internal exception: an exception occurs when the API performs NPU kernel tiling.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_FIND_KERNEL_ERROR</td>
      <td>561003</td>
      <td>Internal exception: the API fails to find the NPU kernel (possibly because the operator binary package is not installed).</td>
    </tr>
  </tbody>
  </table>

## aclnnStemOamPrepPagedKv

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
      <td>Memory address of the workspace allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnStemOamPrepPagedKvGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- kvBlockSize ∈ {64, 128}.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```c++
  #include <iostream>
  #include <vector>
  #include <cstring>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_stem_oam_prep_paged_kv.h"

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
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                      void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    // Initialize the ACL environment and create a stream.
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Operator parameters: batch=1, 8 physical blocks, 4 KV heads, D=128, BNBD layout.
    // maxKvBlocks=2 means the per-batch block index table has 2 slots (up to 2 physical blocks).
    // maxKb = ceil(max(kvSeqLens)/stemBlockSize) = ceil(128/128) = 1, i.e. the 3rd dim of kFlat/vBias.
    int64_t batch = 1;
    int64_t totalBlocks = 8;
    int64_t numKvHeads = 4;
    int64_t dimQk = 128;
    int64_t maxKvBlocks = 2;
    int64_t stemBlockSize = 128;
    int64_t stemStride = 16;
    int64_t maxKb = 1;
    int64_t kflatDim = stemStride * dimQk;
    // kvLayout is a string attribute; initialize it as a char array to match the char* parameter.
    char kvLayout[] = "BNBD";

    // Tensor shapes (BNBD layout: kCache/vCache are [totalBlocks, H_kv, kvBlockSize, D]).
    // kFlat is the BF16 output of [batch, H_kv, maxKb, stemStride*D]; vBias is the FP32 output of [batch, H_kv, maxKb].
    std::vector<int64_t> kCacheShape = {totalBlocks, numKvHeads, 64, dimQk};
    std::vector<int64_t> kvIndicesShape = {batch, maxKvBlocks};
    std::vector<int64_t> kScaleCacheShape = {totalBlocks, numKvHeads, 64, 1};
    std::vector<int64_t> vScaleShape = {numKvHeads};
    std::vector<int64_t> kFlatShape = {batch, numKvHeads, maxKb, kflatDim};
    std::vector<int64_t> vBiasShape = {batch, numKvHeads, maxKb};

    void* kCacheDeviceAddr = nullptr;
    void* vCacheDeviceAddr = nullptr;
    void* kvIndicesDeviceAddr = nullptr;
    void* kScaleCacheDeviceAddr = nullptr;
    void* vScaleDeviceAddr = nullptr;
    void* kFlatDeviceAddr = nullptr;
    void* vBiasDeviceAddr = nullptr;

    aclTensor* kCache = nullptr;
    aclTensor* vCache = nullptr;
    aclTensor* kvIndices = nullptr;
    aclIntArray* kvSeqLens = nullptr;
    aclTensor* kScaleCache = nullptr;
    aclTensor* vScale = nullptr;
    aclTensor* kFlat = nullptr;
    aclTensor* vBias = nullptr;

    // Build host-side data: kvIndices is the block index table (element count matches shape [batch, maxKvBlocks];
    // physical blocks 0 and 1 are used here); kvSeqLens holds the per-batch actual KV lengths (int64 array),
    // whose values must not exceed maxKvBlocks*64=128.
    std::vector<uint8_t> hostKCache(GetShapeSize(kCacheShape), 1);
    std::vector<uint8_t> hostVCache(GetShapeSize(kCacheShape), 1);
    std::vector<int32_t> hostKvIndices({0, 1});
    std::vector<int64_t> hostKvSeqLens({128});
    std::vector<float> hostKScaleCache(GetShapeSize(kScaleCacheShape), 1.0f);
    std::vector<float> hostVScale(numKvHeads, 1.0f);
    std::vector<uint16_t> hostKFlat(GetShapeSize(kFlatShape), 0);
    std::vector<float> hostVBias(GetShapeSize(vBiasShape), 0.0f);

    // Create device-side tensors: kCache/vCache are FP8_E4M3FN, kvIndices is INT32,
    // kScaleCache/vScale are FP32 (required on the FP8 path), kFlat is BF16, and vBias is FP32.
    ret = CreateAclTensor(hostKCache, kCacheShape, &kCacheDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &kCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostVCache, kCacheShape, &vCacheDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &vCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostKvIndices, kvIndicesShape, &kvIndicesDeviceAddr, aclDataType::ACL_INT32, &kvIndices, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    kvSeqLens = aclCreateIntArray(hostKvSeqLens.data(), hostKvSeqLens.size());
    ret = CreateAclTensor(hostKScaleCache, kScaleCacheShape, &kScaleCacheDeviceAddr, aclDataType::ACL_FLOAT, &kScaleCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostVScale, vScaleShape, &vScaleDeviceAddr, aclDataType::ACL_FLOAT, &vScale, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostKFlat, kFlatShape, &kFlatDeviceAddr, aclDataType::ACL_BF16, &kFlat, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostVBias, vBiasShape, &vBiasDeviceAddr, aclDataType::ACL_FLOAT, &vBias, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Phase 1 of the two-phase interface: validates inputs, returns the executor and the required workspace size.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnStemOamPrepPagedKvGetWorkspaceSize(
        kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale,
        0.3, kvLayout, stemBlockSize, stemStride,
        kFlat, vBias, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate workspace on the device according to the size returned by phase 1.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Phase 2 of the two-phase interface: submits the computation task to the specified stream.
    ret = aclnnStemOamPrepPagedKv(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStemOamPrepPagedKv failed. ERROR: %d\n", ret); return ret);

    // Synchronize the stream to wait for the computation to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Copy the kFlat result back: BF16 values are copied as uint16_t and converted to float for printing
    // the first 10 elements; verify the values against the formula in actual use.
    auto size = GetShapeSize(kFlatShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), kFlatDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size && i < 10; i++) {
      float val;
      uint16_t bf16 = resultData[i];
      uint32_t bits = static_cast<uint32_t>(bf16) << 16;
      std::memcpy(&val, &bits, sizeof(val));
      LOG_PRINT("result[%ld] is: %f\n", i, val);
    }

    aclDestroyTensor(kCache);
    aclDestroyTensor(vCache);
    aclDestroyTensor(kvIndices);
    aclDestroyIntArray(kvSeqLens);
    aclDestroyTensor(kScaleCache);
    aclDestroyTensor(vScale);
    aclDestroyTensor(kFlat);
    aclDestroyTensor(vBias);

    aclrtFree(kCacheDeviceAddr);
    aclrtFree(vCacheDeviceAddr);
    aclrtFree(kvIndicesDeviceAddr);
    aclrtFree(kScaleCacheDeviceAddr);
    aclrtFree(vScaleDeviceAddr);
    aclrtFree(kFlatDeviceAddr);
    aclrtFree(vBiasDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```
