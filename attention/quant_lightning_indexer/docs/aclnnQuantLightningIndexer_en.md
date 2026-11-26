# aclnnQuantLightningIndexer

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:35:47.047Z -->

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

- Description: QuantLightningIndexer supports Per-Token-Head quantized input based on LightningIndexer.

- Formula:
    $$
    out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnQuantLightningIndexerGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that includes the operator computation process, and then call the "aclnnQuantLightningIndexer" API to execute the computation.

```Cpp
aclnnStatus aclnnQuantLightningIndexerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    int64_t          queryQuantMode,
    int64_t          keyQuantMode,
    char            *layoutQueryOptional,
    char            *layoutKeyOptional,
    int64_t          sparseCount,
    int64_t          sparseMode,
    int64_t          preTokens,
    int64_t          nextTokens,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```Cpp
aclnnStatus aclnnQuantLightningIndexer(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnQuantLightningIndexerGetWorkspaceSize

- **Parameters:**

> [!NOTE]
>
>- Dimension meanings of the query, key, weights, queryDequantScale, and keyDequantScale parameters: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the size of the hidden layer, N (Head Num) indicates the number of heads, and D (Head Dim) indicates the size of the smallest unit of the hidden layer, satisfying D=H/N. T indicates the cumulative sum of the sequence lengths of input samples across all batches.
>- S1 and S2 respectively indicate the sequence lengths of the query and key input samples, N1 and N2 respectively indicate the number of heads corresponding to query and key, and k indicates the number of indices finally selected. D in the query parameter is equal to D in the key parameter, both being 128. T1 and T2 respectively indicate the cumulative sums of the sequence lengths of the query and key input samples.

<table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
<col style="width: 240px">
<col style="width: 132px">
<col style="width: 232px">
<col style="width: 330px">
<col style="width: 233px">
<col style="width: 119px">
<col style="width: 215px">
<col style="width: 100px">
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
    </tr></thead>
<tbody>
    <tr>
    <td>query</td>
    <td>Input</td>
    <td>Input Q in the formula.</td>
    <td>Empty tensor not supported.</td>
    <td>INT8, FLOAT8_E4M3, HIFLOAT8</td>
    <td>ND</td>
    <td>        <ul>
                <li>When layoutQuery is BSND, the shape is (B,S1,N1,D).</li>
                <li>When layoutQuery is TND, the shape is (T1,N1,D).</li>
        </ul>
    </td>
    <td>x</td>
    </tr>
    <tr>
    <td>key</td>
    <td>Input</td>
    <td>Input K in the formula.</td>
    <td>        <ul>
                <li>Empty tensors are not supported.</li>
                <li>blockNum indicates the total number of blocks in PageAttention, and blockSize indicates the number of tokens in a block.</li>
        </ul>
    </td>
    <td>INT8, FLOAT8_E4M3, HIFLOAT8</td>
    <td>ND</td>
    <td>        <ul>
                <li>If layoutKey is PA_BSND, the shape is (blockNum, blockSize, N2, D).</li>
                <li>If layoutKey is BSND, the shape is (B, S2, N2, D).</li>
                <li>If layoutKey is TND, the shape is (T2, N2, D).</li>
        </ul>
    </td>
    <td>✓</td>
    </tr>
    <tr>
    <td>weights</td>
    <td>Input</td>
    <td>Input W in the formula.</td>
    <td>Empty tensors are not supported.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>        <ul>
                <li>When layoutQuery is BSND, the shape is (B,S1,N1).</li>
                <li>When layoutQuery is TND, the shape is (T1,N1).</li>
        </ul>
    </td>
    <td>x</td>
    </tr>
    <tr>
    <td>queryDequantScale</td>
    <td>Input</td>
    <td>Indicates the dequantization scale Scale_Q of Index Query.</td>
    <td>Empty tensors are not supported.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
    <td>        <ul>
                <li>If layoutQuery is BSND, the shape is (B,S1,N1).</li>
                <li>If layoutQuery is TND, the shape is (T1,N1).</li>
        </ul>
    </td>
    <td>x</td>
    </tr>
    <tr>
    <td>keyDequantScale</td>
    <td>Input</td>
    <td>Indicates the dequantization scale Scale_K of Index Key.</td>
    <td>Empty tensors are not supported.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
    <td>        <ul>
                <li>If layoutKey is BSND, the shape is (B,S2,N2).</li>
                <li>If layoutKey is TND, the shape is (T2,N2).</li>
                <li>If layoutKey is PA_BSND, the shape is (blockNum, blockSize, N2).</li>
                <li>blockNum indicates the total number of blocks in PageAttention, and blockSize indicates the number of tokens in each block.</li>
        </ul>
    </td>
    <td>✓</td>
    </tr>
    <tr>
    <td>actualSeqLengthsQueryOptional</td>
    <td>Input</td>
    <td>Number of valid Query tokens in each Batch.</td>
    <td>        <ul>
                <li>Empty tensors are not supported.</li>
                <li>If seqlen is not specified, None can be passed in, indicating that the length is the same as the S dimension of the query shape.</li>
                <li>In this parameter, the number of valid tokens in each Batch must not exceed the size of dimension S of query and must not be less than 0. A 1-D tensor of length B is supported.</li>
                <li>When layoutQuery is TND, this parameter is mandatory, and the number of elements in this parameter is used as the value of B. The value of each element in this parameter indicates the total number of tokens in the current batch and all preceding batches, that is, the prefix sum. Therefore, the value of the latter element must be greater than or equal to that of the former element.</li>
                <li>Negative values are not allowed.</li>
        </ul>
    </td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B,)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>actualSeqLengthsKeyOptional</td>
    <td>Input</td>
    <td>The number of valid tokens of Key in each batch.</td>
    <td>        <ul>
                <li>Empty tensors are not supported.</li>
                <li>If seqlen is not specified, None can be passed in, indicating that its S length is the same as that of the key shape.</li>
                <li> The number of valid tokens of each batch in this parameter must not exceed the dimension S size of key/value and must not be less than 0. A one-dimensional tensor of length B is supported.</li>
                <li>When layoutKey is TND or PA_BSND, this input parameter must be passed. When layoutKey is TND, each element indicates the total number of tokens of the current batch and all previous batches, that is, the prefix sum. Therefore, the value of the latter element must be greater than or equal to that of the former element.</li>
                <li>Negative values are not allowed.</li>
        </ul>
    </td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B,)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>blockTableOptional</td>
    <td>Input</td>
    <td>Indicates the block mapping table used for KV storage in PageAttention.</td>
    <td>        <ul>
                <li>Empty tensors are not supported.</li>
                <li>In the PageAttention use case, blockTable must be two-dimensional. The length of the first dimension must be equal to B, and the length of the second dimension must not be less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actualSeqLengthsKey in each batch).</li>
                <li>The value of blockSize must be a multiple of 16, and the maximum supported value is 1024.</li>
        </ul>
    </td>
    <td>INT32</td>
    <td>ND</td>
    <td>shape supported: (B,S2_max/blockSize)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>queryQuantMode</td>
    <td>Input</td>
    <td>Used to identify the quantization mode of the input query.</td>
    <td>        <ul>
            <li>Currently supported: Per-Token-Head quantization mode.</li>
            <li>Currently, only 0 is supported.</li>
        </ul>
    </td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>keyQuantMode</td>
    <td>Input</td>
    <td>Used to identify the quantization mode of the input key.</td>
    <td>        <ul>
            <li>The Per-Token-Head quantization mode is currently supported.</li>
            <li>Only the value 0 is currently supported.</li>
        </ul>
    </td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>layoutQueryOptional</td>
    <td>Input</td>
    <td>Used to identify the data layout format of the input Query.</td>
    <td>Currently supported values are BSND and TND.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>layoutKeyOptional</td>
    <td>Input</td>
    <td>Used to identify the data layout format of the input Key.</td>
    <td>Currently supported: PA_BSND, BSND, TND.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sparseCount</td>
    <td>Input</td>
    <td>Number of blocks retained in the topK stage.</td>
    <td>Currently supported: [1, 2048].</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sparseMode</td>
    <td>Input</td>
    <td>Indicates the sparse mode.</td>
    <td>        <ul>
                <li>When sparseMode is 0, it indicates the defaultMask mode.</li>
                <li>When sparseMode is 3, it indicates the mask in rightDownCausal mode, corresponding to the lower-triangular case divided by the right vertex.</li>
        </ul>
    </td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>preTokens</td>
    <td>Input</td>
    <td>Used for sparse computation, indicating that attention needs to be computed in association with the preceding several tokens.</td>
    <td>Recommended value: 2^63-1.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>nextTokens</td>
    <td>Input</td>
    <td>Used for sparse computation. Indicates that the attention computation needs to be associated with the next few tokens.</td>
    <td>Recommended value: 2^63-1.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>out</td>
    <td>Output</td>
    <td>The Indices output in the formula.</td>
    <td>Empty tensor not supported.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>        <ul>
                <li>When layoutQuery is BSND, the output shape is [B, S1, N2, sparseCount].</li>
                <li>When layoutQuery is TND, the output shape is [T1, N2, sparseCount].</li>
        </ul>
    </td>
    <td>x</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the workspace size that needs to be applied on the Device side.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the op executor, which includes the computation flow of the operator.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
</tbody>
</table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>Return Value</th>
            <th>Error Code</th>
            <th>Description</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>If the parameter passed in is a required input, output, or required attribute and is a null pointer, 161001 is returned.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The data types and data formats of query, key, weights, queryDequantScale, keyDequantScale, actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, queryQuantMode, keyQuantMode, layoutQueryOptional, layoutKeyOptional, sparseCount, sparseMode, and out are not supported.</td>
            </tr>
        </tbody>
    </table>

## aclnnQuantLightningIndexer

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 184px">
    <col style="width: 134px">
    <col style="width: 833px">
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
        <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnQuantLightningIndexerGetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, includes computation flow of the operator.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing the task.</td>
        </tr>
    </tbody>

    </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation description: aclnnQuantLightningIndexer defaults to a deterministic implementation.
- N1 in the query parameter supports values less than or equal to 64/32/24/16, and N2 in key supports 1.
- D supports 128.
- blockSize must be a multiple of 16, with a maximum supported value of 1024.
- The data types of the query and key parameters must be consistent.

<!-- npu="A3" id8 -->
- <term>Atlas A3 products</term>:
  - query and key support the `INT8` data type.
  - Only the data types `FLOAT16, FLOAT16, FLOAT16` are supported for weights, queryDequantScale, and keyDequantScale.
  - When layout_key is PA_BSND, key and keyDequantScale support non-contiguous axis 0, and the other axes must be contiguous.
<!-- end id8 -->

<!-- npu="950" id7 -->
- Ascend 950PR&950DT products:
  - query N1 only supports 8, 16, 24, 32, and 64.
  - query and key support the `FLOAT8_E4M3, HIFLOAT8, INT8` data types.
  - When the data types of query and key are `FLOAT8_E4M3`, the data types of weights, queryDequantScale, and keyDequantScale can be `BFLOAT16, FLOAT, FLOAT` or `FLOAT16, FLOAT16, FLOAT16`;
  - When the data types of query and key are `HIFLOAT8`, only the data types `BFLOAT16, FLOAT, FLOAT` are supported for weights, queryDequantScale, and keyDequantScale;
  - When the data types of query and key are `INT8`, only weights, queryDequantScale, and keyDequantScale with the data types `FLOAT16, FLOAT16, FLOAT16` are supported.

<!-- end id7 -->

## Example

The following sample code is for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_lightning_indexer.cpp
 * \brief
 */
//testci
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
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
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
    return ret;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* queryDequantScaleDeviceAddr = nullptr;
    void* keyDequantScaleDeviceAddr = nullptr;
    void* actualSeqLengthsKeyDeviceAddr = nullptr;
    void* blockTableDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* queryDequantScaleTensor = nullptr;
    aclTensor* keyDequantScaleTensor = nullptr;
    aclTensor* actualSeqLengthsKeyTensor = nullptr;
    aclTensor* blockTableTensor = nullptr;
    aclTensor* outTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    std::vector<int64_t> queryShape = {2, 64, 16, 128};
    std::vector<int64_t> keyShape = {32, 16, 1, 128};
    std::vector<int64_t> weightsShape = {2, 64, 16};
    std::vector<int64_t> queryDequantScaleShape = {2, 64, 16};
    std::vector<int64_t> keyDequantScaleShape = {32, 16, 1};
    std::vector<int64_t> actualSeqLengthsKeyShape = {2};
    std::vector<int64_t> blockTableShape = {2, 32};
    std::vector<int64_t> outShape = {2, 64, 1, 2048};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t queryDequantScaleShapeSize = GetShapeSize(queryDequantScaleShape);
    int64_t keyDequantScaleShapeSize = GetShapeSize(keyDequantScaleShape);
    int64_t actualSeqLengthsKeyShapeSize = GetShapeSize(actualSeqLengthsKeyShape);
    int64_t blockTableShapeSize = GetShapeSize(blockTableShape);
    int64_t outShapeSize = GetShapeSize(outShape);

    std::vector<uint8_t> queryHostData(queryShapeSize, 0x38);
    std::vector<uint8_t> keyHostData(keyShapeSize, 0x38);
    std::vector<uint16_t> weightsHostData(weightsShapeSize, 0x3F00);
    std::vector<float> queryDequantScaleHostData(queryDequantScaleShapeSize, 1.0f);
    std::vector<float> keyDequantScaleHostData(keyDequantScaleShapeSize, 1.0f);
    std::vector<int32_t> actualSeqLengthsKeyHostData = {256, 512};
    std::vector<int32_t> blockTableHostData(blockTableShapeSize, 0);
    for (int32_t i = 0; i < 32; i++) {
        blockTableHostData[i] = i;
        blockTableHostData[32 + i] = i;
    }
    std::vector<int32_t> outHostData(outShapeSize, 0);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                              aclDataType::ACL_FLOAT8_E4M3FN, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                          aclDataType::ACL_FLOAT8_E4M3FN, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr,
                          aclDataType::ACL_BF16, &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(queryDequantScaleHostData, queryDequantScaleShape, &resources.queryDequantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.queryDequantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyDequantScaleHostData, keyDequantScaleShape, &resources.keyDequantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.keyDequantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(actualSeqLengthsKeyHostData, actualSeqLengthsKeyShape, &resources.actualSeqLengthsKeyDeviceAddr,
                          aclDataType::ACL_INT32, &resources.actualSeqLengthsKeyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(blockTableHostData, blockTableShape, &resources.blockTableDeviceAddr,
                          aclDataType::ACL_INT32, &resources.blockTableTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(outHostData, outShape, &resources.outDeviceAddr,
                          aclDataType::ACL_INT32, &resources.outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }
    return ACL_SUCCESS;
}

int ExecuteQuantLightningIndexer(TensorResources& resources, aclrtStream stream,
                                 void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t queryQuantMode = 0;
    int64_t keyQuantMode = 0;
    int64_t sparseCount = 2048;
    int64_t sparseMode = 3;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    constexpr const char layoutQueryStr[] = "BSND";
    constexpr const char layoutKeyStr[] = "PA_BSND";
    constexpr size_t layoutQueryLen = sizeof(layoutQueryStr);
    constexpr size_t layoutKeyLen = sizeof(layoutKeyStr);
    char layoutQuery[layoutQueryLen];
    char layoutKey[layoutKeyLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layoutQueryStr, layoutQueryLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layoutKeyStr, layoutKeyLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor;

    int ret = aclnnQuantLightningIndexerGetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor,
        resources.queryDequantScaleTensor, resources.keyDequantScaleTensor,
        nullptr, resources.actualSeqLengthsKeyTensor, resources.blockTableTensor,
        queryQuantMode, keyQuantMode, layoutQuery, layoutKey,
        sparseCount, sparseMode, preTokens, nextTokens,
        resources.outTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnQuantLightningIndexer(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexer failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int32_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("sparse_indices result[%ld] is: %d\n", i, resultData[i]);
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.weightsTensor) {
      aclDestroyTensor(resources.weightsTensor);
    }
    if (resources.queryDequantScaleTensor) {
      aclDestroyTensor(resources.queryDequantScaleTensor);
    }
    if (resources.keyDequantScaleTensor) {
      aclDestroyTensor(resources.keyDequantScaleTensor);
    }
    if (resources.actualSeqLengthsKeyTensor) {
      aclDestroyTensor(resources.actualSeqLengthsKeyTensor);
    }
    if (resources.blockTableTensor) {
      aclDestroyTensor(resources.blockTableTensor);
    }
    if (resources.outTensor) {
      aclDestroyTensor(resources.outTensor);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.weightsDeviceAddr) {
      aclrtFree(resources.weightsDeviceAddr);
    }
    if (resources.queryDequantScaleDeviceAddr) {
      aclrtFree(resources.queryDequantScaleDeviceAddr);
    }
    if (resources.keyDequantScaleDeviceAddr) {
      aclrtFree(resources.keyDequantScaleDeviceAddr);
    }
    if (resources.actualSeqLengthsKeyDeviceAddr) {
      aclrtFree(resources.actualSeqLengthsKeyDeviceAddr);
    }
    if (resources.blockTableDeviceAddr) {
      aclrtFree(resources.blockTableDeviceAddr);
    }
    if (resources.outDeviceAddr) {
      aclrtFree(resources.outDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main() {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> outShape = {2, 64, 1, 2048};
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecuteQuantLightningIndexer(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    PrintOutResult(outShape, &resources.outDeviceAddr);

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}

```
