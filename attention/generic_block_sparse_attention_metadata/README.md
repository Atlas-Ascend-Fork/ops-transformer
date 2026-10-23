# GenericBlockSparseAttentionMetadata

## 产品支持情况

| 产品 | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：

  `GenericBlockSparseAttentionMetadata`是`GenericBlockSparseAttention`的前置算子，根据Query序列长度、Q/KV头数和稀疏块信息生成主算子使用的分核结果`metadataOptional`。本算子不执行Attention计算。

  典型调用流程如下：

  1. 准备`sparseBlockIdx`、`sparseBlockCount`、序列长度和属性等输入。
  2. 调用`aclnnGenericBlockSparseAttentionMetadata`生成`metadataOptional`。
  3. 调用`aclnnGenericBlockSparseAttention`，将上一步得到的`metadataOptional`作为主算子输入。

## 分架构支持范围

| 参数 | Atlas A2/A3 | Ascend 950（A5） |
| ---- | ---- | ---- |
| Query layout | TND | TND |
| KV layout | TND、PA_BBND | PA_BBND |
| headDim | 1～256 | 128 |
| blockShape | X=1，Y∈{1,2,4,8,16,32,64,128} | [1,128] |
| PA 页大小 | 128，与稀疏块 Y 独立 | 128 |
| topK 上限 | 512 | 256 |
| maskType | 0（no-mask）、1（causal） | 1 |
| residualBlockMode | 0、1（非量化） | 0 |
| LSE 输出 | PA_BBND 且 residualBlockMode=0 时支持 | 不支持 |

A2/A3 仅支持 FP16/BF16 非量化；BF16 要求 softmaxPrecision=0，FP16 支持 0/1。两种架构均要求
layoutSparsePattern=4、groupSize≤128、winLeft=winRight=-1。A5 保留本 beta 版本的能力范围。

Dense TND 的 K/V 形状为 [totalKvTokens,numKvHeads,headDim]，须传 cuSeqLengthsKv，不传 blockTable；
PA_BBND 须传 blockTable 和 sequsedKv，不传 cuSeqLengthsKv。序列前缀和从 0 开始，末项对应存储容量；
seqused 不得超过对应存储段。Metadata 与主算子对应的序列、shape、mask 和 residual 等属性必须一致。

每个 Q 块的 sparseBlockIdx 有效前缀须为合法块索引，严格升序且无重复；块可以随机选择，不要求等步长。
residualBlockMode=1 时，当前可见 KV 范围末尾不足 Y 的块由算子补入，不得重复放在有效 sparseBlockIdx 中。
causal 的可见范围为 kvLen-qLen+qToken+1；no-mask 使用整个有效 KV 序列。

## 参数说明

> **说明：**<br>
> B表示Batch Size，`totalQBlocks`表示按各Batch的Query存储长度划分后的Q块总数，`maxSparseBlockCount`表示`sparseBlockIdx`的最后一维。

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 220px">
  <col style="width: 130px">
  <col style="width: 550px">
  <col style="width: 180px">
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
      <td>sparseBlockIdx</td>
      <td>输入</td>
      <td>稀疏块索引。当前shape为[numKeyValueHeads, totalQBlocks, maxSparseBlockCount]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>输入</td>
      <td>每个Q块实际选择的KV块数量，当前shape为[numKeyValueHeads, totalQBlocks]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cuSeqLengthsQOptional</td>
      <td>输入</td>
      <td>各Batch中Query序列存储长度的前缀和，当前必须传入，shape为(B+1,)。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>输入</td>
      <td>各Batch中Key/Value序列存储长度的前缀和，layoutKv为"TND"时必传，非TND时不传，shape为(B+1,)。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>可选输入</td>
      <td>各Batch中Query的实际序列长度，shape为(B,)；不传时使用Query存储长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>输入</td>
      <td>各Batch中Key/Value的实际序列长度，layoutKv为"PA_BBND"时必传，shape为(B,)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>maxQSeqLen</td>
      <td>属性</td>
      <td>主算子Query的最大Sequence Length，必须大于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxKvSeqLen</td>
      <td>属性</td>
      <td>主算子Key/Value的最大Sequence Length，必须大于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numQHeads</td>
      <td>属性</td>
      <td>主算子Query的head数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKvHeads</td>
      <td>属性</td>
      <td>主算子Key/Value的head数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>属性</td>
      <td>主算子 Q/K/V 的 headDim，A2/A3 支持 1～256；A5 支持 128。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>属性</td>
      <td>稀疏块形状[blockShapeX, blockShapeY]，取值见分架构支持范围。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isPackedGQA</td>
      <td>属性</td>
      <td>同一个group内的qHead是否共享相同的稀疏pattern，当前仅支持1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>属性</td>
      <td>主算子Query的数据排布格式，当前仅支持"TND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>属性</td>
      <td>主算子 K/V 排布：A2/A3 支持 TND、PA_BBND；A5 支持 PA_BBND。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>属性</td>
      <td>掩码类型：A2/A3 支持 0/1；A5 支持 1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>属性</td>
      <td>量化类型，当前支持0；Ascend 950上可选5。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxPrecision</td>
      <td>属性</td>
      <td>Softmax计算精度级别，取值为0或1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLeft</td>
      <td>属性</td>
      <td>滑窗Attention向前包含的token数，当前仅支持-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>属性</td>
      <td>滑窗Attention向后包含的token数，当前仅支持-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>输出</td>
      <td>主算子使用的分核结果，shape固定为(1024,)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 本算子必须与`GenericBlockSparseAttention`配套使用。所有对应输入和属性须与随后调用的主算子一致，每次调用主算子前均须重新生成`metadataOptional`。
- layout、headDim、blockShape、mask 和 residual 等约束见分架构支持范围；isPackedGQA=1。
- `numQHeads >= numKvHeads`且`numQHeads % numKvHeads == 0`，`numQHeads / numKvHeads`不超过128。
- `sparseBlockIdx`最后一维须不小于`sparseBlockCount`中所有元素的最大值，A2/A3 不超过512，A5 不超过256。
- 输出`metadataOptional`是固定为INT32、shape为(1024,)的不透明数据，不得解析、修改或跨不同输入和属性复用。
- 完整约束见[aclnnGenericBlockSparseAttentionMetadata](./docs/aclnnGenericBlockSparseAttentionMetadata.md)。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| aclnn API | [test_aclnn_generic_block_sparse_attention_metadata](./examples/test_aclnn_generic_block_sparse_attention_metadata.cpp) | 通过[aclnnGenericBlockSparseAttentionMetadata](./docs/aclnnGenericBlockSparseAttentionMetadata.md)两段式接口调用GenericBlockSparseAttentionMetadata算子。 |
