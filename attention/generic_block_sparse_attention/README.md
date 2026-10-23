# GenericBlockSparseAttention

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √    |
| <term>Atlas A3系列产品</term>     |    √    |
| <term>Atlas A2系列产品</term>     |    √    |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×    |
| <term>Atlas推理系列产品</term>                              |    ×    |
| <term>Atlas训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：

  `GenericBlockSparseAttention`是基于CATLASS模板库实现的高性能块稀疏注意力算子，支持沿S轴的块级稀疏模式。通过`sparseBlockIdx`指定每个Q块选择的KV块，通过`sparseBlockCount`指定每个Q块实际保留的KV块数量。调用前须先执行`aclnnGenericBlockSparseAttentionMetadata`生成`metadataOptional`。

  典型调用流程如下：

  1. 准备`query`、`key`、`value`、`sparseBlockIdx`、`sparseBlockCount`等输入。
  2. 调用`aclnnGenericBlockSparseAttentionMetadata`生成`metadataOptional`。
  3. 调用`aclnnGenericBlockSparseAttention`，将上一步得到的`metadataOptional`传入主算子。

- 计算公式：

  稀疏块大小：$blockShapeX \times blockShapeY$

  $$
  attentionOut = Softmax(scaleValue \cdot query \cdot key_{sparse}^{T} + atten\_mask) \cdot value_{sparse}
  $$

  其中$key_{sparse}$、$value_{sparse}$为按`sparseBlockIdx`/`sparseBlockCount`从 Dense TND 或 Paged KV Cache 中选取的 KV 块。

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
> 参数维度含义：B表示Batch Size，T表示Total tokens，N表示Head Num，D表示Head Dim，topK表示`sparseBlockIdx`最后一维`maxSparseBlockCount`。<br>
> TND中的N为query的headNum（记为N1），PA_BBND中的N为key/value的headNum（记为N2）。GQA下N1与N2可以不同，约束见下文。

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 520px">
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
      <td>query</td>
      <td>输入</td>
      <td>公式中的query。layoutQ为"TND"时，shape为[T, N, D]，N为query的headNum（N1）。</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的key。layoutKv为"PA_BBND"时，shape为[numBlocks, blockSize, N, D]，N为kv的headNum（N2）。</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的value，shape与key一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparseBlockIdx</td>
      <td>输入</td>
      <td>稀疏块索引。TND + isPackedGQA=1时，shape为[N, totalQBlocks, topK]，N为kv的headNum（N2）。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>输入</td>
      <td>每个Q块实际选择的KV块数量。TND + isPackedGQA=1时，shape为[N, totalQBlocks]，N为kv的headNum（N2）。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cuSeqLengthsQOptional</td>
      <td>输入</td>
      <td>各batch中query序列长度前缀和，layoutQ为"TND"时必传，shape为[B+1,]。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>输入</td>
      <td>各batch中key/value序列长度前缀和，layoutKv为"TND"时必传，非TND时不传，shape为[B+1,]。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>可选输入</td>
      <td>各batch中query实际有效长度；不传时按cu前缀和差分得到的存储长度处理。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>输入</td>
      <td>各batch中kv实际有效长度，layoutKv为"PA_BBND"时必传，shape为[B,]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>PagedAttention页表，shape为[B, maxNumBlocksPerBatch]。</td>
      <td>INT32</td>
      <td>ND</td>
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
      <td>同group内qHead是否共享稀疏pattern，当前仅支持1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>属性</td>
      <td>query数据排布格式，当前仅支持"TND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>属性</td>
      <td>key/value数据排布格式，A2/A3 支持 TND、PA_BBND；A5 支持 PA_BBND。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>属性</td>
      <td>缩放系数；传0时算子内按$1/\sqrt{D}$处理。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>属性</td>
      <td>掩码类型：A2/A3 支持 0（no-mask）和 1（causal）；A5 支持 1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>属性</td>
      <td>量化类型；当前支持0，Ascend 950上可选5。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxPrecision</td>
      <td>属性</td>
      <td>Softmax计算精度级别，取值0或1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxlse</td>
      <td>属性</td>
      <td>是否输出 softmaxLse，取值 0/1；支持条件见分架构支持范围。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>输出</td>
      <td>公式中的attentionOut，数据类型和shape与query保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxLseOptional</td>
      <td>输出</td>
      <td>Softmax log-sum-exp 输出。开启时为 FLOAT [T,N1,1]；关闭时传 nullptr。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 调用前须先执行`aclnnGenericBlockSparseAttentionMetadata`生成`metadataOptional`，再调用本接口；metadata须与当前输入/属性配套，每次调用须重新生成。
- headDim、blockShapeY 及 PA 页大小的约束见分架构支持范围。
- TND + isPackedGQA=1时：totalQBlocks按cuSeqLengthsQ差分得到的存储长度分块；sparse分块与QKV寻址均按该存储长度，不以seqused重切分；topK须≥`sparseBlockCount`中所有元素的最大值，A2/A3 上限为512，A5 上限为256。
- seqused与对应cu前缀和同时传入时（Q侧，及layoutKv为TND时的KV侧）：分核/任务空间按各batch实际有效长度（seqused）累加；各batch的seqused元素须≤对应cu存储长度，且须与Metadata侧完全一致。
- layoutKv为"PA_BBND"时须传`sequsedKvOptional`，不传`cuSeqLengthsKvOptional`。
- 输入query、key、value的数据类型必须一致。
- 输入query的headNum为N1，输入key和value的headNum为N2，则N1 >= N2且N1 % N2 == 0；groupSize=N1/N2当前须≤128。
- PA_BBND下key/value仅dim0（物理页轴）可非连续；页内blockSize×N2×D须连续。

## 调用说明

### A2/A3实现

- 内部KV基块为256 token，独立于用户传入的稀疏块Y和PA页大小128；调用方无需设置基块。
- K按128-token子块生成搬运记录，V复用记录；连续片段可合并，满足地址和步长条件的两个稀疏块可用pair搬运。随机索引或随机页表不满足合并条件时使用对应回退路径。
- 满足条件的宽tile路径支持分段V搬运与PV计算重叠；QK的K搬运使用双缓冲。分核任务使用每核游标和KV256阶段记录。
- 搬运记录、宽tile和流水线属于内部实现，不改变稀疏索引、残块或输出语义。

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| aclnn API | [test_aclnn_generic_block_sparse_attention](./examples/test_aclnn_generic_block_sparse_attention.cpp) | 通过[aclnnGenericBlockSparseAttention](./docs/aclnnGenericBlockSparseAttention.md)两段式接口调用GenericBlockSparseAttention算子 |
| PyTorch API | - | 通过[generic_block_sparse_attention](../../attention/generic_block_sparse_attention/docs/torchapi_generic_block_sparse_attention.md)接口调用generic_block_sparse_attention算子。 |
