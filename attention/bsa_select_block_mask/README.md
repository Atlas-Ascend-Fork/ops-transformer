# BSASelectBlockMask

## 产品支持情况

| 产品                                       | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT系列产品</term>     |    √    |
| <term>Atlas A3训练系列产品</term>           |    √    |
| <term>Atlas A3推理系列产品</term>           |    √    |
| <term>Atlas A2训练系列产品</term>           |    √    |
| <term>Atlas A2推理系列产品</term>           |    √    |
| <term>Atlas 200I/500 A2推理产品</term>      |    ×    |
| <term>Atlas推理系列产品</term>              |    ×    |
| <term>Atlas训练系列产品</term>              |    ×    |

## 功能说明

- 算子功能：aclnnBSASelectBlockMask是BSA（BlockSparseAttention）的前置算子，负责根据Query和Key的内容动态生成blockSparseMask，使BSA的调用链从"手动提供掩码"变为"根据Q/K内容自适应选择稀疏模式"。
- 计算公式：

  设blockShape = [blockShapeX, blockShapeY]，Sq是query最大序列长度，Skv是key最大序列长度, 则压缩后块数：

  $$
  Xblocks = \lceil Sq / blockShapeX \rceil,\quad Yblocks = \lceil Skv / blockShapeY \rceil
  $$

  **Step1：均值池化压缩 (Mean Pooling Compression)**

  对每个block，设L为block大小、当前batch从该block起始位置起的剩余序列长度和对应actualBlockLen三者的最小值（未传入actualBlockLen时该项取block大小）。L > 0时，对该block的前L个token取均值；L = 0时，压缩向量填0。即使actualBlockLen为null，尾块的有效token数也可能小于block大小。

  **Step2a：QK Matmul**

  $$
  score[b, n, x, y] = scale \cdot \sum_{d=0}^{D-1} q\_compressed[b, n, x, d] \cdot k\_compressed[b, n, y, d]
  $$

  **Step2b：Softmax**

  $$
  attn\_score[b, n, x, y] = softmax(score[b, n, x, :]) = \frac{\exp(score[b, n, x, y] - m_{final})}{l_{final}}
  $$

  **Step2c：二次池化压缩 (Post-Softmax Mean Pooling)**

  当postBlockShape非null时，对attn_score做二次均值池化，生成粗粒度pooled_score：

  $$
  postXBlocks = \lceil Xblocks / postBlockShapeX \rceil,\quad postYBlocks = \lceil Yblocks / postBlockShapeY \rceil
  $$

  $$
  pooled\_score[b, n, px, py] = \frac{1}{|R_{px,py}|} \sum_{x \in R_x(px)} \sum_{y \in R_y(py)} attn\_score[b, n, x, y]
  $$

  当postBlockShape为null时，跳过此步骤，pooled_score = attn_score。

  **Step3：块选择**

  对每个batch和head，设M为其有效二维块网格的块数（启用二次池化时使用池化后的网格）。当M > 1时：

  - sparsity <= 0.5：保留分数最大的max(1, round(sparsity * M))个块。
  - sparsity > 0.5：剔除分数最小的max(1, round((1 - sparsity) * M))个块，保留其余块。

  这里正数的round按kernel计算方式加0.5后截断。M = 1时保留该块。BNSD场景的最大网格块数须大于1。未启用二次池化时使用attn_score，否则使用pooled_score。保留块的位置集合记为\(\mathcal{indices}\)。

  **Step4：生成BlockSparseMask**

  当postBlockShape非null时，输出直接为**粗粒度**mask（shape为[B, N, postXBlocks, postYBlocks]，二次pooling拆分语义，无需展开）：

  $$
  blockSparseMaskOut[b, n, px, py] =
  \begin{cases}
  1 & (px, py) \in \mathcal{indices} \\
  0 & (px, py) \notin \mathcal{indices}
  \end{cases}
  $$

  当postBlockShape为null时，输出为细粒度mask（shape为[B, N, Xblocks, Yblocks]），直接逐元素生成：

  $$
  blockSparseMaskOut[b, n, x, y] =
  \begin{cases}
  1 & (b, n, x, y) \in \mathcal{indices} \\
  0 & (b, n, x, y) \notin \mathcal{indices}
  \end{cases}
  $$

- 数据排布格式：

  BSASelectBlockMask输入query、key的数据排布格式支持从多种维度排布解读，可通过qInputLayout和kvInputLayout传入。为了方便理解后续支持的具体排布格式（如BNSD、TND等），此处先对排布格式中各缩写字母所代表的维度含义进行统一说明：

  - B：表示输入样本批量大小（Batch）
  - T：B和S合轴紧密排列的长度（Total tokens）
  - S：表示输入样本序列长度（Seq-Length）
  - H：表示隐藏层的大小（Head-Size）
  - N：表示多头数（Head-Num）
  - D：表示隐藏层最小的单元尺寸，需满足D = H / N（Head-Dim）

- 当前支持的布局：

  - qInputLayout: "TND" "BNSD"
  - kvInputLayout: "TND" "BNSD"

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|-----|-----------|------|----------|----------|
| query | 输入 | 注意力计算中的query矩阵，即公式中的`query`。 | FLOAT16、BFLOAT16 | ND |
| key | 输入 | 注意力计算中的key矩阵，即公式中的`key`。 | 数据类型与query保持一致 | ND |
| block_shape | 输入 | 稀疏块形状数组，指定每个稀疏块的二维尺寸（行数和列数），即公式中的`blockShape`。必须包含至少两个元素`[blockShapeX, blockShapeY]`：blockShapeX为Q方向块大小，blockShapeY为KV方向块大小；如不配置（传nullptr），算子将默认blockShapeX = 128，blockShapeY = 128。 | INT64 | - |
| post_block_shape | 输入 | 二次池化块形状数组，指定Softmax后二次pooling的块分组大小（block单位），即公式中的`postBlockShape`。可选输入：传入`[postBlockShapeX, postBlockShapeY]`时启用二次pooling，TopK在粗粒度pooled_score上选择，输出mask为粗粒度[B, N, postXBlocks, postYBlocks]（二次pooling拆分语义，直接输出选择结果，无展开）；传入nullptr时禁用二次pooling（向后兼容），TopK直接在attn_score上选择，输出mask为细粒度[B, N, Xblocks, Yblocks]。postBlockShapeX/postBlockShapeY为8的倍数，表示block分组数。 | INT64 | - |
| actual_seq_lengths | 输入 | 每个batch的query的实际序列长度，即公式中Sq在各batch的实际有效值。用于描述变长序列场景下（即含有Padding填充数据的场景），每个Batch中实际有效的query token数量。 | INT64 | - |
| actual_seq_lengths_kv | 输入 | key的实际序列长度，即公式中Skv在各batch的实际有效值。用于描述变长序列场景下（即含有Padding填充数据的场景），每个Batch中实际有效的key token数量。 | INT64 | - |
| actual_block_len_query | 输入 | 每个query block内实际压缩的有效seq长度，即公式中的`actualBlockLenQuery`。用于部分压缩场景（如末尾不完整块或仅压缩有效token），对每个block内前actual_block_len_query个token取均值，并按当前batch剩余序列长度截断；如不配置（传nullptr），对每块有效token取均值。 | INT64 | - |
| actual_block_len_key | 输入 | 每个key block内实际压缩的有效seq长度，即公式中的`actualBlockLenKey`。用于部分压缩场景（如末尾不完整块或仅压缩有效token），对每个block内前actual_block_len_key个token取均值，并按当前batch剩余序列长度截断；如不配置（传nullptr），对每块有效token取均值。 | INT64 | - |
| q_input_layout | 输入 | query的数据排布格式。指示输入张量在内存中的具体排布。当前仅支持"TND"、"BNSD"，且需与kv_input_layout保持一致。 | String | - |
| kv_input_layout | 输入 | key的数据排布格式。指示输入张量在内存中的具体排布。当前仅支持"TND"、"BNSD"，且需与q_input_layout保持一致。 | String | - |
| num_key_value_heads | 输入 | key的注意力头数，即公式中key张量的多头数N。当前仅支持MHA，必须与query的head数保持一致。 | Int | - |
| scale_value | 输入 | 缩放系数，即公式中的`scale`。用于注意力分数的归一化处理，一般设置为1 / sqrt(D)。 | Float | - |
| sparsity | 输入 | 稀疏度保留比例，即公式中的`sparsity`。指定attn_score中需要保留的块位置占全部块位置的比例。取值范围(0.0, 1.0)。 | Float | - |
| block_sparse_mask_out | 输出 | 块状稀疏掩码输出，即公式中的`blockSparseMaskOut`。表示根据Q/K内容自适应生成的稀疏pattern，可直接作为BSA算子的blockSparseMask输入。shape随post_block_shape变化：post_block_shape非null时为[B, N, postXBlocks, postYBlocks]（粗粒度），postXBlocks = ceilDiv(Xblocks, postBlockShapeX)，postYBlocks = ceilDiv(Yblocks, postBlockShapeY)；post_block_shape为null时为[B, N, Xblocks, Yblocks]（细粒度）。值为1表示该block参与注意力计算，值为0表示不参与。 | INT8 | ND |

## 约束说明

- actual_seq_lengths在q_input_layout为"TND"时必选；actual_seq_lengths_kv在kv_input_layout为"TND"时必选。
- query张量Shape中head维度大小记为N1，key张量Shape中head维度大小记为N2。必须满足N1 = N2（仅支持MHA）。
- headDim = 128。
- block_shape中的blockShapeX和blockShapeY必须为8的倍数。
- 选择网格的最大块数在BNSD场景须大于1，在TND场景须大于0。post_block_shape为null时网格块数为Xblocks * Yblocks，否则为postXBlocks * postYBlocks。
- query和key的数据类型必须一致，仅支持FLOAT16和BFLOAT16。
- block_sparse_mask_out数据类型为INT8（二值：0或1）。
- post_block_shape为可选输入，传入nullptr时禁用二次pooling（向后兼容）；传入`[postBlockShapeX, postBlockShapeY]`时启用，postBlockShapeX/postBlockShapeY为8的倍数，表示block分组数。
- actual_block_len_query / actual_block_len_key若非null，每个元素取值范围[0, blockShapeX] / [0, blockShapeY]；为null时对每块有效token取均值。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| aclnn API | [test_aclnn_bsa_select_block_mask](examples/test_aclnn_bsa_select_block_mask.cpp) | 只支持MHA和格式为TND\BNSD的场景，通过[aclnnBSASelectBlockMask](docs/aclnnBSASelectBlockMask.md)接口方式调用BSASelectBlockMask算子。 |
