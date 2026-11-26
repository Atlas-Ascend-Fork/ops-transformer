# BlockSparseAttentionGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT系列产品</term>|      √     |
|<term>Atlas A3训练系列产品</term>|      √     |
|<term>Atlas A3推理系列产品</term>|      ×     |
|<term>Atlas A2训练系列产品</term>|      √     |
|<term>Atlas A2推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2推理产品</term>                                         |    ×    |
|<term>Atlas推理系列产品</term>                                                 |    ×    |
|<term>Atlas训练系列产品</term>                                                 |    ×    |

## 功能说明

* ​算子功能​：aclnnBlockSparseAttention稀疏注意力反向计算，支持灵活的块级稀疏模式，通过BlockSparseMask指定每个Q块选择的KV块，实现高效的稀疏注意力计算。
* ​计算公式​：

  稀疏块大小：$blockShapeX×blockShapeY$，BlockSparseMask指定稀疏模式。

 blockShapeX 和 blockShapeY 均须大于 0。Ascend 950 另要求两者均为 16 的整数倍。BlockSparseMask 在逻辑块粒度上定义稀疏 pattern，每个 mask 元素控制一个 blockShapeX × blockShapeY 的连续块是否参与计算。
  已知正向计算公式为：

  $$
  attentionOut=Softmax(Mask(scale⋅query⋅key_{sparse}^{T},  atten\_mask))⋅value_{sparse}
  $$

  为方便表达，以变量$S$和$P$表示计算公式：

  $$
  S = Mask(scale⋅query⋅key_{sparse}^{T},atten\_mask)
  $$

  $$
  P = SoftMax(S)
  $$

  $$
  V = value_{sparse}
  $$

  $$
  Out = PV
  $$

  则反向计算公式为：

  $$
  softmax\_grad = softmaxGrad(dOut, attentionOut)
  $$

  $$
  dP=dOut * V^T
  $$

  $$
  dS = P * (dP-softmax\_grad)
  $$

  $$
  dV=P^T * dOut
  $$

  $$
  dQ=(dS*K)*scale
  $$

  $$
  dK=(dS^T*Q)*scale
  $$

BlockSparseAttentionGrad输入dout、query、key、value, attentionOut的数据排布格式支持从多种维度排布解读，可通过qInputLayout和kvInputLayout传入。为了方便理解后续支持的具体排布格式（如 BNSD、TND 等），此处先对排布格式中各缩写字母所代表的维度含义进行统一说明：

* B：表示输入样本批量大小（Batch）
* T：B和S合轴紧密排列的长度（Total tokens）
* S：表示输入样本序列长度（Seq-Length）
* H：表示隐藏层的大小（Head-Size）
* N：表示多头数（Head-Num）
* D：表示隐藏层最小的单元尺寸，需满足D=H/N（Head-Dim）

qInputLayout 和 kvInputLayout 必须一致。各产品支持的布局如下：

- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：BNSD、TND。
- <term>Ascend 950PR&950DT系列产品</term>：BNSD、TND、BSND。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1550px">
<colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 145px">
</colgroup>
<thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>使用说明</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
    <th>非连续Tensor</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>dout（aclTensor*）</td>
    <td>输入</td>
    <td>反向输出梯度，代表最终输出对当前算子的梯度信息。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalQTokens, headNum, headDim]。</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li><li>BSND: [batch, maxQSeqLength, headNum, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>query（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的查询向量，即公式中的query。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalQTokens, headNum, headDim]。</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li><li>BSND: [batch, maxQSeqLength, headNum, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>key（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的键向量，即公式中的key。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalKTokens, numKeyValueHeads, headDim]。</li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li><li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>value（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的值向量，即公式中的value。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalVTokens, numKeyValueHeads, headDim]。</li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li><li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>attentionOut（aclTensor*）</td>
    <td>输入</td>
    <td>正向 BlockSparseAttention 计算的输出结果，即公式中的attentionOut。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalQTokens, headNum, headDim]。</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li><li>BSND: [batch, maxQSeqLength, headNum, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>softmaxLse（aclTensor*）</td>
    <td>输入</td>
    <td>Softmax计算的log-sum-exp中间结果。用于反向计算梯度的对数和指数逆推。</td>
    <td>TND 查询传 [totalQTokens, headNum, 1]。BNSD 和 BSND 查询均传 [batch, headNum, maxQSeqLength, 1]。</td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>blockSparseMaskOptional（aclTensor*）</td>
    <td>输入</td>
    <td>块状稀疏掩码，表示实际的稀疏pattern。决定哪些block实际参与注意力计算。</td>
    <td>必传。shape 为 [batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]，headNum 为 Q 头数。TND 的两个最大长度分别取 Q 和 KV 侧最长样本的长度。1 表示保留该块，0 表示跳过；所有块参与时传全 1 掩码，尾块覆盖剩余 token。</td>
    <td>BOOL, UINT8</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>attenMaskOptional（aclTensor*）</td>
    <td>输入</td>
    <td>注意力掩码，即公式中的atten_mask。用于屏蔽不应参与计算的特定token。</td>
    <td>maskType=0 时必须传 nullptr。maskType=1 时（Atlas A2/A3 训练系列产品）必须传 INT32 张量，shape 为 [batch, headNum, M, 2]。M 的范围为 max(qBlockNum, kvBlockNum) 至 UINT32_MAX，qBlockNum 和 kvBlockNum 是 blockSparseMaskOptional 的后两维。在索引 i 处，最后一维的两个值依次为第 i 个 Q 块和第 i 个 KV 块的有效前缀长度，范围是 0 至对应块大小。取 0 时跳过对应块。<br>某一侧没有对应块时，该侧字段不参与计算，建议填 0。例：Q 有 2 块、KV 有 3 块，M=4，若各块有效长度均为 16，则一个样本的一个头可填 [[16,16], [16,16], [0,16], [0,0]]。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>blockShapeOptional（aclIntArray*）</td>
    <td>输入</td>
    <td>Q 方向和 KV 方向的稀疏块大小。</td>
    <td>必传。至少包含两个元素 [blockShapeX, blockShapeY]，两者均须大于 0，可以不同。Ascend 950 另要求两者均为 16 的整数倍。</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>actualSeqLengthsOptional（aclIntArray*）</td>
    <td>输入</td>
    <td>TND 布局下的 Q 序列边界。</td>
    <td>TND 布局必传，每个样本对应一个元素。例如，两批长度分别为 64、96 时，Atlas A2/A3 传逐样本长度 [64, 96]，Ascend 950 传累计结束位置 [64, 160]。950 的数组恰有 B 个结束位置，省略起始偏移 0，末项必须等于 query 的 T 轴长度。</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>actualSeqLengthsKvOptional（aclIntArray*）</td>
    <td>输入</td>
    <td>TND 布局下的 KV 序列边界。</td>
    <td>TND 布局必传，每个样本对应一个元素。例如，两批长度分别为 64、96 时，Atlas A2/A3 传逐样本长度 [64, 96]，Ascend 950 传累计结束位置 [64, 160]。950 的数组恰有 B 个结束位置，省略起始偏移 0，末项必须等于 key 的 T 轴长度。</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>qInputLayout（char*）</td>
    <td>输入</td>
    <td>query的数据排布格式。指示输入张量在内存中的具体排布（如连续或合轴排列）。</td>
    <td>须与另一布局参数一致。Atlas A2/A3 支持 BNSD、TND，Ascend 950 还支持 BSND。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>kvInputLayout（char*）</td>
    <td>输入</td>
    <td>key和value的数据排布格式。指示输入张量在内存中的具体排布。</td>
    <td>须与另一布局参数一致。Atlas A2/A3 支持 BNSD、TND，Ascend 950 还支持 BSND。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numKeyValueHeads（int64_t）</td>
    <td>输入</td>
    <td>key/value的注意力头数。用于支持GQA（分组查询注意力）机制下的头数比例映射。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maskType（int64_t）</td>
    <td>输入</td>
    <td>注意力计算中的掩码类型。指定采用何种预设规则的掩码逻辑。</td>
    <td>当前支持传0和1：<ul><li>0：仅使用块稀疏掩码，attenMaskOptional 必须传 nullptr。</li><li>1：PerBlockSizeMask场景，按块级变长掩码计算，需配置attenMaskOptional。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scaleValue（double）</td>
    <td>输入</td>
    <td>缩放系数，即公式中的scale。用于注意力分数的归一化处理。</td>
    <td>一般设置为D^-0.5。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>preTokens（int64_t）</td>
    <td>输入</td>
    <td>滑窗向前包含的token数量。限制当前token只能与前方的多少个历史token计算注意力。</td>
    <td>用于滑窗attention场景，当前不支持滑窗attention，只支持传入2147483647。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>nextTokens（int64_t）</td>
    <td>输入</td>
    <td>滑窗向后包含的token数量。限制当前token只能与后方的多少个未来token计算注意力。</td>
    <td>用于滑窗attention场景，当前不支持滑窗attention，只支持传入2147483647。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dq（aclTensor*）</td>
    <td>输出</td>
    <td>query的梯度输出结果，即公式中的dq。</td>
    <td>不支持空Tensor。<br>数据类型和shape与输入query保持一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dk（aclTensor*）</td>
    <td>输出</td>
    <td>key的梯度输出结果，即公式中的dk。</td>
    <td>不支持空Tensor。<br>数据类型和shape与输入key保持一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dv（aclTensor*）</td>
    <td>输出</td>
    <td>value的梯度输出结果，即公式中的dv。</td>
    <td>不支持空Tensor。<br>数据类型和shape与输入value保持一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize（uint64_t*）</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor（aclOpExecutor**）</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子计算流程。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
</tbody>
</table>

<ul>
- <term>Atlas A2训练系列产品</term>、<term>Atlas A3训练系列产品</term>:
不支持FLOAT8_E5M2、FLOAT8_E4M3FN。
</ul>

<ul>
- <term>Ascend 950PR&950DT系列产品</term>:
maskType 仅支持 0，attenMaskOptional 必须传 nullptr。
</ul>

## 约束说明

* 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
* actualSeqLengthsOptional在qInputLayout为“TND”时必选；actualSeqLengthsKvOptional在kvInputLayout为“TND”时必选。
* Ascend 950 上，BNSD/BSND 布局要求 Q、K、V 的序列长度相等，TND 布局要求三者的总 token 数相等。
* Ascend 950 产品要求 HeadDim 等于 128。
* 根据算子支持的输入 Layout，query 张量 Shape 中对应的 head 维度大小记为 N1，key 和 value 张量 Shape 中对应的 head 维度大小记为 N2。必须满足N1 % N2 == 0。支持MQA、MHA、GQA场景。
* actualSeqLengthsOptional与actualSeqLengthsKvOptional相关约束：当qInputLayout或kvInputLayout为非"TND"时，会忽略这两个入参。

## 调用说明

| 调用方式           | 调用样例                                                                                                          | 说明                                                                                                                  |
|----------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_block_sparse_attention_grad](examples/test_aclnn_block_sparse_attention_grad.cpp)                     | 通过[aclnnBlockSparseAttentionGrad](docs/aclnnBlockSparseAttentionGrad.md)接口方式调用BlockSparseAttentionGrad算子。                   |
| PyTorch API | - | 通过[torchapi_block_sparse_attention_grad](docs/torchapi_block_sparse_attention_grad.md)接口调用block_sparse_attention_grad算子。 |
