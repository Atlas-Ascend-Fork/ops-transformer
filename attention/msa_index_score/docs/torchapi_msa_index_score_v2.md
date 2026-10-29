# msa\_index\_score\_v2

## 产品支持情况

<!-- npu="910b" id1 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- **接口功能**：封装 `aclnnMsaIndexScoreV2`，计算 MSA Index Branch 的 block score, 支持指定 layout_query 和 pool_block_size。
- **公式**：

$$
score = Maxpool[(scale\cdot)Q_{idx}@K_{idx}^{T}+atten\_mask]+local\_mask
$$

- `sparse_mode=0`：无因果；`sparse_mode=3`：rightDownCausal（须传 `[2048,2048]` `atten_mask`）。
- `start_loc[B]`：当前 query 所在逻辑 block 索引，用于 `local_mask`。
- `init_blocks` / `local_blocks`：Maxpool 之后的强制高分块；与 Triton raw score 对齐时置 0。默认 `0` / `1`。
- Ascend 950 额外支持 query/key 同型 `torch.float8_e4m3fn` / `torch.float8_e5m2` / `torch_npu.hifloat8`（无 `scale`）。`hifloat8.npu()` 当前不安全，测试脚本跳过；kernel 已注册。
- `q_len` / `kv_len` 允许为 0（含整 batch）。对应请求跳过 QK；空 KV 的 score 为 `-inf`。
- A2/A3 与 Ascend 950 短 decode 按估计 M-task 启动 MIX，不打满空核。950 短 M 长 S 再按 KV stile 切 `kvChunks`。
- PageAttention `key` 允许 dim0（物理 page）非连续，须保持为 view（不要 `.contiguous()`）；size>1 的非首轴必须连续，否则 tiling 拒绝。TND / BSND / BNSD 不允许非连续。
- PageAttention `block_table` 第二维可以大于实际 KV 逻辑 block 数（例如 vLLM 预分配宽表）。950 C2UB 对 score 末维超过 256 列按窗 flush。

## 函数原型

```python
cann_ops_transformer.msa_index_score_v2(
    query,
    key,
    start_loc,
    *,
    block_table=None,
    scale=None,
    atten_mask=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout_key="BBND",
    sparse_mode=3,
    init_blocks=0,
    local_blocks=1,
    layout_query="TND",
    pool_block_size=128,
) -> Tensor
```

## 参数说明

<table style="undefined;table-layout: fixed; width: 1180px"><colgroup>
<col style="width: 160px">
<col style="width: 120px">
<col style="width: 360px">
<col style="width: 280px">
<col style="width: 260px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>dtype</th>
    <th>shape</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>query</td>
    <td>必选</td>
    <td>$Q_{idx}$。由 <code>layout_query</code> 指定：TND / BSND / BNSD，默认 TND。BSND、BNSD 仅 Ascend 950，仅 fp16/bf16。</td>
    <td>fp16/bf16；950 另支持 fp8_e4m3fn / fp8_e5m2 / hifloat8</td>
    <td><code>[T1,N1,D]</code> / <code>[B,S1,N1,D]</code> / <code>[B,N1,S1,D]</code></td>
  </tr>
  <tr>
    <td>key</td>
    <td>必选</td>
    <td>$K_{idx}$。非分页 TND / BSND / BNSD 必须与 query 同布局；分页 BBND / BNBD 可配任意 query。A2/A3 与 950 上 PA 允许 dim0 非连续。TND 缺省逻辑块长 128，每 head 一次块内 max；<code>pool_block_size=16</code> 时走两级 maxpool（仅 950 fp16/bf16）。BSND / BNSD 仅 Ascend 950，逻辑块长 16，仅 fp16/bf16。</td>
    <td>同 query 或 int8（int8 配 fp16 query；仅 TND 与 PA）</td>
    <td><code>[T2,N2,D]</code> / <code>[B,S2,N2,D]</code> / <code>[B,N2,S2,D]</code> / <code>[NP,P,N2,D]</code> / <code>[NP,N2,P,D]</code></td>
  </tr>
  <tr>
    <td>start_loc</td>
    <td>必选</td>
    <td>query 所在逻辑 block</td>
    <td>int32</td>
    <td><code>[B]</code></td>
  </tr>
  <tr>
    <td>block_table</td>
    <td>PA 必选</td>
    <td>逻辑→物理 page；TND / BSND / BNSD 不传</td>
    <td>int32</td>
    <td><code>[B,MB]</code></td>
  </tr>
  <tr>
    <td>scale</td>
    <td>量化必选</td>
    <td>反量化</td>
    <td>float32</td>
    <td>PA <code>[NP,N2,P]</code>；TND <code>[T2,N2]</code>。BSND / BNSD 不支持 int8</td>
  </tr>
  <tr>
    <td>atten_mask</td>
    <td>mode=3 必选</td>
    <td>压缩下三角模板</td>
    <td>int8</td>
    <td><code>[2048,2048]</code></td>
  </tr>
  <tr>
    <td>actual_seq_qlen</td>
    <td>必选</td>
    <td>query 前缀和。BSND / BNSD 时每个请求长度不超过 $S1$</td>
    <td>int32</td>
    <td><code>[B+1]</code></td>
  </tr>
  <tr>
    <td>actual_seq_klen</td>
    <td>必选</td>
    <td>PA 为各请求 S2；TND / BSND / BNSD 为 key 前缀和</td>
    <td>int32</td>
    <td><code>[B]</code> / <code>[B+1]</code></td>
  </tr>
  <tr>
    <td>layout_key</td>
    <td>可选</td>
    <td>key 布局：<code>TND</code> / <code>BSND</code> / <code>BNSD</code> / <code>BBND</code> / <code>BNBD</code>，默认 <code>BBND</code>。非分页必须与 <code>layout_query</code> 相同</td>
    <td>str</td>
    <td>-</td>
  </tr>
  <tr>
    <td>sparse_mode</td>
    <td>可选</td>
    <td>0 / 3，默认 3</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>init_blocks</td>
    <td>可选</td>
    <td><code>local_mask</code> 头部强制块数，默认 0</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>local_blocks</td>
    <td>可选</td>
    <td><code>local_mask</code> 局部窗口，默认 1</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>layout_query</td>
    <td>可选</td>
    <td>query 布局：<code>TND</code> / <code>BSND</code> / <code>BNSD</code>，默认 <code>TND</code>。BSND、BNSD 仅 Ascend 950。分页 key 可配任意 query；非分页 key 必须相同</td>
    <td>str</td>
    <td>-</td>
  </tr>
  <tr>
    <td>pool_block_size</td>
    <td>可选</td>
    <td>打分池化块长：每个 score 归约的 token 数。<code>16</code> / <code>128</code>，默认 128。与 BBND / BNBD 的存储页长不是同一个量；分页时不能超过页长。128 走原单次块内 maxpool；16 走两级 maxpool，仅 Ascend 950、fp16/bf16，且 <code>N1</code> 整除 128。BSND/BNSD 固定 16</td>
    <td>int</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## 输出

`[N1, T1, RoundUp(MB,16)]` float32。`T1` 是 `actual_seq_qlen` 的有效 token 总数。逻辑块长 16（BSND / BNSD，或 BBND / BNBD 页长 16）时，先在每个 head 内对 block 取 max，再对每个 block 取 head 间 max，`N1` 维上是同一个值的复制；`N1` 必须整除 128。块长 128 时每个 head 只做一次块内 max，不做 head 间 max。

## 调用示例

- PageAttention BBND / BNBD：

```python
import torch
import torch_npu
import cann_ops_transformer

T1, N1, N2, D, P = 32, 8, 1, 128, 128
B, NP, MB = 1, 8, 2
query = torch.randn(T1, N1, D, dtype=torch.float16).npu()
key_bbnd = torch.randn(NP, P, N2, D, dtype=torch.float16).npu()
# BNBD 与 BBND 仅 N/P 轴对调：key_bnbd = key_bbnd.permute(0, 2, 1, 3).contiguous()
block_table = torch.arange(B * MB, dtype=torch.int32).view(B, MB).npu()
actual_seq_qlen = torch.tensor([0, T1], dtype=torch.int32).npu()
actual_seq_klen = torch.tensor([256], dtype=torch.int32).npu()
start_loc = torch.tensor([1], dtype=torch.int32).npu()
atten_mask = torch.zeros(2048, 2048, dtype=torch.int8).npu()
score = cann_ops_transformer.msa_index_score_v2(
    query, key_bbnd, start_loc,
    block_table=block_table, atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen,
    layout_key="BBND")
```

- TND packed key（`layout_key="TND"`，不传 `block_table`，`actual_seq_klen` 为 `[B+1]` 前缀和）：

```python
T2 = 256
key_tnd = torch.randn(T2, N2, D, dtype=torch.float16).npu()
actual_seq_klen_tnd = torch.tensor([0, T2], dtype=torch.int32).npu()
score_tnd = cann_ops_transformer.msa_index_score_v2(
    query, key_tnd, start_loc,
    atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen_tnd,
    layout_key="TND")
```

- query BSND、key 仍为 BBND（`layout_query` 与 `layout_key` 可以不同）：

```python
S1 = 32
query_bsnd = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
score_bsnd = cann_ops_transformer.msa_index_score_v2(
    query_bsnd, key_bbnd, start_loc,
    block_table=block_table, atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen,
    layout_key="BBND", layout_query="BSND")
```

- query 与 key 同为 BNSD（仅 Ascend 950，fp16/bf16，不传 `block_table`，逻辑块长 16）：

```python
S2 = 256
query_bnsd = torch.randn(B, N1, S1, D, dtype=torch.float16).npu()
key_bnsd = torch.randn(B, N2, S2, D, dtype=torch.float16).npu()
actual_seq_klen_bnsd = torch.tensor([0, S2], dtype=torch.int32).npu()
score_bnsd = cann_ops_transformer.msa_index_score_v2(
    query_bnsd, key_bnsd, start_loc,
    atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen_bnsd,
    layout_key="BNSD", layout_query="BNSD")
```
