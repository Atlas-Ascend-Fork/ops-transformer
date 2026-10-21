# sparse_lightning_indexer 算子测试框架

基于 pytest 测试框架，实现 sparse_lightning_indexer（candidate consumer / candidate_mode==2）算子的功能验证：

- **CPU 侧**：复现算子功能用以生成 golden 数据（无 leak 两档掩码 → topk，nvalid 不足 topk 槽位 -1 补齐；候选外索引一律不输出）
- **NPU 侧**：通过 TorchNPU 进行算子直调获取实际数据
- **精度对比**：复用 LIV2 的 `result_compare_method.check_result`（多重集合门 →
  `compare_topk_valid` 官方规则 → -1 槽硬校验）

golden 基础设施（GeneralizedLIV2 / 用例数据生成）复用同级 lightning_indexer_v2 算子的
tests/pytest 目录（共享 golden 模块），本目录仅实现 consumer 语义扩展。

## 文件结构

- `sparse_lightning_indexer_golden.py`      # 标杆：consumer golden（无 leak 掩码 + topk + rv=1 values）+ NPU driver + 候选构造
- `test_sparse_lightning_indexer_single.py` # 单用例功能测试套（A2 老功能 + consumer 语义矩阵 + RV1 系列）
- `test_sparse_lightning_indexer_paramset.py` # 单用例入参配置（37 组，仅 ascend910b/910_93 启用）
- `test_sparse_lightning_indexer_reject.py` # host 校验拒绝测试套（C2-C8 / R18，22 条）
- `test_sparse_lightning_indexer_model.py`  # 模型场景测试套（与 LIV2 模型套一一配对）
- `test_run.sh`                             # 执行脚本（single/reject/model 三种命令）
- `pytest.ini`                              # 测试标记
- `sli_candidate_golden.py`                # consumer 候选块级 TopK 的 CPU 标杆（纯 numpy/torch，无重依赖）
- （复用 ../../lightning_indexer_v2/tests/pytest/：lightning_indexer_v2_golden.py、
   result_compare_method.py —— LIV2 标杆基础设施与精度比较器）

## 单用例功能套件（`test_sparse_lightning_indexer_single.py`）

参数集位于 `test_sparse_lightning_indexer_paramset.py`，37 组用例（27 个 LIV2 基础字段 +
3 个 consumer 扩展字段 cand_spec / cand_blocks / cand_block_size）：

| 维度 | 用例 |
|---|---|
| dtype（老功能） | FP16 / BF16 |
| layout 四组合（老功能） | BSND×BSND / TND×TND / BSND×PA / TND×PA |
| mask / cmp_ratio（老功能） | mode3 / mode3+cmp2（残差）/ mode3+cmp128（上界） |
| 变长（老功能） | seqused_k 变长 / TND cu_seqlens 不均匀 |
| topk 边界（C5） | 2048 上界 / topk>可达数（-1 槽补齐） |
| candidate_block_size 全 6 档（新功能） | 2 / 4 / 8 / 16 / 32 / 64 |
| candBlocks 边界（C3） | 2048 上界 / numBlocks<candBlocks 满覆盖退化 |
| 候选来源四形态（新功能） | self / dirty（重复+越界+-1 脏数据免疫）/ missing（缺块，nvalid<topk 触发 -1 补坑）/ full（全候选） |
| regime 2 无 leak（新功能核心语义） | self 档（候选覆盖 < topk < 可达）/ missing 档：剩余槽位全 -1，不输出候选外索引 |
| return_value=1 专项（新功能，RV1 系列 8 组） | FP16 / BF16 / TND / PA / dirty / missing / mask3+cmp2 / regime2（全部 rv=1，values=真实分 fp32） |
| mask3 × 候选交互（新功能） | dirty 候选 + causal |
| PA 块边界（老功能） | block_size 16 / 1024 |
| G 头组（老功能） | N1=32 / 64 |
| TND 变长 + 候选（新功能） | TND 输出布局 [T,N2,topk] 上的候选消费 |

## host 校验拒绝套件（`test_sparse_lightning_indexer_reject.py`）

| 规则 | 用例 |
|---|---|
| C3 candBlocks 非法 | 0（下界外）/ 100（非 64 倍数）/ 2112（>2048）/ 显式 shape 末维 4096 |
| C2 rank/dim 校验 | BSND q + rank3 候选 / TND q + rank4 候选 / B 维错 / S1 维错 |
| C4 blockSize 非法 | 12（非 2 幂）/ 1（<2）/ 128（>64） |
| C6 return_value | rv=2（∉{0,1}）拒绝 / rv=1 接受（开放后正反双面） |
| C7 block_length | 非空（numel=3）拒绝 / 空 tensor（numel=0）接受 |
| C5 topk | 2049（>2048） |
| C8 cmpRatio | 3（非 2 幂）/ 256（2 的幂但 >128） |
| R18 arch22 不消费入参 | max_seqlen_q=4 / seqused_q 非空 / output_idx_offset 非空 / metadata 非空 |

注：C6（return_value ∈ {0,1}）经 torch kwargs 透传触发：rv=2 拒绝 + rv=1 接受两条
（rv=1 开放后 sparse_values 输出真实 TopK 分数）。

## 模型场景（BSND，与 LIV2 模型套一一配对）

| 参数 | 取值 |
|---|---|
| layout | BSND×BSND（泛化：BSND×PA_BBND） |
| N1 (q_head_num) | 32 / 16 / 8 / 4 |
| D (head_dim) | 128 |
| topK | 512 |
| candidate_topk_blocks (candBlocks) | 2048 |
| candidate_block_size | 8 |
| S1 / S2 | 4096 / 4096（S2 为压缩后 K 长度；泛化：16K 满覆盖边界） |
| cmp_ratio | 2 |
| sparse_mode (mask_mode) | 0 / 3 |
| 候选来源 | cand_spec="self"（同 score 过参考 source 行为构造，真实 source→consumer 链路形态） |
| 其余泛化 | dtype FP16/BF16、B>1 变长 seqused_k、PA_BBND key |

用例清单（默认 10 条，`SLI_MODEL_LARGE=1` 追加 16K 边界用例）：

| 用例 | 覆盖点 |
|---|---|
| SLI_MODEL_N1_32_MODE3 | 主规格：B=1，mask_mode=3 causal，cmp_residual_k=[1] |
| SLI_MODEL_N1_16_MODE3 | N1=16，B=2 |
| SLI_MODEL_N1_8_MODE3 | N1=8，B=2 |
| SLI_MODEL_N1_4_MODE3 | N1=4，B=4 |
| SLI_MODEL_N1_32_MODE0 | sparse_mode=0（no mask），cmp_residual_k=None |
| SLI_MODEL_N1_4_MODE0 | sparse_mode=0，N1=4，B=2 |
| SLI_MODEL_N1_32_MODE3_BF16 | BF16 dtype |
| SLI_MODEL_N1_8_MODE0_BF16 | BF16 + sparse_mode=0 |
| SLI_MODEL_VARLEN_MODE3 | B=2 seqused_k=[4096, 2049]（S2 非对齐） |
| SLI_MODEL_PA_MODE3 | BSND q × PA_BBND k（block_table） |
| SLI_MODEL_S2_16K_N1_4_MODE3 | k_seq=16384（numBlocks==candBlocks 满覆盖边界），需 `SLI_MODEL_LARGE=1` |

## 参数限制

- **candidate_block_indices**：REQUIRED，INT32；BSND `[B,S1,N2,candBlocks]`；candBlocks ∈ (0, 2048] 且 64 的倍数
- **candidate_block_size**：[2,64] 且为 2 的幂（本套 8），与 source 侧必须一致
- **topk**：(0, 2048]（无 off 回退）；**return_value**：∈ {0,1}，缺省 0（=1 时 sparse_values 输出真实分，shape 与 sparse_indices 相同、dtype FLOAT32）
- **平台**：仅 ascend910b / ascend910_93

## 环境配置

1. 确认 TorchNPU 为最新版本
2. 参考 Attention 融合算子 Experimental 使用说明激活 CANN 包和自定义算子包

## 运行

```bash
# 单用例功能套（Ascend910B / 910_93）
bash test_run.sh single
python3 -m pytest -rA -s test_sparse_lightning_indexer_single.py -v -m ci

# 指定用例
bash test_run.sh single -C SLI_FP16_01,SLI_CAND_DIRTY_19
SLI_CASE_NAMES=SLI_FP16_01 python3 -m pytest -s test_sparse_lightning_indexer_single.py -m ci

# host 校验拒绝套
bash test_run.sh reject
python3 -m pytest -rA -s test_sparse_lightning_indexer_reject.py -v -m ci

# 模型场景套（全量）
bash test_run.sh model
python3 -m pytest -rA -s test_sparse_lightning_indexer_model.py -v -m ci

# 指定模型用例
SLI_CASE_NAMES=SLI_MODEL_N1_32_MODE3 python3 -m pytest -s test_sparse_lightning_indexer_model.py -m ci

# 追加 16K 满覆盖边界用例
SLI_MODEL_LARGE=1 python3 -m pytest -s test_sparse_lightning_indexer_model.py -m ci
```

算子未注册时自动 skip。
