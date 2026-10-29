# DsaSparseProlog（Experimental）

## 产品支持情况

| 产品                          | 是否支持 |
| ----------------------------- | :------: |
| <term>Atlas A3系列产品</term> |    √     |
| <term>Atlas A2系列产品</term> |    √     |

## 功能说明

`DsaSparseProlog` 是 DSA（DeepSeek Sparse Attention）稀疏路径前向的 prolog 算子（吸收模式），
计算流程如下：

- **Q 路径**：`token_x · weight_dq` → RMSNorm(`gamma_cq`) → `qr`（post-norm，供 Indexer 使用）
  → `· weight_uq_qr` → view 为 `[..., N, D+Dr]` → split 为 `q_nope`/`q_pe`
  → 同乘 `mla_scale_q_lora`，`q_pe` 再做 RoPE。
- **KV 路径**：`token_x · weight_dkv_kr` → split 为 `ct_kv`/`k_pe` → RMSNorm(`gamma_ckv`) 仅作用
  `ct_kv` → 乘 `mla_scale_kv_lora`；`k_pe` 为 MQA 单头，RoPE 后输出。

同时输出反向所需的中间量（`qr`、`q_m`、`q_rstd`、`ckv_m`、`ckv_rstd`）。

## 参数说明

| 参数名             | 输入/输出/属性 | 描述                                              | 数据类型      | 格式           |
| ------------------ | -------------- | ------------------------------------------------- | ------------- | :------------: |
| token_x            | 输入           | `(B, S, He)` 或 `(T, He)`                         | BF16 / FP16   | ND             |
| weight_dq          | 输入           | q_a_proj 权重，`(He, Hcq)`                        | BF16 / FP16   | FRACTAL_NZ / ND |
| weight_uq_qr       | 输入           | q_b_proj 权重，`(Hcq, N*(D+Dr))`                  | BF16 / FP16   | FRACTAL_NZ / ND |
| weight_dkv_kr      | 输入           | kv_a_proj_with_mqa 权重，`(He, Hckv+Dr)`          | BF16 / FP16   | FRACTAL_NZ / ND |
| rmsnorm_gamma_cq   | 输入           | `(Hcq,)`                                          | BF16 / FP16   | ND             |
| rmsnorm_gamma_ckv  | 输入           | `(Hckv,)`                                         | BF16 / FP16   | ND             |
| rope_sin           | 输入           | `(B, S, Dr)` 或 `(T, Dr)`                         | BF16 / FP16   | ND             |
| rope_cos           | 输入           | `(B, S, Dr)` 或 `(T, Dr)`                         | BF16 / FP16   | ND             |
| rmsnorm_epsilon_cq | 可选属性       | q 侧 RMSNorm eps，默认 `1e-5`                     | FLOAT         | -              |
| rmsnorm_epsilon_ckv| 可选属性       | kv 侧 RMSNorm eps，默认 `1e-5`                    | FLOAT         | -              |
| mla_scale_q_lora   | 可选属性       | q 缩放，`<=0` 时按 `sqrt(He/Hcq)` 推导            | FLOAT         | -              |
| mla_scale_kv_lora  | 可选属性       | kv 缩放，`<=0` 时按 `sqrt(He/Hckv)` 推导          | FLOAT         | -              |
| qk_nope_head_dim   | 可选属性       | `D`，默认 `128`；`N` 由 `weight_uq_qr` 宽度推导   | INT           | -              |
| is_type_demotion   | 可选属性       | 是否与 torch 拼接实现精度一致（含手动降精）       | BOOL          | -              |
| apply_rope_fusion  | 可选属性       | RoPE 融合路径选择，默认 `false`                   | BOOL          | -              |
| q_nope             | 输出           | `(B, S, N, D)` 或 `(T, N, D)`                     | BF16 / FP16   | ND             |
| q_pe               | 输出           | `(B, S, N, Dr)` 或 `(T, N, Dr)`                   | BF16 / FP16   | ND             |
| ct_kv              | 输出           | `(B, S, Hckv)` 或 `(T, Hckv)`                     | BF16 / FP16   | ND             |
| k_pe               | 输出           | `(B, S, 1, Dr)` 或 `(T, 1, Dr)`                   | BF16 / FP16   | ND             |
| qr                 | 输出           | `(B, S, Hcq)` 或 `(T, Hcq)`                       | BF16 / FP16   | ND             |
| q_m                | 输出           | `(B, S, Hcq)` 或 `(T, Hcq)`（RMSNorm 前）         | BF16 / FP16   | ND             |
| q_rstd             | 输出           | `(B, S, 1)` 或 `(T, 1)`                           | FLOAT         | ND             |
| ckv_m              | 输出           | `(B, S, Hckv)` 或 `(T, Hckv)`（RMSNorm 前）       | BF16 / FP16   | ND             |
| ckv_rstd           | 输出           | `(B, S, 1)` 或 `(T, 1)`                           | FLOAT         | ND             |

## 构建

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=dsa_sparse_prolog -j8
```

`--soc` 可选 `ascend910b`（Atlas A2 系列）或 `ascend910_93`（Atlas A3 系列）。

## 目录结构

```text
dsa_sparse_prolog/
  CMakeLists.txt
  op_host/
    CMakeLists.txt
    dsa_sparse_prolog_def.cpp                 # OpDef
    dsa_sparse_prolog_infershape.cpp          # InferShape / InferDataType
    dsa_sparse_prolog_tiling.h                # TilingData 定义
    dsa_sparse_prolog_tiling.cpp              # Tiling 实现
    dsa_sparse_prolog_tiling_register.cpp     # Tiling 注册
  op_kernel/
    dsa_sparse_prolog.cpp                     # Ascend C 内核入口
    dsa_sparse_prolog_comm.h
    kernel_dsa_sparse_prolog_vec_s1_cub_s2.h
    service_matmul.h / service_rms_norm.h / service_rotary_position_embedding.h / service_gather_sin_cos.h
  tests/
    pytest/                                        # pytest 功能验证
```

## 测试

参考 [tests/pytest/README.md](./tests/pytest/README.md)。在 `tests/pytest` 目录下激活自定义算子包后执行：

```bash
bash test_run.sh single
```
