#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class DsaSparsePrologConfig:
    """DSA sparse prolog 所需的全部维度与超参（默认取真实 shape）。"""

    def __init__(
        self,
        hidden_size: int = 8192,
        num_attention_heads: int = 64,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        mla_scale_q_lora: bool = True,
        mla_scale_kv_lora: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.mla_scale_q_lora = mla_scale_q_lora
        self.mla_scale_kv_lora = mla_scale_kv_lora

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def q_scale(self) -> float:
        if self.mla_scale_q_lora:
            return (self.hidden_size / self.q_lora_rank) ** 0.5
        return 1.0

    @property
    def kv_scale(self) -> float:
        if self.mla_scale_kv_lora:
            return (self.hidden_size / self.kv_lora_rank) ** 0.5
        return 1.0


def build_rope_cos_sin(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """标准 RoPE 的 cos/sin 生成，shape 均为 [seq_len, dim]。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_mla(
    t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """对交错排列的特征执行 MLA 旋转位置编码。"""
    *batch, h, d = t.shape
    t = t.view(*batch, h, d // 2, 2).transpose(-1, -2).reshape(*batch, h, d)
    cos = cos.reshape(*cos.shape[:-1], 1, d)
    sin = sin.reshape(*sin.shape[:-1], 1, d)
    return (t * cos) + (_rotate_half(t) * sin)


def rand_weights_fp32(
    cfg: DsaSparsePrologConfig, seed: int = 0
) -> Dict[str, torch.Tensor]:
    """fp32 基准权重（op 权重布局：矩阵 [in, out]；gamma [out]）。"""
    g = torch.Generator().manual_seed(seed)
    he, hcq, hckv = cfg.hidden_size, cfg.q_lora_rank, cfg.kv_lora_rank
    d, dr, n = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.num_attention_heads
    scale = 0.02
    return {
        "weight_dq": torch.randn(he, hcq, generator=g) * scale,
        "weight_uq_qr": torch.randn(hcq, n * (d + dr), generator=g) * scale,
        "weight_dkv_kr": torch.randn(he, hckv + dr, generator=g) * scale,
        "gamma_cq": torch.randn(hcq, generator=g),
        "gamma_ckv": torch.randn(hckv, generator=g),
    }


def dsa_sparse_prolog_golden(
    cfg: DsaSparsePrologConfig,
    token_x: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """执行稀疏注意力前处理，返回 9 个输出。"""
    batch_shape = token_x.shape[:-1]
    hckv = cfg.kv_lora_rank
    d, dr, n = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.num_attention_heads

    # Q 路径
    q_m = token_x @ weights["weight_dq"]
    q_rstd = torch.rsqrt(q_m.float().square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    qr = F.rms_norm(q_m, (cfg.q_lora_rank,), weights["gamma_cq"], cfg.rms_norm_eps)
    q_states = (qr @ weights["weight_uq_qr"]).view(*batch_shape, n, d + dr)
    q_nope, q_pe = torch.split(q_states, [d, dr], dim=-1)
    q_nope = q_nope * cfg.q_scale
    q_pe = q_pe * cfg.q_scale

    # KV 路径
    compressed_kv = token_x @ weights["weight_dkv_kr"]
    ckv_m, k_pe = torch.split(compressed_kv, [hckv, dr], dim=-1)
    ckv_rstd = torch.rsqrt(
        ckv_m.float().square().mean(-1, keepdim=True) + cfg.rms_norm_eps
    )
    ct_kv = F.rms_norm(ckv_m, (hckv,), weights["gamma_ckv"], cfg.rms_norm_eps)
    ct_kv = ct_kv * cfg.kv_scale
    k_pe = k_pe.view(*batch_shape, 1, dr)

    # RoPE
    q_pe = apply_rope_mla(q_pe, rope_cos, rope_sin)
    k_pe = apply_rope_mla(k_pe, rope_cos, rope_sin)

    return q_nope, q_pe, ct_kv, k_pe, qr, q_m, q_rstd, ckv_m, ckv_rstd


def _to_fp64_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().double().cpu()


def dsa_sparse_prolog_golden_fp64(
    cfg: DsaSparsePrologConfig,
    token_x: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """fp64 真值 golden：全部输入升 fp64（CPU）。"""
    w64 = {k: _to_fp64_cpu(v) for k, v in weights.items()}
    return dsa_sparse_prolog_golden(
        cfg, _to_fp64_cpu(token_x), w64, _to_fp64_cpu(rope_cos), _to_fp64_cpu(rope_sin)
    )


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return (
        torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-30)
    ).item()


def check_precision_row(row: Dict, demote: bool) -> bool:
    """统一精度判据：op 相对 fp64 的误差不劣于 torch 基线；demote 模式下放宽到 2e-3。"""
    if row["op_vs_fp64"] > max(row["torch_vs_fp64"] * 1.5, 1e-12):
        return False
    if (
        demote
        and row["op_vs_torch"] > 5e-3
        and row["op_vs_fp64"] > row["torch_vs_fp64"]
    ):
        return False
    return True


class _ScaleOverride(DsaSparsePrologConfig):
    """覆盖 q_scale/kv_scale，用于显式 scale 用例的 golden 对齐。"""

    def __init__(self, base, q_scale, kv_scale):
        self.__dict__.update(base.__dict__)
        self._q_scale, self._kv_scale = q_scale, kv_scale

    @property
    def q_scale(self):
        return self._q_scale

    @property
    def kv_scale(self):
        return self._kv_scale


def gen_inputs(params):
    """根据用例参数生成 CPU 侧输入、配置与 fp64 真值。

    params 结构：(batch_shape, dtype, demote, fusion, scale_q, scale_kv, weight_layout, case_name)
    """
    batch_shape, dtype, demote, fusion, scale_q, scale_kv, weight_layout, case_name = (
        params
    )
    cfg = DsaSparsePrologConfig()
    seed = 0

    g = torch.Generator().manual_seed(seed + 1)
    token_x = torch.randn(*batch_shape, cfg.hidden_size, generator=g)
    weights = rand_weights_fp32(cfg, seed=seed)
    s = batch_shape[-1] if len(batch_shape) > 1 else batch_shape[0]
    cos, sin = build_rope_cos_sin(s, cfg.qk_rope_head_dim, cfg.rope_theta)
    if len(batch_shape) > 1:
        cos = cos.unsqueeze(0).expand(batch_shape[0], -1, -1).contiguous()
        sin = sin.unsqueeze(0).expand(batch_shape[0], -1, -1).contiguous()

    golden_cfg = cfg
    if scale_q > 0 and scale_kv > 0:
        golden_cfg = _ScaleOverride(cfg, scale_q, scale_kv)
    fp64_out = dsa_sparse_prolog_golden_fp64(cfg, token_x, weights, cos, sin)

    return {
        "cfg": golden_cfg,
        "batch_shape": batch_shape,
        "dtype": dtype,
        "token_x": token_x,
        "weights": weights,
        "cos": cos,
        "sin": sin,
        "scale_q": scale_q,
        "scale_kv": scale_kv,
        "demote": demote,
        "fusion": fusion,
        "weight_layout": weight_layout,
        "fp64_output": fp64_out,
        "case_name": case_name,
    }


OUT_NAMES = [
    "q_nope",
    "q_pe",
    "ct_kv",
    "k_pe",
    "qr",
    "q_m",
    "q_rstd",
    "ckv_m",
    "ckv_rstd",
]
