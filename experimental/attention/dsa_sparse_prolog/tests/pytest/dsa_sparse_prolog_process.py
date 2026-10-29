#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

import torch
import torch_npu
import aclnn_runner


def _to_npu(input_data):
    dtype = input_data["dtype"]
    wl = input_data["weight_layout"]
    weights = input_data["weights"]
    tx = input_data["token_x"].to(dtype).npu()
    names = ("weight_dq", "weight_uq_qr", "weight_dkv_kr")
    if wl == "nd_kn":
        op_w = {k: weights[k].to(dtype).npu().contiguous() for k in names}
    elif wl == "nd_nk":
        op_w = {k: weights[k].t().contiguous().to(dtype).npu() for k in names}
    else:
        raise ValueError("aclnn path supports nd_kn / nd_nk only, got %s" % wl)
    op_w["gamma_cq"] = weights["gamma_cq"].to(dtype).npu()
    op_w["gamma_ckv"] = weights["gamma_ckv"].to(dtype).npu()
    return (
        tx,
        op_w,
        input_data["cos"].to(dtype).npu(),
        input_data["sin"].to(dtype).npu(),
    )


def call_npu(input_data):
    """直接调用 aclnnDsaSparseProlog，返回 9 个输出。"""
    tx, w, cos, sin = _to_npu(input_data)
    cfg = input_data["cfg"]
    lead = tx.shape[:-1]
    n, d, dr = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim
    hcq, hckv = cfg.q_lora_rank, cfg.kv_lora_rank
    dt, dev = tx.dtype, tx.device
    q_nope = torch.empty(*lead, n, d, dtype=dt, device=dev)
    q_pe = torch.empty(*lead, n, dr, dtype=dt, device=dev)
    ct_kv = torch.empty(*lead, hckv, dtype=dt, device=dev)
    k_pe = torch.empty(*lead, 1, dr, dtype=dt, device=dev)
    qr = torch.empty(*lead, hcq, dtype=dt, device=dev)
    q_m = torch.empty(*lead, hcq, dtype=dt, device=dev)
    q_rstd = torch.empty(*lead, 1, dtype=torch.float32, device=dev)
    ckv_m = torch.empty(*lead, hckv, dtype=dt, device=dev)
    ckv_rstd = torch.empty(*lead, 1, dtype=torch.float32, device=dev)
    aclnn_runner.call(
        "aclnnDsaSparsePrologGetWorkspaceSize",
        "aclnnDsaSparseProlog",
        [
            ("tensor", tx),
            ("tensor", w["weight_dq"]),
            ("tensor", w["weight_uq_qr"]),
            ("tensor", w["weight_dkv_kr"]),
            ("tensor", w["gamma_cq"]),
            ("tensor", w["gamma_ckv"]),
            ("tensor", sin),
            ("tensor", cos),
            ("double", cfg.rms_norm_eps),
            ("double", cfg.rms_norm_eps),
            ("double", input_data["scale_q"]),
            ("double", input_data["scale_kv"]),
            ("int64", cfg.qk_nope_head_dim),
            ("bool", bool(input_data["demote"])),
            ("bool", bool(input_data["fusion"])),
            ("tensor", q_nope),
            ("tensor", q_pe),
            ("tensor", ct_kv),
            ("tensor", k_pe),
            ("tensor", qr),
            ("tensor", q_m),
            ("tensor", q_rstd),
            ("tensor", ckv_m),
            ("tensor", ckv_rstd),
        ],
    )
    return q_nope, q_pe, ct_kv, k_pe, qr, q_m, q_rstd, ckv_m, ckv_rstd


def run_torch_baseline(input_data):
    """CPU 上用目标 dtype 复现 torch 拼接实现，作为基线（不依赖 torch_npu 算子绑定）。"""
    from dsa_sparse_prolog_golden import dsa_sparse_prolog_golden

    dtype = input_data["dtype"]
    cfg = input_data["cfg"]
    tx = input_data["token_x"].to(dtype)
    w = {k: v.to(dtype) for k, v in input_data["weights"].items()}
    cos = input_data["cos"].to(dtype)
    sin = input_data["sin"].to(dtype)
    return dsa_sparse_prolog_golden(cfg, tx, w, cos, sin)
