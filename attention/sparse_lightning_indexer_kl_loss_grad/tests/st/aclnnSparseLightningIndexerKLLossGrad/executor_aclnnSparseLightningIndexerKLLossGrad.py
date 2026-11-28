#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import ctypes
import math
import os
import numpy as np
import torch
import random
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from einops import rearrange
from pathlib import Path
import torch.nn as nn

STR_TO_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

SEQ_USED_MODE_ENV = "KL_LOSS_SEQ_USED_MODE"
SEQ_USED_MODES = ("random", "none", "q_only", "k_only", "full")


def normalize_residuals(residuals, batch_size):
    if isinstance(residuals, int):
        return (residuals,) * batch_size
    residuals = tuple(residuals)
    if len(residuals) != batch_size:
        raise ValueError(
            f"cmp_residual_k batch mismatch: {residuals}, batch_size={batch_size}"
        )
    return residuals


def make_invalid_mask(q_len, kv_len, topk, cmp_ratio, cmp_residual_k=0, sparse_mode=3):
    mask = torch.zeros(q_len, topk, dtype=torch.bool)
    for q_idx in range(q_len):
        if int(sparse_mode) == 0:
            real_k = kv_len
        else:
            real_k = valid_k_count(q_idx, kv_len, q_len, cmp_ratio, cmp_residual_k)
        real_k = max(0, min(int(real_k), kv_len, topk))
        if real_k < topk:
            mask[q_idx, real_k:] = True
    return mask


def make_tensor(shape, dtype, seed, low=None, high=None):
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.rand(*shape, generator=generator, dtype=dtype)
    if low is not None and high is not None:
        tensor = tensor * (high - low) + low
    return tensor


class SparseLightningIndexerKLLossGradRef:
    REDUCTION_TILE = 16
    DIRECT_REDUCTION_LIMIT = 256
    REDUCTION_WORK_LIMIT = 8_000_000
    MATMUL_MAX_BLOCKS = 32
    VECTOR_MAX_BLOCKS = 256
    SCATTER_MAX_PASSES = 64

    def __init__(self, input_data: InputDataset):
        self.layout = input_data.kwargs["layoutQ"].upper()
        self.q_index = input_data.kwargs["q"].cpu()
        self.k_index = input_data.kwargs["k"].cpu()
        self.q = self.q_index
        self.k = self.k_index
        self.w = input_data.kwargs["w"].cpu()
        self.sparse_indices = input_data.kwargs["sparseIndices"].cpu()
        self.attn = input_data.kwargs.get("attnSoftmaxL1Norm", None)
        if self.attn is not None and torch.is_tensor(self.attn):
            self.attn = self.attn.cpu()
        self.allocated_q_len = list(eval(input_data.kwargs["seqlens_list_q"]))
        self.allocated_kv_len = list(eval(input_data.kwargs["seqlens_list_k"]))
        self.actual_q_len = tensor_lengths(
            input_data.kwargs.get("seqUsedQOptional"),
            self.allocated_q_len,
            "seqUsedQOptional",
        )
        self.actual_kv_len = tensor_lengths(
            input_data.kwargs.get("seqUsedKOptional"),
            self.allocated_kv_len,
            "seqUsedKOptional",
        )
        self.cmp_residual_k = list(
            normalize_residuals(
                input_data.kwargs.get("test_cmp_residual", 0), len(self.actual_q_len)
            )
        )
        self.cu_q = get_cu_seqlens(self.allocated_q_len)
        self.cu_kv = get_cu_seqlens(self.allocated_kv_len)
        self.cmp_ratio = input_data.kwargs["cmpRatio"]
        self.sparse_mode = int(input_data.kwargs.get("maskMode", 3))
        self.scale = 1.0 / math.sqrt(self.q.shape[-1])
        self.attn_l1_scale = float(1.0)
        self.input_dtype = self.q_index.dtype
        self.weight_dtype = self.w.dtype
        # no_has 模式（seqUsedQOptional/seqUsedKOptional 均为 None）下使用 mf 的
        # Cube 量化模拟（vec_dtype=float32, mm_dtype=输入精度）；
        # 非 no_has 模式保持原始 yf 行为（vec_dtype=input_dtype, mm_dtype=None，
        # 所有 Cube 模拟方法退化为纯 einsum）。
        self._is_no_has = (
            input_data.kwargs.get("seqUsedQOptional") is None
            and input_data.kwargs.get("seqUsedKOptional") is None
        )
        if self._is_no_has:
            self.vec_dtype = torch.float32
            self.mm_dtype = (
                None if self.input_dtype == torch.float32 else self.input_dtype
            )
        else:
            self.vec_dtype = self.input_dtype
            self.mm_dtype = None

    def _cube_mm(self, equation, a, b):
        """Cube-like GEMM: quantize operands to KV dtype, accumulate in fp32."""
        if self.mm_dtype is None:
            return torch.einsum(equation, a, b)
        return torch.einsum(
            equation, a.to(self.mm_dtype).float(), b.to(self.mm_dtype).float()
        )

    @staticmethod
    def _block_size(contract_size, max_blocks):
        tile = SparseLightningIndexerKLLossGradRef.REDUCTION_TILE
        return max(tile, math.ceil(contract_size / (max_blocks * tile)) * tile)

    def _vector_block_size(self, contract_size, output_elements):
        max_blocks = (
            self.VECTOR_MAX_BLOCKS
            if contract_size * output_elements <= self.REDUCTION_WORK_LIMIT
            else self.MATMUL_MAX_BLOCKS
        )
        return self._block_size(contract_size, max_blocks)

    def _use_direct_reduction(self, contract_size, output_elements):
        return contract_size <= self.REDUCTION_TILE or (
            self.mm_dtype == torch.float16
            and contract_size <= self.DIRECT_REDUCTION_LIMIT
            and contract_size * output_elements <= self.REDUCTION_WORK_LIMIT
        )

    def _cast_accumulator(self, value):
        if self.mm_dtype == torch.float16:
            limit = torch.finfo(torch.float16).max
            value = value.clamp(min=-limit, max=limit)
        return value.to(self.mm_dtype).float()

    def _cube_mm_chunked(self, equation, a, b):
        if self.mm_dtype is None:
            return torch.einsum(equation, a, b)
        a_low = a.to(self.mm_dtype)
        b_low = b.to(self.mm_dtype)
        if equation == "thd,tsd->ths":
            acc = torch.zeros(
                a_low.shape[0], a_low.shape[1], b_low.shape[1], dtype=self.mm_dtype
            )
            contract_size = a_low.shape[-1]
            if self._use_direct_reduction(contract_size, acc.numel()):
                for di in range(contract_size):
                    acc.add_(
                        a_low[:, :, di].unsqueeze(-1) * b_low[:, :, di].unsqueeze(1)
                    )
                return acc.float()
            acc_fp32 = acc.float()
            block = self._block_size(contract_size, self.MATMUL_MAX_BLOCKS)
            for start in range(0, contract_size, block):
                end = min(start + block, contract_size)
                partial = torch.einsum(
                    "thd,tsd->ths",
                    a_low[:, :, start:end].float(),
                    b_low[:, :, start:end].float(),
                )
                acc_fp32 = self._cast_accumulator(acc_fp32 + partial)
            return acc_fp32
        elif equation == "tjs,tsd->tjd":
            acc = torch.zeros(
                a_low.shape[0], a_low.shape[1], b_low.shape[-1], dtype=self.mm_dtype
            )
            contract_size = a_low.shape[-1]
            if self._use_direct_reduction(contract_size, acc.numel()):
                for si in range(contract_size):
                    acc.add_(
                        a_low[:, :, si].unsqueeze(-1) * b_low[:, si, :].unsqueeze(1)
                    )
                return acc.float()
            acc_fp32 = acc.float()
            block = self._block_size(contract_size, self.MATMUL_MAX_BLOCKS)
            for start in range(0, contract_size, block):
                end = min(start + block, contract_size)
                partial = torch.einsum(
                    "tjs,tsd->tjd",
                    a_low[:, :, start:end].float(),
                    b_low[:, start:end, :].float(),
                )
                acc_fp32 = self._cast_accumulator(acc_fp32 + partial)
            return acc_fp32
        elif equation == "tjs,tjd->tsd":
            acc = torch.zeros(
                a_low.shape[0],
                a_low.shape[-1],
                b_low.shape[-1],
                dtype=self.mm_dtype,
            )
            contract_size = a_low.shape[1]
            if self._use_direct_reduction(contract_size, acc.numel()):
                for ji in range(contract_size):
                    acc.add_(
                        a_low[:, ji, :].unsqueeze(-1) * b_low[:, ji, :].unsqueeze(1)
                    )
                return acc.float()
            acc_fp32 = acc.float()
            block = self._block_size(contract_size, self.MATMUL_MAX_BLOCKS)
            for start in range(0, contract_size, block):
                end = min(start + block, contract_size)
                partial = torch.einsum(
                    "tjs,tjd->tsd",
                    a_low[:, start:end, :].float(),
                    b_low[:, start:end, :].float(),
                )
                acc_fp32 = self._cast_accumulator(acc_fp32 + partial)
            return acc_fp32
        raise ValueError(f"unsupported cube equation {equation}")

    def _reduce_dw(self, relu_s, ds_sq):
        contract_size = relu_s.shape[-1]
        if self.mm_dtype is None:
            return torch.einsum("tjs,ts->tj", relu_s, ds_sq)
        relu_low = relu_s.to(self.mm_dtype)
        ds_low = ds_sq.to(self.mm_dtype)
        acc = torch.zeros(relu_s.shape[0], relu_s.shape[1], dtype=self.mm_dtype)
        if self._use_direct_reduction(contract_size, acc.numel()):
            for si in range(contract_size):
                acc.add_(relu_low[:, :, si] * ds_low[:, si].unsqueeze(1))
            return acc.float()
        acc_fp32 = acc.float()
        block = self._vector_block_size(contract_size, acc.numel())
        for start in range(0, contract_size, block):
            end = min(start + block, contract_size)
            partial = torch.einsum(
                "tjs,ts->tj",
                relu_low[:, :, start:end].float(),
                ds_low[:, start:end].float(),
            )
            acc_fp32 = self._cast_accumulator(acc_fp32 + partial)
        return acc_fp32

    def _scatter_add_chunked(self, indices, values, output_size):
        acc = torch.zeros(output_size, values.shape[-1], dtype=torch.float32)
        count = indices.numel()
        if count == 0:
            return acc
        values = values.to(self.mm_dtype).float()
        if count <= self.REDUCTION_TILE:
            for idx in range(count):
                key = int(indices[idx].item())
                acc[key] = self._cast_accumulator(acc[key] + values[idx])
            return acc
        passes = min(self.SCATTER_MAX_PASSES, count)
        block = math.ceil(count / passes)
        for start in range(0, count, block):
            end = min(start + block, count)
            acc.index_add_(0, indices[start:end], values[start:end])
            acc = self._cast_accumulator(acc)
        return acc

    def _cast_round_like(self, x):
        if self.mm_dtype is None:
            return x
        return self._cast_accumulator(x)

    def _compute_p(
        self, q_b, k_b, sparse_b, q_len, kv_len, nk, topk, group, dt, cmp_residual_k=0
    ):
        invalid_row = (sparse_b == -1).all(dim=-1)
        sparse = sparse_b.squeeze(1)
        valid = (sparse >= 0) & (sparse < kv_len)
        safe_id = sparse.clamp(0, kv_len - 1)
        invalid_topk_mask = make_invalid_mask(
            q_len,
            kv_len,
            topk,
            self.cmp_ratio,
            cmp_residual_k,
            sparse_mode=self.sparse_mode,
        )
        k_main = k_b.squeeze(1)
        k_topk = k_main[safe_id].masked_fill(~valid.unsqueeze(-1), 0.0)
        qk = torch.einsum("thd,tsd->ths", q_b, k_topk) * self.scale
        qk = qk.masked_fill(invalid_topk_mask.unsqueeze(1), float("-inf"))
        p_full = nn.functional.softmax(qk, dim=-1)
        p = p_full.reshape(q_len, nk, group, topk).sum(dim=2) / group
        p = p * self.attn_l1_scale
        p = p.masked_fill(invalid_topk_mask.unsqueeze(1), 0.0)
        p = p.masked_fill(invalid_row.unsqueeze(-1), 0.0)
        return p

    def _backward_one_batch(
        self,
        p_b,
        q_index_b,
        k_index_b,
        w_b,
        sparse_b,
        q_len,
        kv_len,
        n_query_index,
        d_index,
        nk,
        topk,
        group_index,
        dt,
        cmp_residual_k=0,
    ):
        vec_dt = self.vec_dtype
        if q_len == 0 or kv_len == 0:
            s_softmax = torch.zeros(q_len, nk, topk, dtype=vec_dt)
            dqi = torch.zeros(q_len, n_query_index, d_index, dtype=vec_dt)
            dki_per_batch = torch.zeros(kv_len, d_index, dtype=vec_dt)
            dw = torch.zeros(q_len, n_query_index, dtype=vec_dt)
            return s_softmax, dqi, dki_per_batch, dw

        invalid_row = (sparse_b == -1).all(dim=-1)
        sparse = sparse_b.squeeze(1)
        valid = (sparse >= 0) & (sparse < kv_len)
        safe_id = sparse.clamp(0, kv_len - 1)
        invalid_topk_mask = make_invalid_mask(
            q_len,
            kv_len,
            topk,
            self.cmp_ratio,
            cmp_residual_k,
            sparse_mode=self.sparse_mode,
        )

        p = p_b.to(vec_dt)
        p = p.masked_fill(invalid_topk_mask.unsqueeze(1), 0.0)
        p = p.masked_fill(invalid_row.unsqueeze(-1), 0.0)

        q_index_b = q_index_b.to(vec_dt)
        k_index_b = k_index_b.to(vec_dt)
        w_b = w_b.to(vec_dt)

        k_index_main = k_index_b.squeeze(1)
        k_index_topk = k_index_main[safe_id].masked_fill(~valid.unsqueeze(-1), 0.0)
        effective_valid = valid & ~invalid_topk_mask
        active_topk = effective_valid.any(dim=0)
        active_topk_count = int(active_topk.sum().item())
        if self.mm_dtype is not None and active_topk_count > 0:
            s_logits = torch.zeros(q_len, n_query_index, topk, dtype=vec_dt)
            s_logits[:, :, active_topk] = self._cube_mm_chunked(
                "thd,tsd->ths",
                q_index_b,
                k_index_topk[:, active_topk, :],
            )
        elif self.mm_dtype is not None:
            s_logits = torch.zeros(q_len, n_query_index, topk, dtype=vec_dt)
        else:
            s_logits = self._cube_mm("thd,tsd->ths", q_index_b, k_index_topk)
        relu_s = torch.relu(s_logits)
        s_weighted = relu_s * w_b.unsqueeze(-1)
        s_reduce = s_weighted.reshape(q_len, nk, group_index, topk).sum(dim=2)
        if self.mm_dtype is not None:
            s_reduce = self._cast_accumulator(s_reduce)
        s_reduce = s_reduce.masked_fill(invalid_topk_mask.unsqueeze(1), float("-inf"))
        s_softmax = nn.functional.softmax(s_reduce, dim=-1)
        s_softmax = s_softmax.masked_fill(invalid_topk_mask.unsqueeze(1), 0.0)
        s_softmax = s_softmax.masked_fill(invalid_row.unsqueeze(-1), 0.0)

        p_reduce = p.sum(dim=-1, keepdim=True)
        ds_reduce = s_softmax * p_reduce - p
        ds_sq = ds_reduce.squeeze(1)

        if self.mm_dtype is not None and active_topk_count > 0:
            dw = self._reduce_dw(
                relu_s[:, :, active_topk],
                ds_sq[:, active_topk],
            )
        elif self.mm_dtype is not None:
            dw = torch.zeros(q_len, n_query_index, dtype=vec_dt)
        else:
            dw = torch.einsum("tjs,ts->tj", relu_s, ds_sq)
        dw = dw.masked_fill(invalid_row, 0.0)

        d_s_logits = ds_sq.unsqueeze(1) * w_b.unsqueeze(-1) * (relu_s > 0).to(vec_dt)
        d_s_logits_hw = self._cast_round_like(d_s_logits)

        if self.mm_dtype is not None and active_topk_count > 0:
            dqi = self._cube_mm_chunked(
                "tjs,tsd->tjd",
                d_s_logits_hw[:, :, active_topk],
                k_index_topk[:, active_topk, :],
            )
        elif self.mm_dtype is not None:
            dqi = torch.zeros(q_len, n_query_index, d_index, dtype=vec_dt)
        else:
            dqi = self._cube_mm("tjs,tsd->tjd", d_s_logits_hw, k_index_topk)
        dqi = dqi.masked_fill(invalid_row.unsqueeze(-1), 0.0)

        if self.mm_dtype is not None and active_topk_count > 0:
            dki_gather = self._cube_mm_chunked(
                "tjs,tjd->tsd",
                d_s_logits_hw[:, :, active_topk],
                q_index_b,
            )
            selected_valid = effective_valid[:, active_topk]
            flat_valid = selected_valid.reshape(-1)
            flat_indices = safe_id[:, active_topk].reshape(-1)[flat_valid]
            flat_values = dki_gather.reshape(-1, d_index)[flat_valid]
            dki_per_batch = self._scatter_add_chunked(flat_indices, flat_values, kv_len)
        elif self.mm_dtype is not None:
            dki_per_batch = torch.zeros(kv_len, d_index, dtype=vec_dt)
        else:
            dki_gather = self._cube_mm("tjs,tjd->tsd", d_s_logits_hw, q_index_b)
            dki_gather = dki_gather.masked_fill(invalid_row.unsqueeze(-1), 0.0)
            dki_gather = dki_gather * valid.unsqueeze(-1)
            dki_per_batch = torch.zeros(kv_len, d_index, dtype=vec_dt)
            dki_per_batch.index_add_(
                0,
                safe_id.reshape(-1),
                dki_gather.reshape(-1, d_index),
            )
        return s_softmax, dqi, dki_per_batch, dw

    def forward(self):
        if self.layout == "BSND":
            return self._forward_bsnd()
        return self._forward_tnd()

    def backward(self):
        if self.attn is None:
            raise ValueError("attnSoftmaxL1Norm is required for CPU backward golden")
        if self.layout == "BSND":
            return self._backward_bsnd()
        return self._backward_tnd()

    def _forward_tnd(self):
        total_q, _, _ = self.q.shape
        _, nk, _ = self.k.shape
        _, n_query, _ = self.q.shape
        topk = self.sparse_indices.shape[-1]
        group = n_query // nk
        dt = self.input_dtype
        attn_softmax_l1_norm = torch.zeros((total_q, nk, topk), dtype=dt)
        for b_idx in range(len(self.actual_q_len)):
            qs = self.cu_q[b_idx].item()
            ks = self.cu_kv[b_idx].item()
            q_len = self.actual_q_len[b_idx]
            kv_len = self.actual_kv_len[b_idx]
            qe = qs + q_len
            ke = ks + kv_len
            cmp_residual_k = self.cmp_residual_k[b_idx]
            if q_len == 0 or kv_len == 0:
                p = torch.zeros(q_len, nk, topk, dtype=dt)
            else:
                p = self._compute_p(
                    self.q[qs:qe].to(dt),
                    self.k[ks:ke].to(dt),
                    self.sparse_indices[qs:qe].to(torch.int64),
                    q_len,
                    kv_len,
                    nk,
                    topk,
                    group,
                    dt,
                    cmp_residual_k=cmp_residual_k,
                )
            attn_softmax_l1_norm[qs:qe] = p
        return attn_softmax_l1_norm

    def _backward_tnd(self):
        total_q, n_query_index, d_index = self.q_index.shape
        total_kv, nk, _ = self.k_index.shape
        topk = self.sparse_indices.shape[-1]
        group_index = n_query_index // nk
        vec_dt = self.vec_dtype

        softmax_out = torch.zeros((total_q, nk, topk), dtype=vec_dt)
        d_query_index = torch.zeros((total_q, n_query_index, d_index), dtype=vec_dt)
        d_key_index = torch.zeros((total_kv, nk, d_index), dtype=vec_dt)
        d_w = torch.zeros((total_q, n_query_index), dtype=vec_dt)

        for b_idx in range(len(self.actual_q_len)):
            qs = self.cu_q[b_idx].item()
            ks = self.cu_kv[b_idx].item()
            q_len = self.actual_q_len[b_idx]
            kv_len = self.actual_kv_len[b_idx]
            qe = qs + q_len
            ke = ks + kv_len
            cmp_residual_k = self.cmp_residual_k[b_idx]

            s_softmax, dqi, dki_per_batch, dw = self._backward_one_batch(
                self.attn[qs:qe],
                self.q_index[qs:qe],
                self.k_index[ks:ke],
                self.w[qs:qe],
                self.sparse_indices[qs:qe].to(torch.int64),
                q_len,
                kv_len,
                n_query_index,
                d_index,
                nk,
                topk,
                group_index,
                vec_dt,
                cmp_residual_k=cmp_residual_k,
            )
            softmax_out[qs:qe] = s_softmax
            d_query_index[qs:qe] = dqi
            d_key_index[ks:ke, 0, :] = dki_per_batch
            d_w[qs:qe] = dw

        return (
            d_query_index.to(self.input_dtype),
            d_key_index.to(self.input_dtype),
            d_w.to(self.weight_dtype),
            softmax_out.to(self.attn.dtype),
        )

    def _forward_bsnd(self):
        b, s1, n_query, _ = self.q.shape
        _, _, nk, _ = self.k.shape
        topk = self.sparse_indices.shape[-1]
        group = n_query // nk
        dt = self.input_dtype

        attn_softmax_l1_norm = torch.zeros((b, s1, nk, topk), dtype=dt)
        for b_idx in range(len(self.actual_q_len)):
            q_len = self.actual_q_len[b_idx]
            kv_len = self.actual_kv_len[b_idx]
            cmp_residual_k = self.cmp_residual_k[b_idx]
            if q_len == 0 or kv_len == 0:
                p = torch.zeros(q_len, nk, topk, dtype=dt)
            else:
                p = self._compute_p(
                    self.q[b_idx, :q_len].to(dt),
                    self.k[b_idx, :kv_len].to(dt),
                    self.sparse_indices[b_idx, :q_len].to(torch.int64),
                    q_len,
                    kv_len,
                    nk,
                    topk,
                    group,
                    dt,
                    cmp_residual_k=cmp_residual_k,
                )
            attn_softmax_l1_norm[b_idx, :q_len] = p
        return attn_softmax_l1_norm

    def _backward_bsnd(self):
        b, s1, n_query_index, d_index = self.q_index.shape
        _, s2, nk, _ = self.k_index.shape
        topk = self.sparse_indices.shape[-1]
        group_index = n_query_index // nk
        vec_dt = self.vec_dtype

        softmax_out = torch.zeros((b, s1, nk, topk), dtype=vec_dt)
        d_query_index = torch.zeros((b, s1, n_query_index, d_index), dtype=vec_dt)
        d_key_index = torch.zeros((b, s2, nk, d_index), dtype=vec_dt)
        d_w = torch.zeros((b, s1, n_query_index), dtype=vec_dt)

        for b_idx in range(len(self.actual_q_len)):
            q_len = self.actual_q_len[b_idx]
            kv_len = self.actual_kv_len[b_idx]
            cmp_residual_k = self.cmp_residual_k[b_idx]

            s_softmax, dqi, dki_per_batch, dw = self._backward_one_batch(
                self.attn[b_idx, :q_len],
                self.q_index[b_idx, :q_len],
                self.k_index[b_idx, :kv_len],
                self.w[b_idx, :q_len],
                self.sparse_indices[b_idx, :q_len].to(torch.int64),
                q_len,
                kv_len,
                n_query_index,
                d_index,
                nk,
                topk,
                group_index,
                vec_dt,
                cmp_residual_k=cmp_residual_k,
            )
            softmax_out[b_idx, :q_len] = s_softmax
            d_query_index[b_idx, :q_len] = dqi
            d_key_index[b_idx, :kv_len, 0, :] = dki_per_batch
            d_w[b_idx, :q_len] = dw

        return (
            d_query_index.to(self.input_dtype),
            d_key_index.to(self.input_dtype),
            d_w.to(self.weight_dtype),
            softmax_out.to(self.attn.dtype),
        )


def normalize_residuals(residuals, batch_size):
    if isinstance(residuals, int):
        return (residuals,) * batch_size
    residuals = tuple(residuals)
    if len(residuals) != batch_size:
        raise ValueError(
            f"cmp_residual_k batch mismatch: {residuals}, batch_size={batch_size}"
        )
    return residuals


def normalize_lengths(lengths):
    if isinstance(lengths, int):
        return (lengths,)
    return tuple(lengths)


def tensor_lengths(tensor, allocated_lengths, name):
    allocated_lengths = tuple(int(value) for value in allocated_lengths)
    if tensor is None:
        return list(allocated_lengths)
    values = tuple(int(value) for value in tensor.detach().cpu().tolist())
    if len(values) != len(allocated_lengths):
        raise ValueError(
            f"{name} batch mismatch: values={values}, allocated={allocated_lengths}"
        )
    for batch_index, (used, allocated) in enumerate(zip(values, allocated_lengths)):
        if used < 0 or used > allocated:
            raise ValueError(
                f"{name}[{batch_index}]={used} is outside [0, {allocated}]"
            )
    return list(values)


def random_used_lengths(allocated_lengths):
    allocated_lengths = [int(value) for value in allocated_lengths]
    # allocated 全为 0 时无法避免，直接返回
    if all(value <= 0 for value in allocated_lengths):
        return [0] * len(allocated_lengths)

    values = [0] * len(allocated_lengths)
    while all(v == 0 for v in values):
        values = []
        for allocated in allocated_lengths:
            if allocated <= 0:
                values.append(0)
            else:
                values.append(int(torch.randint(0, allocated + 1, (1,)).item()))
    return values


def get_seq_used_mode():
    mode = os.getenv(SEQ_USED_MODE_ENV, "random").strip().lower()
    if mode not in SEQ_USED_MODES:
        raise ValueError(
            f"{SEQ_USED_MODE_ENV} must be one of {SEQ_USED_MODES}, got {mode!r}"
        )
    return mode


def make_seq_used_lengths(allocated_lengths, enabled, full_length):
    if not enabled:
        return None
    if full_length:
        return [int(value) for value in allocated_lengths]
    return random_used_lengths(allocated_lengths)


def optional_tensor_to(tensor, device):
    return None if tensor is None else tensor.to(device)


def cast_optional_tensor_nulls(input_args, indices):
    """Preserve aclTensor* type for nullptr inputs during ATK -cp checking."""
    from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor

    null_tensor_ptr = ctypes.cast(ctypes.c_void_p(None), ctypes.POINTER(AclTensor))
    for index in indices:
        if index < len(input_args) and isinstance(input_args[index], ctypes.c_void_p):
            input_args[index] = null_tensor_ptr


def trunc_div(numerator, denominator):
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


def valid_k_count(local_q_idx, kv_len, q_len, cmp_ratio, cmp_residual_k=0):
    if cmp_ratio != 0:
        pre_compress_k_len = kv_len * cmp_ratio + cmp_residual_k
        numerator = pre_compress_k_len - q_len + local_q_idx + 1
        return trunc_div(numerator, cmp_ratio)
    return max(kv_len - q_len + local_q_idx + 1, 0)


class GenCase:
    def __init__(self, input_data: InputDataset):
        self.q = input_data.kwargs["q"]
        self.layout = input_data.kwargs["layoutQ"]
        self.topk = input_data.kwargs["sparseIndices"].shape[-1]
        self.n2 = input_data.kwargs["k"].shape[-2]

        self.cmp_residual_k = input_data.kwargs.get("test_cmp_residual", 0)
        self.q_lens = list(eval(input_data.kwargs["seqlens_list_q"]))
        self.kv_lens = list(eval(input_data.kwargs["seqlens_list_k"]))
        self.used_q_lens = tensor_lengths(
            input_data.kwargs.get("seqUsedQOptional"), self.q_lens, "seqUsedQOptional"
        )
        self.used_kv_lens = tensor_lengths(
            input_data.kwargs.get("seqUsedKOptional"), self.kv_lens, "seqUsedKOptional"
        )
        self.batch_size = len(self.q_lens)
        self.cmp_residual_tuple = normalize_residuals(
            self.cmp_residual_k, self.batch_size
        )
        self.q_lens_tuple = normalize_lengths(self.q_lens)
        self.kv_lens_tuple = normalize_lengths(self.kv_lens)
        self.used_q_lens_tuple = normalize_lengths(self.used_q_lens)
        self.used_kv_lens_tuple = normalize_lengths(self.used_kv_lens)
        self.cmp_ratio = input_data.kwargs["cmpRatio"]
        self.sparse_mode = int(input_data.kwargs.get("maskMode", 3))
        self.total_q = sum(self.q_lens_tuple)
        self.total_k = sum(self.kv_lens_tuple)

    def fill_sparse_tnd(
        self,
        sparse_indices,
        q_start,
        q_len,
        kv_len,
        cmp_ratio,
        topk,
        n_key,
        cmp_residual_k=0,
    ):
        for local_q_idx in range(q_len):
            if int(self.sparse_mode) == 0:
                real_k = kv_len
            else:
                real_k = valid_k_count(
                    local_q_idx, kv_len, q_len, cmp_ratio, cmp_residual_k
                )
            real_k = max(0, min(real_k, kv_len))
            if real_k == 0:
                continue
            used_k = min(real_k, topk)
            sparse_indices[q_start + local_q_idx, 0, :used_k] = torch.randperm(
                real_k, dtype=torch.int32
            )[:used_k]

    def fill_sparse_bsnd(
        self,
        sparse_indices,
        b_idx,
        q_len,
        kv_len,
        cmp_ratio,
        topk,
        n_key,
        cmp_residual_k=0,
    ):
        for s1_idx in range(q_len):
            if int(self.sparse_mode) == 0:
                real_k = kv_len
            else:
                real_k = valid_k_count(s1_idx, kv_len, q_len, cmp_ratio, cmp_residual_k)
            real_k = max(0, min(real_k, kv_len))
            if real_k == 0:
                continue
            used_k = min(real_k, topk)
            sparse_indices[b_idx, s1_idx, 0, :used_k] = torch.randperm(
                real_k, dtype=torch.int32
            )[:used_k]

    def make_sparse_indices(self):
        if self.layout == "BSND":
            b = self.batch_size
            s1 = self.q.shape[1]
            sparse_indices = torch.full(
                (b, s1, self.n2, self.topk), -1, dtype=torch.int32
            )
            for b_idx, (q_len, kv_len, cmp_residual_k) in enumerate(
                zip(
                    self.used_q_lens_tuple,
                    self.used_kv_lens_tuple,
                    self.cmp_residual_tuple,
                )
            ):
                self.fill_sparse_bsnd(
                    sparse_indices,
                    b_idx,
                    q_len,
                    kv_len,
                    self.cmp_ratio,
                    self.topk,
                    self.n2,
                    cmp_residual_k=cmp_residual_k,
                )
            return sparse_indices

        sparse_indices = torch.full(
            (self.total_q, self.n2, self.topk), -1, dtype=torch.int32
        )
        q_start = 0
        for allocated_q_len, q_len, kv_len, cmp_residual_k in zip(
            self.q_lens_tuple,
            self.used_q_lens_tuple,
            self.used_kv_lens_tuple,
            self.cmp_residual_tuple,
        ):
            self.fill_sparse_tnd(
                sparse_indices,
                q_start,
                q_len,
                kv_len,
                self.cmp_ratio,
                self.topk,
                self.n2,
                cmp_residual_k=cmp_residual_k,
            )
            q_start += allocated_q_len

        return sparse_indices


def get_cu_seqlens(seqlens_list):
    seqlens_list = normalize_lengths(seqlens_list)
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int32)
    for i in range(len(seqlens_list) + 1):
        cu[i] = sum(seqlens_list[:i])
    return cu


def prepare_slikg_inputs(input_data: InputDataset, seed: int):
    torch_rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        cr = input_data.kwargs.get("cmpRatio")
        if isinstance(cr, (list, tuple)) and len(cr) == 2:
            import random as _random

            _random.seed(seed)
            input_data.kwargs["cmpRatio"] = int(_random.randint(int(cr[0]), int(cr[1])))
        else:
            input_data.kwargs["cmpRatio"] = int(input_data.kwargs["cmpRatio"])

        q_lens = list(eval(input_data.kwargs["seqlens_list_q"]))
        kv_lens = list(eval(input_data.kwargs["seqlens_list_k"]))
        batch_size = len(q_lens)
        seq_used_mode = input_data.kwargs["seq_used_mode"]
        use_q = seq_used_mode in ("random", "q_only", "full")
        use_k = seq_used_mode in ("random", "k_only", "full")
        full_length = seq_used_mode == "full"
        used_q_lens = make_seq_used_lengths(q_lens, use_q, full_length)
        used_k_lens = make_seq_used_lengths(kv_lens, use_k, full_length)
        input_data.kwargs["cuSeqLensQOptional"] = get_cu_seqlens(q_lens).contiguous()
        input_data.kwargs["cuSeqLensKOptional"] = get_cu_seqlens(kv_lens).contiguous()
        input_data.kwargs["seqUsedQOptional"] = (
            None
            if used_q_lens is None
            else torch.tensor(used_q_lens, dtype=torch.int32).contiguous()
        )
        input_data.kwargs["seqUsedKOptional"] = (
            None
            if used_k_lens is None
            else torch.tensor(used_k_lens, dtype=torch.int32).contiguous()
        )
        input_data.kwargs["cmpResidualKOptional"] = torch.tensor(
            list(
                normalize_residuals(
                    input_data.kwargs.get("test_cmp_residual", 0), batch_size
                )
            ),
            dtype=torch.int32,
        ).contiguous()
        get_case = GenCase(input_data)
        input_data.kwargs["sparseIndices"] = get_case.make_sparse_indices().contiguous()
        ref = SparseLightningIndexerKLLossGradRef(input_data)
        input_data.kwargs["attnSoftmaxL1Norm"] = (
            ref.forward().contiguous().to(torch.float32)
        )
    finally:
        torch.set_rng_state(torch_rng_state)


def build_slikg_metadata(input_data: InputDataset, device_id: int) -> torch.Tensor:
    import cann_ops_transformer

    device = f"npu:{device_id}"
    layout = input_data.kwargs["layoutQ"]
    q = input_data.kwargs["q"]
    k = input_data.kwargs["k"]
    sparse = input_data.kwargs["sparseIndices"]
    if str(layout).upper() == "BSND":
        batch_size, max_seqlen_q, n_query_index, d_index = q.shape
        _, max_seqlen_k, n_key, _ = k.shape
        topk = sparse.shape[-1]
    else:
        n_query_index, d_index = q.shape[-2], q.shape[-1]
        n_key = k.shape[-2]
        topk = sparse.shape[-1]
        q_lens = eval(input_data.kwargs["seqlens_list_q"])
        kv_lens = eval(input_data.kwargs["seqlens_list_k"])
        batch_size = len(q_lens)
        max_seqlen_q = max(q_lens) if q_lens else 0
        max_seqlen_k = max(kv_lens) if kv_lens else 0
    cu_q = input_data.kwargs["cuSeqLensQOptional"].to(device)
    cu_k = input_data.kwargs["cuSeqLensKOptional"].to(device)
    seq_used_q = optional_tensor_to(input_data.kwargs.get("seqUsedQOptional"), device)
    seq_used_k = optional_tensor_to(input_data.kwargs.get("seqUsedKOptional"), device)
    cmp_residual_k = input_data.kwargs["cmpResidualKOptional"].to(device)
    metadata = (
        torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
            n_query_index,
            n_key,
            d_index,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            seqused_q=seq_used_q,
            seqused_k=seq_used_k,
            cmp_residual_k=cmp_residual_k,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            topk=topk,
            layout_q=layout,
            layout_k=input_data.kwargs["layoutK"],
            mask_mode=int(input_data.kwargs["maskMode"]),
            cmp_ratio=int(input_data.kwargs["cmpRatio"]),
        )
    )
    torch.npu.synchronize()
    return metadata.detach().cpu().contiguous()


def is_unranked_tensor(t):
    return torch.is_tensor(t) and t.ndim == 0


def is_unranked_slikg_case(input_data: InputDataset) -> bool:
    return is_unranked_tensor(input_data.kwargs.get("q")) or is_unranked_tensor(
        input_data.kwargs.get("sparseIndices")
    )


def fill_empty_case_defaults(input_data: InputDataset):
    """shape=[] 空 tensor 用例：generator 会提前 return，attrs 可能仍是 None。"""
    kw = input_data.kwargs
    if kw.get("layoutQ") in (None, ""):
        kw["layoutQ"] = "TND"
    if kw.get("layoutK") in (None, ""):
        kw["layoutK"] = kw["layoutQ"]
    if kw.get("maskMode") is None:
        kw["maskMode"] = 3
    if kw.get("cmpRatio") is None:
        kw["cmpRatio"] = 1
    if kw.get("seqlens_list_q") in (None, ""):
        kw["seqlens_list_q"] = "[]"
    if kw.get("seqlens_list_k") in (None, ""):
        kw["seqlens_list_k"] = "[]"


def empty_case_outputs(input_data: InputDataset):
    q = input_data.kwargs["q"]
    k = input_data.kwargs["k"]
    w = input_data.kwargs["w"]
    attn = input_data.kwargs["attnSoftmaxL1Norm"]
    return (
        torch.zeros_like(q),
        torch.zeros_like(k),
        torch.zeros_like(w),
        torch.zeros_like(attn),
    )


@register("execute_aclnnSparseLightningIndexerKLLossGrad")
class AclnnSparseLightningIndexerKLLossGrad(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            if is_unranked_slikg_case(input_data):
                return empty_case_outputs(input_data)
            ref = SparseLightningIndexerKLLossGradRef(input_data)
            d_query_index, d_key_index, d_w, softmax_out = ref.backward()
            return d_query_index, d_key_index, d_w, softmax_out
        elif self.device == "npu":
            if is_unranked_slikg_case(input_data):
                return empty_case_outputs(input_data)
            import cann_ops_transformer

            self.set_device = f"npu:{self.device_id}"
            q = input_data.kwargs["q"].to(self.set_device)
            k = input_data.kwargs["k"].to(self.set_device)
            w = input_data.kwargs["w"].to(self.set_device)
            d_index = q.shape[-1]
            cu_seqlens_q = input_data.kwargs["cuSeqLensQOptional"].to(self.set_device)
            cu_seqlens_k = input_data.kwargs["cuSeqLensKOptional"].to(self.set_device)
            seq_used_q = optional_tensor_to(
                input_data.kwargs.get("seqUsedQOptional"), self.set_device
            )
            seq_used_k = optional_tensor_to(
                input_data.kwargs.get("seqUsedKOptional"), self.set_device
            )
            max_seqlen_q = max(eval(input_data.kwargs["seqlens_list_q"]))
            max_seqlen_k = max(eval(input_data.kwargs["seqlens_list_k"]))
            batch_size = cu_seqlens_q.shape[0] - 1
            cmp_residual_k = input_data.kwargs["cmpResidualKOptional"].to(
                self.set_device
            )
            layout = input_data.kwargs["layoutQ"]
            sparse_indices = input_data.kwargs["sparseIndices"].to(self.set_device)
            topk = sparse_indices.shape[-1]
            sparse_mode = input_data.kwargs["maskMode"]
            cmp_ratio = input_data.kwargs["cmpRatio"]
            n_query_index = q.shape[-2]
            n_key = k.shape[-2]
            attn = input_data.kwargs["attnSoftmaxL1Norm"].to(self.set_device)
            metadata = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
                n_query_index,
                n_key,
                d_index,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seq_used_q,
                seqused_k=seq_used_k,
                cmp_residual_k=cmp_residual_k,
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                topk=topk,
                layout_q=layout,
                layout_k=layout,
                mask_mode=sparse_mode,
                cmp_ratio=cmp_ratio,
            )

            dq, dk, dw, softmax_out = (
                torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
                    q,
                    k,
                    w,
                    sparse_indices,
                    attn,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    seqused_q=seq_used_q,
                    seqused_k=seq_used_k,
                    cmp_residual_k=cmp_residual_k,
                    metadata=metadata,
                    layout_q=layout,
                    layout_k=layout,
                    mask_mode=sparse_mode,
                    cmp_ratio=cmp_ratio,
                )
            )
            torch.npu.synchronize()
            return dq, dk, dw, softmax_out

    def init_by_input_data(self, input_data: InputDataset):
        prepare_slikg_inputs(input_data, seed=self.task_result.case_config.id)

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *w, const aclTensor *sparseIndices,\
                const aclTensor *attnSoftmaxL1Norm, const aclTensor *cuSeqLensQOptional, const aclTensor *cuSeqLensKOptional, const aclTensor *seqUsedQOptional, const aclTensor *seqUsedKOptional,\
                const aclTensor *cmpResidualKOptional, const aclTensor *metadataOptional, char *layoutQ, char *layoutK, int64_t maskMode, int64_t cmpRatio, const aclTensor *dq, const aclTensor *dk,\
                const aclTensor *dw, const aclTensor *softmaxOut, uint64_t *workspaceSize, aclOpExecutor **executor)"


@register("pyaclnn_execute_aclnnSparseLightningIndexerKLLossGrad")
class PyaclnnSparseLightningIndexerKLLossGradApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        prepare_slikg_inputs(input_data, seed=self.task_result.case_config.id)
        input_data.kwargs["metadataOptional"] = build_slikg_metadata(
            input_data, device_id=int(self.device_id)
        )
        input_data.kwargs.pop("orgdtype", None)
        input_data.kwargs.pop("seqlens_list_q", None)
        input_data.kwargs.pop("seqlens_list_k", None)
        input_data.kwargs.pop("seq_used_mode", None)
        input_data.kwargs.pop("test_cmp_residual", None)
        input_args, output_packages = super().init_by_input_data(input_data)
        # GetWorkspaceSize args 7/8 are seqUsedQOptional/seqUsedKOptional.
        cast_optional_tensor_nulls(input_args, (7, 8))
        return input_args, output_packages


@register("execute_aclnnSparseLightningIndexerKLLossGrad_Abnormal")
class AclnnSparseLightningIndexerKLLossGrad_Abnormal(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            q = input_data.kwargs["q"]
            k = input_data.kwargs["k"]
            w = input_data.kwargs["w"]
            return q, k, w, torch.tensor(0)
        elif self.device == "npu":
            import cann_ops_transformer

            self.set_device = f"npu:{self.device_id}"
            q = input_data.kwargs["q"].to(self.set_device)
            k = input_data.kwargs["k"].to(self.set_device)
            w = input_data.kwargs["w"].to(self.set_device)
            d_index = q.shape[-1]
            cu_seqlens_q = input_data.kwargs["cuSeqLensQOptional"].to(self.set_device)
            cu_seqlens_k = input_data.kwargs["cuSeqLensKOptional"].to(self.set_device)
            seq_used_q = optional_tensor_to(
                input_data.kwargs.get("seqUsedQOptional"), self.set_device
            )
            seq_used_k = optional_tensor_to(
                input_data.kwargs.get("seqUsedKOptional"), self.set_device
            )
            max_seqlen_q = max(eval(input_data.kwargs["seqlens_list_q"]))
            max_seqlen_k = max(eval(input_data.kwargs["seqlens_list_k"]))
            batch_size = cu_seqlens_q.shape[0] - 1
            cmp_residual_k = input_data.kwargs["cmpResidualKOptional"].to(
                self.set_device
            )
            layout = input_data.kwargs["layoutQ"]
            sparse_indices = input_data.kwargs["sparseIndices"].to(self.set_device)
            topk = sparse_indices.shape[-1]
            sparse_mode = input_data.kwargs["maskMode"]
            cmp_ratio = input_data.kwargs["cmpRatio"]
            n_query_index = q.shape[-2]
            n_key = k.shape[-2]
            attn = input_data.kwargs["attnSoftmaxL1Norm"].to(self.set_device)
            metadata = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
                n_query_index,
                n_key,
                d_index,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seq_used_q,
                seqused_k=seq_used_k,
                cmp_residual_k=cmp_residual_k,
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                topk=topk,
                layout_q=layout,
                layout_k=layout,
                mask_mode=sparse_mode,
                cmp_ratio=cmp_ratio,
            )

            dq, dk, dw, softmax_out = (
                torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
                    q,
                    k,
                    w,
                    sparse_indices,
                    attn,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    seqused_q=seq_used_q,
                    seqused_k=seq_used_k,
                    cmp_residual_k=cmp_residual_k,
                    metadata=metadata,
                    layout_q=layout,
                    layout_k=layout,
                    mask_mode=sparse_mode,
                    cmp_ratio=cmp_ratio,
                )
            )
            torch.npu.synchronize()
            return dq, dk, dw, softmax_out

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "npu" and (self.name == "perf" or self.name == "memo"):
            prepare_slikg_inputs(input_data, seed=self.task_result.case_config.id)
