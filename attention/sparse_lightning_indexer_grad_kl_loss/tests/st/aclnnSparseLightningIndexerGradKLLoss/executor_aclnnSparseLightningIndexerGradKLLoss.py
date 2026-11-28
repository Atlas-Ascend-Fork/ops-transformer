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
import hashlib
import math
import os
import numpy as np
import torch
import torch.nn as nn
import random
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from einops import rearrange


MAX_MATMUL_CORE_NUM = 24
LOAD_ALIGN_SIZE = 512


def ceil_div(num1, num2):
    if num2 == 0:
        return 0
    return (num1 + num2 - 1) // num2


def align_up(num1, num2):
    if num2 == 0:
        return 0
    return ceil_div(num1, num2) * num2


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int64)
    for i in range(1, len(seqlens_list) + 1):
        cu[i] = cu[i - 1] + seqlens_list[i - 1]
    return cu


def getCaseInfo(q, k, q_index, k_index, sparse_indices):
    D = q.shape[3]
    B = q.shape[0]
    S1 = q.shape[1]
    S2 = k.shape[1]
    N1 = q.shape[2]
    N2 = k.shape[2]
    G = N1 // N2
    Nqi = q_index.shape[2]
    Nki = k_index.shape[2]
    G_index = Nqi // Nki
    D_index = q_index.shape[3]
    topKSize = sparse_indices.shape[3]

    return B, S1, S2, N1, N2, G, Nqi, Nki, G_index, D, D_index, topKSize


def getTNDCaseInfo(
    q,
    k,
    q_index,
    k_index,
    sparse_indices,
    query_rope,
    key_rope,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
):
    B = len(actual_seq_lengths_query)
    T1 = q.shape[0]
    T2 = k.shape[0]
    NQuery = q.shape[1]
    N2 = k.shape[1]
    NQueryIndex = q_index.shape[1]
    D1 = q.shape[2] if query_rope is None else q.shape[2] + query_rope.shape[2]
    D2 = q_index.shape[2]
    return B, T1, T2, NQuery, N2, NQueryIndex, D1, D2


class LightningIndicesKLLoss:
    # input     1. query, key, topk_index, scale, max, sum
    #           2. query_index, key_index, weight
    # output    1. KLLoss
    #           2. dW, dQ, dK
    def __init__(
        self,
        q,
        k,
        topk_index,
        scale,
        layout,
        query_index,
        key_index,
        weight,
        q_rope=None,
        k_rope=None,
        actual_seq_q_len=None,
        actual_seq_kv_len=None,
        ext_softmax_max=None,
        ext_softmax_sum=None,
    ):
        self.ext_softmax_max = ext_softmax_max
        self.ext_softmax_sum = ext_softmax_sum
        self.layout = layout
        if layout == "TND":
            self.B = len(actual_seq_q_len)
            self.N1 = q.shape[1]
            self.N2 = k.shape[1]
            self.Nqi = query_index.shape[1]
            self.Nki = key_index.shape[1]
            self.G = self.N1 // self.N2
            self.G_index = self.Nqi // self.Nki
            self.T1 = actual_seq_q_len[-1]
            self.T2 = actual_seq_kv_len[-1]
            self.D1 = q.shape[2]
            self.D2 = key_index.shape[2]
            self.topKSize = topk_index.shape[1]
        else:  # BSND
            self.B = q.shape[0]
            self.N1 = q.shape[2]
            self.N2 = k.shape[2]
            self.Nqi = query_index.shape[2]
            self.Nki = key_index.shape[2]
            self.G = self.N1 // self.N2
            self.G_index = self.Nqi // self.Nki
            self.S1 = q.shape[1]
            self.S2 = k.shape[1]
            self.D1 = q.shape[3]
            self.D2 = key_index.shape[3]
            self.topKSize = topk_index.shape[2]

        self.q = q
        self.k = k
        self.topk_index = topk_index
        self.scale = scale
        self.q_index = query_index
        self.k_index = key_index
        self.weight = weight

        self.q_rope = q_rope
        self.k_rope = k_rope
        self.actual_seq_q_len = actual_seq_q_len
        self.actual_seq_kv_len = actual_seq_kv_len

    def get_spoint_by_bIdx(self, bIdx):
        Sq, Skv = 0, 0
        if bIdx == 0:
            Sq = self.actual_seq_q_len[0]
            Skv = self.actual_seq_kv_len[0]
        elif bIdx > 0:
            Sq = self.actual_seq_q_len[bIdx]
            Skv = self.actual_seq_kv_len[bIdx]
        return Sq, Skv

    def run_unpad(self):
        if self.layout != "TND":
            raise ValueError(
                f"[error] case layout is not invalid!, case layout is {self.layout}, please check correct interface!"
            )
        # 初始化zeros tensor 接收每个B的输出
        B, T1, T2, NQuery, N2, NQueryIndex, DQuery, DQuerIndex = getTNDCaseInfo(
            self.q,
            self.k,
            self.q_index,
            self.k_index,
            self.topk_index,
            self.q_rope,
            self.k_rope,
            self.actual_seq_q_len,
            self.actual_seq_kv_len,
        )
        d_query_index = torch.zeros(T1, NQueryIndex, DQuerIndex)
        d_key_index = torch.zeros(T2, N2, DQuerIndex)
        d_weight = torch.zeros(T1, NQueryIndex)
        loss = torch.zeros(1)
        p_max = None
        p_sum = None

        for bIdx in range(B):
            q_start, kv_start = self.get_spoint_by_bIdx(bIdx - 1)
            q_end, kv_end = self.get_spoint_by_bIdx(bIdx)
            if q_start == q_end:
                continue
            # transpose to BSND
            q_i = self.q[q_start:q_end, :, :].unsqueeze(
                0
            )  # (T1, N1, DQuery) -> (1, S1, N1, DQuery)
            k_i = self.k[kv_start:kv_end, :, :].unsqueeze(
                0
            )  # (T1, N2, DQuery) -> (1, S2, N2, DQuery)
            q_index_i = self.q_index[q_start:q_end, :, :].unsqueeze(0)
            k_index_i = self.k_index[kv_start:kv_end, :, :].unsqueeze(0)
            q_rope_i = (
                self.q_rope[q_start:q_end, :, :].unsqueeze(0)
                if self.q_rope is not None
                else None
            )
            k_rope_i = (
                self.k_rope[kv_start:kv_end, :, :].unsqueeze(0)
                if self.k_rope is not None
                else None
            )
            # [N,T] -> [N,S,1]

            # TBD:
            weight = self.weight[q_start:q_end, :].unsqueeze(0)
            sparse_indices = self.topk_index[q_start:q_end, :, :].unsqueeze(
                0
            )  # [S1, N2, K] -> [1, S1, N2, K]

            saved_ext_max = self.ext_softmax_max
            saved_ext_sum = self.ext_softmax_sum
            if saved_ext_max is not None:
                self.ext_softmax_max = saved_ext_max[:, q_start:q_end, :]
                self.ext_softmax_sum = saved_ext_sum[:, q_start:q_end, :]
            d_query_index_i, d_key_index_i, d_weight_i, loss_i, p_max_i, p_sum_i = (
                self.golden(
                    q_i, k_i, q_index_i, k_index_i, self.scale, weight, sparse_indices
                )
            )
            if saved_ext_max is not None:
                self.ext_softmax_max = saved_ext_max
                self.ext_softmax_sum = saved_ext_sum

            # 填入数据
            d_query_index[q_start:q_end, :, :] = d_query_index_i
            d_key_index[kv_start:kv_end, :, :] = d_key_index_i
            d_weight[q_start:q_end, :] = d_weight_i

            # 确认维度 1 done
            loss[0] += loss_i

            if p_max is None:
                p_max = p_max_i
                p_sum = p_sum_i
            else:
                p_max = torch.cat((p_max, p_max_i), dim=0)
                p_sum = torch.cat((p_sum, p_sum_i), dim=0)
        p_max.unsqueeze(0)
        p_sum.unsqueeze(0)
        return d_query_index, d_key_index, d_weight, loss, p_max, p_sum

    def tsoftmax(self, x):
        x_max = torch.max(x, dim=-1, keepdims=True)[0]
        x_sub = x.sub(x_max)
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True)
        ans = y.div(x_sum)
        return ans, x_max, x_sum

    def safe_gather(self, key, topk_index, S1):
        """Gather key[:, S2, D] by sparse indices without materializing [B,S1,S2,D]."""
        S2 = key.size(1)
        # topk_index may be [B,S1,K] or [B,S1,K,D] (last dim broadcast of ids)
        idx = topk_index[..., 0] if topk_index.dim() == 4 else topk_index
        mask = (idx < 0) | (idx >= S2)
        safe_index = idx.clamp(min=0, max=max(S2 - 1, 0))
        B = key.size(0)
        batch_idx = (
            torch.arange(B, device=key.device).view(B, 1, 1).expand_as(safe_index)
        )
        key_topk = key[batch_idx, safe_index]
        key_topk = key_topk.masked_fill(mask.unsqueeze(-1), 0.0)
        return key_topk

    def safe_scatter_add(self, beforeScatterAdd, sparse_indices, S2):
        """Scatter-add [B,S1,K,D] -> [B,S2,1,D] without allocating [B,S1,S2,D]."""
        B, S1, K, D_index = beforeScatterAdd.shape
        mask = sparse_indices != -1
        safe_indices = sparse_indices.clone()
        safe_indices[~mask] = 0
        masked = beforeScatterAdd * mask.unsqueeze(-1)

        d_key = getattr(self, "_stream_d_key", None)
        if d_key is None:
            d_key = torch.zeros(
                B,
                S2,
                D_index,
                dtype=beforeScatterAdd.dtype,
                device=beforeScatterAdd.device,
            )
        flat_src = masked.reshape(-1, D_index)
        flat_idx = safe_indices.reshape(B, -1)
        offsets = (
            torch.arange(B, device=beforeScatterAdd.device, dtype=flat_idx.dtype).view(
                B, 1
            )
            * S2
        )
        global_idx = (flat_idx + offsets).reshape(-1)
        d_key.view(B * S2, D_index).index_add_(0, global_idx, flat_src)
        return d_key.unsqueeze(2)  # [B, S2, 1, D]

    def deal_invalid_token(self, B, S1, S2, topKSize, src_tensor):
        if hasattr(self, "_stream_origin_s1"):
            # Preserve the original causal position, rather than treating a tile as a new sequence.
            for j in range(S1):
                real = S2 - self._stream_origin_s1 + self._stream_start + j + 1
                if real <= 0:
                    real = S2
                if real < topKSize:
                    src_tensor[:, j, :, real:] = -float("inf")
            return src_tensor
        # Fast path: for S2 >> S1 (typical large-S), every row has enough valid K tokens.
        min_real = (S2 - S1) + 1
        if min_real <= 0:
            min_real = S2
        if min_real >= topKSize:
            return src_tensor
        dst_tensor = src_tensor
        for b_idx in range(B):
            for s1_idx in range(S1):
                s2_realsize = (int)((S2 - S1) + s1_idx + 1)
                if s2_realsize <= 0:
                    s2_realsize = S2
                invalid_len = topKSize - s2_realsize
                if invalid_len > 0:
                    dst_tensor[b_idx, s1_idx, :, s2_realsize:] = -float("inf")
        return dst_tensor

    def get_s2_real_size_for_split(self, s1_size, s2_size, s1_idx, topk_size):
        s2_real_size = (s2_size - s1_size) + s1_idx + 1
        if s2_real_size <= 0:
            s2_real_size = s2_size
        return min(align_up(int(s2_real_size), LOAD_ALIGN_SIZE), int(topk_size))

    def build_sparse_valid_array_for_split(self, B, S1, S2, topKSize):
        sparse_valid_array = []
        for _ in range(B):
            for s1_idx in range(S1):
                sparse_valid_array.append(
                    self.get_s2_real_size_for_split(S1, S2, s1_idx, topKSize)
                )
        return sparse_valid_array

    def init_load_value(
        self, sparse_valid_array, valid_core_num, total_size, sparse_start_idx
    ):
        local_value = [0 for _ in range(valid_core_num)]
        for idx in range(valid_core_num):
            start = sparse_start_idx[idx]
            end = sparse_start_idx[idx + 1] if idx + 1 < valid_core_num else total_size
            if start < total_size:
                local_value[idx] = sum(sparse_valid_array[start:end])
        return local_value

    def balance_load(
        self, sparse_valid_array, local_value, sparse_start_idx, total_size
    ):
        if not local_value:
            return False
        max_val = max(local_value)
        tmp_max_val = max_val
        valid_core_num = len(local_value)

        for idx in range(1, valid_core_num):
            start = sparse_start_idx[idx]
            if (
                start < total_size
                and start > 0
                and (local_value[idx - 1] + sparse_valid_array[start]) < max_val
            ):
                local_value[idx - 1] += sparse_valid_array[start]
                local_value[idx] -= sparse_valid_array[start]
                sparse_start_idx[idx] += 1
            elif start == total_size:
                break

        tmp_max_val = max(local_value)
        for idx in range(valid_core_num - 1, 0, -1):
            start = sparse_start_idx[idx]
            if start == total_size:
                if sparse_start_idx[idx - 1] == total_size:
                    continue
                local_value[idx - 1] -= sparse_valid_array[start - 1]
                local_value[idx] = sparse_valid_array[start - 1]
                sparse_start_idx[idx] -= 1
            elif start > 0:
                if (local_value[idx] + sparse_valid_array[start - 1]) >= tmp_max_val:
                    continue
                local_value[idx - 1] -= sparse_valid_array[start - 1]
                local_value[idx] += sparse_valid_array[start - 1]
                sparse_start_idx[idx] -= 1
            else:
                break

        return max(local_value) < max_val

    def balance_4d_load(self, sparse_valid_array, valid_core_num):
        if valid_core_num <= 0:
            return []
        sparse_start_idx = [0 for _ in range(valid_core_num)]
        sparse_array_sum = sum(sparse_valid_array)
        balance_num = ceil_div(sparse_array_sum, valid_core_num)
        tmp_index = 0
        sum_tmp_array = 0
        for idx, sparse_value in enumerate(sparse_valid_array):
            if tmp_index == valid_core_num - 1:
                break
            sum_tmp_array_last = sum_tmp_array
            sum_tmp_array += sparse_value
            if sum_tmp_array == balance_num:
                tmp_index = min(tmp_index + 1, valid_core_num - 1)
                sparse_start_idx[tmp_index] = idx + 1
                sum_tmp_array = 0
            elif sum_tmp_array > balance_num:
                tmp_index = min(tmp_index + 1, valid_core_num - 1)
                if balance_num - sum_tmp_array_last >= sum_tmp_array - balance_num:
                    sparse_start_idx[tmp_index] = idx + 1
                    sum_tmp_array = 0
                else:
                    sparse_start_idx[tmp_index] = idx
                    sum_tmp_array = sparse_value
        return sparse_start_idx

    def get_non_deterministic_split_ranges(self, B, S1, S2, topKSize):
        total_size = B * S1
        if total_size <= 0:
            return []
        valid_core_num = min(total_size, MAX_MATMUL_CORE_NUM)
        sparse_valid_array = self.build_sparse_valid_array_for_split(
            B, S1, S2, topKSize
        )

        if self.layout == "TND":
            split_factor_size = ceil_div(total_size, valid_core_num)
            even_start_idx = [
                min(idx * split_factor_size, total_size)
                for idx in range(valid_core_num)
            ]
            even_value = self.init_load_value(
                sparse_valid_array, valid_core_num, total_size, even_start_idx
            )

            sparse_array_sum = sum(sparse_valid_array)
            avg_val = ceil_div(sparse_array_sum, valid_core_num)
            balanced_start_idx = [total_size for _ in range(valid_core_num)]
            balanced_start_idx[0] = 0
            for idx in range(1, valid_core_num):
                start = balanced_start_idx[idx - 1]
                single_load_value = 0
                balanced_start_idx[idx] = start
                while (
                    single_load_value < avg_val and balanced_start_idx[idx] < total_size
                ):
                    single_load_value += sparse_valid_array[balanced_start_idx[idx]]
                    balanced_start_idx[idx] += 1
                if (start + 1) < balanced_start_idx[idx]:
                    redo_single_load_value = (
                        single_load_value
                        - sparse_valid_array[balanced_start_idx[idx] - 1]
                    )
                    if (single_load_value - avg_val) > (
                        avg_val - redo_single_load_value
                    ):
                        balanced_start_idx[idx] -= 1
                        single_load_value = redo_single_load_value
                    sparse_array_sum -= single_load_value
                    avg_val = ceil_div(sparse_array_sum, valid_core_num - idx)

            balanced_value = self.init_load_value(
                sparse_valid_array, valid_core_num, total_size, balanced_start_idx
            )
            while self.balance_load(
                sparse_valid_array, balanced_value, balanced_start_idx, total_size
            ):
                pass

            if max(even_value) <= max(balanced_value):
                start_idx = even_start_idx
            else:
                start_idx = balanced_start_idx
        else:
            start_idx = self.balance_4d_load(sparse_valid_array, valid_core_num)

        ranges = []
        for idx in range(valid_core_num):
            start = start_idx[idx]
            end = start_idx[idx + 1] if idx + 1 < valid_core_num else total_size
            if start < end:
                ranges.append((start, end))
        return ranges

    def blocked_matmul_by_bs1(self, lhs, rhs, B, S1, S2, topKSize):
        ranges = self.get_non_deterministic_split_ranges(B, S1, S2, topKSize)
        flat_lhs = lhs.reshape(B * S1, *lhs.shape[2:])
        flat_rhs = rhs.reshape(B * S1, *rhs.shape[2:])
        out_shape = (B * S1, flat_lhs.shape[-2], flat_rhs.shape[-1])
        flat_out = torch.empty(
            out_shape, dtype=torch.result_type(lhs, rhs), device=lhs.device
        )
        for start, end in ranges:
            flat_out[start:end] = torch.matmul(flat_lhs[start:end], flat_rhs[start:end])
        return flat_out.reshape(B, S1, flat_lhs.shape[-2], flat_rhs.shape[-1])

    def _golden_chunk(self, q, k, q_index, k_index, scale, weight, sparse_indices):
        # P 和S'Y的N实际是不同的, 需要额外处理  done
        # sparse 处理 done
        # case info
        B, S1, S2, N1, N2, G, Nqi, Nki, G_index, D, D_index, topKSize = getCaseInfo(
            q, k, q_index, k_index, sparse_indices
        )

        self.input_dtype = q.dtype
        use_ext_softmax = self.ext_softmax_max is not None
        query = q  # (B, S1, N1, D) -> (B, N1, S1, D)
        key = k.squeeze(2)
        sparse_indices = sparse_indices.squeeze(2).to(torch.int64)
        topk_index = sparse_indices.unsqueeze(-1).expand(-1, -1, -1, D)

        key_topk = self.safe_gather(key, topk_index, S1)  # B,S1,K,D
        # p = torch.matmul(query, key_topk.permute(0, 1, 3, 2).contiguous())  # (B, S1, N1, topK)
        p = self.blocked_matmul_by_bs1(
            query, key_topk.permute(0, 1, 3, 2).contiguous(), B, S1, S2, topKSize
        )
        # print("p.dtype===================================", p.dtype)

        # TODO scale mul 需要补
        p = p * scale
        p = self.deal_invalid_token(B, S1, S2, topKSize, p)
        if use_ext_softmax:
            if self.layout == "BSND":
                p_max = self.ext_softmax_max.permute(0, 2, 3, 1).to(
                    p.device, dtype=torch.float
                )
                p_sum = self.ext_softmax_sum.permute(0, 2, 3, 1).to(
                    p.device, dtype=torch.float
                )
            else:
                p_max = self.ext_softmax_max.unsqueeze(-1).to(
                    p.device, dtype=torch.float
                )
                p_sum = self.ext_softmax_sum.unsqueeze(-1).to(
                    p.device, dtype=torch.float
                )
            pSoftmaxRes = torch.exp(p.to(torch.float) - p_max) / p_sum
        else:
            pSoftmaxRes, p_max, p_sum = self.tsoftmax(p)
        # simplesoftmax
        reduceGShape = [B, S1, N2, G, topKSize]
        hideGShape = [B, S1, N2, topKSize]
        pSoftmaxRes = pSoftmaxRes.reshape(reduceGShape)
        pReduceGRes = torch.sum(
            pSoftmaxRes, dim=3
        )  # (B, N1, S1, topK) -> (B, N2, G, S1, topK) -> (B, N2, S1, topK)
        gRes = 1.0 / float(G)
        pReduceGRes = pReduceGRes * gRes
        if not use_ext_softmax:
            pReduceGRes = pReduceGRes.to(self.input_dtype)

        # calc S', Y
        query_index = q_index.to(torch.float32)  # (B, S1, N1, D) -> (B, N1, S1, D)
        key_index = k_index.squeeze(2).to(torch.float32)  # -> [B,S2,D]
        topk_index = sparse_indices.unsqueeze(-1).expand(
            -1, -1, -1, D_index
        )  # -> [B,S1,K,D]

        # key_index_topk = torch.gather(key_index.unsqueeze(1).expand(-1,S1,-1,-1), 2, topk_index)    # key_index [B,S2,D] -> [B,S1,S2,D]
        key_index_topk = self.safe_gather(key_index, topk_index, S1)
        # s = torch.matmul(query_index, key_index_topk.permute(0, 1, 3, 2).contiguous())  # (B, S1, N1, topK)
        s = self.blocked_matmul_by_bs1(
            query_index,
            key_index_topk.permute(0, 1, 3, 2).contiguous(),
            B,
            S1,
            S2,
            topKSize,
        )

        # relu res 复用
        reluRes = torch.relu(s)  #
        weight = weight.unsqueeze(-1).to(
            torch.float32
        )  # (B,S1,N1) -> (B,N1,S1) -> (B,N1,S1,1)
        sReluRes = reluRes * weight  #
        reduceGShape = [B, S1, Nki, G_index, topKSize]
        hideGShape = [B, S1, Nki, topKSize]
        sReduceGRes = torch.sum(sReluRes.reshape(reduceGShape), dim=3).reshape(
            hideGShape
        )  #
        # S2无效行场景 kSize 选择的行大于S2实际的行 末尾赋值-1 把矩阵末尾的数据修改成-inf
        sReduceGRes = self.deal_invalid_token(B, S1, S2, topKSize, sReduceGRes)

        softmax_func = nn.Softmax(dim=3)
        sSoftmaxRes = softmax_func(sReduceGRes)  # (B, S1, N2, topK)

        # KLLoss part
        # pReduceGRes is true, sSoftmaxRes is predict
        min_value = torch.tensor([1e-8])
        pReduceGResClip = torch.max(pReduceGRes, min_value)
        sSoftmaxResClip = torch.max(sSoftmaxRes, min_value)
        log_p = torch.log(pReduceGResClip)
        log_s = torch.log(sSoftmaxResClip)
        sub_res = log_p - log_s
        mul_res = sub_res * pReduceGRes
        loss = torch.sum(mul_res)

        # dW, dQ, dK
        if use_ext_softmax:
            q_sub_p = sSoftmaxRes.float() - pReduceGRes.float()
            d_weight = self.blocked_matmul_by_bs1(
                q_sub_p, reluRes.permute(0, 1, 3, 2).contiguous(), B, S1, S2, topKSize
            ).squeeze(2)
            relu_grad = q_sub_p * weight * (reluRes > 0).to(self.input_dtype)
        else:
            q_sub_p = sSoftmaxRes - pReduceGRes  # (B, N2, S1, topK) TOCHECK
            d_weight = self.blocked_matmul_by_bs1(
                q_sub_p, reluRes.permute(0, 1, 3, 2).contiguous(), B, S1, S2, topKSize
            ).squeeze(2)
            relu_grad = q_sub_p * weight * (reluRes > 0).float().to(self.input_dtype)

        # d_query_index = torch.matmul(relu_grad.to(self.input_dtype),
        #                              key_index_topk)  # (B, S1, N1, topK) * (B, S1, K, D)

        # beforeScatterAdd = torch.matmul(relu_grad.permute(0, 1, 3, 2).contiguous().to(self.input_dtype),
        #                                 query_index)  # [B,S1,N,D]
        # Retain the kernel's relu-gradient quantization boundary and the
        # existing low-precision dQ/dK matmul/scatter paths in this minimal patch.
        relu_grad = relu_grad.to(self.input_dtype)
        key_index_topk = key_index_topk.to(self.input_dtype)
        query_index = query_index.to(self.input_dtype)
        d_query_index = self.blocked_matmul_by_bs1(
            relu_grad, key_index_topk, B, S1, S2, topKSize
        )  # (B, S1, N1, topK) * (B, S1, K, D)
        beforeScatterAdd = self.blocked_matmul_by_bs1(
            relu_grad.permute(0, 1, 3, 2).contiguous(), query_index, B, S1, S2, topKSize
        )

        # Scatter across S1 into [B,S2,1,D] (do NOT allocate [B,S1,S2,D] — OOM on large S).
        d_key_index = self.safe_scatter_add(beforeScatterAdd, sparse_indices, S2)
        if self.layout == "TND":
            p_max = p_max.squeeze(0)
            p_sum = p_sum.squeeze(0)
            p_max = p_max.squeeze(-1)
            p_sum = p_sum.squeeze(-1)
        elif self.layout == "BSND":
            p_max = p_max.permute(0, 3, 1, 2)  # 输入为 (B S1 G N2) --->  (B N2 S1 G)
            p_sum = p_sum.permute(0, 3, 1, 2)
            # print(d_key_index[0, :, 0, 0:1])
        return d_query_index, d_key_index, d_weight, loss, p_max, p_sum

    def golden(self, q, k, q_index, k_index, scale, weight, sparse_indices):
        """Bound all gathered keys, attention scores and gradient temporaries by the tile size."""
        B, S1, _, _ = q.shape
        S2, Di = k.shape[1], k_index.shape[-1]
        chunk = int(os.environ.get("SLIGK_CPU_CHUNK", "64"))
        if chunk < 1:
            raise ValueError("SLIGK_CPU_CHUNK must be positive")
        dq = torch.empty(q_index.shape, dtype=q.dtype, device=q.device)
        dk = torch.zeros((B, S2, Di), dtype=q.dtype, device=q.device)
        dw = torch.empty(weight.shape, dtype=q.dtype, device=q.device)
        loss = torch.zeros((), dtype=torch.float32, device=q.device)
        saved_max, saved_sum = self.ext_softmax_max, self.ext_softmax_sum
        all_max, all_sum = [], []
        self._stream_origin_s1 = S1
        try:
            for b in range(B):
                max_tiles, sum_tiles = [], []
                self._stream_d_key = dk[b : b + 1]
                for start in range(0, S1, chunk):
                    end = min(start + chunk, S1)
                    self._stream_start = start
                    if saved_max is not None:
                        if self.layout == "BSND":
                            self.ext_softmax_max = saved_max[b : b + 1, :, start:end, :]
                            self.ext_softmax_sum = saved_sum[b : b + 1, :, start:end, :]
                        else:
                            self.ext_softmax_max = saved_max[:, start:end, :]
                            self.ext_softmax_sum = saved_sum[:, start:end, :]
                    out = self._golden_chunk(
                        q[b : b + 1, start:end],
                        k[b : b + 1],
                        q_index[b : b + 1, start:end],
                        k_index[b : b + 1],
                        scale,
                        weight[b : b + 1, start:end],
                        sparse_indices[b : b + 1, start:end],
                    )
                    dq[b : b + 1, start:end], dw[b : b + 1, start:end] = out[0], out[2]
                    loss += out[3].float()
                    max_tiles.append(out[4])
                    sum_tiles.append(out[5])
                    del out
                axis = 2 if self.layout == "BSND" else 0
                all_max.append(torch.cat(max_tiles, dim=axis))
                all_sum.append(torch.cat(sum_tiles, dim=axis))
        finally:
            self.ext_softmax_max, self.ext_softmax_sum = saved_max, saved_sum
            for name in ("_stream_d_key", "_stream_origin_s1", "_stream_start"):
                if hasattr(self, name):
                    delattr(self, name)
        return (
            dq,
            dk.unsqueeze(2),
            dw,
            loss.float(),
            torch.cat(all_max, dim=0),
            torch.cat(all_sum, dim=0),
        )

    def run(self):
        if self.layout != "BSND":
            raise ValueError(
                f"[error] case layout is not invalid!, case layout is {self.layout}, please check correct interface!"
            )
        return self.golden(
            self.q,
            self.k,
            self.q_index,
            self.k_index,
            self.scale,
            self.weight,
            self.topk_index,
        )


def lig_train_golden(
    q,
    k,
    sparse_indices,
    scale,
    input_layout,
    q_index,
    k_index,
    weights,
    q_rope,
    k_rope,
    actual_seq_qlen,
    actual_seq_kvlen,
    softmax_max=None,
    softmax_sum=None,
):
    op = LightningIndicesKLLoss(
        q,
        k,
        sparse_indices,
        scale,
        input_layout,
        q_index,
        k_index,
        weights,
        q_rope,
        k_rope,
        actual_seq_qlen,
        actual_seq_kvlen,
        softmax_max,
        softmax_sum,
    )
    if input_layout == "TND":
        return op.run_unpad()
    else:
        return op.run()


def safe_gather(key, topk_index, S1):
    """Gather key[:, S2, D] by sparse indices without materializing [B,S1,S2,D]."""
    S2 = key.size(1)
    idx = topk_index[..., 0] if topk_index.dim() == 4 else topk_index
    mask = (idx < 0) | (idx >= S2)
    safe_index = idx.clamp(min=0, max=max(S2 - 1, 0))
    B = key.size(0)
    batch_idx = torch.arange(B, device=key.device).view(B, 1, 1).expand_as(safe_index)
    key_topk = key[batch_idx, safe_index]
    key_topk = key_topk.masked_fill(mask.unsqueeze(-1), 0.0)
    return key_topk


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


def deal_invalid_token(B, S1, S2, topKSize, src_tensor):
    min_real = (S2 - S1) + 1
    if min_real <= 0:
        min_real = S2
    if min_real >= topKSize:
        return src_tensor
    dst_tensor = src_tensor
    for b_idx in range(B):
        for s1_idx in range(S1):
            s2_realsize = (int)((S2 - S1) + s1_idx + 1)
            if s2_realsize <= 0:
                s2_realsize = S2
            invalid_len = topKSize - s2_realsize
            if invalid_len > 0:
                dst_tensor[b_idx, s1_idx, :, s2_realsize:] = -float("inf")
    return dst_tensor


def get_spoint_by_bIdx(bIdx, actual_seq_q_len, actual_seq_kv_len):
    Sq, Skv = 0, 0
    if bIdx == 0:
        Sq = actual_seq_q_len[0]
        Skv = actual_seq_kv_len[0]
    elif bIdx > 0:
        Sq = actual_seq_q_len[bIdx]
        Skv = actual_seq_kv_len[bIdx]
    return Sq, Skv


def chunked_softmax(q, key_topk, sparse_indices, scale_value, chunk_size=8192):
    """
    q: (B, S1, N1, D)
    key_topk: (B, S1, K, D)
    sparse_indices: (B, S1, K)  # 含-1表示无效位置
    scale_value: float
    返回: softmax_max (B, S1, N1), softmax_sum (B, S1, N1)
    """
    B, S1, N1, D = q.shape
    K = key_topk.shape[2]
    max_list, sum_list = [], []
    for start in range(0, S1, chunk_size):
        end = min(start + chunk_size, S1)
        q_chunk = q[:, start:end, :, :]  # (B, chunk, N1, D)
        key_chunk = key_topk[:, start:end, :, :]  # (B, chunk, K, D)
        sparse_chunk = sparse_indices[:, start:end, :]  # (B, chunk, K)

        # 矩阵乘法
        p_chunk = (
            torch.matmul(q_chunk, key_chunk.permute(0, 1, 3, 2).contiguous())
            * scale_value
        )
        # 无效掩码
        mask = (
            (sparse_chunk != -1).unsqueeze(2).expand(-1, -1, N1, -1)
        )  # (B, chunk, N1, K)
        p_chunk = p_chunk.masked_fill(~mask, -float("inf"))

        # softmax 计算 max 和 sum (沿K维)
        max_vals = torch.max(p_chunk, dim=-1, keepdim=True)[0]  # (B, chunk, N1, 1)
        exp_vals = torch.exp(p_chunk - max_vals)
        sum_vals = torch.sum(exp_vals, dim=-1, keepdim=True)  # (B, chunk, N1, 1)
        max_vals = max_vals.squeeze(-1)  # (B, chunk, N1)
        sum_vals = sum_vals.squeeze(-1)  # (B, chunk, N1)
        max_list.append(max_vals)
        sum_list.append(sum_vals)

    softmax_max = torch.cat(max_list, dim=1)  # (B, S1, N1)
    softmax_sum = torch.cat(sum_list, dim=1)  # (B, S1, N1)
    return softmax_max, softmax_sum


def streaming_softmax(q, key, sparse_indices, scale_value):
    """Slice S1 first: never materialize [B, full_S1, K, D]."""
    B, S1, N1, _ = q.shape
    chunk = int(
        os.environ.get("SLIGK_PREP_CHUNK", os.environ.get("SLIGK_CPU_CHUNK", "64"))
    )
    if chunk < 1:
        raise ValueError("SLIGK_CPU_CHUNK must be positive")
    maximum = torch.empty((B, S1, N1), dtype=q.dtype, device=q.device)
    total = torch.empty_like(maximum)
    for b in range(B):
        for start in range(0, S1, chunk):
            end = min(start + chunk, S1)
            ids = sparse_indices[b : b + 1, start:end]
            gathered = safe_gather(key[b : b + 1], ids, end - start)
            m, z = chunked_softmax(
                q[b : b + 1, start:end], gathered, ids, scale_value, chunk_size=chunk
            )
            maximum[b : b + 1, start:end] = m
            total[b : b + 1, start:end] = z
            del gathered, m, z
    return maximum, total


@register("execute_aclnnSparseLightningIndexerGradKLLoss")
class AclnnSparseLightningIndexerGradKLLoss(BaseApi):
    def __init__(self, config):
        super().__init__(config)
        self.spag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        query = input_data.kwargs["query"]
        key = input_data.kwargs["key"]
        queryIndex = input_data.kwargs["queryIndex"]
        keyIndex = input_data.kwargs["keyIndex"]
        weights = input_data.kwargs["weights"]
        queryRope = input_data.kwargs["queryRope"]
        keyRope = input_data.kwargs["keyRope"]
        scaleValue = input_data.kwargs["scaleValue"]
        layout = input_data.kwargs["layout"]
        actualSeqLengthsQ = input_data.kwargs["actualSeqLengthsQ"]
        actualSeqLengthsK = input_data.kwargs["actualSeqLengthsK"]
        input_dtype = query.dtype
        q_all = torch.cat((query, queryRope), -1) if queryRope is not None else query
        k_all = torch.cat((key, keyRope), -1) if keyRope is not None else key
        q_all = q_all.to(input_dtype)
        k_all = k_all.to(input_dtype)
        if self.device == "cpu":
            prepare_sparse_lightning_indexer_inputs(
                input_data, int(self.task_result.case_config.id)
            )
            sparseIndices = input_data.kwargs["sparseIndices"]
            softmaxMax = input_data.kwargs["softmaxMax"]
            softmaxSum = input_data.kwargs["softmaxSum"]
            (
                cpu_d_query_index,
                cpu_d_key_index,
                cpu_d_weights,
                cpu_loss,
                softmax_max,
                softmax_sum,
            ) = lig_train_golden(
                q_all,
                k_all,
                sparseIndices,
                scaleValue,
                layout,
                queryIndex,
                keyIndex,
                weights,
                queryRope,
                keyRope,
                actualSeqLengthsQ,
                actualSeqLengthsK,
                softmaxMax,
                softmaxSum,
            )
            if cpu_loss.ndim == 0:
                cpu_loss = cpu_loss.reshape(1)
            return (
                cpu_d_query_index.to(input_dtype),
                cpu_d_key_index.to(input_dtype),
                cpu_d_weights.to(weights.dtype),
                cpu_loss,
            )
        return

    def init_by_input_data(self, input_data: InputDataset):
        print(
            "self.device=====================================================",
            self.device,
        )
        prepare_sparse_lightning_indexer_inputs(
            input_data, int(self.task_result.case_config.id)
        )

    @staticmethod
    def prepare_sparse_lightning_indexer_inputs(
        input_data: InputDataset, case_seed=128
    ):
        torch.manual_seed(case_seed)
        # Force CPU first to avoid NPU IndexCheck during golden/prepare.
        for _k, _v in list(input_data.kwargs.items()):
            if torch.is_tensor(_v):
                input_data.kwargs[_k] = _v.detach().to("cpu")
        query = input_data.kwargs["query"]
        key = input_data.kwargs["key"]
        queryIndex = input_data.kwargs["queryIndex"]
        keyIndex = input_data.kwargs["keyIndex"]
        weights = input_data.kwargs["weights"]
        queryRope = input_data.kwargs["queryRope"]
        keyRope = input_data.kwargs["keyRope"]
        scaleValue = input_data.kwargs["scaleValue"]
        layout = input_data.kwargs["layout"]
        sparseIndices = input_data.kwargs["sparseIndices"]

        q_all = torch.cat((query, queryRope), -1) if queryRope is not None else query
        k_all = torch.cat((key, keyRope), -1) if keyRope is not None else key
        q_all = q_all.to(torch.float)
        k_all = k_all.to(torch.float)

        sparse_indices = torch.zeros_like(sparseIndices).to(torch.int32)
        device = "cpu"

        query = query.to(device)
        key = key.to(device)
        q_all = q_all.to(device)
        k_all = k_all.to(device)
        sparse_indices = sparse_indices.to(device)

        if layout == "TND":
            T1 = query.shape[0]
            T2 = key.shape[0]
            seqlens_list_q = eval(input_data.kwargs["seqlens_list_q"])
            seqlens_list_kv = eval(input_data.kwargs["seqlens_list_k"])
            B = len(seqlens_list_q)
            seqlens_list_array = np.array(seqlens_list_q)
            seqlens_list_kv_array = np.array(seqlens_list_kv)
            actual_seq_qlen = tuple(
                get_cu_seqlens(seqlens_list_q)[1:].cpu().numpy().tolist()
            )
            actual_seq_kvlen = tuple(
                get_cu_seqlens(seqlens_list_kv)[1:].cpu().numpy().tolist()
            )

            tIdx = 0
            input_data.kwargs["actualSeqLengthsQ"] = actual_seq_qlen
            input_data.kwargs["actualSeqLengthsK"] = actual_seq_kvlen
            topKSize = sparse_indices.shape[2]
            for bIdx in range(B):
                for s1Idx in range(seqlens_list_array[bIdx]):
                    s2RealSize = (int)(
                        (seqlens_list_kv_array[bIdx] - seqlens_list_array[bIdx])
                        + s1Idx
                        + 1
                    )
                    if s2RealSize <= 0:
                        s2RealSize = seqlens_list_kv_array[bIdx]
                    if s2RealSize > topKSize:
                        s2RealLen = topKSize
                    else:
                        s2RealLen = s2RealSize
                    # 处理S2无效行场景，把对应的sparse indices置为-1
                    torch.manual_seed(case_seed)
                    sparse_indices[tIdx, :, 0:s2RealLen] = torch.randint(
                        0, s2RealSize, (s2RealLen,)
                    ).to(torch.int32)
                    sparse_indices[tIdx, :, s2RealLen:topKSize] = -1
                    tIdx = tIdx + 1
            input_data.kwargs["sparseIndices"] = sparse_indices.contiguous()
            softmax_max = None
            softmax_sum = None
            for bIdx in range(B):
                q_start, kv_start = get_spoint_by_bIdx(
                    bIdx - 1, actual_seq_qlen, actual_seq_kvlen
                )
                q_end, kv_end = get_spoint_by_bIdx(
                    bIdx, actual_seq_qlen, actual_seq_kvlen
                )
                if q_start == q_end:
                    continue
                # transpose to BSND
                q_i = q_all[q_start:q_end, :, :].unsqueeze(
                    0
                )  # (T1, N1, DQuery) -> (1, S1, N1, DQuery)
                k_i = k_all[kv_start:kv_end, :, :].unsqueeze(
                    0
                )  # (T1, N2, DQuery) -> (1, S2, N2, DQuery)
                sparse_indices_i = sparse_indices[q_start:q_end, :, :].unsqueeze(
                    0
                )  # [S1, N2, K] -> [1, S1, N2, K]
                B_i = q_i.shape[0]
                S1 = q_i.shape[1]
                S2 = k_i.shape[1]
                D = q_i.shape[3]
                k_i = k_i.squeeze(2).to(torch.float)  # N2轴压缩
                q_i = q_i.to(torch.float)
                sparse_indices_i = sparse_indices_i.squeeze(2).to(torch.int64)

                softmax_max_i, softmax_sum_i = streaming_softmax(
                    q_i, k_i, sparse_indices_i, scaleValue
                )
                softmax_max_i = softmax_max_i.squeeze(0)
                softmax_sum_i = softmax_sum_i.squeeze(0)

                if softmax_max is None:
                    softmax_max = softmax_max_i
                    softmax_sum = softmax_sum_i
                else:
                    softmax_max = torch.cat((softmax_max, softmax_max_i), dim=0)
                    softmax_sum = torch.cat((softmax_sum, softmax_sum_i), dim=0)
            softmax_max = softmax_max.unsqueeze(0)
            softmax_sum = softmax_sum.unsqueeze(0)

        else:
            q = q_all  # (B, S1, N1, D) -> (B, N1, S1, D)
            k = k_all.squeeze(2)  # N2轴压缩
            B = q.shape[0]
            S1 = q.shape[1]
            S2 = k.shape[1]
            D = q.shape[3]
            topKSize = sparse_indices.shape[3]
            # Vectorized sparse-index fill (avoid per-S1 Python loop on large S).
            # s2_real[s1] = max(S2, (S2 - S1) + s1 + 1) when (S2-S1)+s1+1 > 0 else S2
            s1_arange = torch.arange(S1, device=sparse_indices.device)
            s2_real = (S2 - S1) + s1_arange + 1
            s2_real = torch.where(s2_real <= 0, torch.full_like(s2_real, S2), s2_real)
            s2_real_len = torch.clamp(s2_real, max=topKSize)  # [S1]
            # Sample in [0, s2_real); invalid tail -> -1
            rand = torch.rand(B, S1, 1, topKSize, device=sparse_indices.device)
            sampled = (rand * s2_real.view(1, S1, 1, 1).to(rand.dtype)).to(torch.int32)
            k_arange = torch.arange(topKSize, device=sparse_indices.device).view(
                1, 1, 1, topKSize
            )
            valid = k_arange < s2_real_len.view(1, S1, 1, 1)
            sparse_indices = torch.where(valid, sampled, torch.full_like(sampled, -1))
            input_data.kwargs["sparseIndices"] = sparse_indices.contiguous()
            # BSND: optional cu_seqlens unused; pass None to match aclnn optional nullptr.
            input_data.kwargs["actualSeqLengthsQ"] = None
            input_data.kwargs["actualSeqLengthsK"] = None
            sparse_indices = sparse_indices.squeeze(2).to(torch.int64)
            softmax_max, softmax_sum = streaming_softmax(
                q, k, sparse_indices, scaleValue
            )
            # (B, S1, N1) -> (B, 1, S1, N1)
            softmax_max = softmax_max.unsqueeze(-1).permute(0, 3, 1, 2)
            softmax_sum = softmax_sum.unsqueeze(-1).permute(0, 3, 1, 2)

        if layout == "TND" and query.shape[0] > softmax_max.shape[1]:
            padding_len = query.shape[0] - softmax_max.shape[1]
            softmax_max = torch.cat(
                [
                    softmax_max,
                    torch.zeros(
                        softmax_max.shape[0],
                        padding_len,
                        softmax_max.shape[2],
                        device=softmax_max.device,
                    ),
                ],
                dim=1,
            )
            softmax_sum = torch.cat(
                [
                    softmax_sum,
                    torch.zeros(
                        softmax_max.shape[0],
                        padding_len,
                        softmax_sum.shape[2],
                        device=softmax_sum.device,
                    ),
                ],
                dim=1,
            )
        print(
            "RECHECK_PREPARE",
            case_seed,
            hashlib.sha256(
                sparse_indices.reshape(-1)[:4096].numpy().tobytes()
            ).hexdigest(),
            hashlib.sha256(
                query.reshape(-1)[:4096].to(torch.float16).numpy().tobytes()
            ).hexdigest(),
            flush=True,
        )
        input_data.kwargs["softmaxMax"] = softmax_max.contiguous()
        input_data.kwargs["softmaxSum"] = softmax_sum.contiguous()
        # # test
        # save_file = os.path.join(save_dir, "sparseIndices.pt")
        # torch.save(input_data.kwargs['sparseIndices'], save_file)
        # save_file = os.path.join(save_dir, "softmaxMax.pt")
        # torch.save(input_data.kwargs['softmaxMax'], save_file)
        # save_file = os.path.join(save_dir, "softmaxSum.pt")
        # torch.save(input_data.kwargs['softmaxSum'], save_file)


_prepare_uncached = (
    AclnnSparseLightningIndexerGradKLLoss.prepare_sparse_lightning_indexer_inputs
)
_prepare_cache = None
_PREPARE_INPUT_NAMES = (
    "query",
    "key",
    "queryIndex",
    "keyIndex",
    "weights",
    "queryRope",
    "keyRope",
)
_PREPARE_OUTPUT_NAMES = ("sparseIndices", "softmaxMax", "softmaxSum")


def _tensor_signature(t):
    if not torch.is_tensor(t):
        return repr(t)
    return (
        t.data_ptr(),
        tuple(t.shape),
        tuple(t.stride()),
        str(t.dtype),
        str(t.device),
        t._version,
    )


def prepare_sparse_lightning_indexer_inputs(input_data, case_seed=128):
    global _prepare_cache
    values = tuple(input_data.kwargs.get(k) for k in _PREPARE_INPUT_NAMES)
    attrs = tuple(
        repr(input_data.kwargs.get(k))
        for k in ("layout", "scaleValue", "seqlens_list_q", "seqlens_list_k")
    )
    signature = (case_seed, attrs, tuple(_tensor_signature(t) for t in values))
    outputs = tuple(
        _tensor_signature(input_data.kwargs.get(k)) for k in _PREPARE_OUTPUT_NAMES
    )
    if (
        _prepare_cache is not None
        and _prepare_cache[0] == signature
        and _prepare_cache[1] == outputs
    ):
        print("PREPARE_CACHE_HIT", case_seed, flush=True)
        return
    _prepare_cache = None
    started = __import__("time").monotonic()
    _prepare_uncached(input_data, case_seed)
    # Hold source refs only for the most recent input, preventing allocator-address reuse hits.
    values = tuple(input_data.kwargs.get(k) for k in _PREPARE_INPUT_NAMES)
    signature = (case_seed, attrs, tuple(_tensor_signature(t) for t in values))
    outputs = tuple(
        _tensor_signature(input_data.kwargs.get(k)) for k in _PREPARE_OUTPUT_NAMES
    )
    _prepare_cache = (signature, outputs, values)
    print(
        "PREPARE_COMPUTE_SECONDS",
        case_seed,
        __import__("time").monotonic() - started,
        flush=True,
    )


@register("execute_aclnnSparseLightningIndexerGradKLLoss_perf")
class AclnnSparseLightningIndexerGradKLLoss(BaseApi):
    def __init__(self, config):
        super().__init__(config)
        self.spag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        queryIndex = input_data.kwargs["queryIndex"]
        keyIndex = input_data.kwargs["keyIndex"]
        weights = input_data.kwargs["weights"]

        return queryIndex, keyIndex, weights, torch.tensor([1]).to(torch.float32)


from atk.tasks.backends.lib_interface.acl_wrapper import (
    AclTensor,
    AclTensorStruct,
    AclIntArray,
)
import ctypes


@register("pyaclnn_execute_aclnnSparseLightningIndexerGradKLLoss")
class SparseLightningIndexerGradKLLossApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        prepare_sparse_lightning_indexer_inputs(
            input_data, int(self.task_result.case_config.id)
        )
        # seqlens_list_* are ATK helpers only; not in aclnn signature.
        input_data.kwargs.pop("seqlens_list_q", None)
        input_data.kwargs.pop("seqlens_list_k", None)
        input_args, output_packages = super().init_by_input_data(input_data)
        # BSND sets actualSeqLengths*=None → ATK converts to c_void_p(0).
        # After a TND case binds GetWorkspaceSize with LP_AclIntArray, later BSND
        # calls fail with: expected LP_AclIntArray instead of c_void_p.
        # Use a typed NULL pointer so argtypes stay consistent and ABI is nullptr.
        null_int_array = ctypes.POINTER(AclIntArray)()
        fixed_args = []
        for arg in input_args:
            if isinstance(arg, ctypes.c_void_p) and not arg.value:
                fixed_args.append(null_int_array)
            else:
                fixed_args.append(arg)
        return fixed_args, output_packages

    def after_call(self, output_packages: AclTensorStruct):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        # print("output[3]======", output[3])
        return output

    def get_format(self, input_data: InputDataset, index=None, name=None):
        """
        :param input_data: 参数列表
        :param index: 参数位置
        :param name: 参数名字
        :return:
        format at this index or name
        """
        from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

        if "is_nd" in input_data.kwargs:
            is_nd = input_data.kwargs["is_nd"]
            if is_nd is False and name == "query":
                return AclFormat.ACL_FORMAT_FRACTAL_NZ
        return AclFormat.ACL_FORMAT_ND
