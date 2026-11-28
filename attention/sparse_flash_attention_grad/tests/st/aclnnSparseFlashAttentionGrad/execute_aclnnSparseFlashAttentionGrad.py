# -----------------------------------------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2023. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
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


def split_number(total1, parts1, total2, parts2):
    # 生成 parts-1 个随机分割点
    dividers1 = sorted(random.sample(range(1, total1), parts1 - 1))
    # 计算每个区间
    result1 = (
        [dividers1[0]]
        + [dividers1[i + 1] - dividers1[i] for i in range(len(dividers1) - 1)]
        + [total1 - dividers1[-1]]
    )
    # 生成 parts-1 个随机分割点
    dividers2 = sorted(random.sample(range(1, total2), parts2 - 1))
    # 计算每个区间
    result2 = (
        [dividers2[0]]
        + [dividers2[i + 1] - dividers2[i] for i in range(len(dividers2) - 1)]
        + [total2 - dividers2[-1]]
    )
    return result1, result2


def cal_attenmask(topk, s1_idx, selected_block_size, select_s2, G, cur_s2, cur_s1):
    atten_msk = torch.zeros(select_s2)
    sparse_tail_idx = math.ceil(cur_s2 / selected_block_size)
    sparse_tail_seq_len = cur_s2 % selected_block_size
    if sparse_tail_seq_len == 0:
        sparse_tail_seq_len = selected_block_size

    # ###### sparseMode=3 ######
    delta_s = cur_s2 - cur_s1
    threshold = delta_s + s1_idx + 1

    s_idx = 0
    for _, sparse_id in enumerate(topk):
        if sparse_id == -1:
            break
        begin_idx = sparse_id * selected_block_size
        block_len = (
            selected_block_size
            if sparse_id != sparse_tail_idx - 1
            else sparse_tail_seq_len
        )
        end_idx = begin_idx + block_len
        # print("begin_idx:", begin_idx)
        # print("end_idx:", end_idx)
        if begin_idx < threshold and end_idx <= threshold:
            s_idx += block_len
            continue
        elif end_idx > threshold:
            local_offset = 0 if threshold <= begin_idx else threshold - begin_idx
            mask_begin = s_idx + local_offset
            mask_end = s_idx + block_len

            atten_msk[mask_begin:mask_end] = 1
        s_idx += block_len

    atten_msk = atten_msk.repeat(G, 1)
    return atten_msk


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    softmax_res = y.div(x_sum)
    return softmax_res, x_max, x_sum


def simpleSoftmax(x, x_max, x_sum):
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    softmax_res = y.div(x_sum)
    return softmax_res


def tsoftmax_grad(p, dp, out, outGrad):
    # print("0 count in p:", (p == 0).sum().item())
    muls = outGrad * out
    muls_res = muls.sum(dim=-1, keepdims=True)
    sub_res = dp - muls_res
    if torch.isnan(sub_res).any():
        print("sub_res has nan")
    if torch.isinf(sub_res).any():
        print("sub_res has inf")
    if torch.isnan(p).any():
        print("p has nan")
    if torch.isinf(p).any():
        print("p has inf")
    # print("sub_res:", sub_res)
    res = sub_res * p
    if torch.isnan(res).any():
        print("res has nan")
    if torch.isinf(res).any():
        print("res has inf")
    nan_mask = torch.isnan(res)
    if nan_mask.any():
        print("nan count:", nan_mask.sum().item())
        print("sub_res at nan:", sub_res[nan_mask][:20])
        print("p at nan:", p[nan_mask][:20])
    return res


def get_currentB_index(BS_index, actual_seq_len):
    for i in range(len(actual_seq_len)):
        if BS_index < actual_seq_len[i]:
            return i
    raise RuntimeError("BS_index is greater than max(actual_seq_len).")


def get_tnd_idx(actual_q_len, t_idx):
    b_idx = 0
    while t_idx >= actual_q_len[b_idx]:
        b_idx += 1
    if b_idx == 0:
        s1_offset = 0
    else:
        s1_offset = actual_q_len[b_idx - 1]
    s1_idx = t_idx - s1_offset

    return b_idx, s1_idx


class NsaSelectedAttention:
    def __init__(
        self,
        T1,
        T2,
        N1,
        N2,
        D,
        B,
        G,
        query,
        key,
        value,
        out,
        dy,
        topkIndices,
        attenMsk,
        dtype,
        actualSeqLenQ,
        actualSeqLenKV,
        scaleValue=1.0,
        blockCount=16,
        blockSize=64,
        atten_enable=True,
        softmaxMax=None,
        softmaxSum=None,
        q_rope=None,
        k_rope=None,
    ):
        self.query = query
        self.key = key
        self.value = value
        self.out = out
        self.dy = dy
        self.topkIndices = topkIndices
        self.attenMsk = attenMsk
        self.actualSeqLenQ = actualSeqLenQ
        self.actualSeqLenKV = actualSeqLenKV
        self.q_rope = q_rope if q_rope.numel() != 0 else None
        self.k_rope = k_rope if k_rope.numel() != 0 else None
        # param
        self.scaleValue = scaleValue
        self.blockCount = blockCount
        self.blockSize = blockSize
        self.atten_enable = atten_enable
        self.dtype = query.dtype
        self.softmaxMax = softmaxMax
        self.softmaxSum = softmaxSum
        self.orgDtype = dtype
        self.T1, self.T2, self.N1, self.N2, self.D, self.B, self.G = (
            T1,
            T2,
            N1,
            N2,
            D,
            B,
            G,
        )

    def gather_kv(self, cur_k, cur_v, topk, cur_s2):
        actual_sel_s2 = 0
        s2_sparse = list()
        for sparse_id in topk:
            if sparse_id == -1:
                break
            begin_idx = sparse_id * self.blockSize
            end_idx = (
                begin_idx + self.blockSize
                if begin_idx + self.blockSize <= cur_s2
                else cur_s2
            )
            s2_sparse.extend(np.arange(begin_idx, end_idx))
            actual_sel_s2 += end_idx - begin_idx
        k_cal, v_cal = cur_k[s2_sparse, :], cur_v[s2_sparse, :]
        return k_cal, v_cal, actual_sel_s2

    def scatter_kv(self, dkv_out, dkv, topk, cur_s2):
        """
        dkv_out: [cur_s2, D] or [cur_s2, D2]
        dkv: [actualSelLen, D] or [actualSelLen, D2]
        topk: [blockCount]
        """
        dkv_start = 0
        for sparse_id in topk:
            if sparse_id == -1:
                break
            begin_idx = sparse_id * self.blockSize
            end_idx = (
                begin_idx + self.blockSize
                if begin_idx + self.blockSize <= cur_s2
                else cur_s2
            )
            dkv_end = dkv_start + (end_idx - begin_idx)
            dkv_out[begin_idx:end_idx, :] += dkv[dkv_start:dkv_end, :]
            dkv_start = dkv_end
        return dkv_out

    def forward(self):
        query = self.query.float()
        key = self.key.float()
        value = self.value.float()
        topk_indices = self.topkIndices
        actual_q_len = self.actualSeqLenQ
        actual_kv_len = self.actualSeqLenKV

        # param
        scaleValue = self.scaleValue
        selected_block_count = self.blockCount
        selected_block_size = self.blockSize
        select_s2 = selected_block_size * selected_block_count

        # shape
        S1 = max(self.actualSeqLenQ)
        S2 = max(self.actualSeqLenKV)
        # reshape
        query = query.reshape(self.T1, self.N2, self.G, self.D)
        key = key.reshape(self.T2, self.N2, self.D)
        value = value.reshape(self.T2, self.N2, self.D)
        D_qk = self.D
        if self.q_rope != None and self.k_rope != None:
            q_rope = self.q_rope.float()
            k_rope = self.k_rope.reshape(
                self.T2, self.N2, self.k_rope.shape[-1]
            ).float()
            q_rope = q_rope.reshape(self.T1, self.N2, self.G, q_rope.shape[-1])
            query = torch.cat([query, q_rope], dim=-1)
            key = torch.cat([key, k_rope], dim=-1)
            D_qk += q_rope.shape[-1]

        out = torch.zeros(self.T1, self.N2, self.G, self.out.shape[-1]).to(torch.float)
        x_max_out = torch.zeros(self.T1, self.N2, self.G, 1).to(torch.float)
        x_sum_out = torch.zeros(self.T1, self.N2, self.G, 1).to(torch.float)
        for i in range(self.T1):
            b_idx, s1_idx = get_tnd_idx(actual_q_len, i)
            for n2_idx in range(self.N2):
                topk = topk_indices[i][n2_idx]
                q_cal = query[i][n2_idx]  # [G, D1]

                s2_start = 0 if b_idx == 0 else actual_kv_len[b_idx - 1]
                s2_end = actual_kv_len[b_idx]
                cur_s2 = (
                    actual_kv_len[b_idx]
                    if b_idx == 0
                    else (actual_kv_len[b_idx] - actual_kv_len[b_idx - 1])
                )
                cur_s1 = (
                    actual_q_len[b_idx]
                    if b_idx == 0
                    else (actual_q_len[b_idx] - actual_q_len[b_idx - 1])
                )
                cur_k = key[s2_start:s2_end, n2_idx, :]
                cur_v = value[s2_start:s2_end, n2_idx, :]

                k_cal, v_cal, actual_sel_s2 = self.gather_kv(cur_k, cur_v, topk, cur_s2)
                if actual_sel_s2 > 0:
                    qk = torch.matmul(q_cal, k_cal.permute(1, 0)).mul(
                        scaleValue
                    )  # [G, D] * [D, selLen]
                    if self.atten_enable:
                        atten_msk_cal = cal_attenmask(
                            topk,
                            s1_idx,
                            selected_block_size,
                            actual_sel_s2,
                            self.G,
                            cur_s2,
                            cur_s1,
                        ).to(qk.device)

                        qk = qk + atten_msk_cal * (-2e35)

                    softmax_res, x_max, x_sum = tsoftmax(qk)
                    x_max_out[i][n2_idx] = x_max
                    x_sum_out[i][n2_idx] = x_sum

                    outi = torch.matmul(softmax_res, v_cal)
                    out[i][n2_idx] = outi
        x_max_out = x_max_out.expand(self.T1, self.N2, self.G, 8)
        x_sum_out = x_sum_out.expand(self.T1, self.N2, self.G, 8)

        self.softmaxMax = x_max_out.reshape(self.T1, self.N2 * self.G, 8)
        self.softmaxSum = x_sum_out.reshape(self.T1, self.N2 * self.G, 8)
        # self.out = out.reshape(self.T1, self.N2 * self.G, out.shape[-1]).to(self.orgDtype)

        return self.out, self.softmaxMax, self.softmaxSum

    def backward(self):
        query = self.query.float()
        key = self.key.reshape(self.T2, self.N2, self.D).float()
        value = self.value.reshape(self.T2, self.N2, self.D).float()
        out = self.out.float()
        dy = self.dy.float()
        softmaxMax = self.softmaxMax.float()
        softmaxSum = self.softmaxSum.float()
        topk_indices = self.topkIndices
        actual_q_len = self.actualSeqLenQ
        actual_kv_len = self.actualSeqLenKV

        # param
        scaleValue = self.scaleValue
        selected_block_count = self.blockCount
        selected_block_size = self.blockSize
        select_s2 = selected_block_size * selected_block_count

        # shape
        S1 = max(self.actualSeqLenQ)
        S2 = max(self.actualSeqLenKV)
        # reshape
        query = query.reshape(self.T1, self.N2, self.G, self.D)
        out = out.reshape(self.T1, self.N2, self.G, self.D)
        dy = dy.reshape(self.T1, self.N2, self.G, self.D)
        softmaxMax = softmaxMax.reshape(self.T1, self.N2, self.G, 8)
        softmaxSum = softmaxSum.reshape(self.T1, self.N2, self.G, 8)
        D_qk = self.D
        if self.q_rope != None and self.k_rope != None:
            q_rope = self.q_rope
            k_rope = self.k_rope.reshape(self.T2, self.N2, self.k_rope.shape[-1])
            q_rope = q_rope.reshape(self.T1, self.N2, self.G, q_rope.shape[-1])
            query = torch.cat([query, q_rope], dim=-1)
            key = torch.cat([key, k_rope], dim=-1)
            D_qk += q_rope.shape[-1]

        dq_out = torch.zeros(self.T1, self.N2, self.G, D_qk).to(torch.float)
        x_max_out = torch.zeros(self.T1, self.N2, self.G, 1).to(torch.float)
        x_sum_out = torch.zeros(self.T1, self.N2, self.G, 1).to(torch.float)
        dk_out = torch.zeros(S2, self.N2, D_qk).to(torch.float)
        dv_out = torch.zeros(S2, self.N2, self.D).to(torch.float)

        for i in range(self.T1):
            b_idx, s1_idx = get_tnd_idx(actual_q_len, i)

            for n2_idx in range(self.N2):
                # gather
                topk = topk_indices[i][n2_idx]
                q_cal = query[i][n2_idx]
                out_cal = out[i][n2_idx]
                dy_cal = dy[i][n2_idx]

                s2_start = 0 if b_idx == 0 else actual_kv_len[b_idx - 1]
                s2_end = actual_kv_len[b_idx]
                cur_s2 = (
                    actual_kv_len[b_idx]
                    if b_idx == 0
                    else (actual_kv_len[b_idx] - actual_kv_len[b_idx - 1])
                )
                cur_s1 = (
                    actual_q_len[b_idx]
                    if b_idx == 0
                    else (actual_q_len[b_idx] - actual_q_len[b_idx - 1])
                )
                cur_k = key[s2_start:s2_end, n2_idx, :]
                cur_v = value[s2_start:s2_end, n2_idx, :]
                k_cal, v_cal, actual_sel_s2 = self.gather_kv(cur_k, cur_v, topk, cur_s2)

                if actual_sel_s2 > 0:
                    # fag cal
                    qk = torch.matmul(q_cal, k_cal.permute(1, 0)).mul(scaleValue)
                    if self.atten_enable:
                        atten_msk_cal = cal_attenmask(
                            topk,
                            s1_idx,
                            selected_block_size,
                            actual_sel_s2,
                            self.G,
                            cur_s2,
                            cur_s1,
                        )
                        qk = qk + atten_msk_cal * (-2e35)

                    x_max = softmaxMax[i][n2_idx][:, [0]].reshape(-1, 1)
                    x_sum = softmaxSum[i][n2_idx][:, [0]].reshape(-1, 1)
                    softmax_res = simpleSoftmax(qk, x_max, x_sum)

                    dp = torch.matmul(dy_cal, v_cal.permute(1, 0))
                    softmax_grad_res = tsoftmax_grad(softmax_res, dp, out_cal, dy_cal)
                    softmax_grad_res = softmax_grad_res.to(self.orgDtype).to(
                        torch.float
                    )
                    dq = torch.matmul(softmax_grad_res, k_cal)
                    if torch.isnan(softmax_grad_res).any():
                        print("softmax_grad_res has nan")
                    if torch.isinf(softmax_grad_res).any():
                        print("softmax_grad_res has inf")
                    if (softmax_grad_res == 0).sum().item() > 0:
                        print("softmax_grad_res has 0")
                    if torch.isnan(q_cal).any():
                        print("q_cal has nan")
                    if torch.isinf(q_cal.to(torch.bfloat16)).any():
                        print("q_cal has inf")
                    if (q_cal.to(torch.bfloat16) == 0).sum().item() > 0:
                        print("q_cal.to(torch.bfloat16) has 0")
                    if torch.isnan(q_cal).any():
                        print("q_cal has nan")
                    if torch.isinf(q_cal).any():
                        print("q_cal has inf")
                    if (q_cal == 0).sum().item() > 0:
                        print("q_cal has 0")
                    dk = torch.matmul(softmax_grad_res.permute(1, 0), q_cal)
                    if torch.isnan(dk).any():
                        print("dk has nan")
                        nan_idx = torch.isnan(dk).nonzero()
                        print("nan_idx:", nan_idx[:10])
                        m, n = nan_idx[0].tolist()
                        a = softmax_grad_res.permute(1, 0)[m, :]  # [softmax_grad_res K]
                        b = q_cal[:, n]  # [q_cal K]
                        prod = a * b

                        print("a has inf:", torch.isinf(a).any().item())
                        print("b has inf:", torch.isinf(b).any().item())
                        print("prod has nan:", torch.isnan(prod).any().item())
                        print("prod has +inf:", torch.isposinf(prod).any().item())
                        print("prod has -inf:", torch.isneginf(prod).any().item())
                        print("sum(prod):", prod.sum())
                    dv = torch.matmul(
                        softmax_res.permute(1, 0).to(self.orgDtype).to(torch.float),
                        dy_cal,
                    )

                    # scatter
                    dq_out[i][n2_idx] = dq
                    dk_out[s2_start:s2_end, n2_idx, :] = self.scatter_kv(
                        dk_out[s2_start:s2_end, n2_idx, :], dk, topk, cur_s2
                    )
                    dv_out[s2_start:s2_end, n2_idx, :] = self.scatter_kv(
                        dv_out[s2_start:s2_end, n2_idx, :], dv, topk, cur_s2
                    )

        dq_out = dq_out * scaleValue
        dk_out = dk_out * scaleValue

        dq_out = dq_out.reshape(self.T1, self.N1, D_qk)

        if self.q_rope != None and self.k_rope != None:
            dq_rope = dq_out[:, :, self.D :]
            dk_rope = dk_out[:, :, self.D :]
            dk = dk_out[:, :, : self.D]
            dq = dq_out[:, :, : self.D]
        else:
            dq = dq_out
            dk = dk_out
            dq_rope = None
            dk_rope = None

        return (
            dq.to(self.orgDtype),
            dk.to(self.orgDtype),
            dv_out.to(self.orgDtype),
            dq_rope.to(self.orgDtype),
            dk_rope.to(self.orgDtype),
        )


def get_topk(
    atten_enable,
    B,
    T1,
    N2,
    selected_block_size,
    selected_block_count,
    actual_kv_len,
    actual_q_len,
):
    if atten_enable:
        topk_indices = (
            torch.ones(
                T1,
                N2,
                selected_block_count,
            ).to(torch.int)
            * -1
        )
        for i in range(T1):
            b_idx, s1_idx = get_tnd_idx(actual_q_len, i)
            cur_s2 = (
                actual_kv_len[b_idx]
                if b_idx == 0
                else (actual_kv_len[b_idx] - actual_kv_len[b_idx - 1])
            )
            cur_s1 = (
                actual_q_len[b_idx]
                if b_idx == 0
                else (actual_q_len[b_idx] - actual_q_len[b_idx - 1])
            )
            delta_s = cur_s2 - cur_s1

            for j in range(N2):
                threshold = delta_s + s1_idx + 1 if delta_s + s1_idx + 1 >= 0 else 0
                max_valid_count = min(
                    math.ceil(threshold / selected_block_size), selected_block_count
                )
                if threshold % selected_block_size != 0:
                    # 保证非对齐partial尾块在有效部分最后
                    topk_indices[i][j][: (max_valid_count - 1)] = torch.randperm(
                        max_valid_count - 1
                    )
                    topk_indices[i][j][max_valid_count - 1] = max_valid_count - 1
                else:
                    topk_indices[i][j][:max_valid_count] = torch.randperm(
                        max_valid_count
                    )
    else:
        topk_indices = (
            torch.ones([T1, N2, selected_block_count], dtype=torch.int32) * -1
        )
        for i in range(B):
            cur_s2 = (
                actual_kv_len[i]
                if i == 0
                else (actual_kv_len[i] - actual_kv_len[i - 1])
            )
            s1_start = 0 if i == 0 else actual_q_len[i - 1]
            s1_end = actual_q_len[i]

            # 当前batch对应的S2最大能选出的块数
            max_sel_count = min(
                math.ceil(cur_s2 / selected_block_size), selected_block_count
            )
            batch_size = s1_end - s1_start
            cols = topk_indices.shape[1]
            if cur_s2 % selected_block_size != 0:
                # 保证非对齐partial尾块在有效部分最后
                random_perms = torch.stack(
                    [
                        torch.randperm(max_sel_count - 1, dtype=torch.int32)
                        for _ in range(batch_size * cols)
                    ]
                ).reshape(batch_size, cols, max_sel_count - 1)
                topk_indices[s1_start:s1_end, :, : (max_sel_count - 1)] = random_perms
                topk_indices[s1_start:s1_end, :, max_sel_count - 1] = max_sel_count - 1
            else:
                random_perms = torch.stack(
                    [
                        torch.randperm(max_sel_count, dtype=torch.int32)
                        for _ in range(batch_size * cols)
                    ]
                ).reshape(batch_size, cols, max_sel_count)
                topk_indices[s1_start:s1_end, :, :max_sel_count] = random_perms
    return topk_indices


@register("execute_aclnnSparseFlashAttentionGrad")
class AclnnSparseFlashAttentionGrad(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        golden_path = f"./golden_path/{self.task_result.case_config.id}/"
        if self.device == "cpu":
            query = input_data.kwargs["query"]
            key = input_data.kwargs["key"]
            value = key

            dOut = input_data.kwargs["dOut"]
            out = input_data.kwargs["out"]
            queryRope = input_data.kwargs["queryRope"]
            keyRope = input_data.kwargs["keyRope"]
            scaleValue = input_data.kwargs["scaleValue"]
            sparseBlockSize = input_data.kwargs["sparseBlockSize"]
            layout = input_data.kwargs["layout"]
            actualSeqLengthsQ = input_data.kwargs["actualSeqLengthsQ"]
            actualSeqLengthskv = input_data.kwargs["actualSeqLengthskv"]
            sparseMode = input_data.kwargs["sparseMode"]
            topk_indices = input_data.kwargs["sparseIndices"]
            softmaxMax = input_data.kwargs["softmaxMax"]
            softmaxSum = input_data.kwargs["softmaxSum"]
            B = actualSeqLengthsQ.shape[0]
            if layout == "TND":
                T1 = query.shape[0]
                T2 = key.shape[0]
                N1 = query.shape[1]
                N2 = key.shape[1]
                D = query.shape[2]
                if B > 1:
                    q_list, kv_list = split_number(T1, B, T2, B)
                else:
                    q_list = [T1]
                    kv_list = [T2]
            else:
                S1 = query.shape[1]
                S2 = key.shape[1]
                T1 = S1 * B
                T2 = S2 * B
                N1 = query.shape[2]
                N2 = key.shape[2]
                D = query.shape[3]
                q_list = [S1 for _ in range(B)]
                kv_list = [S2 for _ in range(B)]
            atten_enable = True if sparseMode != 0 else False
            selected_block_count = topk_indices.shape[-1]
            pttype = query.dtype
            G = N1 // N2
            if not os.path.exists(golden_path):
                actualSeqLengthsQ = [sum(q_list[: x + 1]) for x in range(B)]
                actualSeqLengthskv = [sum(kv_list[: x + 1]) for x in range(B)]

                topk_indices = get_topk(
                    atten_enable,
                    B,
                    T1,
                    N2,
                    sparseBlockSize,
                    selected_block_count,
                    actualSeqLengthskv,
                    actualSeqLengthsQ,
                )
            else:
                softmaxMax = torch.load(golden_path + "softmaxMax_cpu.pt")
                softmaxSum = torch.load(golden_path + "softmaxSum_cpu.pt")
                actualSeqLengthsQ = torch.load(golden_path + "actualSeqLengthsQ.pt")
                actualSeqLengthskv = torch.load(golden_path + "actualSeqLengthskv.pt")
                topk_indices = torch.load(golden_path + "topk_indices.pt")
            spag = NsaSelectedAttention(
                T1,
                T2,
                N1,
                N2,
                D,
                B,
                G,
                query,
                key,
                value,
                out,
                dOut,
                topk_indices,
                None,
                pttype,
                actualSeqLengthsQ,
                actualSeqLengthskv,
                scaleValue,
                selected_block_count,
                sparseBlockSize,
                atten_enable,
                softmaxMax,
                softmaxSum,
                queryRope,
                keyRope,
            )
            if not os.path.exists(golden_path):
                out, softmaxMax, softmaxSum = spag.forward()
                actualSeqLengthsQ = torch.tensor(actualSeqLengthsQ)
                actualSeqLengthskv = torch.tensor(actualSeqLengthskv)
                os.system(f"mkdir -p {golden_path}")
                torch.save(softmaxMax.detach(), golden_path + "softmaxMax_cpu.pt")
                torch.save(softmaxSum.detach(), golden_path + "softmaxSum_cpu.pt")
                torch.save(
                    actualSeqLengthsQ.contiguous().detach(),
                    golden_path + "actualSeqLengthsQ.pt",
                )
                torch.save(
                    actualSeqLengthskv.contiguous().detach(),
                    golden_path + "actualSeqLengthskv.pt",
                )
                torch.save(
                    topk_indices.contiguous().detach(), golden_path + "topk_indices.pt"
                )
                x_max = softmaxMax[..., 0].reshape(T1, N2, G)
                x_sum = softmaxSum[..., 0].reshape(T1, N2, G)
                if layout == "TND":
                    x_max = x_max.permute(1, 0, 2).to(torch.float)
                    x_sum = x_sum.permute(1, 0, 2).to(torch.float)
                    sparseIndices = topk_indices.reshape(T1, N2, selected_block_count)
                else:
                    x_max = (
                        x_max.reshape(B, S1, N2, G).permute(0, 2, 1, 3).to(torch.float)
                    )
                    x_sum = (
                        x_sum.reshape(B, S1, N2, G).permute(0, 2, 1, 3).to(torch.float)
                    )
                    sparseIndices = topk_indices.reshape(
                        B, S1, N2, selected_block_count
                    )
                torch.save(x_max.contiguous().detach(), golden_path + "softmaxMax.pt")
                torch.save(x_sum.contiguous().detach(), golden_path + "softmaxSum.pt")
                torch.save(
                    sparseIndices.contiguous().detach(),
                    golden_path + "sparseIndices.pt",
                )
            dq_out, dk_out, dv_out, dq_rope, dk_rope = spag.backward()
            return dq_out, dk_out, dv_out, dq_rope, dk_rope

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnSparseFlashAttentionGradGetWorkspaceSize(const aclTensor *query, const aclTensor *key,const aclTensor *value,const aclTensor *sparseIndices,\
                const aclTensor *dOut,const aclTensor  *out,const aclTensor  *softmaxMax,const aclTensor     *softmaxSum,const aclTensor *actualSeqLengthsQueryOptional,\
                const aclTensor  *actualSeqLengthskvOptional,const aclTensor  *queryRopeOptional,const aclTensor *keyRopeOptional,double scaleValue,int64_t sparseBlockSize,\
                char  *layoutOptional,int64_t sparseMode,int64_t preTokens,int64_t nextTokens,bool deterministic,const aclTensor *dQueryOut,const aclTensor *dKeyOut,\
                const aclTensor *dValueOut,const aclTensor *dQueryRopeOutOptional,const aclTensor *dKeyRopeOutOptional,uint64_t *workspaceSize,aclOpExecutor **executor)"


@register("pyaclnn_execute_aclnnSparseFlashAttentionGrad")
class PyaclnnSparseFlashAttentionGradApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        golden_path = f"./golden_path/{self.task_result.case_config.id}/"
        torch.npu.synchronize()
        softmaxMax = torch.load(golden_path + "softmaxMax.pt")
        softmaxSum = torch.load(golden_path + "softmaxSum.pt")
        actualSeqLengthsQ = torch.load(golden_path + "actualSeqLengthsQ.pt").to(
            torch.int32
        )
        actualSeqLengthskv = torch.load(golden_path + "actualSeqLengthskv.pt").to(
            torch.int32
        )
        topk_indices = torch.load(golden_path + "sparseIndices.pt")

        input_data.kwargs["value"] = input_data.kwargs["key"]
        input_data.kwargs["softmaxMax"] = softmaxMax.npu()
        input_data.kwargs["softmaxSum"] = softmaxSum.npu()
        input_data.kwargs["actualSeqLengthsQ"] = actualSeqLengthsQ.npu()
        input_data.kwargs["actualSeqLengthskv"] = actualSeqLengthskv.npu()
        input_data.kwargs["sparseIndices"] = topk_indices.npu()
        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages


@register("execute_aclnnSparseFlashAttentionGrad_Abnormal")
class AclnnSparseFlashAttentionGrad_Abnormal(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        query = input_data.kwargs["query"]
        key = input_data.kwargs["key"]
        value = key
        queryRope = input_data.kwargs["queryRope"]
        keyRope = input_data.kwargs["keyRope"]
        return query, key, value, queryRope, keyRope


@register("pyaclnn_execute_aclnnSparseFlashAttentionGrad_Abnormal")
class PyaclnnSparseFlashAttentionGrad_AbnormalApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        if "is_nd" in input_data.kwargs:
            is_nd = input_data.kwargs["is_nd"]
            if is_nd is False:
                input_args.pop(-6)  # 去除入参列表中的最后一个值
        return input_args, output_packages

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


@register("execute_aclnnSparseFlashAttentionGrad_spec")
class AclnnSparseFlashAttentionGrad(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        golden_path = f"./golden_path_spec/{self.task_result.case_config.id}/"
        if self.device == "cpu":
            query = input_data.kwargs["query"]
            key = input_data.kwargs["key"]
            value = key

            dOut = input_data.kwargs["dOut"]
            out = input_data.kwargs["out"]
            queryRope = input_data.kwargs["queryRope"]
            keyRope = input_data.kwargs["keyRope"]
            scaleValue = input_data.kwargs["scaleValue"]
            sparseBlockSize = input_data.kwargs["sparseBlockSize"]
            layout = input_data.kwargs["layout"]
            actualSeqLengthsQ = input_data.kwargs["actualSeqLengthsQ"]
            actualSeqLengthskv = input_data.kwargs["actualSeqLengthskv"]
            sparseMode = input_data.kwargs["sparseMode"]
            topk_indices = input_data.kwargs["sparseIndices"]
            softmaxMax = input_data.kwargs["softmaxMax"]
            softmaxSum = input_data.kwargs["softmaxSum"]
            B = actualSeqLengthsQ.shape[0]
            if layout == "TND":
                T1 = query.shape[0]
                T2 = key.shape[0]
                N1 = query.shape[1]
                N2 = key.shape[1]
                D = query.shape[2]
                if B > 1:
                    q_list, kv_list = split_number(T1, B, T2, B)
                else:
                    q_list = [T1]
                    kv_list = [T2]
            else:
                S1 = query.shape[1]
                S2 = key.shape[1]
                T1 = S1 * B
                T2 = S2 * B
                N1 = query.shape[2]
                N2 = key.shape[2]
                D = query.shape[3]
                q_list = [S1 for _ in range(B)]
                kv_list = [S2 for _ in range(B)]
            atten_enable = True if sparseMode != 0 else False
            selected_block_count = topk_indices.shape[-1]
            pttype = query.dtype
            G = N1 // N2
            if not os.path.exists(golden_path):
                actualSeqLengthsQ = [sum(q_list[: x + 1]) for x in range(B)]
                actualSeqLengthskv = [sum(kv_list[: x + 1]) for x in range(B)]

                topk_indices = get_topk(
                    atten_enable,
                    B,
                    T1,
                    N2,
                    sparseBlockSize,
                    selected_block_count,
                    actualSeqLengthskv,
                    actualSeqLengthsQ,
                )
            else:
                softmaxMax = torch.load(golden_path + "softmaxMax_cpu.pt")
                softmaxSum = torch.load(golden_path + "softmaxSum_cpu.pt")
                actualSeqLengthsQ = torch.load(golden_path + "actualSeqLengthsQ.pt")
                actualSeqLengthskv = torch.load(golden_path + "actualSeqLengthskv.pt")
                topk_indices = torch.load(golden_path + "topk_indices.pt")
            spag = NsaSelectedAttention(
                T1,
                T2,
                N1,
                N2,
                D,
                B,
                G,
                query,
                key,
                value,
                out,
                dOut,
                topk_indices,
                None,
                pttype,
                actualSeqLengthsQ,
                actualSeqLengthskv,
                scaleValue,
                selected_block_count,
                sparseBlockSize,
                atten_enable,
                softmaxMax,
                softmaxSum,
                queryRope,
                keyRope,
            )
            if not os.path.exists(golden_path):
                out, softmaxMax, softmaxSum = spag.forward()
                actualSeqLengthsQ = torch.tensor(actualSeqLengthsQ)
                actualSeqLengthskv = torch.tensor(actualSeqLengthskv)
                os.system(f"mkdir -p {golden_path}")
                torch.save(softmaxMax.detach(), golden_path + "softmaxMax_cpu.pt")
                torch.save(softmaxSum.detach(), golden_path + "softmaxSum_cpu.pt")
                torch.save(
                    actualSeqLengthsQ.contiguous().detach(),
                    golden_path + "actualSeqLengthsQ.pt",
                )
                torch.save(
                    actualSeqLengthskv.contiguous().detach(),
                    golden_path + "actualSeqLengthskv.pt",
                )
                torch.save(
                    topk_indices.contiguous().detach(), golden_path + "topk_indices.pt"
                )
                x_max = softmaxMax[..., 0].reshape(T1, N2, G)
                x_sum = softmaxSum[..., 0].reshape(T1, N2, G)
                if layout == "TND":
                    x_max = x_max.permute(1, 0, 2).to(torch.float)
                    x_sum = x_sum.permute(1, 0, 2).to(torch.float)
                    sparseIndices = topk_indices.reshape(T1, N2, selected_block_count)
                else:
                    x_max = (
                        x_max.reshape(B, S1, N2, G).permute(0, 2, 1, 3).to(torch.float)
                    )
                    x_sum = (
                        x_sum.reshape(B, S1, N2, G).permute(0, 2, 1, 3).to(torch.float)
                    )
                    sparseIndices = topk_indices.reshape(
                        B, S1, N2, selected_block_count
                    )
                torch.save(x_max.contiguous().detach(), golden_path + "softmaxMax.pt")
                torch.save(x_sum.contiguous().detach(), golden_path + "softmaxSum.pt")
                torch.save(
                    sparseIndices.contiguous().detach(),
                    golden_path + "sparseIndices.pt",
                )
            dq_out, dk_out, dv_out, dq_rope, dk_rope = spag.backward()
            return dq_out, dk_out, dv_out, dq_rope, dk_rope

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnSparseFlashAttentionGradGetWorkspaceSize(const aclTensor *query, const aclTensor *key,const aclTensor *value,const aclTensor *sparseIndices,\
                const aclTensor *dOut,const aclTensor  *out,const aclTensor  *softmaxMax,const aclTensor     *softmaxSum,const aclTensor *actualSeqLengthsQueryOptional,\
                const aclTensor  *actualSeqLengthskvOptional,const aclTensor  *queryRopeOptional,const aclTensor *keyRopeOptional,double scaleValue,int64_t sparseBlockSize,\
                char  *layoutOptional,int64_t sparseMode,int64_t preTokens,int64_t nextTokens,bool deterministic,const aclTensor *dQueryOut,const aclTensor *dKeyOut,\
                const aclTensor *dValueOut,const aclTensor *dQueryRopeOutOptional,const aclTensor *dKeyRopeOutOptional,uint64_t *workspaceSize,aclOpExecutor **executor)"


@register("pyaclnn_execute_aclnnSparseFlashAttentionGrad_spec")
class PyaclnnSparseFlashAttentionGradApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        golden_path = f"./golden_path_spec/{self.task_result.case_config.id}/"
        torch.npu.synchronize()
        softmaxMax = torch.load(golden_path + "softmaxMax.pt")
        softmaxSum = torch.load(golden_path + "softmaxSum.pt")
        actualSeqLengthsQ = torch.load(golden_path + "actualSeqLengthsQ.pt").to(
            torch.int32
        )
        actualSeqLengthskv = torch.load(golden_path + "actualSeqLengthskv.pt").to(
            torch.int32
        )
        topk_indices = torch.load(golden_path + "sparseIndices.pt")

        input_data.kwargs["value"] = input_data.kwargs["key"]
        input_data.kwargs["softmaxMax"] = softmaxMax.npu()
        input_data.kwargs["softmaxSum"] = softmaxSum.npu()
        input_data.kwargs["actualSeqLengthsQ"] = actualSeqLengthsQ.npu()
        input_data.kwargs["actualSeqLengthskv"] = actualSeqLengthskv.npu()
        input_data.kwargs["sparseIndices"] = topk_indices.npu()
        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages
