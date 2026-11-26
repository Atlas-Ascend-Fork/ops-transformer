# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import numpy as np
import torch

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr


def compute_select_attention(
    query,
    key,
    value,
    topK_index,
    atten_mask,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    scalar_value=1.0,
    block_size=64,
    block_count=16,
):
    """
    完整的注意力计算流程
    Args:
        query: 查询张量 [BS1, N1, queryKD]
        key: 键张量 [BS2, N2, queryKD]
        value: 值张量 [BS2, N2, VD]
        topK_index: 索引张量 [BS1, N2, block_count]
        atten_mask: 索引张量 [S1, S2]
    Returns:
        output: 最终输出结果 [BS1, N1, VD]
    """
    # 步骤1：处理QKV，以及top_k的维度，将维度保持一致
    ori_dtype = query.dtype
    query = query.float()
    key = key.float()
    value = value.float()
    BS1, N1, QKD = query.shape
    BS2, N2, VD = value.shape
    G = N1 // N2
    query = query.reshape(BS1, N2 * G, QKD)  # [BS1, N1, QKD]
    key = key  # [BS2, N2, QKD]
    value = value  # [BS2, N2, VD]
    topK_index = topK_index  # [BS1, N2, 16]
    N1 = N2 * G
    key_reshaped = key.view(BS2 // block_size, block_size, N2, QKD)
    value_reshaped = value.view(BS2 // block_size, block_size, N2, VD)
    query_reshaped = query.view(BS1, N2, G, QKD)

    # output
    output = torch.zeros(
        BS1, N2, G, VD, dtype=torch.float32, device=query_reshaped.device
    )
    softmax_max = torch.zeros(
        BS1, N2, G, 1, dtype=torch.float32, device=query_reshaped.device
    )
    softmax_sum = torch.zeros(
        BS1, N2, G, 1, dtype=torch.float32, device=query_reshaped.device
    )

    # print("=====================calculate======================")
    for bs1_index in range(BS1):
        b_index = get_currentB_index(bs1_index, actual_seq_qlen)
        s1_index = (
            bs1_index if b_index == 0 else bs1_index - actual_seq_qlen[b_index - 1]
        )
        start_BS2_index = 0 if b_index == 0 else actual_seq_kvlen[b_index - 1]
        start_block_index = start_BS2_index // block_size
        current_S2_size = actual_seq_kvlen[b_index] - start_BS2_index
        current_block_count = current_S2_size // block_size
        end_block_index = start_block_index + current_block_count
        if start_BS2_index % block_size != 0:
            raise RuntimeError(f"当前golden只支持S2 {block_size}对齐.")

        for n2_index in range(N2):
            topk_index_ = topK_index[bs1_index, n2_index, :]
            selectedK = torch.index_select(
                key_reshaped[start_block_index:end_block_index, :, n2_index, :],
                0,
                topk_index_,
            )  # (block_count, block_size, QKD)
            selectedK = selectedK.reshape(block_count * block_size, QKD)
            qk_ = torch.matmul(
                query_reshaped[bs1_index, n2_index, :, :], selectedK.transpose(-1, -2)
            )  # (G, block_count * block_size)
            qk_ = torch.mul(qk_, scalar_value)
            softmax_res, x_max, x_sum = softmax_torch(qk_)
            selectedV = torch.index_select(
                value_reshaped[start_block_index:end_block_index, :, n2_index, :],
                0,
                topk_index_,
            )  # (block_count, block_size, VD)
            selectedV = selectedV.reshape(block_count * block_size, VD)
            # out_ = torch.matmul(softmax_res.to(ori_dtype), selectedV.to(ori_dtype))  # (G, VD)
            out_ = torch.matmul(softmax_res, selectedV)

            output[bs1_index, n2_index] = out_
            softmax_max[bs1_index, n2_index] = x_max
            softmax_sum[bs1_index, n2_index] = x_sum

    output = output.reshape(BS1, N1, VD)
    softmax_max = softmax_max.reshape(BS1, N1, 1)
    softmax_sum = softmax_sum.reshape(BS1, N1, 1)

    softmax_max_8 = softmax_max.broadcast_to(BS1, N1, 8)
    softmax_sum_8 = softmax_sum.broadcast_to(BS1, N1, 8)

    return output.cpu().to(ori_dtype), softmax_max_8.cpu(), softmax_sum_8.cpu()


def softmax_torch(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = torch.softmax(x, dim=-1)
    return ans, x_max, x_sum


def get_currentB_index(BS_index, actual_seq_len):
    for i in range(actual_seq_len.size(0)):
        if BS_index < actual_seq_len[i]:
            return i
    raise RuntimeError("BS_index is greater than max(actual_seq_len).")


def build_topk(kwargs):
    query, key = kwargs["query"], kwargs["key"]
    q_ends = list(kwargs["actualSeqQLenOptional"])
    kv_ends = list(kwargs["actualSeqKvLenOptional"])
    block_size = kwargs["selectedBlockSize"]
    count = kwargs["selectedBlockCount"]
    if (
        not q_ends
        or len(q_ends) != len(kv_ends)
        or q_ends[-1] != query.shape[0]
        or kv_ends[-1] != key.shape[0]
    ):
        raise ValueError("Cumulative query/KV lengths must match the TND tensors")
    rng = np.random.RandomState(2023)
    rows = []
    q_start = kv_start = 0
    for q_end, kv_end in zip(q_ends, kv_ends):
        q_len, kv_len = q_end - q_start, kv_end - kv_start
        blocks, rem = divmod(kv_len, block_size)
        if q_len <= 0 or rem or not 0 < count < blocks:
            raise ValueError(
                "Each batch must have query tokens and more aligned KV blocks than TopK"
            )
        scores = rng.rand(q_len, key.shape[1], blocks)
        rows.append(
            torch.from_numpy(scores.argsort(axis=-1)[..., :count]).to(torch.int32)
        )
        q_start, kv_start = q_end, kv_end
    result = torch.cat(rows)
    if result.shape != kwargs["topkIndices"].shape:
        raise ValueError("Generated TopK does not match the declared tensor shape")
    return result


@register("function_aclnn_nsa_selected_attention")
class NsaSelectedAttentionReference(BaseApi):
    def __call__(self, input_data, with_output=False):
        kw = input_data.kwargs
        out, maximum, total = compute_select_attention(
            kw["query"].cpu(),
            kw["key"].cpu(),
            kw["value"].cpu(),
            build_topk(kw),
            None,
            torch.tensor(kw["actualSeqQLenOptional"]),
            torch.tensor(kw["actualSeqKvLenOptional"]),
            kw["scaleValue"],
            kw["selectedBlockSize"],
            kw["selectedBlockCount"],
        )
        return maximum, total, out


@register("pyaclnn_nsa_selected_attention")
class NsaSelectedAttentionAclnn(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        kwargs = input_data.kwargs
        kwargs["topkIndices"] = build_topk(kwargs).to(f"npu:{self.device_id}")
        kwargs["attenMaskOptional"] = None
        args, outputs = super().init_by_input_data(input_data)
        if len(args) != 16:
            raise RuntimeError(
                f"Expected 13 inputs and 3 outputs, got {len(args)} arguments"
            )
        args[4] = TensorPtr()
        return args, outputs
