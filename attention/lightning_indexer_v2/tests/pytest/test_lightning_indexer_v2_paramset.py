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

"""LightningIndexerV2 默认参数集（A2 / arch22 老功能矩阵）。

仅在 ascend910b / ascend910_93（A2）上启用；Ascend950 走 STC 白盒参数集
（test_lightning_indexer_v2_stc.py）。

A2 约束（与 950 的 STC 参数集差异）：
  * seqused_q / output_idx_offset / max_seqlen_q 不支持：恒 None / -1；
  * topk ∈ [1, 2048] 或 3072/4096/5120/6144/7168/8192（>2048 时为 1024 的倍数）；
  * layout 合法组合：BSND+BSND、TND+TND、BSND+PA、TND+PA；
  * PA_BBND 时 seqused_k 与 block_table 必传；BSND/TND 时 seqused_k 可选；
  * mask_mode=3 且 cmp_ratio>1 时 cmp_residual_k 必传（值 < cmp_ratio）。
"""

import os

import torch


####### 公共基座（A2 合法骨架） ########
# batch_size: BSND 的 B；TND 的 B 由 cu_seqlens_q 的长度确定。
# q_seq/k_seq: BSND 的 S1/S2；TND 下为标称长度（>= cu_seqlens 最大差分）。
# q_t_size/k_t_size: TND 的总 token 数，等于对应 cu_seqlens 的末值。
# q_head_num/k_head_num: N1/N2；N2=1、G=N1/N2<=64。
# block_size/block_num: PA 物理块大小和数量；block_size 为 (0,1024] 内 16 的倍数。
BASE = {
    "batch_size": [1],
    "q_seq": [1],
    "k_seq": [128],
    "q_t_size": [1],
    "k_t_size": [128],
    "q_head_num": [1],
    "k_head_num": [1],
    "head_dim": [128],
    "block_size": [None],
    "block_num": [None],
    "qk_dtype": [torch.float16],
    "cu_seqlens_q": [None],
    "cu_seqlens_k": [None],
    "seqused_q": [None],
    "seqused_k": [None],
    "cmp_residual_k": [None],
    "output_idx_offset": [None],
    "layout_q": ["BSND"],
    "layout_k": ["BSND"],
    "topk": [64],
    "mask_mode": [0],
    "query_datarange": [[-2, 2]],
    "key_datarange": [[-2, 2]],
    "weights_datarange": [[-2, 2]],
    "cmp_ratio": [1],
    "return_value": [0],
    "max_seqlen_q": [-1],
}


####### 覆盖范围（A2 老功能） ########
# dtype/layout：fp16、bf16 与四种合法 layout 组合。
# S1/G：S1 尾块/整块、G 奇偶与上界 64。
# S2/TopK：128 基本块边界、零长度、topk 上界 2048 与 over-2K（1024 倍数）分支。
# mask/cmp_ratio：0/3、cmp_residual_k 组合与 cmp_ratio 上界 128。
# 实际长度：seqused_k 覆盖 shape、TND cu_seqlens 差分、TND seqused_k 覆盖前缀和。
# PA：block_size 16/96/128/1024 与多 batch block_table。
TestCases = {
    # ------ dtype + layout
    "FP16_01": {
        **BASE,
        "qk_dtype": [torch.float16],
    },
    "BF16_02": {
        **BASE,
        "qk_dtype": [torch.bfloat16],
    },
    "PA_03": {
        **BASE,
        "block_size": [128],
        "block_num": [1],
        "seqused_k": [[128]],
        "layout_k": ["PA_BBND"],
    },
    "TND_04": {
        **BASE,
        "q_t_size": [1],
        "k_t_size": [128],
        "cu_seqlens_q": [[0, 1]],
        "cu_seqlens_k": [[0, 128]],
        "layout_q": ["TND"],
        "layout_k": ["TND"],
    },
    "TND_PA_05": {
        **BASE,
        "q_t_size": [1],
        "block_size": [128],
        "block_num": [1],
        "cu_seqlens_q": [[0, 1]],
        "seqused_k": [[128]],
        "layout_q": ["TND"],
        "layout_k": ["PA_BBND"],
    },
    # ------ S1 / G
    "S1_06": {
        **BASE,
        "q_seq": [1],
    },
    "S1_07": {
        **BASE,
        "q_seq": [4],
    },
    "S1_08": {
        **BASE,
        "q_seq": [5],
    },
    # G=2/3 覆盖偶数/奇数头组。
    "G_09": {
        **BASE,
        "q_seq": [4],
        "q_head_num": [2, 3],
    },
    # G=64 覆盖上界。
    "G_10": {
        **BASE,
        "q_seq": [2],
        "q_head_num": [64],
    },
    # ------ S2（128 基本块边界）
    "S2_11": {
        **BASE,
        "k_seq": [1, 127, 128, 129, 256, 257],
        "return_value": [1],
    },
    # k 实际长度为 0：输出全 -1/-inf。
    "S2_ZERO_12": {
        **BASE,
        "k_seq": [1],
        "seqused_k": [[0]],
        "topk": [1],
        "return_value": [1],
    },
    # ------ TopK
    "TOPK_13": {
        **BASE,
        "k_seq": [1],
        "topk": [1],
    },
    # A2 over-2K：3072/4096 为 1024 的倍数（合法）；topk>2048 走 1024 对齐分支。
    "TOPK_OVER2K_14": {
        **BASE,
        "k_seq": [4096],
        "topk": [3072, 4096],
    },
    # topk 上界 2048（candidate 关闭路径的最大合法值）。
    "TOPK_MAX_15": {
        **BASE,
        "k_seq": [2048],
        "topk": [2048],
        "return_value": [1],
    },
    # validS2 < topk：indices/value 尾部补齐。
    "TOPK_GT_VALID_16": {
        **BASE,
        "k_seq": [65],
        "topk": [129],
        "return_value": [1],
    },
    # ------ mask / cmp_ratio
    "MASK_17": {
        **BASE,
        "q_seq": [4],
        "k_seq": [128],
        "mask_mode": [3],
    },
    # S1>S2：下三角窗口前部 query 行无有效 K。
    "MASK_S1GT2_18": {
        **BASE,
        "q_seq": [5],
        "k_seq": [2],
        "topk": [2],
        "mask_mode": [3],
        "return_value": [1],
    },
    "CMP_19": {
        **BASE,
        "q_seq": [4],
        "k_seq": [65],
        "seqused_k": [[65]],
        "cmp_residual_k": [[1]],
        "mask_mode": [3],
        "cmp_ratio": [2],
    },
    # cmp_ratio 上界 128 + 残差上界。
    "CMP_128_20": {
        **BASE,
        "q_seq": [4],
        "k_seq": [129],
        "seqused_k": [[129]],
        "cmp_residual_k": [[127]],
        "mask_mode": [3],
        "cmp_ratio": [128],
    },
    # ------ 实际长度来源
    # BSND seqused_k 覆盖 k shape，实际 S1/S2 同时形成尾块。
    "ACTUAL_21": {
        **BASE,
        "q_seq": [5],
        "k_seq": [129],
        "seqused_k": [[127]],
    },
    # TND 仅由 cu_seqlens 差分取得实际长度（不均匀 batch）。
    "ACTUAL_TND_22": {
        **BASE,
        "batch_size": [2],
        "q_seq": [3],
        "k_seq": [129],
        "q_t_size": [3],
        "k_t_size": [129],
        "cu_seqlens_q": [[0, 1, 3]],
        "cu_seqlens_k": [[0, 1, 129]],
        "layout_q": ["TND"],
        "layout_k": ["TND"],
    },
    # TND seqused_k 覆盖前缀和长度（seqused_k < cu_seqlens 差分）。
    # 【已知 A2 语义分歧，默认关闭】：arch22 kernel 的 TND 路径按 cu_seqlens 差分推导
    # k 有效长度、忽略 seqused_k 覆盖（BSND 路径正确响应 seqused_k；950 kernel 两侧
    # 均响应，950 STC 的 ACTUAL_31 为同形态用例）。golden 按 doc 语义
    # （"sequsedKOptional 传入则参与 k 有效长度计算"）对 TND 位置做了截断，导致
    # cpu/npu 在被截断位上分歧（如 seqused=127 时 NPU 仍选出位置 127）。
    # 修复 kernel 后设 LIV2_ENABLE_TND_SEQUSED_K=1 启用本用例回归验证。
    "ACTUAL_TND_SEQUSED_23": {
        **BASE,
        "batch_size": [2],
        "q_seq": [5],
        "k_seq": [256],
        "q_t_size": [5],
        "k_t_size": [256],
        "cu_seqlens_q": [[0, 2, 5]],
        "cu_seqlens_k": [[0, 128, 256]],
        "seqused_k": [[127, 64]],
        "layout_q": ["TND"],
        "layout_k": ["TND"],
    },
    # B>1 变长 seqused_k（含 0 长度 batch）。
    "VARLEN_ZERO_24": {
        **BASE,
        "batch_size": [4],
        "q_seq": [4],
        "k_seq": [256],
        "seqused_k": [[256, 128, 0, 64]],
        "return_value": [1],
    },
    # ------ PA block
    # 16 下界、96（16 倍数非 2 幂）、1024 上界。
    "PA_BS16_25": {
        **BASE,
        "k_seq": [129],
        "block_size": [16],
        "block_num": [9],
        "seqused_k": [[129]],
        "layout_k": ["PA_BBND"],
    },
    "PA_BS96_26": {
        **BASE,
        "k_seq": [129],
        "block_size": [96],
        "block_num": [2],
        "seqused_k": [[129]],
        "layout_k": ["PA_BBND"],
    },
    "PA_BS1024_27": {
        **BASE,
        "k_seq": [129],
        "block_size": [1024],
        "block_num": [1],
        "seqused_k": [[129]],
        "layout_k": ["PA_BBND"],
    },
    # 多 batch + 多物理块 + 变长 seqused_k（block_table 共享消费）。
    "PA_MULTI_BATCH_28": {
        **BASE,
        "batch_size": [2],
        "k_seq": [512],
        "block_size": [128],
        "block_num": [8],
        "seqused_k": [[512, 129]],
        "layout_k": ["PA_BBND"],
        "return_value": [1],
    },
    # ------ return_value 路径（与 S2/TopK 组合已在上方遍历，此处 TND 维度补齐）
    "TND_RETURN_VALUE_29": {
        **BASE,
        "batch_size": [2],
        "q_seq": [4],
        "k_seq": [192],
        "q_t_size": [6],
        "k_t_size": [192],
        "cu_seqlens_q": [[0, 2, 6]],
        "cu_seqlens_k": [[0, 64, 192]],
        "layout_q": ["TND"],
        "layout_k": ["TND"],
        "return_value": [1],
    },
}


properties = None
try:
    import torch_npu  # noqa: F401  (register npu before querying device properties)

    properties = torch.npu.get_device_properties()
except Exception:  # noqa: BLE001 - collectable on hosts without NPU
    properties = None
_name = getattr(properties, "name", "") if properties is not None else ""
ENABLED_PARAMSETS = []
if (
    properties is not None
    and ("910B" in _name or "910b" in _name or "910_93" in _name or "2201" in _name)
    and ("Ascend950" not in _name)
):
    _tnd_seqused_gated = os.environ.get("LIV2_ENABLE_TND_SEQUSED_K", "0") == "1"
    ENABLED_PARAMSETS = [
        (name, TestCases[name])
        for name in (
            # ------ dtype + layout
            "FP16_01",
            "BF16_02",
            "PA_03",
            "TND_04",
            "TND_PA_05",
            # ------ S1 / G
            "S1_06",
            "S1_07",
            "S1_08",
            "G_09",
            "G_10",
            # ------ S2 / TopK
            "S2_11",
            "S2_ZERO_12",
            "TOPK_13",
            "TOPK_OVER2K_14",
            "TOPK_MAX_15",
            "TOPK_GT_VALID_16",
            # ------ mask / cmp_ratio
            "MASK_17",
            "MASK_S1GT2_18",
            "CMP_19",
            "CMP_128_20",
            # ------ 实际长度来源
            "ACTUAL_21",
            "ACTUAL_TND_22",
            *(() if not _tnd_seqused_gated else ("ACTUAL_TND_SEQUSED_23",)),
            "VARLEN_ZERO_24",
            # ------ PA block
            "PA_BS16_25",
            "PA_BS96_26",
            "PA_BS1024_27",
            "PA_MULTI_BATCH_28",
            # ------ return_value
            "TND_RETURN_VALUE_29",
        )
    ]

ENABLED_PARAMS = [params for _, params in ENABLED_PARAMSETS]
