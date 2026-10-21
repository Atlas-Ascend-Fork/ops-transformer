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

"""SparseLightningIndexer 单用例参数集（candidate consumer / candidate_mode==2）。

仅在 ascend910b / ascend910_93（arch22，A2）上启用；Ascend950 上算子未注册，
本参数集不生成用例。

参数说明（27 个 LIV2 基础字段 + 3 个 consumer 扩展字段）：
  基础字段与 lightning_indexer_v2 共用同一 golden 入参（BASE_FIELDS）；
  A2 约束：seqused_q / output_idx_offset 不支持（恒 None），max_seqlen_q 恒 -1。
  return_value：0（或缺省，sparse_values 恒空 (0,)）/ 1（sparse_values 输出真实 TopK 分数，
  shape 与 sparse_indices 相同、dtype FLOAT32）；其它值 host 拒绝。
  consumer 扩展字段：
    cand_spec:      候选来源 self（同 score 过 source 参考行为）/ dirty（重复+越界+-1
                    脏数据免疫）/ missing（故意缺块，nvalid<topk 触发 -1 补坑）/ full（全候选）
    cand_blocks:    candidate_block_indices 末维宽度 K ∈ (0,2048] 且 64 的倍数
    cand_block_size: candidate_block_size ∈ [2,64] 的 2 的幂
"""

import torch


####### 公共基座（A2 合法骨架） ########
# BSND×BSND：seqused_q/output_idx_offset 恒 None（arch22 host 拒绝），max_seqlen_q 恒 -1
BASE = {
    "batch_size": [1],
    "q_seq": [8],
    "k_seq": [256],
    "q_t_size": [0],
    "k_t_size": [0],
    "q_head_num": [4],
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
    # ---- consumer 扩展字段
    "cand_spec": ["self"],
    "cand_blocks": [64],
    "cand_block_size": [8],
}


####### 覆盖范围 ########
# 老功能（consumer 打分框架）：dtype × layout 四组合、变长（seqused_k/cu_seqlens）、
#   mask/cmp_ratio、PA block、topk 边界。
# 新功能（candidate consumer 语义）：候选来源 self/dirty/missing/full、candidate_block_size
#   全 6 档、candBlocks 边界（64/2048）、topk 边界 2048、regime 2 无 leak（-1 补坑）、
#   numBlocks<candBlocks 满覆盖退化。
# 新功能（return_value 开放）：RV1 系列 8 组专项（rv=1 values 真实分输出，继承各自基座）。
TestCases = {
    # ------ dtype（老功能）
    "FP16_01": {**BASE},
    "BF16_02": {**BASE, "qk_dtype": [torch.bfloat16]},
    # ------ layout 四组合（老功能；layoutK 非 PA 时两侧一致，PA 时 q 可 BSND/TND）
    "TND_TND_03": {
        **BASE,
        "batch_size": [2],
        "q_seq": [4],
        "k_seq": [128],
        "q_t_size": [6],
        "k_t_size": [192],
        "cu_seqlens_q": [[0, 2, 6]],
        "cu_seqlens_k": [[0, 64, 192]],
        "layout_q": ["TND"],
        "layout_k": ["TND"],
    },
    "PA_04": {
        **BASE,
        "k_seq": [256],
        "block_size": [128],
        "block_num": [2],
        "seqused_k": [[256]],
        "layout_k": ["PA_BBND"],
    },
    "TND_PA_05": {
        **BASE,
        "batch_size": [2],
        "q_seq": [4],
        "q_t_size": [6],
        "k_seq": [256],
        "block_size": [128],
        "block_num": [4],
        "cu_seqlens_q": [[0, 2, 6]],
        "seqused_k": [[256, 256]],
        "layout_q": ["TND"],
        "layout_k": ["PA_BBND"],
    },
    # ------ mask / cmp_ratio（老功能；mask3 + cmp_ratio>1 时 cmp_residual_k 必传）
    "MASK3_06": {**BASE, "mask_mode": [3]},
    "MASK3_CMP2_07": {
        **BASE,
        "k_seq": [257],
        "seqused_k": [[257]],
        "cmp_residual_k": [[1]],
        "mask_mode": [3],
        "cmp_ratio": [2],
    },
    "MASK3_CMP128_08": {
        **BASE,
        "k_seq": [260],
        "seqused_k": [[260]],
        "cmp_residual_k": [[127]],
        "mask_mode": [3],
        "cmp_ratio": [128],
    },
    # ------ 变长（老功能）：seqused_k 覆盖 k shape、B>1 不均匀 batch
    "VARLEN_09": {
        **BASE,
        "batch_size": [2],
        "k_seq": [256],
        "seqused_k": [[256, 129]],
    },
    # ------ topk 边界（老功能 + C5）
    "TOPK_MAX_10": {**BASE, "k_seq": [2048], "topk": [2048], "cand_blocks": [2048]},
    "TOPK_GT_VALID_11": {
        # topk > 可达位置数：-1 槽尾部补齐（不可达位恒 -1）
        **BASE,
        "k_seq": [65],
        "topk": [128],
    },
    # ------ candidate_block_size 全 6 档（新功能：C4 粒度扫描）
    "CAND_BS_2_12": {**BASE, "k_seq": [512], "cand_block_size": [2]},
    "CAND_BS_4_13": {**BASE, "k_seq": [512], "cand_block_size": [4]},
    "CAND_BS_16_14": {**BASE, "k_seq": [512], "cand_block_size": [16]},
    "CAND_BS_32_15": {**BASE, "k_seq": [512], "cand_block_size": [32]},
    "CAND_BS_64_16": {**BASE, "k_seq": [512], "cand_block_size": [64]},
    # ------ candBlocks 边界（新功能：C3）
    "CAND_BLOCKS_MAX_17": {
        # candBlocks=2048 上界；numBlocks=256/8=32 < 2048，满覆盖退化（-1 槽为主）
        **BASE,
        "k_seq": [2048],
        "cand_blocks": [2048],
        "topk": [128],
    },
    "CAND_BLOCKS_GT_NUMBLOCKS_18": {
        # numBlocks(32) < candBlocks(128)：块全选 + 尾部 -1 槽
        **BASE,
        "k_seq": [256],
        "cand_blocks": [128],
    },
    # ------ 候选来源（新功能：候选契约四形态）
    "CAND_DIRTY_19": {
        # 重复块号 + 越界块号 + -1 槽（P6f 鲁棒性：脏数据按"非候选"处理，结果不变）
        **BASE,
        "k_seq": [512],
        "cand_spec": ["dirty"],
    },
    "CAND_MISSING_20": {
        # 故意缺块（覆盖不全）：nvalid < topk，无效槽（含候选外可达位）全 -1 补齐
        **BASE,
        "k_seq": [512],
        "topk": [64],
        "cand_spec": ["missing"],
    },
    "CAND_FULL_21": {
        # 全候选：等价于无降级（与 source 行为一致性检查）
        **BASE,
        "k_seq": [512],
        "cand_spec": ["full"],
    },
    # ------ regime 2 无 leak（-1 补坑，新功能核心语义）
    "REGIME2_SELF_22": {
        # self 候选仅覆盖 candBlocks×bs=64×2=128 位 < topk=192 < 可达 512：
        # 剩余槽位全 -1，不输出候选外索引（无 leak 核心语义）
        **BASE,
        "k_seq": [512],
        "topk": [192],
        "cand_blocks": [64],
        "cand_block_size": [2],
    },
    "REGIME2_MISSING_23": {
        # missing 保留 ~50% 块（16 块=128 位）< topk=160 < 可达 256：
        # 剩余槽位全 -1，不输出候选外索引（无 leak 核心语义）
        **BASE,
        "k_seq": [256],
        "topk": [160],
        "cand_spec": ["missing"],
    },
    # ------ mask3 + 候选交互（新功能：causal 可达前缀上的候选降级）
    "CAND_MASK3_24": {
        **BASE,
        "q_seq": [8],
        "k_seq": [512],
        "cmp_residual_k": [[1]],
        "mask_mode": [3],
        "cmp_ratio": [2],
        "cand_spec": ["dirty"],
    },
    # ------ PA 块边界（老功能：block_size 16/1024 上下界）
    "PA_BS16_25": {
        **BASE,
        "k_seq": [260],
        "block_size": [16],
        "block_num": [17],
        "seqused_k": [[260]],
        "layout_k": ["PA_BBND"],
    },
    "PA_BS1024_26": {
        **BASE,
        "k_seq": [1024],
        "block_size": [1024],
        "block_num": [1],
        "seqused_k": [[1024]],
        "layout_k": ["PA_BBND"],
    },
    # ------ G 头组（老功能：N1 泛化，G=N1/N2<=64）
    "G_32_27": {**BASE, "q_head_num": [32], "k_seq": [512]},
    "G_64_28": {**BASE, "q_head_num": [64], "k_seq": [512]},
    # ------ TND 变长 + 候选（新功能：TND 输出布局 [T,N2,topk] 上的候选消费）
    "TND_VARLEN_CAND_29": {
        # cu_seqlens_k 差分 [129, 63]：不均匀 batch + 129 非对齐（% bs=8 != 0）；
        # TND 标称 k_seq 需 >= 最大差分
        **BASE,
        "batch_size": [2],
        "q_seq": [4],
        "k_seq": [192],
        "q_t_size": [6],
        "k_t_size": [192],
        "cu_seqlens_q": [[0, 2, 6]],
        "cu_seqlens_k": [[0, 129, 192]],
        "cmp_residual_k": [[1, 1]],
        "mask_mode": [3],
        "cmp_ratio": [2],
        "layout_q": ["TND"],
        "layout_k": ["TND"],
        "cand_spec": ["missing"],
    },
    # ------ return_value=1 专项（新功能：values 真实分输出，全部继承各自基座） ------
    "RV1_FP16_30": {
        # rv=1 主口径：候选充足（self 全覆盖），values=入选位真实分（fp32）
        **BASE,
        "return_value": [1],
    },
    "RV1_BF16_31": {
        # bf16 输入 × rv=1（dtype 矩阵第二行；values dtype 恒 FLOAT32）
        **BASE,
        "qk_dtype": [torch.bfloat16],
        "return_value": [1],
    },
    "RV1_TND_32": {
        # TND×TND × rv=1：输出 (T,1,topk)，values 同 shape
        **BASE,
        "batch_size": [2],
        "q_seq": [4],
        "k_seq": [128],
        "q_t_size": [6],
        "k_t_size": [192],
        "cu_seqlens_q": [[0, 2, 6]],
        "cu_seqlens_k": [[0, 64, 192]],
        "layout_q": ["TND"],
        "layout_k": ["TND"],
        "return_value": [1],
    },
    "RV1_PA_33": {
        # BSND×PA_BBND × rv=1：block_table 寻址下 values 通路
        **BASE,
        "k_seq": [256],
        "block_size": [128],
        "block_num": [2],
        "seqused_k": [[256]],
        "layout_k": ["PA_BBND"],
        "return_value": [1],
    },
    "RV1_DIRTY_34": {
        # 脏候选（重复+越界正块号+-1 三形态混合）× rv=1：脏块号按非候选免疫，
        # values 仅真实入选位有值
        **BASE,
        "k_seq": [512],
        "cand_spec": ["dirty"],
        "return_value": [1],
    },
    "RV1_MISSING_35": {
        # 缺块 × rv=1：nvalid < topk，indices 尾部 -1；values 有效槽真实分、-1 槽 0
        **BASE,
        "k_seq": [512],
        "topk": [64],
        "cand_spec": ["missing"],
        "return_value": [1],
    },
    "RV1_MASK3_36": {
        # mask_mode=3 + cmp_ratio=2 + rv=1：causal 短前缀行（行首 vl 小、-1 槽多），
        # vl 公式逐行生效
        **BASE,
        "k_seq": [257],
        "seqused_k": [[257]],
        "cmp_residual_k": [[1]],
        "mask_mode": [3],
        "cmp_ratio": [2],
        "return_value": [1],
    },
    "RV1_REGIME2_37": {
        # regime 2 × rv=1：nvalid < topk < 可达，剩余槽位全 -1 / value 0，
        # 不含任何候选外索引（无 leak 核心语义 × values 开放）
        **BASE,
        "k_seq": [512],
        "topk": [192],
        "cand_blocks": [64],
        "cand_block_size": [2],
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
    ENABLED_PARAMSETS = [
        (name, TestCases[name])
        for name in (
            # ------ dtype / layout（老功能）
            "FP16_01",
            "BF16_02",
            "TND_TND_03",
            "PA_04",
            "TND_PA_05",
            # ------ mask / cmp_ratio / 变长（老功能）
            "MASK3_06",
            "MASK3_CMP2_07",
            "MASK3_CMP128_08",
            "VARLEN_09",
            # ------ topk 边界（老功能 + C5）
            "TOPK_MAX_10",
            "TOPK_GT_VALID_11",
            # ------ candidate_block_size 全 6 档（新功能）
            "CAND_BS_2_12",
            "CAND_BS_4_13",
            "CAND_BS_16_14",
            "CAND_BS_32_15",
            "CAND_BS_64_16",
            # ------ candBlocks 边界（新功能）
            "CAND_BLOCKS_MAX_17",
            "CAND_BLOCKS_GT_NUMBLOCKS_18",
            # ------ 候选来源四形态（新功能）
            "CAND_DIRTY_19",
            "CAND_MISSING_20",
            "CAND_FULL_21",
            # ------ regime 2 无 leak（-1 补坑，新功能核心语义）
            "REGIME2_SELF_22",
            "REGIME2_MISSING_23",
            # ------ mask3 × 候选 / PA 块边界 / G（老功能）
            "CAND_MASK3_24",
            "PA_BS16_25",
            "PA_BS1024_26",
            "G_32_27",
            "G_64_28",
            # ------ TND 变长 + 候选（新功能）
            "TND_VARLEN_CAND_29",
            # ------ return_value=1 专项（新功能）
            "RV1_FP16_30",
            "RV1_BF16_31",
            "RV1_TND_32",
            "RV1_PA_33",
            "RV1_DIRTY_34",
            "RV1_MISSING_35",
            "RV1_MASK3_36",
            "RV1_REGIME2_37",
        )
    ]

ENABLED_PARAMS = [params for _, params in ENABLED_PARAMSETS]
