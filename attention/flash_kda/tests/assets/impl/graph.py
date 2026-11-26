# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ACLGraph target module for FlashKDA.

Reuses the metadata tensor prepared by ``npu_preprocess`` before graph capture
so ``forward`` captures only ``torch.ops.cann_ops_transformer.flash_kda``.
"""

import torch


class FlashKdaAclGraph(torch.nn.Module):
    """ACLGraph module that reuses TTK-prepared tensors and metadata.

    ``flash_kda_metadata._capture_schedule`` verifies object identity on the
    ``metadata``, ``cu_seqlens``, and ``seqused`` tensors produced by
    ``npu_preprocess``. Keeping those scheduling tensors on ``self`` avoids
    ``npugraph_ex`` ``clone_input`` replacing them with schedule-less copies
    while keeping the data tensors as graph forward inputs.
    """

    def __init__(
        self,
        *,
        metadata=None,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        seqused=None,
        scale=128 ** (-0.5),
        output_final_state=False,
        return_intermediate_states=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        disable_recompute=False,
        lower_bound=None,
        layout="TND",
    ):
        super().__init__()
        if metadata is None or metadata.numel() == 0:
            raise ValueError(
                "FlashKDA graph requires metadata prepared by npu_preprocess"
            )
        self.metadata = metadata
        self.cu_seqlens = cu_seqlens
        self.cu_seqlens_cpu = cu_seqlens_cpu
        self.seqused = seqused
        self.attrs = dict(
            scale=scale,
            output_final_state=output_final_state,
            return_intermediate_states=return_intermediate_states,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            disable_recompute=disable_recompute,
            lower_bound=lower_bound,
            layout=layout,
        )

    def forward(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        initial_state=None,
        A_log=None,
        dt_bias=None,
    ):
        return torch.ops.cann_ops_transformer.flash_kda(
            q,
            k,
            v,
            g,
            beta,
            self.metadata,
            initial_state=initial_state,
            A_log=A_log,
            dt_bias=dt_bias,
            cu_seqlens=self.cu_seqlens,
            cu_seqlens_cpu=self.cu_seqlens_cpu,
            seqused=self.seqused,
            **self.attrs,
        )
