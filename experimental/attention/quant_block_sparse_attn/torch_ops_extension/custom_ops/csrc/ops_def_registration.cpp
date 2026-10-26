/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

// Register quant_block_sparse_attn in the custom namespace; add a definition before introducing each custom ATen IR.
// Step 1: add a definition for the new custom operator.
//   - Schema argument order: required tensors and scalars first; optional tensors and defaulted attributes follow '*'.
//   - Match tensor data types and attribute defaults in op_host/quant_block_sparse_attn_def.cpp.
TORCH_LIBRARY(custom, m)
{
    m.def("npu_quant_block_sparse_attn(Tensor query, Tensor key, Tensor value, "
          "Tensor q_descale, Tensor k_descale, Tensor v_descale, Tensor? p_scale, "
          "Tensor sparse_indices, Tensor sparse_seq_len, Tensor? atten_mask, "
          "float softmax_scale, int sparse_q_block_size, int sparse_kv_block_size, *, "
          "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
          "Tensor? seqused_q=None, Tensor? seqused_kv=None, "
          "Tensor? block_table=None, Tensor? metadata=None, "
          "str layout_kv='PA_BNBD', str layout_q='TND', str layout_sparse_indices='B_N_Qb_Kb', "
          "str layout_out='TND', "
          "int quant_mode=1, int mask_mode=3, bool return_softmax_lse=False) -> (Tensor, Tensor)");
    m.def("npu_quant_block_sparse_attn_metadata(Tensor sparse_seq_len, int num_heads_q, int num_heads_kv, "
          "int head_dim, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
          "Tensor? seqused_q=None, Tensor? seqused_kv=None, int batch_size=0, "
          "int sparse_block_size_q=128, int sparse_block_size_k=128, int quant_mode=1, int mask_mode=3, "
          "str layout_q='TND', str layout_kv='PA_BNBD', str layout_sparse_indices='B_N_Qb_Kb') -> Tensor");
}

// Bind the C++ and Python interfaces through pybind; this binds an interface, not an operator.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
