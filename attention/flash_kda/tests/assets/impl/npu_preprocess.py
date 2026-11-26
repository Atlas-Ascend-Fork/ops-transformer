# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build dynamic FlashKDA metadata and manage inference/grad mode before operator calls."""

from contextlib import ExitStack
import logging
import os

import torch

from . import inputs as inputs_mod
from . import kda_ttk_ops
from .arch_config import architecture
from .performance import options as perf_options

METADATA_INDEX = 5
_MODE_STACK = None
_ACTIVE_CASE = None


def reset_mode():
    """Exit any inference/grad context entered by a prior npu_preprocess call."""
    global _MODE_STACK
    if _MODE_STACK is not None:
        stack = _MODE_STACK
        _MODE_STACK = None
        stack.close()


def enter_mode(c):
    """Enter the testcase's requested inference_mode and grad state."""
    global _MODE_STACK
    reset_mode()
    stack = ExitStack()
    stack.enter_context(torch.inference_mode(bool(c["inference"])))
    if c["inference"] or c.get("no_grad"):
        stack.enter_context(torch.no_grad())
    else:
        stack.enter_context(torch.enable_grad())
    if torch.is_inference_mode_enabled() != bool(c["inference"]):
        stack.close()
        raise AssertionError("FlashKDA inference_mode state mismatch")
    _MODE_STACK = stack


def set_active_case(arch, config):
    global _ACTIVE_CASE
    _ACTIVE_CASE = (arch, config)


def get_active_case():
    return _ACTIVE_CASE


def _metadata_slot_declared(tensor_dtypes):
    if tensor_dtypes is None:
        return True
    if (
        not isinstance(tensor_dtypes, (tuple, list))
        or len(tensor_dtypes) <= METADATA_INDEX
    ):
        raise ValueError("tensor_dtypes must declare the FlashKDA metadata slot")
    return tensor_dtypes[METADATA_INDEX] is not None


def build_hook_attributes(
    *,
    scale,
    output_final_state,
    return_intermediate_states,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    use_beta_sigmoid_in_kernel,
    allow_neg_eigval,
    disable_recompute,
    lower_bound,
    layout,
    cu_seqlens_cpu,
    extra_kwargs,
):
    attrs = dict(extra_kwargs)
    attrs.update(
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
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    return attrs


def run(
    q,
    k,
    v,
    g,
    beta,
    metadata=None,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    seqused=None,
    *,
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
    **kwargs,
):
    """Build dynamic NPU metadata (returning ``{"metadata": tensor}`` for ``(-1,)`` CSVs).

    When a pre-allocated ``metadata`` tensor is passed (legacy fixed-shape CSV),
    updates it in-place and returns ``None``.
    """
    reset_mode()
    arch = architecture(metadata=metadata)
    inputs = [
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
    ]
    attrs = build_hook_attributes(
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
        cu_seqlens_cpu=cu_seqlens_cpu,
        extra_kwargs=kwargs,
    )
    c = inputs_mod.configuration(inputs, attrs)
    set_active_case(arch, c)

    forced_template = None
    if c["suite"] == "performance" or "KDA_PERF_TEMPLATE" in os.environ:
        _, template = perf_options()
        forced_template = None if template == "auto" else template

    # Enter testcase mode before generating metadata; enable_grad is controlled by c["enable_grad"].
    enter_mode(c)
    if metadata is None:
        if not _metadata_slot_declared(kwargs.get("tensor_dtypes")):
            return None
        generated = kda_ttk_ops.produce(inputs, c, template=forced_template)
        if generated.dtype != torch.int32 or generated.device != q.device:
            generated = generated.to(dtype=torch.int32, device=q.device)
        logging.info(
            "[%s] prepared dynamic FlashKDA %s metadata: shape=%s inference=%s enable_grad=%s",
            c["id"],
            arch,
            tuple(generated.shape),
            c["inference"],
            c["enable_grad"],
        )
        return {"metadata": generated}

    generated = kda_ttk_ops.produce(inputs, c, template=forced_template)
    generated = generated.to(dtype=metadata.dtype, device=metadata.device)
    if metadata.is_inference() and not torch.is_inference_mode_enabled():
        with torch.inference_mode(True):
            metadata.resize_(generated.shape)
            metadata.copy_(generated)
    else:
        metadata.resize_(generated.shape)
        metadata.copy_(generated)

    logging.info(
        "[%s] prepared FlashKDA %s metadata: shape=%s inference=%s enable_grad=%s",
        c["id"],
        arch,
        tuple(metadata.shape),
        c["inference"],
        c["enable_grad"],
    )
    return None


def run_adapter(
    q,
    k,
    v,
    g,
    beta,
    metadata=None,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    seqused=None,
    **kwargs,
):
    """Materialize dynamic ``(-1,)`` metadata for ``spec.flash_kda`` suites.

    Unlike direct ``run()``, adapter suites manage ``torch.inference_mode`` locally
    inside ``kda_ttk_ops.flash_kda``, so this hook never leaves inference mode active.
    """
    if cu_seqlens_cpu is not None:
        kwargs = dict(kwargs, cu_seqlens_cpu=cu_seqlens_cpu)
    reset_mode()
    if metadata is not None:
        return None
    if not _metadata_slot_declared(kwargs.get("tensor_dtypes")):
        return None
    arch = architecture(metadata=metadata)
    inputs = [
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
    ]
    c = inputs_mod.configuration(inputs, kwargs)
    set_active_case(arch, c)
    if c["suite"] in ("meta", "meta_negative") or c["pending"]:
        return {
            "metadata": torch.empty(
                (inputs_mod.capacity(c, arch=arch),),
                dtype=torch.int32,
                device=q.device,
            )
        }
    with kda_ttk_ops.mode(c):
        generated = kda_ttk_ops.produce(inputs, c)
    if generated.dtype != torch.int32 or generated.device != q.device:
        generated = generated.to(dtype=torch.int32, device=q.device)
    logging.info(
        "[%s] prepared dynamic FlashKDA %s adapter metadata: shape=%s suite=%s inference=%s enable_grad=%s",
        c["id"],
        arch,
        tuple(generated.shape),
        c["suite"],
        c["inference"],
        c["enable_grad"],
    )
    return {"metadata": generated}
