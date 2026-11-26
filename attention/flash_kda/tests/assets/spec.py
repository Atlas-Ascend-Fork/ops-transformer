#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Top-level TTK TestSpec registration for FlashKDA (arch35 & arch92).

Registers both:
1. ``torch.ops.cann_ops_transformer.flash_kda`` -> ``FlashKdaOpTestSpec``
   (direct public operator testing for redline, stc, generalized, profiling;
   registered by the installed wheel via ``cann_ops_transformer.ops`` entry
   points and TTK ``TorchOpsPackageLoader``)
2. ``spec.flash_kda`` -> ``FlashKdaAdapterTestSpec``
   (negative, recovery, Meta, stride, and whitebox contract testing; expose
   ``tests/assets`` on ``PYTHONPATH`` for TTK testcase validation)

``impl`` modules are loaded lazily without importing ``torch`` or ``torch_npu``
at spec discovery time so forked TTK workers initialize NPU cleanly.
"""

import importlib.util
from pathlib import Path
import sys

_IMPL_DIR = Path(__file__).with_name("impl")
_PACKAGE_NAME = "_flash_kda_assets_impl"


def _ensure_impl_package():
    if _PACKAGE_NAME in sys.modules:
        return sys.modules[_PACKAGE_NAME]
    spec = importlib.util.spec_from_file_location(
        _PACKAGE_NAME,
        _IMPL_DIR / "__init__.py",
        submodule_search_locations=[str(_IMPL_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _load_impl(stem):
    _ensure_impl_package()
    return importlib.import_module(f"{_PACKAGE_NAME}.{stem}")


def __getattr__(name):
    if name == "flash_kda":
        return _load_impl("kda_ttk_ops").flash_kda
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _remote_xpu_enabled():
    try:
        from ttk.remote import is_remote_configured
    except Exception:
        return False
    try:
        return bool(is_remote_configured())
    except Exception:
        return False


def _cli_compare_method():
    try:
        from ttk.switches import get_global_storage
    except Exception:
        return None
    try:
        method = getattr(get_global_storage(), "compare_method", None)
    except Exception:
        return None
    return str(method).lower() if method else None


def _use_remote_cross_check():
    """Enable native remote cross_check when xpu_server endpoints are configured."""
    if not _remote_xpu_enabled():
        return False
    return _cli_compare_method() in (None, "cross_check")


def _use_local_cross_check():
    """Fall back to local three-party cross_check L2 when no xpu_server is configured."""
    if _remote_xpu_enabled():
        return False
    return _cli_compare_method() in (None, "cross_check")


class _ToleranceHook:
    """Resolve native remote cross_check tolerance only when xpu_server is configured.

    Without remote endpoints (such as CI ``ops_st_test.sh``), omit
    ``standard="cross_check"`` so TTK does not dispatch to missing XPU servers
    while still enabling Promote golden and keeping ``level="L2"`` for
    ``--compare cross_check``.
    """

    def __init__(self, remote_tol, local_tol):
        self._remote_tol = remote_tol
        self._local_tol = local_tol

    def __get__(self, instance, owner):
        return self._remote_tol if _use_remote_cross_check() else self._local_tol


class _CompareHook:
    """Resolve local cross_check callback only when remote xpu_server is absent."""

    def __init__(self, func):
        self._func = func

    def __get__(self, instance, owner):
        return self._func if _use_local_cross_check() else None


class _LazyAttrHook:
    """Resolve an impl attribute lazily on class access without eager torch imports."""

    def __init__(self, stem, attr_name):
        self._stem = stem
        self._attr_name = attr_name

    def __get__(self, instance, owner):
        return getattr(_load_impl(self._stem), self._attr_name)


def third_party_flash_kda(*args, **kwargs):
    """Native TTK third-party peer for ``torch.ops.cann_ops_transformer.flash_kda``."""
    return _load_impl("golden").op_third_party(*args, **kwargs)


def third_party_flash_kda_adapter(*args, **kwargs):
    """Native TTK third-party peer for ``spec.flash_kda``."""
    return _load_impl("golden").adapter_third_party(*args, **kwargs)


def _local_compare_op(*outputs, compare_context=None, **kwargs):
    return _load_impl("cross_check").compare_op(
        *outputs, compare_context=compare_context, **kwargs
    )


def _local_compare_adapter(*outputs, compare_context=None, **kwargs):
    return _load_impl("cross_check").compare_adapter(
        *outputs, compare_context=compare_context, **kwargs
    )


class FlashKdaOpTestSpec:
    """TestSpec for direct ``torch.ops.cann_ops_transformer.flash_kda`` calls.

    When TTK remote endpoints are configured (``--config`` / ``ttk.conf.yaml``),
    uses TTK's native remote ``cross_check L2`` comparison with ``third_party``.
    When no remote ``xpu_server`` is configured (for example CI ``ops_st_test.sh``),
    falls back to local three-party ``CrossCheckComparison L2`` via ``compare``
    while keeping ``--compare <other>`` CLI overrides intact.
    """

    tolerance = _ToleranceHook(
        remote_tol={
            "bfloat16": {"standard": "cross_check", "level": "L2"},
            "float32": {"standard": "cross_check", "level": "L2"},
        },
        local_tol={
            "bfloat16": {"level": "L2"},
            "float32": {"level": "L2"},
        },
    )
    third_party = {"torch": third_party_flash_kda}
    compare = _CompareHook(_local_compare_op)
    torch_graph = _LazyAttrHook("graph", "FlashKdaAclGraph")

    @staticmethod
    def customize_inputs(*args, **kwargs):
        return _load_impl("inputs").customize_op_inputs(*args, **kwargs)

    @staticmethod
    def npu_preprocess(*args, **kwargs):
        return _load_impl("npu_preprocess").run(*args, **kwargs)

    @staticmethod
    def pre_compare(*outputs):
        # Exit inference_mode entered by npu_preprocess before the next testcase allocates inputs,
        # and zero out unwritten padding/unused arch35 chunk_states slots before TTK comparison.
        _load_impl("npu_preprocess").reset_mode()
        if not outputs:
            return None
        return _load_impl("output_contract").pre_compare_public(*outputs)

    @staticmethod
    def golden(*args, **kwargs):
        return _load_impl("golden").op_golden(*args, **kwargs)


class FlashKdaAdapterTestSpec:
    """TestSpec for ``spec.flash_kda`` contract and whitebox suites."""

    tolerance = _ToleranceHook(
        remote_tol={
            "bfloat16": {"standard": "cross_check", "level": "L2"},
            "float32": {"standard": "cross_check", "level": "L2"},
            "bool": {"standard": "binary_equal"},
        },
        local_tol={
            "bfloat16": {"level": "L2"},
            "float32": {"level": "L2"},
            "bool": {"standard": "binary_equal"},
        },
    )
    third_party = {"torch": third_party_flash_kda_adapter}
    compare = _CompareHook(_local_compare_adapter)

    @staticmethod
    def customize_inputs(*args, **kwargs):
        return _load_impl("inputs").customize_inputs(*args, **kwargs)

    @staticmethod
    def npu_preprocess(*args, **kwargs):
        return _load_impl("npu_preprocess").run_adapter(*args, **kwargs)

    @staticmethod
    def golden(*args, **kwargs):
        return _load_impl("golden").golden(*args, **kwargs)


__spec__ = {
    "torch.ops.cann_ops_transformer.flash_kda": "FlashKdaOpTestSpec",
    "spec.flash_kda": "FlashKdaAdapterTestSpec",
}
