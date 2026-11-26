# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Architecture configuration and TTK ``--plat`` resolution for FlashKDA (arch35/arch92).

Example (offline golden dump):
    ttk e2e ... --backend cpu --plat arch92 --no-prof --dump in,golden
Omitting ``--plat`` defaults to ``arch35``. No device or TTK dependency is
loaded at import time.
"""

import os
import sys

ARCHITECTURES = ("arch35", "arch92")
# (magic, abi_version, max_aic_cores)
ARCH_PROFILES = {
    "arch35": (0x4B444132, 4, 32),
    "arch92": (0x4B444136, 5, 64),
}
ABI_ARCH = {
    (0x4B444132, 4): "arch35",
    (0x4B444136, 5): "arch92",
}


def parse_plat_alias(platform):
    """Interpret TTK ``--plat`` values (arch35/arch92 or Ascend950/960).

    Returns ``None`` when ``--plat`` is omitted/AUTO/cpu so device auto-detection
    or default ``arch35`` can apply.
    """
    value = str(platform or "").strip().lower()
    if value in ("", "auto", "cpu"):
        return None
    if value in ARCHITECTURES:
        return value
    if value.startswith("ascend950"):
        return "arch35"
    if value.startswith("ascend960"):
        return "arch92"
    raise ValueError(
        f"FlashKDA: unsupported --plat {platform!r}; use --plat arch35 or --plat arch92"
    )


def get_ttk_switches():
    get_storage = None
    for module_name in (
        "ttk.switches",
        "ttk.utilities.container_utils",
        "ttk.utilities",
    ):
        mod = sys.modules.get(module_name)
        if mod is not None and hasattr(mod, "get_global_storage"):
            get_storage = mod.get_global_storage
            break
    if get_storage is None:
        try:
            from ttk.utilities import get_global_storage as get_storage
        except ImportError:
            return None
    return get_storage()


_get_ttk_switches = get_ttk_switches


def _detect_ttk_plat():
    """Read TTK's native ``--plat`` switch without hard-depending on TTK at import time."""
    switches = _get_ttk_switches()
    platform = getattr(switches, "dev_plat", None) if switches is not None else None
    return parse_plat_alias(platform)


def _is_cpu_backend():
    switches = _get_ttk_switches()
    if switches is None:
        return False
    return str(getattr(switches, "backend", "") or "").strip().lower() == "cpu"


def _detect_npu_arch():
    torch_mod = sys.modules.get("torch")
    npu = getattr(torch_mod, "npu", None) if torch_mod is not None else None
    if npu is None or not npu.is_available():
        return None
    name = npu.get_device_properties(npu.current_device()).name.upper()
    if name.startswith("ASCEND950"):
        return "arch35"
    if name.startswith("ASCEND960"):
        return "arch92"
    raise ValueError(f"FlashKDA does not support NPU device {name}")


def _detect_metadata_arch(metadata):
    if metadata is None or not hasattr(metadata, "numel") or metadata.numel() < 2:
        return None
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and getattr(metadata, "dtype", None) != torch_mod.int32:
        return None
    if getattr(metadata, "is_meta", False):
        return None
    try:
        key = tuple(int(v) for v in metadata[:2].detach().cpu().tolist())
    except Exception:
        return None
    return ABI_ARCH.get(key)


def architecture(metadata=None):
    """Select arch35 or arch92 from TTK --plat, FLASH_KDA_ARCH, NPU device, or metadata ABI.

    When neither ``--plat`` nor ``FLASH_KDA_ARCH`` nor an NPU device is active
    (e.g. ``ttk e2e ... --backend cpu --no-prof --dump in,golden``), defaults to
    ``arch35``.
    """
    plat_arch = _detect_ttk_plat()
    env_arch = os.environ.get("FLASH_KDA_ARCH")
    if env_arch is not None and env_arch not in ARCHITECTURES:
        raise ValueError("FLASH_KDA_ARCH must be arch35 or arch92")
    if plat_arch is not None and env_arch is not None and plat_arch != env_arch:
        raise ValueError(f"--plat {plat_arch} conflicts with FLASH_KDA_ARCH={env_arch}")
    requested = plat_arch or env_arch
    detected_npu = None if _is_cpu_backend() else _detect_npu_arch()
    if requested is not None and detected_npu is not None and requested != detected_npu:
        raise ValueError(
            f"Requested architecture {requested} does not match detected NPU ({detected_npu})"
        )
    if requested is not None:
        return requested
    if detected_npu is not None:
        return detected_npu
    detected_md = _detect_metadata_arch(metadata)
    if detected_md is not None:
        return detected_md
    return "arch35"


def arch_profile(arch=None, *, metadata=None):
    """Return ``(magic, abi_version, max_aic_cores)`` for the active architecture."""
    selected = arch or architecture(metadata=metadata)
    if selected not in ARCH_PROFILES:
        raise ValueError(f"Unsupported FlashKDA architecture: {selected!r}")
    return ARCH_PROFILES[selected]
