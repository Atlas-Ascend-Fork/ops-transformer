# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""flash_attn 参考实现：按行分块、头维批量、支持 fp32/fp64。

支持布局：Q ∈ {BNSD, BSND, TND}，KV ∈ {BNSD, BSND, TND, PA_BBND, PA_BNBD, PA_NZ}。

同一实现覆盖两种口径：
- ``p_dtype=None``：单趟 softmax 高精度真值；
- ``p_dtype="out_dtype"``：按 kernel ``s2BaseSize`` 分块在线 softmax，**P 落盘到 V 的
  dtype 后参与 P@V**，而分母/lse 仍用未落盘的 fp32 P——与算子精度口径一致（三方 peer）。
  该算子非量化，P 落盘只服务 P@V 的 MM 输入精度。

所有 (batch, head) 的计算都只在头维上批量、行维上分块，避免逐 tile 的 Python 循环；
行块大小由 ``chunk_bytes`` 控制，峰值内存与 S1*S2 解耦。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from .layout import PagedKVLoader, pa_to_bnsd, trans_bnsd_to_layout
from .policy import resolve_p_block

_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
GOLDEN_CHUNK_DEVICE = 1024 * 1024 * 1024
GOLDEN_CHUNK_HOST = 128 * 1024 * 1024
GOLDEN_SCRATCH_FACTOR = 3


def golden_chunk_bytes(device: str) -> int:
    return GOLDEN_CHUNK_DEVICE if device == "npu" else GOLDEN_CHUNK_HOST


def golden_peak(
    footprint: Dict[str, int],
    params: Optional[Dict[str, Any]] = None,
    *,
    device: str,
    chunk_bytes: int,
    p_dtype=None,
    p_block: int = 0,
    softmax_mode: str = "auto",
    softmax_block_size: int = 0,
    kv_loader: str = "dense",
    keep_on_device: bool = True,
    include_golden: bool = True,
) -> Dict[str, int]:
    """golden 阶段峰值；中间量公式与 compute 的分块实现保持同源。"""
    scratch = GOLDEN_SCRATCH_FACTOR * chunk_bytes
    if include_golden:
        kv_seg = footprint["kv_seg"]
        n2, d, dv = footprint["n2"], footprint["d"], footprint["dv"]
        layout_kv = str((params or {}).get("layout_kv", "BNSD"))
        if softmax_mode == "auto":
            full = p_dtype in (None, "fp32", "float32") or footprint["size"] == 4
            softmax_mode = "full" if full else "online"
        paged = kv_loader == "paged" and layout_kv.startswith("PA_")
        if paged:
            if softmax_mode == "full":
                raise ValueError("full softmax does not support paged KV")
            block = softmax_block_size or p_block or resolve_p_block(params or {})
            block = min(max(int(block), 1), max(kv_seg, 1))
            kv_extra = (d + dv) * block * n2 * 4 + n2 * block * d * 4
            pa_dense = 0
        else:
            full_k = softmax_mode != "online" and (
                p_dtype in (None, "fp32", "float32") or footprint["size"] == 4
            )
            if full_k:
                kv_extra = (2 * d + dv) * kv_seg * n2 * 4
            else:
                block = p_block or resolve_p_block(params or {})
                block = min(max(int(block), 1), max(kv_seg, 1))
                kv_extra = (d + dv) * kv_seg * n2 * 4 + n2 * block * d * 4
            pa_dense = footprint["pa_dense"]
        scratch += pa_dense + footprint["out_extra"] + kv_extra
        device_total = footprint["inputs"] + footprint["golden"] + scratch
    else:
        device_total = scratch
    if device == "npu":
        host = 0
        if include_golden and not keep_on_device:
            host = footprint["golden"]
        return {"host_peak": host, "device_peak": device_total}
    return {"host_peak": device_total, "device_peak": 0}


def _as_int(value, default: int) -> int:
    """容忍字符串/浮点/列表形态的整型参数。"""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_bnsd_q(t: torch.Tensor, layout: str, cu=None) -> List[torch.Tensor]:
    """返回 per-batch 的 (N, S, D) 视图列表。"""
    if layout == "BNSD":
        return [t[b] for b in range(t.shape[0])]
    if layout == "BSND":
        return [t[b].permute(1, 0, 2) for b in range(t.shape[0])]
    if layout == "TND":
        out = []
        for b in range(len(cu) - 1):
            start, end = int(cu[b]), int(cu[b + 1])
            out.append(t[start:end].permute(1, 0, 2))
        return out
    raise ValueError(f"unsupported layout_q: {layout}")


def _to_bnsd_kv(t: torch.Tensor, layout: str, params: Dict[str, Any], cu=None):
    if layout in ("BNSD", "BSND"):
        return [
            t[b].permute(1, 0, 2) if layout == "BSND" else t[b]
            for b in range(t.shape[0])
        ]
    if layout == "TND":
        out = []
        for b in range(len(cu) - 1):
            start, end = int(cu[b]), int(cu[b + 1])
            out.append(t[start:end].permute(1, 0, 2))
        return out
    if layout.startswith("PA_"):
        return pa_to_bnsd(t, layout, params)
    raise ValueError(f"unsupported layout_kv: {layout}")


def _lengths(params: Dict[str, Any], key: str, batch: int, fallback: int) -> List[int]:
    used = params.get(key)
    if used is not None:
        if (
            isinstance(used, (list, tuple))
            and len(used) == 1
            and isinstance(used[0], (list, tuple))
        ):
            used = used[0]
        return [int(v) for v in used]
    cu = params.get("cu_seqlens_q" if key == "seqused_q" else "cu_seqlens_kv")
    if cu is not None:
        if (
            isinstance(cu, (list, tuple))
            and len(cu) == 1
            and isinstance(cu[0], (list, tuple))
        ):
            cu = cu[0]
        return [int(cu[i + 1]) - int(cu[i]) for i in range(len(cu) - 1)]
    return [int(fallback)] * batch


def _elem_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _scores(q: torch.Tensor, k_t: torch.Tensor, group: int, n2: int) -> torch.Tensor:
    """(N, rows, D) @ (N, D, S) -> (N, rows, S)；group>1 时按 KV 头广播。"""
    if group == 1:
        return torch.bmm(q, k_t)
    n1, rows, d = q.shape
    out = torch.matmul(q.reshape(n2, group, rows, d), k_t.unsqueeze(1))
    return out.reshape(n1, rows, -1)


def _pv(p: torch.Tensor, v: torch.Tensor, group: int, n2: int) -> torch.Tensor:
    """(N, rows, S) @ (N, S, DV) -> (N, rows, DV)。"""
    if group == 1:
        return torch.bmm(p, v)
    n1, rows, s = p.shape
    out = torch.matmul(p.reshape(n2, group, rows, s), v.unsqueeze(1))
    return out.reshape(n1, rows, -1)


def _tile_mask(
    start: int,
    end: int,
    k_start: int,
    k_stop: int,
    valid_s1: int,
    valid_s2: int,
    mask_mode: int,
    attn_mask: Optional[torch.Tensor],
    win_left: int,
    win_right: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """行块 [start, end) × KV 列块 [k_start, k_stop) 的 bool mask（True=屏蔽）。

    mask_mode 3/4 用位置关系直接构造（不落全量 S1×S2 mask）；显式 attn_mask 只切片。
    """
    delta = valid_s2 - valid_s1
    q_pos = torch.arange(start, end, device=device) + delta
    if mask_mode == 3:
        k_pos = torch.arange(k_start, k_stop, device=device)
        return k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
    if mask_mode == 4:
        k_pos = torch.arange(k_start, k_stop, device=device).unsqueeze(0)
        q_pos = q_pos.unsqueeze(1)
        # -1 表示该侧无界：用大于任何合法位置距离的常量
        unbounded = valid_s1 + valid_s2 + 1
        left = -win_left if win_left >= 0 else -unbounded
        right = win_right if win_right >= 0 else unbounded
        return (k_pos < (q_pos + left)) | (k_pos > (q_pos + right))
    if attn_mask is not None:
        return attn_mask[start:end, k_start:k_stop].to(torch.bool)
    return None


def _resolve_p_dtype(p_dtype, v_dtype: torch.dtype) -> Optional[torch.dtype]:
    """P 落盘精度：fp32（不落盘）/ out_dtype（落盘到 V 的 dtype）。非量化算子只有这两档。"""
    if p_dtype in (None, "fp32", "float32"):
        return None
    if p_dtype == "out_dtype":
        if v_dtype == torch.float32:
            return None
        return v_dtype
    raise ValueError(
        f"p_dtype must be 'fp32' or 'out_dtype' for a non-quantized operator, got {p_dtype!r}"
    )


def _attend_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_s1: int,
    valid_s2: int,
    *,
    dtype: torch.dtype,
    scale: float,
    group: int,
    n2: int,
    mask_mode: int,
    attn_mask: Optional[torch.Tensor],
    win_left: int,
    win_right: int,
    sinks: Optional[torch.Tensor],
    p_dtype: Optional[torch.dtype],
    p_block: int,
    chunk_bytes: int,
    softmax_mode: str,
    softmax_block_size: int,
    k_loader=None,
    v_loader=None,
    kv_seq_index: int = 0,
    dv: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """单个 batch 的全部 head：返回 (N1, S1, DV) fp32/fp64 输出与 (N1, S1) lse。"""
    n1, s1, _ = q.shape
    dv = dv or v.shape[-1]
    out = torch.zeros(n1, s1, dv, dtype=dtype, device=q.device)
    lse = torch.full((n1, s1), float("inf"), dtype=torch.float32, device=q.device)
    if valid_s1 <= 0 or valid_s2 <= 0:
        return out, lse

    if k_loader is None:
        k = k[:, :valid_s2, :].to(dtype)
        v = v[:, :valid_s2, :].to(dtype)
        k_t = k.transpose(1, 2).contiguous() if softmax_mode == "full" else None
        sink = sinks[:n1].to(dtype) if sinks is not None else None
        block = (
            valid_s2
            if softmax_mode == "full"
            else min(max(softmax_block_size or p_block, 1), valid_s2)
        )
    else:
        if softmax_mode == "full":
            raise ValueError("full softmax does not support paged KV")
        k = v = k_t = None
        sink = sinks[:n1].to(dtype) if sinks is not None else None
        block = min(max(softmax_block_size or p_block, 1), valid_s2)
    rows = max(1, chunk_bytes // max(n1 * block * _elem_size(dtype), 1))

    for start in range(0, valid_s1, rows):
        end = min(start + rows, valid_s1)
        rows_here = end - start
        q_chunk = q[:, start:end, :].to(dtype)
        if softmax_mode == "full":
            mask = _tile_mask(
                start,
                end,
                0,
                valid_s2,
                valid_s1,
                valid_s2,
                mask_mode,
                attn_mask,
                win_left,
                win_right,
                q.device,
            )
            scores = _scores(q_chunk, k_t, group, n2) * scale
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
            row_max = scores.amax(dim=-1, keepdim=True)
            if sink is not None:
                row_max = torch.maximum(row_max, sink.view(n1, 1, 1))
            row_max = torch.where(
                torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
            )
            probs = torch.exp(scores - row_max)
            denom = probs.sum(dim=-1, keepdim=True)
            if sink is not None:
                denom = denom + torch.exp(sink.view(n1, 1, 1) - row_max)
            prob_mm = probs if p_dtype is None else probs.to(p_dtype).float()
            numer = _pv(prob_mm, v, group, n2)
            _write_chunk(out, lse, numer, denom, row_max, start, end)
            continue

        m = torch.full(
            (n1, rows_here, 1),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
            device=q.device,
        )
        s = torch.zeros((n1, rows_here, 1), dtype=torch.float32, device=q.device)
        acc = torch.zeros((n1, rows_here, dv), dtype=torch.float32, device=q.device)
        if sink is not None:
            m = sink.view(n1, 1, 1).expand(n1, rows_here, 1).contiguous()
            s = torch.ones((n1, rows_here, 1), dtype=torch.float32, device=q.device)
        for j in range(0, valid_s2, block):
            stop = min(j + block, valid_s2)
            if k_loader is None:
                k_blk = k[:, j:stop, :]
                v_blk = v[:, j:stop, :]
            else:
                k_blk = k_loader.load(kv_seq_index, j, stop - j).to(dtype)
                v_blk = v_loader.load(kv_seq_index, j, stop - j).to(dtype)
            k_blk_t = k_blk.transpose(1, 2).contiguous()
            scores = _scores(q_chunk, k_blk_t, group, n2) * scale
            mask = _tile_mask(
                start,
                end,
                j,
                stop,
                valid_s1,
                valid_s2,
                mask_mode,
                attn_mask,
                win_left,
                win_right,
                q.device,
            )
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
            m_new = torch.maximum(m, scores.amax(dim=-1, keepdim=True))
            probs = torch.exp(scores - m_new)
            # kernel: 分母/lse 用未落盘的 fp32 P，只有 P@V 用落盘后的 P
            prob_mm = probs if p_dtype is None else probs.to(p_dtype).float()
            alpha = torch.exp(m - m_new)
            s = s * alpha + probs.sum(dim=-1, keepdim=True)
            acc = acc * alpha + _pv(prob_mm, v_blk, group, n2)
            m = m_new
        valid = s > 0
        out[:, start:end, :] = torch.where(
            valid,
            acc / s.clamp_min(torch.finfo(torch.float32).tiny),
            torch.zeros_like(acc),
        )
        lse_rows = torch.log(s.clamp_min(torch.finfo(torch.float32).tiny)) + m
        lse[:, start:end] = torch.where(
            valid.squeeze(-1),
            lse_rows.squeeze(-1),
            torch.full_like(lse_rows.squeeze(-1), float("inf")),
        )
    return out, lse


def _write_chunk(out, lse, numer, denom, row_max, start, end):
    tiny = torch.finfo(out.dtype).tiny
    valid = denom > 0
    out[:, start:end, :] = torch.where(
        valid, numer / denom.clamp_min(tiny), torch.zeros_like(numer)
    )
    lse_rows = torch.log(denom.clamp_min(tiny)) + row_max
    lse[:, start:end] = torch.where(
        valid.squeeze(-1),
        lse_rows.squeeze(-1),
        torch.full_like(lse_rows.squeeze(-1), float("inf")),
    )


def compute(
    inputs: Dict[str, Any],
    params: Dict[str, Any],
    compute_dtype: str = "fp32",
    chunk_bytes: int = 128 * 1024 * 1024,
    device: str = "cpu",
    p_dtype=None,
    p_block: int = 0,
    softmax_mode: str = "auto",
    softmax_block_size: int = 0,
    kv_loader: str = "dense",
    keep_on_device: bool = False,
) -> Dict[str, Any]:
    target = torch.device(device if device != "gpu" else "cuda")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("golden device=gpu requested but CUDA is not available")
    layout_q = params.get("layout_q", params.get("input_layout", "BNSD"))
    layout_kv = params.get("layout_kv", layout_q)
    layout_out = params.get("layout_out", layout_q)
    cu_q = params.get("cu_seqlens_q")
    cu_kv = params.get("cu_seqlens_kv")
    if isinstance(cu_q, list) and cu_q and isinstance(cu_q[0], (list, tuple)):
        cu_q = cu_q[0]
    if isinstance(cu_kv, list) and cu_kv and isinstance(cu_kv[0], (list, tuple)):
        cu_kv = cu_kv[0]
    out_dtype = _DTYPE_MAP.get(
        str(params.get("dtype", params.get("Dtype", "fp16"))), torch.float16
    )
    v_dtype = (
        inputs["v"].dtype if isinstance(inputs.get("v"), torch.Tensor) else out_dtype
    )
    p_cast = _resolve_p_dtype(p_dtype, v_dtype)
    if softmax_mode == "auto":
        softmax_mode = "full" if p_cast is None else "online"
    if softmax_mode not in ("full", "online"):
        raise ValueError(f"unsupported softmax_mode: {softmax_mode!r}")
    if kv_loader not in ("dense", "paged"):
        raise ValueError(f"unsupported kv_loader: {kv_loader!r}")
    use_paged = kv_loader == "paged" and str(layout_kv).startswith("PA_")
    if use_paged and softmax_mode == "full":
        raise ValueError("full softmax does not support paged KV")
    # P 落盘口径与 kernel 一致时恒用 fp32 累加（fp64 只服务于高精度真值）
    dtype = torch.float32
    if p_cast is None and compute_dtype == "fp64":
        dtype = torch.float64

    def _move(t):
        return t.to(target) if isinstance(t, torch.Tensor) else t

    q_list = [x.to(target) for x in _to_bnsd_q(_move(inputs["q"]), layout_q, cu_q)]
    if use_paged:
        k_loader = PagedKVLoader(_move(inputs["k"]), layout_kv, params)
        v_loader = PagedKVLoader(_move(inputs["v"]), layout_kv, params)
        k_list = v_list = None
    else:
        k_loader = v_loader = None
        k_list = [
            x.to(target)
            for x in _to_bnsd_kv(_move(inputs["k"]), layout_kv, params, cu_kv)
        ]
        v_list = [
            x.to(target)
            for x in _to_bnsd_kv(_move(inputs["v"]), layout_kv, params, cu_kv)
        ]
    batch = len(q_list)
    if k_loader is not None and k_loader.num_sequences() < batch:
        raise ValueError("paged KV has fewer sequences than Q batches")
    n1, n2 = int(params["N1"]), int(params.get("N2", params["N1"]))
    group = max(n1 // max(n2, 1), 1)
    raw_scale = params.get("scale")
    scale = (
        float(raw_scale) if raw_scale is not None else 1.0 / math.sqrt(int(params["D"]))
    )
    mask_mode = int(params.get("mask_mode", 0) or 0)
    win_left = int(
        params.get("win_left", -1) if params.get("win_left") is not None else -1
    )
    win_right = int(
        params.get("win_right", -1) if params.get("win_right") is not None else -1
    )
    sinks = inputs.get("sinks")
    if sinks is not None:
        sinks = sinks.to(target).float()
    want_lse = bool(params.get("return_softmax_lse", False))
    # mask_mode 3/4 由位置关系构造；只有 mask_mode=0 才消费显式 mask
    attn_mask = inputs.get("attn_mask")
    if attn_mask is not None and mask_mode in (3, 4):
        attn_mask = None
    if attn_mask is not None:
        attn_mask = attn_mask.to(target)
        if attn_mask.dim() == 4:
            attn_mask = attn_mask[:, 0]

    q_lens = _lengths(params, "seqused_q", batch, q_list[0].shape[1] if q_list else 1)
    kv_lens = _lengths(params, "seqused_kv", batch, k_list[0].shape[1] if k_list else 1)
    block = softmax_block_size or p_block or resolve_p_block(params)

    lse_list: List[torch.Tensor] = []
    dv = v_list[0].shape[-1] if v_list else int(params.get("DV", params["D"]))
    if layout_q == "TND":
        out = torch.empty(
            (sum(x.shape[1] for x in q_list), n1, dv),
            dtype=out_dtype,
            device=q_list[0].device,
        )
        out_bnsd = None
        offset = 0
    else:
        s1 = q_list[0].shape[1]
        if layout_out in ("BNSD", "BSND"):
            shape = (batch, n1, s1, dv) if layout_out == "BNSD" else (batch, s1, n1, dv)
            out = torch.empty(shape, dtype=out_dtype, device=q_list[0].device)
        else:
            out = None
        out_bnsd = None
        offset = 0
    for b in range(batch):
        qb = q_list[b]
        if k_loader is None:
            kb, vb = k_list[b], v_list[b]
            s2 = kb.shape[1]
        else:
            kb = vb = None
            s2 = kv_lens[b] if b < len(kv_lens) else k_loader._seq_len(b)
        s1 = qb.shape[1]
        valid_s1 = min(int(q_lens[b]) if b < len(q_lens) else s1, s1)
        valid_s2 = min(int(kv_lens[b]) if b < len(kv_lens) else s2, s2)
        mask_b = (
            attn_mask[b]
            if attn_mask is not None and attn_mask.dim() == 3
            else attn_mask
        )
        out_b, lse_b = _attend_batch(
            qb,
            kb,
            vb,
            valid_s1,
            valid_s2,
            dtype=dtype,
            scale=scale,
            group=group,
            n2=n2 if k_loader is not None else kb.shape[0],
            mask_mode=mask_mode,
            attn_mask=mask_b,
            win_left=win_left,
            win_right=win_right,
            sinks=sinks,
            p_dtype=p_cast,
            p_block=block,
            chunk_bytes=chunk_bytes,
            softmax_mode=softmax_mode,
            softmax_block_size=block,
            k_loader=k_loader,
            v_loader=v_loader,
            kv_seq_index=b,
            dv=dv,
        )
        if layout_q == "TND":
            rows = out_b.shape[1]
            out[offset : offset + rows].copy_(out_b.permute(1, 0, 2))
            offset += rows
        elif layout_out == "BNSD":
            out[b].copy_(out_b)
        elif layout_out == "BSND":
            out[b].copy_(out_b.permute(1, 0, 2))
        else:
            if out_bnsd is None:
                out_bnsd = torch.empty(
                    (batch, n1, s1, dv), dtype=out_dtype, device=q_list[0].device
                )
            out_bnsd[b].copy_(out_b)
        lse_list.append(lse_b)
        del out_b

    if not keep_on_device:
        lse_list = [x.cpu() for x in lse_list]
    if layout_q != "TND" and layout_out not in ("BNSD", "BSND"):
        out = trans_bnsd_to_layout(out_bnsd, layout_out, **params)
        if out is None:
            out = out_bnsd

    if not want_lse:
        lse_out = torch.zeros(1, dtype=torch.float32, device=out.device)
    elif layout_q == "TND":
        # 与算子契约一致：(N1, T)
        lse_out = torch.cat(lse_list, dim=1)
    else:
        lse_out = torch.stack(lse_list, dim=0)  # (B, N1, S1)
    return {"out": out, "lse": lse_out}
