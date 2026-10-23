# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FlashMla golden reference and TTK entry point."""

from typing import Optional, Tuple
from functools import partial

import torch


def attention_single(
    q,
    k,
    v,
    scale,
    tile_size,
    mask_fn,
    sink=None,
    kv_bounds=None,
    rounding_block_size=None,
):
    sq, skv, dv = q.shape[0], k.shape[0], v.shape[-1]
    out = torch.zeros((sq, dv), dtype=torch.float32, device=q.device)
    lse = torch.full((sq,), float("inf"), dtype=torch.float32, device=q.device)
    for qs in range(0, sq, tile_size):
        qe = min(qs + tile_size, sq)
        qi = q[qs:qe].float()
        acc = torch.zeros((qe - qs, dv), dtype=torch.float32, device=q.device)
        total = torch.full(
            (qe - qs, 1),
            0.0 if sink is None else 1.0,
            dtype=torch.float32,
            device=q.device,
        )
        if isinstance(sink, torch.Tensor):
            maximum = (
                sink[qs:qe]
                .to(device=q.device, dtype=torch.float32)
                .reshape(-1, 1)
                .clone()
            )
        else:
            maximum = torch.full_like(total, float("-inf") if sink is None else sink)
        first, stop = (0, skv) if kv_bounds is None else kv_bounds(qs, qe)
        first = max(0, first // tile_size * tile_size)
        stop = min(skv, max(0, stop))
        for ks in range(first, stop, tile_size):
            ke = min(ks + tile_size, skv)
            scores = (qi @ k[ks:ke].float().T).mul_(scale)
            mask = mask_fn(qs, qe, ks, ke)
            if mask is not None:
                scores.masked_fill_(mask, float("-inf"))
            block = min(rounding_block_size or (ke - ks), ke - ks)
            if (
                rounding_block_size is not None
                and q.dtype in (torch.float16, torch.bfloat16)
                and block < ke - ks
            ):
                # Preserve rounding at each KV block before merging PV terms.
                width = ke - ks
                padded = (width + block - 1) // block * block
                if padded != width:
                    scores = torch.nn.functional.pad(
                        scores, (0, padded - width), value=float("-inf")
                    )
                blocks = scores.reshape(qe - qs, -1, block)
                running = torch.maximum(
                    blocks.amax(-1, keepdim=True).cummax(1).values, maximum[:, None, :]
                )
                new_max = running[:, -1, :]
                safe = torch.where(torch.isneginf(running), 0.0, running)
                safe_max = torch.where(torch.isneginf(new_max), 0.0, new_max)
                weights = blocks.sub_(safe).exp_()
                rescale = (running - safe_max[:, None, :]).exp_()
                correction = (maximum - safe_max).exp_()
                total.mul_(correction).add_(
                    (weights.sum(-1, keepdim=True) * rescale).sum(1)
                )
                rounded = (
                    weights.to(q.dtype)
                    .float()
                    .mul_(rescale)
                    .reshape(qe - qs, padded)[:, :width]
                )
                del blocks, running, safe, rescale
            else:
                new_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
                # Entirely masked tiles must not produce exp(-inf - -inf).
                safe_max = torch.where(torch.isneginf(new_max), 0.0, new_max)
                weights = scores.sub_(safe_max).exp_()
                correction = (maximum - safe_max).exp_()
                total.mul_(correction).add_(weights.sum(-1, keepdim=True))
                # Sum uses unrounded weights, PV uses low-precision weights.
                rounded = weights.to(q.dtype).float()
            del scores, weights
            acc.mul_(correction).add_(rounded @ v[ks:ke].float())
            del rounded
            maximum = new_max
        denom = torch.where(total > 0, total, 1.0)
        out[qs:qe] = acc / denom
        rows_lse = (total.log() + maximum).squeeze(-1)
        lse[qs:qe] = torch.where(total.squeeze(-1) == 0, float("inf"), rows_lse)
    return out, lse


def gather_pages(cache, block_table, batch, length, layout):
    """Restore one batch from the referenced cache pages."""
    bs = (
        cache.shape[3]
        if layout == "PA_NZ"
        else cache.shape[1 if layout == "PA_BBND" else 2]
    )
    heads = cache.shape[2] if layout == "PA_BBND" else cache.shape[1]
    width = cache.shape[2] * cache.shape[4] if layout == "PA_NZ" else cache.shape[-1]
    if length < 0:
        raise ValueError("KV length must be nonnegative")
    count = (length + bs - 1) // bs
    if (
        block_table.ndim != 2
        or batch >= block_table.shape[0]
        or count > block_table.shape[1]
    ):
        raise ValueError("block_table does not cover the requested KV length")
    ids = block_table[batch, :count].to(device="cpu", dtype=torch.int64)
    if ((ids < 0) | (ids >= cache.shape[0])).any():
        raise ValueError("referenced block_table index is outside the cache")
    result = torch.empty(
        (1, heads, length, width), dtype=cache.dtype, device=cache.device
    )
    ids = ids.to(cache.device)
    for start in range(0, count, 128):
        pages = cache.index_select(0, ids[start : start + 128])
        if layout == "PA_NZ":
            pages = pages.permute(1, 0, 3, 2, 4).reshape(heads, -1, width)
        elif layout == "PA_BBND":
            pages = pages.permute(2, 0, 1, 3).reshape(heads, -1, width)
        else:
            pages = pages.permute(1, 0, 2, 3).reshape(heads, -1, width)
        offset = start * bs
        valid = min(pages.shape[1], length - offset)
        result[0, :, offset : offset + valid] = pages[:, :valid]
    return result


_NZ_BLK_ELEM = 16  # PA_NZ blk_elem (fp16/bf16)


class FlashMlaGolden:
    """Compute FlashMla reference outputs."""

    # Layout/length handling is shared; numerical backends may override this hook.
    # flash_mla.cpp fixes S2BaseSize at 112; P is rounded before each PV.
    _attention_single = staticmethod(partial(attention_single, rounding_block_size=112))

    TILE_SIZE = 2048
    EPSILON = 1e-20
    TILED_CROSSOVER = 3 * 1024 * 1024  # S1*S2 > this -> tiled saves memory

    @staticmethod
    def _derive_geometry(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        layout_q: str,
        layout_kv: str,
        head_dim_v: int,
    ) -> dict:
        """Extract geometry from inputs and layouts."""
        if layout_q not in ("BNSD", "BSND", "TND"):
            raise ValueError(f"unsupported layout_q: {layout_q}")
        if layout_kv not in ("PA_NZ", "PA_BBND", "PA_BNBD"):
            raise ValueError(
                f"unsupported layout_kv: {layout_kv} (仅 PA_NZ/PA_BBND/PA_BNBD)"
            )

        head_dim_qk = q.shape[-1]
        if head_dim_qk != 576:
            raise ValueError(
                f"q last dim (head_dim_qk) 必须为 576(MLA 硬约束, nope 512 + "
                f"rope 64), got {head_dim_qk}"
            )
        # MLA 硬约束: V 宽 = nope 宽, 必须为 512
        if head_dim_v != 512:
            raise ValueError(f"head_dim_v 必须为 512(MLA 硬约束), got {head_dim_v}")

        # KV 布局维度校验与块长/宽提取
        if layout_kv == "PA_NZ":
            if k_cache.dim() != 5:
                raise ValueError(
                    f"layout_kv=PA_NZ 要求 k_cache 5 维 (NB, kv, DSUB, BS, "
                    f"16), got dim={k_cache.dim()}"
                )
            if k_cache.shape[1] != 1:
                raise ValueError(
                    f"k_cache dim1 (kv heads) must be 1 (MLA 硬约束), "
                    f"got {k_cache.shape[1]}"
                )
            if k_cache.shape[-1] != _NZ_BLK_ELEM:
                raise ValueError(
                    f"k_cache last dim must be NZ blk_elem={_NZ_BLK_ELEM}, "
                    f"got {k_cache.shape[-1]}"
                )
            block_size = k_cache.shape[3]  # BS
            k_dim = k_cache.shape[2] * k_cache.shape[4]  # DSUB*BLK = 576
            if k_dim != head_dim_qk:
                raise ValueError(f"k_cache 宽度 {k_dim} != head_dim_qk({head_dim_qk})")
        elif layout_kv == "PA_BBND":
            if k_cache.dim() != 4:
                raise ValueError(
                    f"layout_kv=PA_BBND 要求 k_cache 4 维 (NB, BS, kv, D), "
                    f"got dim={k_cache.dim()}"
                )
            if k_cache.shape[2] != 1:
                raise ValueError(
                    f"k_cache dim2 (kv heads) must be 1 (MLA 硬约束), "
                    f"got {k_cache.shape[2]}"
                )
            block_size = k_cache.shape[1]  # BS
            k_dim = k_cache.shape[-1]  # D
            if k_dim != head_dim_qk:
                raise ValueError(f"k_cache 宽度 {k_dim} != head_dim_qk({head_dim_qk})")
        else:  # PA_BNBD
            if k_cache.dim() != 4:
                raise ValueError(
                    f"layout_kv=PA_BNBD 要求 k_cache 4 维 (NB, kv, BS, D), "
                    f"got dim={k_cache.dim()}"
                )
            if k_cache.shape[1] != 1:
                raise ValueError(
                    f"k_cache dim1 (kv heads) must be 1 (MLA 硬约束), "
                    f"got {k_cache.shape[1]}"
                )
            block_size = k_cache.shape[2]  # BS
            k_dim = k_cache.shape[-1]  # D
            if k_dim != head_dim_qk:
                raise ValueError(f"k_cache 宽度 {k_dim} != head_dim_qk({head_dim_qk})")

        # N 位置随 layout: TND/BNSD 在 dim1, BSND (B,S,N,D) 在 dim2
        if layout_q == "BSND":
            num_heads_q, seq_decl, batch = q.shape[2], q.shape[1], q.shape[0]
        elif layout_q == "BNSD":
            num_heads_q, seq_decl, batch = q.shape[1], q.shape[2], q.shape[0]
        else:  # TND
            num_heads_q, seq_decl, batch = q.shape[1], None, None

        return {
            "num_heads_q": num_heads_q,
            "head_dim_qk": head_dim_qk,
            "head_dim_v": head_dim_v,
            "rope_dim": head_dim_qk - head_dim_v,  # = 64
            "block_size": block_size,
            "batch": batch,
            "seq_decl": seq_decl,
            "layout_kv": layout_kv,
        }

    @torch.no_grad()
    def forward_golden(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        head_dim_v: int = 512,
        softmax_scale: float = 1.0,
        mask_mode: int = 0,
        win_left: int = -1,
        win_right: int = -1,
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        layout_q: str = "BSND",
        layout_kv: str = "PA_BBND",
        layout_out: Optional[str] = "BSND",
        return_softmax_lse: bool = False,
        device_mode: str = "npu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention output and optional LSE."""
        if device_mode not in ("cpu", "npu"):
            raise ValueError(f"device_mode must be 'cpu' or 'npu', got {device_mode}")
        layout_out = "BSND" if layout_out is None else layout_out
        valid_outputs = (layout_q, "NTD") if layout_q == "TND" else (layout_q,)
        if layout_out not in valid_outputs:
            raise ValueError(f"unsupported output layout: {layout_q} -> {layout_out}")
        if softmax_scale is None:
            raise ValueError("softmax_scale must be a number")
        if layout_kv not in ("PA_NZ", "PA_BBND", "PA_BNBD"):
            raise ValueError(
                f"unsupported layout_kv: {layout_kv} (仅 PA_NZ/PA_BBND/PA_BNBD)"
            )
        if mask_mode not in (0, 3, 4):
            raise ValueError(f"mask_mode 仅支持 0/3/4, got {mask_mode}")
        if layout_q == "TND" and cu_seqlens_q is None:
            raise ValueError("layout_q=TND 要求 cu_seqlens_q")
        # Modes 3/4 construct the right-aligned mask from logical positions.
        if mask_mode in (3, 4):
            attn_mask = None

        geo = self._derive_geometry(q, k_cache, layout_q, layout_kv, head_dim_v)
        self._geo = geo  # 供 _compute_per_batch 使用

        if device_mode == "npu":
            for name, t in [("q", q), ("k_cache", k_cache)]:
                if t.device.type != "npu":
                    raise ValueError(
                        f"device_mode='npu' requires {name} to be on NPU, "
                        f"got {t.device}"
                    )

        orig_dtype = q.dtype
        scale = softmax_scale

        # 1. cpu 模式: 辅助张量搬 CPU(q/k 由调用方自备, 计算留在 q.device)
        if device_mode == "cpu":
            block_table = block_table.cpu()
            cache_seqlens = cache_seqlens.cpu()
            if cu_seqlens_q is not None:
                cu_seqlens_q = cu_seqlens_q.cpu()
            if seqused_q is not None:
                seqused_q = seqused_q.cpu()

        cu_q_list = cu_seqlens_q.tolist() if cu_seqlens_q is not None else None
        cache_lens = [int(x) for x in cache_seqlens.tolist()]

        # 2. 推导每 batch 序列长: seqused_q 优先, cu diff 次之, 否则声明 S
        if seqused_q is not None:
            q_lens = seqused_q.tolist()
        elif cu_q_list is not None:
            q_lens = [
                cu_q_list[b + 1] - cu_q_list[b] for b in range(len(cu_q_list) - 1)
            ]
        else:
            q_lens = None
        kv_lens = cache_lens  # PA cache 长度 = cache_seqlens(接口参数)

        batch = len(cu_q_list) - 1 if layout_q == "TND" else geo["batch"]
        if len(cache_lens) != batch or any(x < 0 for x in cache_lens):
            raise ValueError(f"cache_seqlens must contain {batch} nonnegative lengths")
        if q_lens is not None and len(q_lens) != batch:
            raise ValueError(f"seqused_q must contain {batch} lengths")
        if layout_q == "TND" and (
            cu_q_list[0] != 0
            or cu_q_list[-1] != q.shape[0]
            or any(a > b for a, b in zip(cu_q_list, cu_q_list[1:]))
        ):
            raise ValueError(
                "cu_seqlens_q must monotonically cover the TND tensor from 0"
            )

        if q_lens is not None:
            for b_idx, ql in enumerate(q_lens):
                if layout_q == "TND":
                    phys_len = cu_q_list[b_idx + 1] - cu_q_list[b_idx]
                else:
                    phys_len = geo["seq_decl"]
                if not (0 <= ql <= phys_len):
                    raise ValueError(
                        f"seqused_q[{b_idx}]={ql} 超出该 batch 物理/声明序列长 "
                        f"{phys_len}(layout_q={layout_q}); 序列实际长度不得越过张量长度"
                    )

        # Resolve auto compute_mode (同 flash golden 口径)
        if q_lens is not None:
            max_sq = max(q_lens, default=0)
            max_product = max((sq * skv for sq, skv in zip(q_lens, kv_lens)), default=0)
        else:
            max_sq = geo["seq_decl"]
            max_skv = max(kv_lens, default=0)
            max_product = max_sq * max_skv
        if max_product * geo["num_heads_q"] > self.TILED_CROSSOVER:
            compute_mode = "tiled"
        else:
            compute_mode = "full"
        self.last_compute_mode = compute_mode

        # 3. 逐 batch 计算(防 OOM)
        out_native, lse_f32 = self._compute_per_batch(
            q,
            k_cache,
            q_lens,
            kv_lens,
            layout_q,
            cu_q_list,
            block_table,
            mask_mode,
            attn_mask,
            scale,
            compute_mode,
            win_left,
            win_right,
            sinks,
        )

        # 4. 输出布局转换 + dtype
        if layout_q == "TND":
            attn_out = out_native.to(orig_dtype)  # (T, N, DV)
        else:
            out_bnsd = out_native.to(orig_dtype)  # (B, N, S, DV)
            if layout_q == "BNSD":
                attn_out = out_bnsd
            else:  # BSND
                attn_out = out_bnsd.permute(0, 2, 1, 3).contiguous()

        # 4b. 目标输出布局(与 fia golden 同口径; golden 内部完成, 外部不得变换)
        if layout_out != layout_q:
            if layout_q == "TND" and layout_out == "NTD":
                attn_out = attn_out.permute(1, 0, 2).contiguous()
            else:
                raise ValueError(
                    f"Unsupported golden output conversion {layout_q} -> {layout_out}"
                )

        # 5. softmax_lse (算子契约: TND -> (N, T), 非 TND -> (B, N, S))
        if not return_softmax_lse:
            softmax_lse = torch.zeros(1, dtype=torch.float32, device=attn_out.device)
        elif layout_q == "TND":
            softmax_lse = lse_f32.t().contiguous()  # (T, N) -> (N, T)
        else:
            softmax_lse = lse_f32

        return attn_out, softmax_lse

    def _compute_per_batch(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        q_lens: Optional[list],
        kv_lens: list,
        layout_q: str,
        cu_q: Optional[list],
        block_table: torch.Tensor,
        mask_mode: int,
        attn_mask: Optional[torch.Tensor],
        scale: float,
        compute_mode: str,
        win_left: int,
        win_right: int,
        sinks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention one batch at a time."""
        N_q = self._geo["num_heads_q"]
        if sinks is not None and sinks.shape == (0,):
            sinks = None
        if sinks is not None:
            if sinks.dtype != torch.float32 or sinks.shape != (N_q,):
                raise ValueError(
                    "sinks must be a float32 tensor with shape (num_heads_q,)"
                )
            sinks = sinks.to(q.device)
        dv = self._geo["head_dim_v"]
        seq_decl = self._geo["seq_decl"]  # BNSD/BSND 声明序列长(TND=None)
        group = N_q  # num_heads_kv == 1, 单 KV 头广播到所有 Q 头

        if layout_q == "TND":
            B = len(cu_q) - 1
        else:
            B = self._geo["batch"]

        if q_lens is None:
            if layout_q == "TND":
                q_lens = [cu_q[b + 1] - cu_q[b] for b in range(B)]
            else:  # BNSD/BSND: 声明长度
                q_lens = [seq_decl] * B

        is_tnd_out = layout_q == "TND"
        if is_tnd_out:
            total_q = q.shape[0]
            out_tnd = torch.zeros(total_q, N_q, dv, dtype=q.dtype, device=q.device)
            lse_tnd = torch.zeros(total_q, N_q, dtype=torch.float32, device=q.device)
            out = lse = None
        else:
            out = torch.zeros(B, N_q, seq_decl, dv, dtype=q.dtype, device=q.device)
            lse = torch.zeros(B, N_q, seq_decl, dtype=torch.float32, device=q.device)
            out_tnd = lse_tnd = None

        for b in range(B):
            sq = q_lens[b]
            skv = kv_lens[b]
            if is_tnd_out:
                q_start = cu_q[b]
            # Set truncated-row LSE before skipping empty batches.
            if is_tnd_out:
                phys = cu_q[b + 1] - cu_q[b]
                if phys > sq:
                    lse_tnd[q_start + sq : q_start + phys, :] = float("inf")
            elif sq < seq_decl:
                lse[b, :, sq:] = float("inf")
            if sq <= 0 or skv <= 0:
                # 空 batch / 空 KV: 算子 LSE 输出 +inf
                if is_tnd_out:
                    lse_tnd[q_start : q_start + sq, :] = float("inf")
                else:
                    lse[b, :, :] = float("inf")
                continue

            q_b, k_b = self._extract_batch_bnsd(
                q,
                k_cache,
                b,
                sq,
                skv,
                layout_q,
                self._geo["layout_kv"],
                cu_q,
                block_table,
            )
            # V = K 的 nope 部分 (kv 复用, rope 不参与 V 投影)
            v_b = k_b[..., :dv]

            # 单 KV 头 -> 所有 Q 头广播; masked 用右对齐 causal 自构造
            mask_b = (
                None
                if (compute_mode == "tiled" and mask_mode in (3, 4))
                else self._build_batch_mask(
                    sq,
                    skv,
                    mask_mode,
                    attn_mask,
                    q.device,
                    b,
                    win_left=win_left,
                    win_right=win_right,
                )
            )

            delta = skv - sq
            k_bh = k_b[0, 0, :skv, :]  # (Skv, D=576)
            v_bh = v_b[0, 0, :skv, :]  # (Skv, DV=512)
            # Reuse each KV tile across heads.
            per_head_mask = (
                attn_mask is not None
                and attn_mask.ndim == 4
                and attn_mask.shape[1] != 1
            )
            head_step = 1 if per_head_mask else group
            for first in range(0, group, head_step):
                last = min(first + head_step, group)
                q_heads = q_b[0, first:last, :sq, :]
                selected_mask = mask_b
                if per_head_mask:
                    selected_mask = self._build_batch_mask(
                        sq,
                        skv,
                        mask_mode,
                        attn_mask,
                        q.device,
                        b,
                        head_index=first,
                        win_left=win_left,
                        win_right=win_right,
                    )
                flat_q = q_heads.reshape(-1, q_heads.shape[-1])
                sink_rows = (
                    None if sinks is None else sinks[first:last].repeat_interleave(sq)
                )
                if compute_mode == "tiled":

                    def mask_fn(qs, qe, ks, ke):
                        positions = torch.arange(qs, qe, device=q.device) % sq
                        if mask_mode in (3, 4):
                            return self._build_mask_mode_mask(
                                positions,
                                torch.arange(ks, ke, device=q.device),
                                delta,
                                mask_mode,
                                win_left,
                                win_right,
                            )
                        if selected_mask is None:
                            return None
                        return selected_mask[:, ks:ke].index_select(0, positions)

                    def kv_bounds(qs, qe):
                        if mask_mode not in (3, 4):
                            return 0, skv
                        # A row tile can straddle heads; each head restarts Q.
                        crosses_head = qs // sq != (qe - 1) // sq
                        first_q = 0 if crosses_head else qs % sq
                        last_q = sq - 1 if crosses_head else (qe - 1) % sq
                        if mask_mode == 3:
                            return 0, last_q + delta + 1
                        left = 0 if win_left == -1 else first_q + delta - win_left
                        right = (
                            skv if win_right == -1 else last_q + delta + win_right + 1
                        )
                        return left, right

                    o, l = self._attention_single(
                        flat_q,
                        k_bh,
                        v_bh,
                        scale,
                        self.TILE_SIZE,
                        mask_fn,
                        sink=sink_rows,
                        kv_bounds=kv_bounds,
                    )
                else:
                    if selected_mask is not None and last - first > 1:
                        selected_mask = selected_mask.repeat(last - first, 1)
                    o, l = self._compute_full_single(
                        flat_q, k_bh, v_bh, selected_mask, scale, sink_rows
                    )
                if mask_mode in (3, 4):
                    pos = torch.arange(sq, device=q.device) + delta
                    lower = (
                        torch.zeros_like(pos)
                        if mask_mode == 3 or win_left == -1
                        else pos - win_left
                    )
                    upper = (
                        pos
                        if mask_mode == 3
                        else (
                            torch.full_like(pos, skv - 1)
                            if win_right == -1
                            else pos + win_right
                        )
                    )
                    valid = lower.clamp(min=0) <= upper.clamp(max=skv - 1)
                    # Use positions to clear masked rows without hiding visible NaN/Inf.
                    fully_masked = ~valid.repeat(last - first)
                    o = o.masked_fill(fully_masked[:, None], 0.0)
                    l = l.masked_fill(fully_masked, float("inf"))
                o = o.reshape(last - first, sq, dv)
                l = l.reshape(last - first, sq)
                if is_tnd_out:
                    out_tnd[q_start : q_start + sq, first:last, :] = o.permute(1, 0, 2)
                    lse_tnd[q_start : q_start + sq, first:last] = l.T
                else:
                    out[b, first:last, :sq, :] = o
                    lse[b, first:last, :sq] = l

            del q_b, k_b, v_b

        if is_tnd_out:
            return out_tnd, lse_tnd
        return out, lse

    def _extract_batch_bnsd(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        b: int,
        sq: int,
        skv: int,
        layout_q: str,
        layout_kv: str,
        cu_q: Optional[list],
        block_table: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Restore one batch of query and paged KV tensors."""
        # Q
        if layout_q == "BNSD":
            q_b = q[b : b + 1, :, :sq, :]
        elif layout_q == "BSND":
            q_b = q[b : b + 1, :sq, :, :].permute(0, 2, 1, 3)
        else:  # TND
            s, e = cu_q[b], cu_q[b + 1]
            q_b = q[s:e].unsqueeze(0).permute(0, 2, 1, 3)

        k_b = gather_pages(k_cache, block_table, b, skv, layout_kv)
        return q_b, k_b

    def _compute_tiled_single(
        self,
        q: torch.Tensor,  # (Sq, D) input dtype
        k: torch.Tensor,  # (Skv, D) input dtype
        v: torch.Tensor,  # (Skv, DV) input dtype
        mask: Optional[torch.Tensor],  # (Sq, Skv) bool, True=mask out
        scale: float,
        mask_mode: int = 0,
        delta: int = 0,
        win_left: int = -1,
        win_right: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Online softmax with input-dtype Cube operands and FP32 accumulators."""

        def mask_fn(qs, qe, ks, ke):
            if mask_mode in (3, 4):
                q_idx = torch.arange(qs, qe, device=q.device)
                k_idx = torch.arange(ks, ke, device=q.device)
                return self._build_mask_mode_mask(
                    q_idx, k_idx, delta, mask_mode, win_left, win_right
                )
            return None if mask is None else mask[qs:qe, ks:ke]

        return self._attention_single(
            q,
            k,
            v,
            scale,
            self.TILE_SIZE,
            mask_fn,
            kv_bounds=(lambda qs, qe: (0, qe + delta)) if mask_mode == 3 else None,
        )

    def _compute_full_single(
        self,
        q: torch.Tensor,  # (Sq, D) input dtype
        k: torch.Tensor,  # (Skv, D) input dtype
        v: torch.Tensor,  # (Skv, DV) input dtype
        mask: Optional[torch.Tensor],  # (Sq, Skv) bool
        scale: float,
        sink: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-tile mixed-precision reference; exp weights round before PV."""
        return self._attention_single(
            q,
            k,
            v,
            scale,
            max(q.shape[0], k.shape[0], 1),
            lambda qs, qe, ks, ke: mask,
            sink,
        )

    @staticmethod
    def _build_mask_mode_mask(
        q_idx: torch.Tensor,
        k_idx: torch.Tensor,
        delta: int,
        mask_mode: int = 3,
        win_left: int = -1,
        win_right: int = -1,
    ) -> Optional[torch.Tensor]:
        """Build a right-aligned mask; True marks masked positions."""
        q_pos = q_idx.unsqueeze(1) + delta
        k_pos = k_idx.unsqueeze(0)
        if mask_mode == 3:
            return k_pos > q_pos
        masked = torch.zeros(
            (q_idx.numel(), k_idx.numel()), dtype=torch.bool, device=q_idx.device
        )
        if win_left >= 0:
            masked |= k_pos < q_pos - win_left
        if win_right >= 0:
            masked |= k_pos > q_pos + win_right
        return masked

    def _build_batch_mask(
        self,
        sq: int,
        skv: int,
        mask_mode: int,
        attn_mask: Optional[torch.Tensor],
        device: torch.device,
        batch_index: int,
        head_index: int = 0,
        win_left: int = -1,
        win_right: int = -1,
    ) -> Optional[torch.Tensor]:
        """Build (sq, skv) bool mask for a single batch. True = mask out."""
        if mask_mode in (3, 4):
            q_idx = torch.arange(sq, device=device)
            k_idx = torch.arange(skv, device=device)
            return self._build_mask_mode_mask(
                q_idx, k_idx, skv - sq, mask_mode, win_left, win_right
            )
        if attn_mask is not None:
            if attn_mask.dim() == 4:
                attn_mask = attn_mask[
                    0 if attn_mask.shape[0] == 1 else batch_index,
                    0 if attn_mask.shape[1] == 1 else head_index,
                ]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[0 if attn_mask.shape[0] == 1 else batch_index]
            return attn_mask[:sq, :skv].to(device=device, dtype=torch.bool)
        return None


def cpu_flash_mla(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
    **unused,
):
    arguments = dict(locals())
    from .cross_check import enabled, get_case

    if enabled():
        return get_case(unused["testcase_name"])[0]
    arguments.pop("unused")
    arguments.pop("metadata")
    for name, value in arguments.items():
        if torch.is_tensor(value):
            arguments[name] = value.detach().cpu()
    out, lse = FlashMlaGolden().forward_golden(**arguments, device_mode="cpu")
    # The kernel does not define the LSE output when it is disabled.
    return out, lse if return_softmax_lse else None
