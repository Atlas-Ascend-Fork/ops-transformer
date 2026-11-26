#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""compressor_grad 大值边界回归（arch22 / A3/A2 原生 AscendC）。

背景：arch22 的 bf16-io 工作区统一落盘为 fp16（11-bit 尾数），合法 bf16 输入
（|v|>65504）会超出 fp16 范围。实现上以全局 2 的幂缩放解决：普通数据（全部输入
<= 2^15）指数为 0，数值路径逐位不变；超出时按 2^-a（dn）/ 2^-b（x/wkv/wgate）
缩放后落盘，GEMM 输出再乘 2^(a+b) 还原（2 的幂乘除无舍入）。本文件覆盖：

1. bf16 合法输入 ~1e5（BSH/TH，逐操作数）：四输出必须有限，且与官方 cpu golden
   在 mixed_tolerance 判定下一致（缩放路径不引入可见误差）。
2. 极端量级（kv=±1e35 等）：四输出有限且量级/符号正确。
3. fp16 HiLo 范围悬崖：|dn|>65504 时落盘为 ±Inf 而非 NaN（与 golden/A5 单平面
   cast 一致）。
4. 中间量探针（恒等权重暴露 dn_sc）、padding 行投毒、双跑逐位（确定性）。

另有 fp16 普通幅值非回归（与 high@fp64 golden 在 fp16 容差内一致）。

注：arch35（A5）/arch92（960dt）的 kernel 本文件不覆盖，运行时自动跳过。

运行：pytest test_compressor_grad_large_value.py -q
"""

import torch
import pytest

from compressor_grad_cpu_golden import compressor_grad_golden
from compressor_grad_check import check_result, get_pct_thd
from cann_ops_transformer.ops.attention.compressor.compressor import (
    _compressor_backward,
)

_DT = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _device(request):
    return f"npu:{request.config.getoption('--device')}"


def _skip_if_not_arch22():
    """本文件只针对 arch22（A3/A2）；A5/960dt 的 arch35/arch92 未修正，跳过。"""
    try:
        from test_compressor_grad import _KERNEL_ARCH
    except Exception:  # pragma: no cover - 仅本地无 custom 包时走到这里
        return
    if _KERNEL_ARCH is not None:
        pytest.skip("large-value regression targets arch22 (A3/A2) native AscendC")


def _gen_bsh(B, S, H, D, cr, coff, dtype, seed=20261119, scale=1.0, kv_scale=None):
    """BSH 反向输入：x (B*S,H)、dc (B,nblk,D)、sm (B,nblk,N,D)、kv (B,nblk,N,D)。"""
    g = torch.Generator().manual_seed(seed)
    td = _DT[dtype]
    C = coff * D
    nblk = (S + cr - 1) // cr
    N = coff * cr
    ksc = scale if kv_scale is None else kv_scale
    x = (torch.randn(B * S, H, generator=g) * scale).to(td)
    wkv = (torch.randn(C, H, generator=g) * scale).to(td)
    wgate = (torch.randn(C, H, generator=g) * scale).to(td)
    dc = (torch.randn(B, nblk, D, generator=g) * scale).to(td)
    sm = torch.softmax(torch.randn(B, nblk, N, D, generator=g), dim=2)
    kv = torch.randn(B, nblk, N, D, generator=g) * ksc
    return x, wkv, wgate, dc, sm, kv


def _gen_th(T, lengths, H, D, cr, coff, dtype, seed=20261119, scale=1.0, kv_scale=None):
    """TH 反向输入：x (T,H)、dc (rows,D)、sm/kv (rows,N,D)、cu_seqlens (B+1,)。"""
    g = torch.Generator().manual_seed(seed)
    td = _DT[dtype]
    C = coff * D
    N = coff * cr
    B = len(lengths)
    rows = min(T, T // cr + B)  # 官方 outputRows 公式
    ksc = scale if kv_scale is None else kv_scale
    x = (torch.randn(T, H, generator=g) * scale).to(td)
    wkv = (torch.randn(C, H, generator=g) * scale).to(td)
    wgate = (torch.randn(C, H, generator=g) * scale).to(td)
    dc = (torch.randn(rows, D, generator=g) * scale).to(td)
    sm = torch.softmax(torch.randn(rows, N, D, generator=g), dim=1)
    kv = torch.randn(rows, N, D, generator=g) * ksc
    cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lengths), 0).tolist()), dtype=torch.int32
    )
    return x, wkv, wgate, dc, sm, kv, cu


def _run_bsh(x, wkv, wgate, dc, sm, kv, cr, coff, device):
    return _compressor_backward(
        dc.to(device),
        x.to(device),
        wkv.to(device),
        wgate.to(device),
        sm.to(device),
        kv.to(device),
        cmp_ratio=cr,
        coff=coff,
    )


def _run_th(x, wkv, wgate, dc, sm, kv, cu, cr, coff, device):
    return _compressor_backward(
        dc.to(device),
        x.to(device),
        wkv.to(device),
        wgate.to(device),
        sm.to(device),
        kv.to(device),
        cmp_ratio=cr,
        coff=coff,
        cu_seqlens=cu.to(device),
    )


def _assert_all_finite(outs):
    for name, out in zip(("d_x", "d_wkv", "d_wgate", "d_ape"), outs):
        assert torch.isfinite(out.float()).all().item(), f"{name} 含 NaN/Inf"


def _assert_match(outs, golden, dtype):
    names = ("d_x", "d_wkv", "d_wgate")
    pct = get_pct_thd(dtype)
    # 统一转 fp32 再比对：阈值仍按 dtype 取，避免部分 torch 版本
    # numpy() 不支持 bfloat16 标量类型
    for name, expect, result in zip(names, golden[:3], outs[:3]):
        _, verdict = check_result(expect.float(), result.float(), dtype, pct)
        assert verdict == "Pass", f"{name} 与官方 golden 不一致"
    _, verdict = check_result(
        golden[3].float(), outs[3].float(), "float32", get_pct_thd("float32")
    )
    assert verdict == "Pass", "d_ape 与官方 golden 不一致"


@pytest.mark.parametrize(
    "B,S,H,D,cr,coff,dtype",
    [
        (2, 256, 1024, 128, 8, 1, "bfloat16"),
        (1, 512, 2048, 128, 16, 2, "bfloat16"),
        (2, 256, 1024, 128, 8, 1, "float16"),
    ],
    ids=["bsh_bf16_cr8", "bsh_bf16_cr16_coff2", "bsh_fp16_cr8"],
)
def test_large_value_bsh(request, B, S, H, D, cr, coff, dtype):
    """BSH：输入 ~1e5（bf16 超过 fp16 上限 65504；dC 约 1e5）四输出有限并与
    官方 golden 一致；fp16 用例为普通幅值非回归。"""
    _skip_if_not_arch22()
    device = _device(request)
    torch.npu.set_device(device)
    scale = 1e5 if dtype == "bfloat16" else 1.0
    x, wkv, wgate, dc, sm, kv = _gen_bsh(B, S, H, D, cr, coff, dtype, scale=scale)
    x3 = x.view(B, S, H)  # BSH 直调接口的 x 为 3D (B,S,H)
    outs = _run_bsh(x3, wkv, wgate, dc, sm, kv, cr, coff, device)
    torch.npu.synchronize()
    _assert_all_finite(outs)
    if dtype == "bfloat16":
        golden = compressor_grad_golden(
            x=x3,
            wkv=wkv,
            wgate=wgate,
            d_cpm_kv=dc,
            softmax_score=sm,
            kv=kv,
            cmp_ratio=cr,
            coff=coff,
            compute_dtype=torch.float32,
            matmul_mode="same",
        )
    else:
        golden = compressor_grad_golden(
            x=x3,
            wkv=wkv,
            wgate=wgate,
            d_cpm_kv=dc,
            softmax_score=sm,
            kv=kv,
            cmp_ratio=cr,
            coff=coff,
            compute_dtype=torch.float64,
            matmul_mode="high",
        )
    _assert_match(outs, golden, dtype)


def test_large_value_th(request):
    """TH：x [T,H] + cu_seqlens，输入 ~1e5 四输出有限并与官方 golden 一致。"""
    _skip_if_not_arch22()
    device = _device(request)
    torch.npu.set_device(device)
    lengths = [64, 96]
    T, H, D, cr, coff = sum(lengths), 2048, 128, 8, 1
    x, wkv, wgate, dc, sm, kv, cu = _gen_th(
        T, lengths, H, D, cr, coff, "bfloat16", scale=1e5
    )
    outs = _run_th(x, wkv, wgate, dc, sm, kv, cu, cr, coff, device)
    torch.npu.synchronize()
    _assert_all_finite(outs)
    golden = compressor_grad_golden(
        x=x,
        wkv=wkv,
        wgate=wgate,
        d_cpm_kv=dc,
        softmax_score=sm,
        kv=kv,
        cu_seqlens=cu,
        cmp_ratio=cr,
        coff=coff,
        compute_dtype=torch.float64,
        matmul_mode="high",
    )
    _assert_match(outs, golden, "bfloat16")


def test_extreme_kv_finite(request):
    """极端 kv（±1e35）：四输出必须有限、量级/符号正确，且与 high@fp64 golden 一致。"""
    _skip_if_not_arch22()
    device = _device(request)
    torch.npu.set_device(device)
    B, S, H, D, cr, coff = 1, 8, 1024, 128, 2, 1
    C = coff * D
    N = coff * cr
    nblk = (S + cr - 1) // cr
    x = torch.zeros(B * S, H, dtype=torch.bfloat16)
    wkv = torch.zeros(C, H, dtype=torch.bfloat16)
    wgate = torch.zeros(C, H, dtype=torch.bfloat16)
    wgate[:, :C] = torch.eye(C, dtype=torch.bfloat16)
    dc = torch.ones(B, nblk, D, dtype=torch.bfloat16)
    sm = torch.full((B, nblk, N, D), 0.5, dtype=torch.float32)
    kv = torch.zeros(B, nblk, N, D, dtype=torch.float32)
    kv[:, :, 0, 0] = 1e35
    kv[:, :, 1, 0] = -1e35
    outs = _run_bsh(x.view(B, S, H), wkv, wgate, dc, sm, kv, cr, coff, device)
    torch.npu.synchronize()
    _assert_all_finite(outs)
    # 列 0 的 dn_sc = sm*kv = ±5e34（bf16 可表示），d_x 列 0 应同量级且符号正确
    dx0 = outs[0].float().reshape(-1, H)[:, 0]  # BSH 输出为 (B,S,H)，拉平取列 0
    assert torch.isfinite(dx0).all().item()
    assert (dx0.abs() > 1e34).all().item(), "大值相消结果幅值异常"
    assert dx0[0].item() > 0 and dx0[1].item() < 0, "大值相消符号错误"
    # 与官方 golden 对比：防"有限但错误"的实现（如漏缩放/高低位颠倒）
    golden = compressor_grad_golden(
        x=x.view(B, S, H),
        wkv=wkv,
        wgate=wgate,
        d_cpm_kv=dc,
        softmax_score=sm,
        kv=kv,
        cmp_ratio=cr,
        coff=coff,
        compute_dtype=torch.float64,
        matmul_mode="high",
    )
    _assert_match(outs, golden, "bfloat16")


def test_fp16_dn_overflow_inf(request):
    """fp16 HiLo 范围悬崖：d_score 超 fp16 上限时 hi=±Inf，staging 不得把 lo
    变成 ∓Inf（Inf-Inf=NaN 会让 GEMM 得 NaN，而 golden/A5 单平面 cast 是 Inf）。

    取 wkv=0、wgate=[I|0]，使 d_x 列 0 直接暴露 dn_sc：期望 [inf, -inf]。
    """
    _skip_if_not_arch22()
    device = _device(request)
    torch.npu.set_device(device)
    B, S, H, D, cr, coff = 1, 2, 1024, 128, 2, 1
    C = coff * D
    N = coff * cr
    nblk = 1
    x = torch.zeros(B * S, H, dtype=torch.float16)
    wkv = torch.zeros(C, H, dtype=torch.float16)
    wgate = torch.zeros(C, H, dtype=torch.float16)
    wgate[:, :C] = torch.eye(C, dtype=torch.float16)
    dc = torch.zeros(B, nblk, D, dtype=torch.float16)
    dc[:, :, 0] = 65504.0
    sm = torch.full((B, nblk, N, D), 0.5, dtype=torch.float32)
    kv = torch.zeros(B, nblk, N, D, dtype=torch.float32)
    kv[:, :, 0, 0] = 2e5  # d_score = ±0.5*65504*(2e5-1e5) = ±3.2752e9 > 65504
    outs = _run_bsh(x.view(B, S, H), wkv, wgate, dc, sm, kv, cr, coff, device)
    torch.npu.synchronize()
    dx0 = outs[0].float().reshape(-1, H)[:, 0]  # BSH 输出为 (B,S,H)，拉平取列 0
    assert torch.isinf(dx0).all().item(), "d_x 列 0 应为 ±Inf（HiLo 不得变 NaN）"
    assert dx0[0].item() > 0 and dx0[1].item() < 0, "d_x 列 0 符号应与 dn_sc 一致"
    # d_wkv/d_wgate 由 dn_kv 驱动（本构造为有限值），d_ape 为 fp32 有限
    assert torch.isfinite(outs[1].float()).all().item()
    assert torch.isfinite(outs[3].float()).all().item()


def test_dn_identity_probe_bitwise(request):
    """恒等权重探针：wkv=0、wgate=[I|0] 时 d_x 前 C 列就是 dn_sc。

    直接对中间量做逐位断言，能暴露被 bf16/容差吞掉的 1 ulp 级约定错误。
    """
    _skip_if_not_arch22()
    device = _device(request)
    torch.npu.set_device(device)
    B, S, H, D, cr, coff = 1, 64, 1024, 128, 8, 1
    C = coff * D
    x, _, wgate, dc, sm, kv = _gen_bsh(B, S, H, D, cr, coff, "bfloat16", kv_scale=10.0)
    x = x.view(B, S, H)
    wkv = torch.zeros(C, H, dtype=torch.bfloat16)
    wgate = torch.zeros(C, H, dtype=torch.bfloat16)
    wgate[:, :C] = torch.eye(C, dtype=torch.bfloat16)
    outs = _run_bsh(x.view(B, S, H), wkv, wgate, dc, sm, kv, cr, coff, device)
    torch.npu.synchronize()
    golden = compressor_grad_golden(
        x=x,
        wkv=wkv,
        wgate=wgate,
        d_cpm_kv=dc,
        softmax_score=sm,
        kv=kv,
        cmp_ratio=cr,
        coff=coff,
        compute_dtype=torch.float64,
        matmul_mode="high",
    )
    # d_x: NPU 为 (B,S,H)，golden 为 (T,H)，统一拉平到 (T,H) 再取前 C 列。
    # dn_sc 以 fp16 单平面落盘（~2^-11 相对），故按相对容差 4e-3 断言。
    dx_npu = outs[0].float().reshape(-1, H)[:, :C].cpu()
    dx_ref = golden[0].float().reshape(-1, H)[:, :C]
    denom = dx_ref.abs().clamp_min(1e-3)
    rel = ((dx_npu - dx_ref).abs() / denom).max().item()
    assert rel < 4e-3, f"d_x 前 C 列（=dn_sc）与 high golden 相对偏差 {rel:.3e} 超限"


@pytest.mark.parametrize(
    "tensor,scale",
    [
        ("x", 1e30),
        ("wkv", 1e30),
        ("wgate", 1e30),
        ("d_cmp_kv", 1e30),
        ("kv", 1e5),
        ("kv", 1e35),
    ],
    ids=["x_1e30", "wkv_1e30", "wgate_1e30", "dc_1e30", "kv_1e5", "kv_1e35"],
)
def test_large_value_per_operand(request, tensor, scale):
    """逐操作数大值隔离：只放大一个张量，其余正常，四输出有限并与 golden 一致。"""
    _skip_if_not_arch22()
    device = _device(request)
    torch.npu.set_device(device)
    B, S, H, D, cr, coff = 1, 64, 1024, 128, 8, 1
    x, wkv, wgate, dc, sm, kv = _gen_bsh(B, S, H, D, cr, coff, "bfloat16", kv_scale=2.0)
    vals = {"x": x, "wkv": wkv, "wgate": wgate, "d_cmp_kv": dc, "kv": kv}
    vals[tensor] = vals[tensor] * scale
    outs = _run_bsh(
        vals["x"].view(B, S, H),
        vals["wkv"],
        vals["wgate"],
        vals["d_cmp_kv"],
        sm,
        vals["kv"],
        cr,
        coff,
        device,
    )
    torch.npu.synchronize()
    _assert_all_finite(outs)
    golden = compressor_grad_golden(
        x=vals["x"].view(B, S, H),
        wkv=vals["wkv"],
        wgate=vals["wgate"],
        d_cpm_kv=vals["d_cmp_kv"],
        softmax_score=sm,
        kv=vals["kv"],
        cmp_ratio=cr,
        coff=coff,
        compute_dtype=torch.float64,
        matmul_mode="high",
    )
    _assert_match(outs, golden, "bfloat16")


def test_zero_gradient_kahan_deterministic(request):
    """S=2048（1024 块，触发 d_ape 每核跨块 Kahan）+ 相消零梯度：

    四输出应精确为 0；同一输入连跑两次四输出逐位一致（同时守 alias 类偶发脏读）。
    """
    _skip_if_not_arch22()
    device = _device(request)
    torch.npu.set_device(device)
    B, S, H, D, cr, coff = 1, 2048, 1024, 128, 2, 1
    C = coff * D
    N = coff * cr
    nblk = (S + cr - 1) // cr
    x = torch.zeros(B * S, H, dtype=torch.bfloat16)
    wkv = torch.zeros(C, H, dtype=torch.bfloat16)
    wgate = torch.zeros(C, H, dtype=torch.bfloat16)
    wgate[:, :C] = torch.eye(C, dtype=torch.bfloat16)
    dc = torch.zeros(B, nblk, D, dtype=torch.bfloat16)
    dc[:, :, 0] = 65536.0
    sm = torch.zeros(B, nblk, N, D, dtype=torch.float32)
    sm[:, :, 0, :] = 0.7874732613563538
    sm[:, :, 1, :] = 0.21252673864364624
    kv = torch.full((B, nblk, N, D), 8237.3447265625, dtype=torch.float32)
    runs = []
    x3 = x.view(B, S, H)
    for _ in range(2):
        outs = _run_bsh(x3, wkv, wgate, dc, sm, kv, cr, coff, device)
        torch.npu.synchronize()
        assert all(torch.equal(o, torch.zeros_like(o)) for o in outs), (
            "相消零梯度四输出应精确为 0"
        )
        runs.append(outs)
    for a, b in zip(*runs):
        assert torch.equal(a, b), "run-to-run 非逐位一致"


def test_padding_rows_poison(request):
    """padding 行投毒：nblk_in=10、实际只处理 8 块，把第 8/9 行置 1e30 后
    四输出必须与置 0 时逐位相同（不得读取未使用行）。"""
    _skip_if_not_arch22()
    device = _device(request)
    torch.npu.set_device(device)
    B, S, H, D, cr, coff = 1, 64, 1024, 128, 8, 1
    N = coff * cr
    nblk_in = 10
    base = _gen_bsh(B, S, H, D, cr, coff, "bfloat16", kv_scale=2.0)
    x, wkv, wgate, dc8, sm8, kv8 = base
    x = x.view(B, S, H)
    # 扩到 10 个块，未用行先置 0
    dc = torch.zeros(B, nblk_in, D, dtype=torch.bfloat16)
    dc[:, :8] = dc8
    sm = torch.zeros(B, nblk_in, N, D, dtype=torch.float32)
    sm[:, :8] = sm8
    kv = torch.zeros(B, nblk_in, N, D, dtype=torch.float32)
    kv[:, :8] = kv8
    outs0 = _run_bsh(x, wkv, wgate, dc, sm, kv, cr, coff, device)
    torch.npu.synchronize()
    dc2 = dc.clone()
    sm2 = sm.clone()
    kv2 = kv.clone()
    dc2[:, 8:] = 1e30
    sm2[:, 8:] = 1e30
    kv2[:, 8:] = 1e30
    outs1 = _run_bsh(x, wkv, wgate, dc2, sm2, kv2, cr, coff, device)
    torch.npu.synchronize()
    for name, a, b in zip(("d_x", "d_wkv", "d_wgate", "d_ape"), outs0, outs1):
        assert torch.equal(a, b), f"{name} 受 padding 行影响（读取了未使用行）"


def test_run_to_run_bitwise_random(request):
    """随机输入的 run-to-run 逐位一致（守 alias 类偶发脏读/非确定性）。"""
    _skip_if_not_arch22()
    device = _device(request)
    torch.npu.set_device(device)
    B, S, H, D, cr, coff = 1, 64, 1024, 128, 8, 1
    x, wkv, wgate, dc, sm, kv = _gen_bsh(B, S, H, D, cr, coff, "bfloat16", kv_scale=5.0)
    x3 = x.view(B, S, H)
    runs = []
    for _ in range(2):
        outs = _run_bsh(x3, wkv, wgate, dc, sm, kv, cr, coff, device)
        torch.npu.synchronize()
        runs.append(outs)
    for name, a, b in zip(("d_x", "d_wkv", "d_wgate", "d_ape"), runs[0], runs[1]):
        assert torch.equal(a, b), f"{name} run-to-run 非逐位一致"


_TINY_KV = 2.0**-114 + 2.0**-137
