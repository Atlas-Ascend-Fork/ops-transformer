# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU reference and legal scatter inputs for the Ascend 950 V1 scenario.

Both ACLNN interfaces support this scenario; V2 must receive vOptional=None.
The reference follows the documented RMSNorm/interleaved RoPE equations.
It does not import TTK or call the operator under test.
Promote requires TTK's original_*_dtypes metadata to preserve storage semantics.
"""

import numpy as np

__spec__ = {
    "kv_rms_norm_rope_cache": "KvRmsNormRopeCacheTestSpec",
    "aclnnKvRmsNormRopeCache": "AclnnKvRmsNormRopeCacheTestSpec",
    "aclnnKvRmsNormRopeCacheV2": "AclnnKvRmsNormRopeCacheV2TestSpec",
}

_MODES = ("Norm", "PA", "PA_BNSD", "PA_NZ", "PA_BLK_BNSD", "PA_BLK_NZ")
_FLOAT8_MAX = {"float8_e4m3fn": 448.0, "float8_e5m2": 57344.0, "hifloat8": 32768.0}


def _numpy_dtype(name):
    name = str(name).replace("torch.", "", 1)
    if name in ("bfloat16", "float8_e4m3fn", "float8_e5m2"):
        import ml_dtypes

        return np.dtype(getattr(ml_dtypes, name))
    if name == "hifloat8":
        from en_dtypes import hifloat8

        return np.dtype(hifloat8)
    return np.dtype(name)


def _numpy(value):
    if value is None or isinstance(value, np.ndarray):
        return value
    value = value.detach().cpu()
    dtype = _numpy_dtype(value.dtype)
    if dtype.name in ("bfloat16", "float8_e4m3fn", "float8_e5m2"):
        return value.float().numpy().astype(dtype)
    return value.numpy()


def _torch_outputs(outputs, use_numpy):
    if use_numpy:
        return outputs
    import torch

    result = []
    for output in outputs:
        if output is None:
            result.append(None)
        elif output.dtype.name in ("bfloat16", "float8_e4m3fn", "float8_e5m2"):
            result.append(
                torch.from_numpy(output.astype(np.float32)).to(
                    getattr(torch, output.dtype.name)
                )
            )
        else:
            result.append(torch.from_numpy(output))
    return result


def _index_geometry(kv, index, k_cache, ckv_cache, cache_mode):
    if cache_mode not in _MODES:
        raise ValueError(f"Unsupported cache_mode: {cache_mode}")
    if len(kv.shape) != 4 or kv.shape[1] != 1:
        raise ValueError("The Ascend 950 V1 reference requires kv in [B, 1, S, D]")
    if len(k_cache.shape) != 4 or k_cache.shape[:3] != ckv_cache.shape[:3]:
        raise ValueError(
            "K and V caches must have matching four-dimensional slot geometry"
        )
    batch, _, sequence, _ = kv.shape
    if cache_mode == "Norm":
        if k_cache.shape[:2] != kv.shape[:2]:
            raise ValueError("Norm cache batch/head dimensions must match kv")
        expected = (batch, sequence)
        if tuple(index.shape) != expected and not (
            sequence == 1 and tuple(index.shape) == (batch,)
        ):
            raise ValueError(f"Norm index must have shape {expected}, or [B] when S=1")
        return batch, sequence, k_cache.shape[2], 1
    blocks, block_size, heads, _ = k_cache.shape
    if block_size <= 0 or heads != kv.shape[1]:
        raise ValueError("Invalid PA cache block size or head dimension")
    blocked = cache_mode.startswith("PA_BLK")
    count = batch * ((sequence + block_size - 1) // block_size if blocked else sequence)
    if tuple(index.shape) != (count,):
        raise ValueError(f"{cache_mode} index must have shape ({count},)")
    return (
        1,
        count,
        blocks if blocked else blocks * block_size,
        block_size if blocked else 1,
    )


def _check_indices(index, groups, count, capacity, stride):
    if index.dtype != np.dtype("int64"):
        raise ValueError("index must have dtype int64")
    rows = index.reshape(groups, count)
    if np.any(rows < -1) or np.any(rows >= capacity * stride):
        raise ValueError(
            "index is outside the cache range; only -1 denotes a skipped write"
        )
    for row in rows:
        slots = row[row >= 0] // stride
        if np.unique(slots).size != slots.size:
            raise ValueError(
                "Duplicate cache destinations have no deterministic scatter reference"
            )


def _generate_index(kv, index, k_cache, ckv_cache, cache_mode):
    """Keep valid explicit indices; replace random invalid indices safely.

    Sampling uses O(number of tokens) memory, even for very large backing
    caches. Excess tokens and existing -1 entries remain skipped.
    """
    geometry = _index_geometry(kv, index, k_cache, ckv_cache, cache_mode)
    original = _numpy(index)
    try:
        _check_indices(original, *geometry)
        return original.copy()
    except ValueError:
        if original.dtype != np.dtype("int64"):
            raise
    groups, count, capacity, stride = geometry
    result = np.full((groups, count), -1, dtype=original.dtype)
    for source, row in zip(original.reshape(groups, count), result):
        positions = np.flatnonzero(source != -1)
        np.random.shuffle(positions)
        positions = positions[:capacity]
        # Floyd's sampling avoids np.random.permutation(capacity), which can
        # allocate gigabytes for a small view into a large shared KV cache.
        chosen = set()
        slots = []
        for upper in range(capacity - len(positions), capacity):
            candidate = int(np.random.randint(upper + 1))
            slot = upper if candidate in chosen else candidate
            chosen.add(slot)
            slots.append(slot * stride)
        np.random.shuffle(slots)
        row[positions] = slots
    return result.reshape(original.shape)


def _scatter(cache, values, index, cache_mode):
    result = cache.copy()
    batch, heads, sequence, width = values.shape
    if cache_mode == "Norm":
        for b, row in enumerate(index.reshape(batch, sequence)):
            valid = row >= 0
            result[b].transpose(1, 0, 2)[row[valid]] = values[b].transpose(1, 0, 2)[
                valid
            ]
        return result
    blocks, block_size = cache.shape[:2]
    if cache_mode.startswith("PA_BLK"):
        chunks = (sequence + block_size - 1) // block_size
        base = np.repeat(index.reshape(batch, chunks), block_size, axis=1)[:, :sequence]
        destinations = np.where(
            base < 0,
            -1,
            base // block_size * block_size + np.arange(sequence) % block_size,
        )
    else:
        destinations = index
    destinations = destinations.reshape(-1)
    valid = destinations >= 0
    destinations = destinations[valid]
    tokens = values.transpose(0, 2, 1, 3).reshape(batch * sequence, heads, width)[valid]
    if cache_mode.endswith("NZ"):
        # ND tensor metadata carries an NZ-packed byte buffer. C0 depends
        # independently on each cache's dtype, including mixed K/V quantization.
        c0 = 32 // cache.dtype.itemsize
        if width % c0:
            raise ValueError("NZ cache head dimension must be 32-byte aligned")
        physical = result.reshape(blocks, heads, width // c0, block_size, c0)
        physical[destinations // block_size, :, :, destinations % block_size, :] = (
            tokens.reshape(-1, heads, width // c0, c0)
        )
    else:
        result.reshape(blocks * block_size, heads, width)[destinations] = tokens
    return result


def _quantize(values, scale, offset, dtype):
    quantized = dtype.name == "int8" or dtype.name in _FLOAT8_MAX
    if not quantized:
        if scale is not None or offset is not None:
            raise ValueError("Non-quantized caches require scale=None and offset=None")
        return values.astype(dtype)
    if scale is None:
        raise ValueError("Quantized caches require a scale")
    values = values * scale
    if offset is not None:
        values = values + offset
    if dtype.name == "int8":
        # RINT (ties to even), then saturate; quantization multiplies by scale.
        return np.clip(np.rint(values), -128, 127).astype(dtype)
    limit = _FLOAT8_MAX[dtype.name]
    return np.clip(values, -limit, limit).astype(dtype)


def kv_rms_norm_rope_cache_golden(
    kv,
    gamma,
    cos,
    sin,
    index,
    k_cache,
    ckv_cache,
    k_rope_scale=None,
    c_kv_scale=None,
    k_rope_offset=None,
    c_kv_offset=None,
    v=None,
    epsilon=1e-5,
    cache_mode="Norm",
    is_output_kv=False,
    **kwargs,
):
    """RMSNorm + interleaved RoPE + scatter; independently checked by analytic tests."""
    if v is not None:
        raise ValueError("Ascend 950 supports the V1 scenario only: v must be None")
    if (
        kwargs.get("golden_mode") == "Promote"
        and "original_output_dtypes" not in kwargs
    ):
        raise ValueError("Promote requires TTK original_output_dtypes metadata")
    geometry = _index_geometry(kv, index, k_cache, ckv_cache, cache_mode)
    _check_indices(index, *geometry)
    output_dtypes = kwargs.get("original_output_dtypes", kwargs.get("output_dtypes"))
    if output_dtypes:
        # Promote may change input carriers; physical cache layout still follows
        # the declared output dtype, not the temporary computation dtype.
        k_cache = k_cache.astype(_numpy_dtype(output_dtypes[0]))
        ckv_cache = ckv_cache.astype(_numpy_dtype(output_dtypes[1]))
    compute_dtype = np.float64 if kv.dtype == np.float64 else np.float32
    x = kv.astype(compute_dtype)
    dv, dk = gamma.size, cos.shape[-1]
    if dv + dk != kv.shape[-1] or dk % 2:
        raise ValueError("kv last dimension must equal Dv + Dk, with even Dk")
    rms_input, rope_input = x[..., :dv], x[..., dv:]
    rms = rms_input / np.sqrt(
        np.mean(rms_input * rms_input, axis=-1, keepdims=True) + epsilon
    )
    rms = rms * gamma.astype(compute_dtype).reshape(-1)
    even, odd = rope_input[..., ::2], rope_input[..., 1::2]
    rope = np.concatenate((even, odd), axis=-1) * cos.astype(compute_dtype)
    rope += np.concatenate((-odd, even), axis=-1) * sin.astype(compute_dtype)
    k_values = _quantize(rope, k_rope_scale, k_rope_offset, k_cache.dtype)
    v_values = _quantize(rms, c_kv_scale, c_kv_offset, ckv_cache.dtype)
    return [
        _scatter(k_cache, k_values, index, cache_mode),
        _scatter(ckv_cache, v_values, index, cache_mode),
        rope.astype(kv.dtype) if is_output_kv else None,
        rms.astype(kv.dtype) if is_output_kv else None,
    ]


def kv_rms_norm_rope_cache_inputs(
    kv,
    gamma,
    cos,
    sin,
    index,
    k_cache,
    ckv_cache,
    k_rope_scale=None,
    c_kv_scale=None,
    k_rope_offset=None,
    c_kv_offset=None,
    v=None,
    epsilon=1e-5,
    cache_mode="Norm",
    is_output_kv=False,
    **kwargs,
):
    index = _generate_index(kv, index, k_cache, ckv_cache, cache_mode)
    return (
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache,
        ckv_cache,
        k_rope_scale,
        c_kv_scale,
        k_rope_offset,
        c_kv_offset,
        v,
    )


def _aclnn_golden(inputs, epsilon, cache_mode, is_output_kv, kwargs):
    if (
        kwargs.get("golden_mode") == "Promote"
        and "original_tensor_dtypes" not in kwargs
    ):
        raise ValueError("Promote requires TTK original_tensor_dtypes metadata")
    arrays = [_numpy(value) for value in inputs]
    metadata = kwargs.get("original_tensor_dtypes", kwargs.get("tensor_dtypes"))
    core_kwargs = {"output_dtypes": (metadata[5], metadata[6])} if metadata else {}
    outputs = kv_rms_norm_rope_cache_golden(
        *arrays,
        epsilon=epsilon,
        cache_mode=cache_mode,
        is_output_kv=is_output_kv,
        **core_kwargs,
    )
    return _torch_outputs(
        outputs, kwargs.get("use_numpy", isinstance(inputs[0], np.ndarray))
    )


def _aclnn_inputs(inputs, cache_mode):
    result = list(inputs)
    index = _generate_index(inputs[0], inputs[4], inputs[5], inputs[6], cache_mode)
    if isinstance(inputs[4], np.ndarray):
        result[4] = index
    else:
        import torch

        result[4] = torch.from_numpy(index).to(dtype=inputs[4].dtype)
    return tuple(result)


class KvRmsNormRopeCacheTestSpec:
    """Kernel/GEIR: numpy inputs in op_def order, including optional v."""

    golden = staticmethod(kv_rms_norm_rope_cache_golden)
    customize_inputs = staticmethod(kv_rms_norm_rope_cache_inputs)
    tolerance = {"int8": {"standard": "quant"}}


class AclnnKvRmsNormRopeCacheTestSpec:
    """ACLNN V1: parameters follow the complete GetWorkspaceSize prototype."""

    tolerance = {"int8": {"standard": "quant"}}

    @staticmethod
    def golden(
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache_ref,
        ckv_cache_ref,
        k_rope_scale_optional=None,
        c_kv_scale_optional=None,
        k_rope_offset_optional=None,
        c_kv_offset_optional=None,
        epsilon=1e-5,
        cache_mode_optional="Norm",
        is_output_kv=False,
        k_rope_out=None,
        c_kv_out=None,
        **kwargs,
    ):
        inputs = (
            kv,
            gamma,
            cos,
            sin,
            index,
            k_cache_ref,
            ckv_cache_ref,
            k_rope_scale_optional,
            c_kv_scale_optional,
            k_rope_offset_optional,
            c_kv_offset_optional,
        )
        return _aclnn_golden(inputs, epsilon, cache_mode_optional, is_output_kv, kwargs)

    @staticmethod
    def customize_inputs(
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache_ref,
        ckv_cache_ref,
        k_rope_scale_optional=None,
        c_kv_scale_optional=None,
        k_rope_offset_optional=None,
        c_kv_offset_optional=None,
        epsilon=1e-5,
        cache_mode_optional="Norm",
        is_output_kv=False,
        k_rope_out=None,
        c_kv_out=None,
        **kwargs,
    ):
        inputs = (
            kv,
            gamma,
            cos,
            sin,
            index,
            k_cache_ref,
            ckv_cache_ref,
            k_rope_scale_optional,
            c_kv_scale_optional,
            k_rope_offset_optional,
            c_kv_offset_optional,
        )
        return _aclnn_inputs(inputs, cache_mode_optional)


class AclnnKvRmsNormRopeCacheV2TestSpec:
    """ACLNN V2 entry, restricted to the V1 scenario supported on Ascend 950."""

    tolerance = {"int8": {"standard": "quant"}}

    @staticmethod
    def golden(
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache_ref,
        ckv_cache_ref,
        k_rope_scale_optional=None,
        c_kv_scale_optional=None,
        k_rope_offset_optional=None,
        c_kv_offset_optional=None,
        v_optional=None,
        epsilon=1e-5,
        cache_mode_optional="Norm",
        is_output_kv=False,
        k_rope_out=None,
        c_kv_out=None,
        **kwargs,
    ):
        inputs = (
            kv,
            gamma,
            cos,
            sin,
            index,
            k_cache_ref,
            ckv_cache_ref,
            k_rope_scale_optional,
            c_kv_scale_optional,
            k_rope_offset_optional,
            c_kv_offset_optional,
            v_optional,
        )
        return _aclnn_golden(inputs, epsilon, cache_mode_optional, is_output_kv, kwargs)

    @staticmethod
    def customize_inputs(
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache_ref,
        ckv_cache_ref,
        k_rope_scale_optional=None,
        c_kv_scale_optional=None,
        k_rope_offset_optional=None,
        c_kv_offset_optional=None,
        v_optional=None,
        epsilon=1e-5,
        cache_mode_optional="Norm",
        is_output_kv=False,
        k_rope_out=None,
        c_kv_out=None,
        **kwargs,
    ):
        inputs = (
            kv,
            gamma,
            cos,
            sin,
            index,
            k_cache_ref,
            ckv_cache_ref,
            k_rope_scale_optional,
            c_kv_scale_optional,
            k_rope_offset_optional,
            c_kv_offset_optional,
            v_optional,
        )
        return _aclnn_inputs(inputs, cache_mode_optional)
