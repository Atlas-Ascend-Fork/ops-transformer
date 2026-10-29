# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes
import os
from pathlib import Path

import numpy as np

ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT32 = 3
ACL_INT64 = 9
ACL_BF16 = 27
ACL_FORMAT_ND = 2

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

DTYPE_TO_ACL = {
    np.float32: ACL_FLOAT,
    np.float16: ACL_FLOAT16,
    np.uint16: ACL_BF16,
    np.int32: ACL_INT32,
    np.int64: ACL_INT64,
}

C_INT64_P = ctypes.POINTER(ctypes.c_int64)
C_VOID_P_P = ctypes.POINTER(ctypes.c_void_p)


def _first_existing(paths):
    for path in paths:
        if path and Path(path).is_file():
            return Path(path)
    return None


def _cann_lib_dirs():
    roots = [
        os.environ.get("ASCEND_HOME_PATH"),
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/cann-8.5.0",
        "/usr/local/Ascend/latest",
    ]
    dirs = []
    for root in roots:
        if not root:
            continue
        for sub in ("lib64", "lib", "runtime/lib64"):
            path = Path(root) / sub
            if path.is_dir() and path not in dirs:
                dirs.append(path)
    for item in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        path = Path(item)
        if item and path.is_dir() and path not in dirs:
            dirs.append(path)
    return dirs


def _custom_opapi_candidates():
    candidates = []
    explicit = os.environ.get("CUSTOM_OPAPI_LIB")
    if explicit:
        candidates.append(Path(explicit))

    for root in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if root:
            candidates.append(Path(root) / "op_api" / "lib" / "libcust_opapi.so")

    repo_root = Path(__file__).resolve().parents[5]
    for root in (repo_root / "build_out", repo_root / "output", repo_root / "build"):
        if not root.is_dir():
            continue
        for path in root.rglob("libcust_opapi.so"):
            candidates.append(path)

    for lib_dir in _cann_lib_dirs():
        opp_root = lib_dir.parent / "opp" / "vendors"
        if opp_root.is_dir():
            for path in opp_root.glob("*/op_api/lib/libcust_opapi.so"):
                candidates.append(path)
    return candidates


def _load_library(name, extra_dirs=None):
    dirs = list(extra_dirs or []) + _cann_lib_dirs()
    for directory in dirs:
        path = Path(directory) / name
        if path.is_file():
            return ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)


class AclnnTensor:
    def __init__(self, runtime, array):
        self.runtime = runtime
        self.host = np.ascontiguousarray(array)
        self.shape = list(self.host.shape)
        self.strides = [1] * len(self.shape)
        for index in range(len(self.shape) - 2, -1, -1):
            self.strides[index] = self.shape[index + 1] * self.strides[index + 1]
        self.nbytes = int(self.host.nbytes)
        self.device = runtime.malloc(self.nbytes) if self.nbytes else ctypes.c_void_p()
        if self.nbytes:
            runtime.h2d(self.device, self.host)

        self._shape = (ctypes.c_int64 * len(self.shape))(*self.shape)
        self._strides = (ctypes.c_int64 * len(self.strides))(*self.strides)
        self._storage = (ctypes.c_int64 * len(self.shape))(*self.shape)
        acl_dtype = DTYPE_TO_ACL[self.host.dtype.type]
        self.ptr = runtime.nnopbase.aclCreateTensor(
            ctypes.cast(self._shape, C_INT64_P),
            len(self.shape),
            acl_dtype,
            ctypes.cast(self._strides, C_INT64_P),
            0,
            ACL_FORMAT_ND,
            ctypes.cast(self._storage, C_INT64_P),
            len(self.shape),
            self.device,
        )
        if not self.ptr:
            raise RuntimeError("aclCreateTensor failed")

    def copy_back(self, dtype, shape):
        output = np.empty(shape, dtype=dtype)
        if output.nbytes:
            self.runtime.d2h(output, self.device)
        return output

    def close(self):
        if getattr(self, "ptr", None):
            self.runtime.nnopbase.aclDestroyTensor(self.ptr)
            self.ptr = None
        if getattr(self, "device", None):
            self.runtime.free(self.device)
            self.device = None


class AclnnRuntime:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.acl = _load_library("libascendcl.so")
        self.nnopbase = _load_library("libnnopbase.so")
        opapi = _first_existing(_custom_opapi_candidates())
        self.opapi_path = str(opapi) if opapi else None
        self.opapi = _load_library(
            "libcust_opapi.so", [opapi.parent] if opapi else None
        )

        self.acl.aclInit.argtypes = [ctypes.c_char_p]
        self.acl.aclInit.restype = ctypes.c_int
        self.acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
        self.acl.aclrtSetDevice.restype = ctypes.c_int
        self.acl.aclrtCreateStream.argtypes = [C_VOID_P_P]
        self.acl.aclrtCreateStream.restype = ctypes.c_int
        self.acl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
        self.acl.aclrtSynchronizeStream.restype = ctypes.c_int
        self.acl.aclrtMalloc.argtypes = [C_VOID_P_P, ctypes.c_size_t, ctypes.c_int32]
        self.acl.aclrtMalloc.restype = ctypes.c_int
        self.acl.aclrtFree.argtypes = [ctypes.c_void_p]
        self.acl.aclrtFree.restype = ctypes.c_int
        self.acl.aclrtMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
        ]
        self.acl.aclrtMemcpy.restype = ctypes.c_int

        self.nnopbase.aclCreateTensor.argtypes = [
            C_INT64_P,
            ctypes.c_uint64,
            ctypes.c_int32,
            C_INT64_P,
            ctypes.c_int64,
            ctypes.c_int32,
            C_INT64_P,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.nnopbase.aclCreateTensor.restype = ctypes.c_void_p
        self.nnopbase.aclDestroyTensor.argtypes = [ctypes.c_void_p]
        self.nnopbase.aclDestroyTensor.restype = ctypes.c_int

        if self.acl.aclInit(None) not in (0, 100002, 500000):
            raise RuntimeError("aclInit failed")
        if self.acl.aclrtSetDevice(0) not in (0, 100002, 500000, 107001):
            raise RuntimeError("aclrtSetDevice failed")
        self.stream = ctypes.c_void_p()
        if self.acl.aclrtCreateStream(ctypes.byref(self.stream)) != 0:
            raise RuntimeError("aclrtCreateStream failed")

    def malloc(self, size):
        device = ctypes.c_void_p()
        if (
            self.acl.aclrtMalloc(ctypes.byref(device), size, ACL_MEM_MALLOC_HUGE_FIRST)
            != 0
        ):
            raise RuntimeError("aclrtMalloc failed")
        return device

    def free(self, device):
        if device:
            self.acl.aclrtFree(device)

    def h2d(self, device, array):
        size = int(array.nbytes)
        source = array.ctypes.data_as(ctypes.c_void_p)
        if (
            self.acl.aclrtMemcpy(device, size, source, size, ACL_MEMCPY_HOST_TO_DEVICE)
            != 0
        ):
            raise RuntimeError("aclrtMemcpy H2D failed")

    def d2h(self, array, device):
        size = int(array.nbytes)
        target = array.ctypes.data_as(ctypes.c_void_p)
        if (
            self.acl.aclrtMemcpy(target, size, device, size, ACL_MEMCPY_DEVICE_TO_HOST)
            != 0
        ):
            raise RuntimeError("aclrtMemcpy D2H failed")

    def sync(self):
        self.acl.aclrtSynchronizeStream(self.stream)

    def tensor(self, array):
        return AclnnTensor(self, array)

    def set_opapi(self, name, argtypes):
        try:
            function = getattr(self.opapi, name)
        except AttributeError:
            for candidate in _custom_opapi_candidates():
                if not candidate.is_file():
                    continue
                try:
                    library = ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                    function = getattr(library, name)
                except (OSError, AttributeError):
                    continue
                self.opapi = library
                self.opapi_path = str(candidate)
                break
            else:
                raise RuntimeError(
                    f"{name} not found in libcust_opapi.so; build and install the operator package first"
                )
        function.restype = ctypes.c_int
        function.argtypes = argtypes
        return function

    def get_workspace(self, function, *args):
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        ret = function(*args, ctypes.byref(workspace_size), ctypes.byref(executor))
        return ret, workspace_size.value, executor

    def execute(self, function, workspace_size, executor):
        workspace = self.malloc(workspace_size) if workspace_size else ctypes.c_void_p()
        try:
            ret = function(workspace, workspace_size, executor, self.stream)
            self.sync()
            return ret
        finally:
            self.free(workspace)
