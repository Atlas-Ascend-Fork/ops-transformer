# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pure import-time layout primitives shared by the FAG resource families.

The allocation policy stays in each family's fag_mem module: explicit mutex
reuse/reservation and CV tables must not leak between the resource families.
Only the resulting integer/list tables are consumed by kernel trace helpers.
"""


class _MutexPoolBase:
    """Common owner bookkeeping; subclasses keep their allocation policy."""

    LIMIT = 32

    def __init__(self, side, tag, cv_reserved, cv_mutex):
        self._side, self._tag = side, tag
        self._next = cv_reserved
        self.owners = {i: f"{n}(CV)" for n, ids in cv_mutex.items() for i in ids}

    def _take(self, label, count, check_reserved=False):
        if self._next + count > self.LIMIT:
            raise AssertionError(
                f"{self._side}({self._tag}) mutex overflow: "
                f"need {self._next + count} ids, LIMIT={self.LIMIT}"
            )
        ids = list(range(self._next, self._next + count))
        if check_reserved and any(i in self.owners for i in ids):
            raise AssertionError(
                f"{self._side}({self._tag}) mutex reservation collision: {ids}"
            )
        self._next += count
        for i in ids:
            self.owners[i] = label
        return ids

    def report(self):
        free = sorted(set(range(self.LIMIT)) - set(self.owners))
        lines = [
            f"{self._side}({self._tag}): 用了 {len(self.owners)}/{self.LIMIT}，"
            f"空闲 {len(free)} 个 -> {free}"
        ]
        for i in sorted(self.owners):
            lines.append(f"   {i:>2}  {self.owners[i]}")
        return "\n".join(lines)


class _ArenaBase:
    """Common address placement; subclasses keep their buffer-id policy."""

    def __init__(self, name, capacity, pool, base=0):
        self._name = name
        self._cap = capacity
        self._pool = pool
        self._cur = base
        self.map = []  # [(标签, 地址, 字节数, mutex id)]

    def _declare_buffer(self, label, nbytes, slots, align, cv, cv_mutex, ids=None):
        """Allocate an address/id record using the caller's explicit policy."""
        if ids is None:
            ids = list(cv_mutex[cv]) if cv else self._pool.take(label, slots)
        else:
            ids = list(ids)
            for i in ids:
                self._pool.owners[i] = f"{label}(reuse)"
        return {"addrs": self._place(label, nbytes, len(ids), align, ids), "mutex": ids}

    def _place(self, label, nbytes, count, align, ids):
        addrs = []
        for i in range(count):
            self._cur = (self._cur + align - 1) & ~(align - 1)
            addrs.append(self._cur)
            self.map.append(
                (f"{label}[{i}]" if count > 1 else label, self._cur, nbytes, ids[i])
            )
            self._cur += nbytes
        if self._cur > self._cap:
            raise AssertionError(
                f"{self._name}: used 0x{self._cur:X} > cap 0x{self._cap:X}"
            )
        return addrs

    def describe(self):
        def size(n):
            return f"{n // 1024}KB" if n >= 1024 else f"{n}B"

        lines = [
            f"{self._name}: 用到 0x{self._cur:06X}"
            f"（{size(self._cur)} / {size(self._cap)}，"
            f"余 {size(self._cap - self._cur)}）"
        ]
        for label, addr, nbytes, mid in self.map:
            lock = f"mutex {mid}" if mid is not None else ""
            lines.append(f"  0x{addr:06X}  {size(nbytes):>5}  {label:<12}{lock}")
        return "\n".join(lines)


def _addrs(table, field):
    """``{档: {字段: buffer}}`` → ``{档: 地址列表}``。"""
    return {w: table[w][field]["addrs"] for w in table}


def _addr(table, field):
    """单槽 buffer 的标量地址表。"""
    return {w: table[w][field]["addrs"][0] for w in table}


def _mutex(table, field):
    """``{档: {字段: buffer}}`` → ``{档: id 列表}``。与 _addrs 同源，不会对不上。"""
    return {w: table[w][field]["mutex"] for w in table}
