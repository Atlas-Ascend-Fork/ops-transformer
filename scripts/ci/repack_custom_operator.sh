#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# repack_custom_operator.sh - 自定义算子 host/kernel 分离编译后的二次打包脚本
#
# 背景:
#   算子编译任务被拆分为 host 侧与 kernel 侧两个独立任务:
#     * host  侧: build.sh --pkg --jit --soc=<soc> -f <filelist> -j<N>   -> 只带 host 内容的 custom run 包
#     * kernel侧: build.sh --opkernel       --soc=<soc> -f <filelist> -j<N> -> 算子 *.o / *.json 等交付件
#   本脚本把 host run 包与 kernel 交付件合并, 重新打包成与"一次性编译"
#   (build.sh --pkg --soc=<soc> -f <filelist> -j<N>) 完全一致的 custom 算子 run 包。
#
# 合并规则 (与 cmake/custom_build.cmake 的安装布局保持一致):
#   <kernel-dir>/<op>/                  -> .../tbe/kernel/<soc>/<op>/
#   <kernel-dir>/<op>.json              -> .../tbe/kernel/config/<soc>/<op>.json
#   <kernel-dir>/binary_info_config.json-> .../tbe/kernel/config/<soc>/binary_info_config.json
#
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST_PKG=""
HOST_DIR=""
KERNEL_DIR=""
SOC=""
VENDOR_NAME=""
OUTPUT=""
MAKESELF_SH=""
HELP_HEADER=""
WORK_DIR=""
KEEP_WORK="false"
ARCH=""

DEFAULT_SOC="ascend950"
MAKESELF_LABEL="version:1.0"
MAKESELF_COMPLEVEL="4"

log()  { echo "[repack][$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
err()  { echo "[repack][ERROR] $*" >&2; }
die()  { err "$*"; exit 1; }

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS]

二次打包: 将 host 侧(--pkg --jit) run 包与 kernel 侧(--opkernel)交付件
合并为与一次性编译(--pkg)一致的 custom 算子 run 包。

必选 (二选一, 指定 host 侧输入):
  --host-pkg <file.run>   host 侧编译产生的 custom run 包
  --host-dir <dir>        已解压的 host 侧 run 包目录 (与 --host-pkg 互斥)

必选:
  --kernel-dir <dir>      kernel 侧交付件目录; 支持以下两种形式:
                            * build/binary/<soc>/bin           (推荐, 标准产物目录)
                            * build/binary/<soc>               (自动向下寻找 bin 目录)

可选:
  --soc <soc>             目标 SoC, 默认 ${DEFAULT_SOC}; 未指定时优先从 kernel 目录路径推断
  --vendor-name <name>    vendor 名, 默认从 host 包自动识别 (custom)
  --output <file.run>     输出 run 包路径, 默认 ./cann-ops-transformer-<vendor>_linux-<arch>.run
  --makeself <path>       指定 makeself.sh 路径, 默认自动探测 (CANN 安装目录 / PATH)
  --help-header <file>    指定包内 help.info, 默认使用 host 包内的 help.info
  --work-dir <dir>        指定临时工作目录, 默认 mktemp -d
  --keep-work             保留临时工作目录 (排障用)
  -h, --help              显示帮助

示例:
  # 1) 分别编译
  bash build.sh --pkg --jit --soc=${DEFAULT_SOC} --ops=moe_fused_topk,grouped_matmul_add -j16
  bash build.sh --opkernel  --soc=${DEFAULT_SOC} --ops=moe_fused_topk,grouped_matmul_add -j16

  # 2) 二次打包
  bash ${SCRIPT_NAME} \\
      --host-pkg   build_out/cann-ops-transformer-custom_linux-aarch64.run \\
      --kernel-dir build/binary/${DEFAULT_SOC}/bin \\
      --soc ${DEFAULT_SOC} \\
      --output     build_out/cann-ops-transformer-custom_linux-aarch64.run
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --host-pkg)      HOST_PKG="$2"; shift 2 ;;
            --host-dir)      HOST_DIR="$2"; shift 2 ;;
            --kernel-dir)    KERNEL_DIR="$2"; shift 2 ;;
            --soc)           SOC="$2"; shift 2 ;;
            --vendor-name)   VENDOR_NAME="$2"; shift 2 ;;
            --output)        OUTPUT="$2"; shift 2 ;;
            --makeself)      MAKESELF_SH="$2"; shift 2 ;;
            --help-header)   HELP_HEADER="$2"; shift 2 ;;
            --work-dir)      WORK_DIR="$2"; shift 2 ;;
            --keep-work)     KEEP_WORK="true"; shift ;;
            -h|--help)       usage; exit 0 ;;
            *) die "未知参数: $1 (使用 --help 查看用法)" ;;
        esac
    done
}

detect_makeself() {
    if [[ -n "$MAKESELF_SH" ]]; then
        [[ -x "$MAKESELF_SH" ]] || die "指定的 makeself 不可执行: ${MAKESELF_SH}"
        return
    fi
    local candidates=()
    local home="${ASCEND_HOME_PATH:-}"
    if [[ -n "$home" ]]; then
        candidates+=("$home/aarch64-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/util/makeself/makeself.sh")
        candidates+=("$home/x86_64-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/util/makeself/makeself.sh")
        for c in "${candidates[@]}"; do
            [[ -x "$c" ]] && { MAKESELF_SH="$c"; return; }
        done
        local found
        found="$(find "$home" -maxdepth 7 -name makeself.sh 2>/dev/null | head -1 || true)"
        [[ -n "$found" ]] && candidates+=("$found")
    fi
    if command -v makeself >/dev/null 2>&1; then
        candidates+=("$(command -v makeself)")
    fi
    for c in "${candidates[@]}"; do
        if [[ -x "$c" ]]; then
            MAKESELF_SH="$c"
            return
        fi
    done
    die "未找到 makeself.sh, 请使用 --makeself 指定 (或设置 ASCEND_HOME_PATH)"
}

detect_arch() {
    if [[ -n "$ARCH" ]]; then return; fi
    if [[ -n "$HOST_PKG" ]]; then
        case "$(basename "$HOST_PKG")" in
            *aarch64*) ARCH="aarch64"; return ;;
            *x86_64*)  ARCH="x86_64";  return ;;
        esac
    fi
    case "$(uname -m)" in
        aarch64|arm64) ARCH="aarch64" ;;
        x86_64|amd64)  ARCH="x86_64" ;;
        *) die "无法识别架构: $(uname -m)" ;;
    esac
}

resolve_kernel_bin() {
    [[ -d "$KERNEL_DIR" ]] || die "kernel 交付件目录不存在: ${KERNEL_DIR}"
    KERNEL_DIR="$(cd "$KERNEL_DIR" && pwd)"
    if [[ -d "${KERNEL_DIR}/bin" ]]; then
        KERNEL_BIN="${KERNEL_DIR}/bin"
    elif [[ -f "${KERNEL_DIR}/binary_info_config.json" ]]; then
        KERNEL_BIN="${KERNEL_DIR}"
    else
        die "kernel 交付件目录中未找到 bin/ 或 binary_info_config.json: ${KERNEL_DIR}"
    fi
    [[ -n "$(ls -A "${KERNEL_BIN}" 2>/dev/null)" ]] || die "kernel 交付件目录为空: ${KERNEL_BIN}"
}

detect_soc() {
    if [[ -n "$SOC" ]]; then return; fi
    local cand
    cand="$(basename "$(dirname "$(dirname "${KERNEL_BIN}")")")"
    if [[ "$cand" == ascend* || "$cand" == kirin* || "$cand" == mc* ]]; then
        SOC="$cand"
    else
        SOC="$DEFAULT_SOC"
    fi
    log "未显式指定 --soc, 自动推断为: ${SOC}"
}

extract_host_pkg() {
    local target="$1"
    local pkg_abs
    pkg_abs="$(cd "$(dirname "$HOST_PKG")" && pwd)/$(basename "$HOST_PKG")"
    [[ -x "$pkg_abs" ]] || die "host run 包不可执行: ${pkg_abs}"
    log "解压 host run 包: ${pkg_abs} -> ${target}"
    ( cd "$(dirname "$pkg_abs")" && "./$(basename "$pkg_abs")" --noexec --extract="$target" --nox11 >/dev/null 2>&1 ) \
        || die "解压 host run 包失败"
    [[ -d "${target}/packages" ]] || die "host run 包结构异常, 未找到 packages 目录"
}

resolve_vendor_dir() {
    local vendors_root="${STAGING}/packages/vendors"
    [[ -d "$vendors_root" ]] || die "host 包中未找到 packages/vendors 目录"
    local dirs=()
    while IFS= read -r d; do dirs+=("$d"); done < <(find "$vendors_root" -mindepth 1 -maxdepth 1 -type d)
    [[ ${#dirs[@]} -eq 1 ]] || die "host 包 vendors 目录应仅包含一个 vendor, 实际: ${dirs[*]:-空}"
    VENDOR_DIR_NAME="$(basename "${dirs[0]}")"
    if [[ -z "$VENDOR_NAME" ]]; then
        VENDOR_NAME="${VENDOR_DIR_NAME%_transformer}"
    fi
    log "vendor 目录: packages/vendors/${VENDOR_DIR_NAME} (vendor_name=${VENDOR_NAME})"
}

merge_kernel() {
    local kernel_root="${STAGING}/packages/vendors/${VENDOR_DIR_NAME}/op_impl/ai_core/tbe/kernel"
    local bin_target="${kernel_root}/${SOC}"
    local cfg_target="${kernel_root}/config/${SOC}"
    mkdir -p "$bin_target" "$cfg_target"

    log "合并 kernel 交付件: ${KERNEL_BIN}"
    log "  *.o / 算子子目录 -> ${bin_target}"
    log "  *.json 配置       -> ${cfg_target}"

    local count_o=0 count_json=0
    shopt -s nullglob
    for entry in "${KERNEL_BIN}"/*; do
        local base; base="$(basename "$entry")"
        if [[ -d "$entry" ]]; then
            cp -a "$entry" "${bin_target}/${base}"
            local n; n="$(find "${bin_target}/${base}" -type f | wc -l)"
            count_o=$((count_o + n))
        elif [[ "$base" == *.json ]]; then
            cp -a "$entry" "${cfg_target}/${base}"
            count_json=$((count_json + 1))
        else
            cp -a "$entry" "${bin_target}/${base}"
            count_o=$((count_o + 1))
        fi
    done
    shopt -u nullglob

    [[ $((count_o + count_json)) -gt 0 ]] || die "kernel 交付件目录未包含可合并文件: ${KERNEL_BIN}"
    log "合并完成: kernel 文件 ${count_o} 个, 配置 json ${count_json} 个"
}

run_makeself() {
    local out_name="$1"
    local makeself_header
    makeself_header="$(dirname "$MAKESELF_SH")/makeself-header.sh"
    [[ -f "$makeself_header" ]] || die "未找到 makeself-header.sh: ${makeself_header}"

    local help_header="${HELP_HEADER:-./help.info}"
    [[ -f "${STAGING}/help.info" || -f "$help_header" ]] || die "未找到 help.info (包内或 --help-header 指定)"

    log "调用 makeself 重新生成 run 包: ${out_name}"
    ( cd "$STAGING" && bash "$MAKESELF_SH" \
        --header "$makeself_header" \
        --help-header "$help_header" \
        --tar-format posix \
        --gzip \
        --complevel "$MAKESELF_COMPLEVEL" \
        --nomd5 \
        --sha256 \
        ./ "$out_name" "$MAKESELF_LABEL" ./install.sh ) \
        || die "makeself 打包失败"

    [[ -f "${STAGING}/${out_name}" ]] || die "makeself 未生成 run 包: ${STAGING}/${out_name}"
    chmod +x "${STAGING}/${out_name}"
}

main() {
    parse_args "$@"

    if [[ -n "$HOST_PKG" && -n "$HOST_DIR" ]]; then
        die "--host-pkg 与 --host-dir 互斥, 只能指定一个"
    fi
    if [[ -z "$HOST_PKG" && -z "$HOST_DIR" ]]; then
        usage; die "必须指定 --host-pkg 或 --host-dir"
    fi
    [[ -n "$KERNEL_DIR" ]] || { usage; die "必须指定 --kernel-dir"; }

    detect_makeself
    detect_arch
    resolve_kernel_bin
    detect_soc

    local cleanup="false"
    if [[ -z "$WORK_DIR" ]]; then
        WORK_DIR="$(mktemp -d -t repack_custom.XXXXXX)"
        cleanup="true"
    else
        mkdir -p "$WORK_DIR"
    fi
    STAGING="${WORK_DIR}/staging"

    if [[ -n "$HOST_PKG" ]]; then
        rm -rf "$STAGING"
        extract_host_pkg "$STAGING"
    else
        [[ -d "$HOST_DIR" ]] || die "host 目录不存在: ${HOST_DIR}"
        log "使用已解压的 host 目录: ${HOST_DIR}"
        rm -rf "$STAGING"
        mkdir -p "$STAGING"
        cp -a "${HOST_DIR}/." "$STAGING/"
    fi

    resolve_vendor_dir
    merge_kernel

    local out_name="cann-ops-transformer-${VENDOR_NAME}_linux-${ARCH}.run"
    run_makeself "$out_name"

    local final_out="${OUTPUT:-${PWD}/cann-ops-transformer-${VENDOR_NAME}_linux-${ARCH}.run}"
    mkdir -p "$(dirname "$final_out")"
    cp -f "${STAGING}/${out_name}" "$final_out"
    chmod +x "$final_out"

    log "二次打包完成: ${final_out}"
    log "包大小: $(du -h "$final_out" | cut -f1)"

    if [[ "$cleanup" == "true" && "$KEEP_WORK" != "true" ]]; then
        rm -rf "$WORK_DIR"
    else
        log "保留工作目录: ${WORK_DIR}"
    fi
}

main "$@"
