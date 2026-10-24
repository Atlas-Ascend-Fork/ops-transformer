#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set +e

REPOSITORY_NAME="ops-transformer"
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
if [[ "${task_name}" == *ubuntu24* ]]; then
    if [[ "${GIT_TARGET_BRANCH}" = "master" ]]; then
        sudo update-alternatives --set gcc /usr/bin/gcc-15
    else
        sudo update-alternatives --set gcc /usr/bin/gcc-14
    fi
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
fi

if gcc --version | head -n1 | grep -q "15\."; then
    rm -rf /home/jenkins/opensource/lib_cache
    if [ -d  /home/jenkins/opensource/gcc15 ];then
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15/lib_cache /home/jenkins/opensource/lib_cache
    elif [ -d  /home/jenkins/opensource/gcc15x86 ];then
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15x86/lib_cache /home/jenkins/opensource/lib_cache
    fi
else
    gcc --version
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi
gcc --version
rm -rf /home/jenkins/opensource/json

if [ -z "${ASCEND_3RD_LIB_PATH}" ]; then
    export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
fi

if [ -z "${OS_TYPE}" ]; then
    OS_TYPE=$(uname -m)
fi

if [ -f /home/jenkins/Ascend/cann/bin/setenv.bash ]; then
    source /home/jenkins/Ascend/cann/bin/setenv.bash
fi
LOG_INFO() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "[INFO] ${date_time} ${assert_msg}"
}
LOG_HEAD()
{
    echo "========================================"
    echo "  $1"
    echo "========================================"
}

LOG_DO()
{
    echo "[LOG_DO] $*"
    "$@"
}
DP_ASSERT_CHECK_SKIP() {
    local actual_value=${1}
    local assert_msg=${2}
    if [ "${actual_value}" != "0" ] && [ "${actual_value}" != "200" ]; then
        LOG_ERROR "${assert_msg} is failed."
        exit 1
    else
        LOG_INFO "${assert_msg} is success."
    fi
}
DP_ASSERT_EQUAL()
{
    local actual="$1"
    local expected="$2"
    local msg="$3"
    if [ "${actual}" != "${expected}" ]; then
        echo "::error::ASSERT FAILED: ${msg} (expected=${expected}, actual=${actual})"
        exit 1
    fi
}

# 解析 PR 变更命中且带 examples 的算子。stdout 只输出分号分隔的结果串（可能为空）;
# 解析脚本自身日志走 stderr,不会污染捕获。pr_filelist或命中为空均或命中为空均输出空串。
# kirin 过滤：只保留属于 kirinx90/kirin9030 的算子(存在<module>/<op>/ophost/config/{kirinx90|kirin9030}),
# 全都不属于则输出空串，由调用方走默认算子分支
IS_KIRIN_OP()
{
    local op="$1"
    local cfg_dir
    for cfg_dir in "${WORKSPACE}"/*/"${op}"/op_host/config;  do
        [[ -d "${cfg_dir}" ]] || continue
        if [[ -d "${cfg_dir}/kirinx90" || -d "${cfg_dir}/kirin9030" ]]; then
            return 0
        fi
    done
    return 1
}

GET_PR_RELATED_OPS()
{
    local ops=""
    local filtered=""
    local op op_arr
    if [ -f "${WORKSPACE}/pr_filelist.txt" ]; then
        ops=$(python3 "${WORKSPACE}/cmake/scripts/parse_changed_files.py" \
              -c "${WORKSPACE}/tests/test_config.yaml" \
              -f "${WORKSPACE}/pr_filelist.txt" get_related_examples 2>/dev/null)
    else
        echo "pr_filelist.txt not found" >&2
    fi
    LOG_INFO "ops_name raw: ${ops}" >&2
    IFS=';' read -r -a op_arr <<< "${ops}"
    for op in "${op_arr[@]}"; do
        [[ -z "${op}" ]] && continue
        if IS_KIRIN_OP "${op}"; then
            filtered="${filtered}${op};"
            LOG_INFO "kirin filter keep: ${op}" >&2
        else
            LOG_INFO "kirin filter drop: ${op}" >&2
        fi
    done
    LOG_INFO "ops_name kirin filtered: ${filtered}" >&2
    echo "${filtered}"
}
# Log error
function LOG_ERROR() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BRed}[ERROR] ${date_time} ${assert_msg}${Color_Off}"
}


LOG_HEAD "Build ${REPOSITORY_NAME}."
cd "${WORKSPACE}/" || exit 1

non_skip_count=$(grep -vE '(\.md$|^tests/)' "${WORKSPACE}/pr_filelist.txt" | grep -cv '^$')
if [ "${non_skip_count}" -eq 0 ]; then
    LOG_HEAD "pr_filelist.txt only contains .md or tests/ files, skip build"
    mkdir -p build_out
    # host/kernel 拆分任务: touch 对应空占位产物, 保证 package 任务可下载并跳过 repack
    case "${task_name}" in
        *host*910B*)    touch build_out/custom_host_ascend910b.run;  exit 0 ;;
        *host*950*)     touch build_out/custom_host_ascend950.run;   exit 0 ;;
        *kernel*910B*)  touch build_out/kernel_ascend910b.tar.gz;    exit 0 ;;
        *kernel*950*)   touch build_out/kernel_ascend950.tar.gz;     exit 0 ;;
        *oneshot*910B*) touch build_out/custom_oneshot_ascend910b.run; exit 0 ;;
        *oneshot*950*)  touch build_out/custom_oneshot_ascend950.run;  exit 0 ;;
    esac
    touch build_out/skip_build.run
    touch single.tar.gz
    echo "api-check=continue" >> "${ATOMGIT_OUTPUT}"
    exit 0
fi
if [[ "${task_name}" =~ Compile_Ascend_X86_ubuntu24 ]]; then
    sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
    echo "api-check=compile" >> "${ATOMGIT_OUTPUT}"
else
    echo "api-check=continue" >> "${ATOMGIT_OUTPUT}"
fi
if [ "${task_name}" == "Pre_Compile" ];then
    LOG_DO bash build.sh --PR_PKG ./pr_filelist.txt -j32 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}
    ret=$?
    if [ $ret -eq 200 ]; then
        LOG_HEAD "NOTE: changed files are not supported in pre_smoke"
        exit 0
    fi
    DP_ASSERT_EQUAL "$ret" "0" "Build ${REPOSITORY_NAME}"
elif [[ "${task_name}" == "Pre_Compile_host_910B" || "${task_name}" == "Pre_Compile_host_950" ]]; then
    echo "==========pre compile host ${task_name} start:================="
    if [ "${task_name}" == "Pre_Compile_host_950" ]; then
        SOC_ARG="--soc=ascend950"
        SOC_NAME="ascend950"
    else
        SOC_ARG=""
        SOC_NAME="ascend910b"
    fi
    # HOST: 只编 host, 产物为不含 kernel bin 的 custom run 包(供 repackage 任务下载合并)
    LOG_DO bash build.sh --pkg --jit -f "pr_filelist.txt" ${SOC_ARG} --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
    BUILD_EXIT_CODE=$?
    # 空跑兼容: build.sh 返回 200 表示无编译范围, 跳过产物收集, touch 空包占位
    if [ $BUILD_EXIT_CODE -eq 200 ]; then
        LOG_HEAD "NOTE: no custom packages to build for this PR (exit 200), touch placeholder"
        mkdir -p "${WORKSPACE}/build_out"
        touch "${WORKSPACE}/build_out/custom_host_${SOC_NAME}.run"
        exit 0
    fi
    DP_ASSERT_CHECK_SKIP "$BUILD_EXIT_CODE" "Build ${REPOSITORY_NAME}."
    run_file=$(find "${WORKSPACE}/build_out" "${WORKSPACE}/build" -maxdepth 2 -type f -name "*.run" 2>/dev/null | head -n 1)
    if [[ -z "${run_file}" ]]; then
        # 增量编译空跑兼容: 修改不涉及算子编译时无产物, touch 空包占位, 保证 package 任务可下载
        LOG_HEAD "NOTE: no host run package produced (empty compile scope), touch placeholder"
        mkdir -p "${WORKSPACE}/build_out"
        touch "${WORKSPACE}/build_out/custom_host_${SOC_NAME}.run"
        exit 0
    fi
    mkdir -p "${WORKSPACE}/build_out"
    cp -f "${run_file}" "${WORKSPACE}/build_out/custom_host_${SOC_NAME}.run"
    # 清理其余 run 包, 避免干扰 rename_package 步骤
    find "${WORKSPACE}/build_out" -maxdepth 1 -type f -name "*.run" ! -name "custom_host_${SOC_NAME}.run" -exec rm -f {} +
elif [[ "${task_name}" == "Pre_package_oneshot_910B" || "${task_name}" == "Pre_package_oneshot_950" ]]; then
    echo "==========pre package oneshot ${task_name} start:================="
    if [ "${task_name}" == "Pre_package_oneshot_950" ]; then
        SOC_ARG="--soc=ascend950"
        SOC_NAME="ascend950"
    else
        SOC_ARG=""
        SOC_NAME="ascend910b"
    fi
    # 一次性编包: --pkg -f (不带 --jit), 与 A(host+kernel+package) 区分产物名
    LOG_DO bash build.sh --pkg -f "pr_filelist.txt" ${SOC_ARG} --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j32
    BUILD_EXIT_CODE=$?
    # 空跑兼容: build.sh 返回 200 表示无编译范围, 跳过产物收集, touch 空包占位
    if [ $BUILD_EXIT_CODE -eq 200 ]; then
        LOG_HEAD "NOTE: no custom packages to build for this PR (exit 200), touch placeholder"
        mkdir -p "${WORKSPACE}/build_out"
        touch "${WORKSPACE}/build_out/custom_oneshot_${SOC_NAME}.run"
        exit 0
    fi
    DP_ASSERT_CHECK_SKIP "$BUILD_EXIT_CODE" "Build ${REPOSITORY_NAME}."
    run_file=$(find "${WORKSPACE}/build_out" "${WORKSPACE}/build" -maxdepth 2 -type f -name "*.run" 2>/dev/null | head -n 1)
    if [[ -z "${run_file}" ]]; then
        # 兜底: 构建成功但无产物, 同样 touch 空包占位
        LOG_HEAD "NOTE: no oneshot run package produced (empty compile scope), touch placeholder"
        mkdir -p "${WORKSPACE}/build_out"
        touch "${WORKSPACE}/build_out/custom_oneshot_${SOC_NAME}.run"
        exit 0
    fi
    mkdir -p "${WORKSPACE}/build_out"
    cp -f "${run_file}" "${WORKSPACE}/build_out/custom_oneshot_${SOC_NAME}.run"
    find "${WORKSPACE}/build_out" -maxdepth 1 -type f -name "*.run" ! -name "custom_oneshot_${SOC_NAME}.run" -exec rm -f {} +
elif [[ "${task_name}" == "Pre_Compile_kernel_910B" || "${task_name}" == "Pre_Compile_kernel_950" ]]; then
    echo "==========pre compile kernel ${task_name} ${task_type} start:================="
    if [ "${task_name}" == "Pre_Compile_kernel_950" ]; then
        SOC_ARG="--soc=ascend950"
        SOC_NAME="ascend950"
    else
        SOC_ARG=""
        SOC_NAME="ascend910b"
    fi
    # kernel: 只编 kernel, 产物为 *.o/*.json(供 repackage 任务下载合并)
    if [ "${task_type}X" = "disX" ]; then
        LOG_DO bash build.sh --opkernel -f "pr_filelist.txt" ${SOC_ARG} -j96
    else
        LOG_DO bash build.sh --opkernel -f "pr_filelist.txt" ${SOC_ARG} --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j32
    fi
    BUILD_EXIT_CODE=$?
    # 空跑兼容: build.sh 返回 200 表示无编译范围, 跳过产物收集, touch 空包占位
    if [ $BUILD_EXIT_CODE -eq 200 ]; then
        LOG_HEAD "NOTE: no custom packages to build for this PR (exit 200), touch placeholder"
        mkdir -p "${WORKSPACE}/build_out"
        touch "${WORKSPACE}/build_out/kernel_${SOC_NAME}.tar.gz"
        exit 0
    fi
    DP_ASSERT_CHECK_SKIP "$BUILD_EXIT_CODE" "Build ${REPOSITORY_NAME}."
    if [[ ! -d "${WORKSPACE}/build/binary/${SOC_NAME}/bin" ]]; then
        # 增量编译空跑兼容: 修改不涉及算子编译(或该 soc 无编译范围)时无 kernel 产物, touch 空包占位
        LOG_HEAD "NOTE: no kernel products under build/binary/${SOC_NAME} (empty compile scope), touch placeholder"
        mkdir -p "${WORKSPACE}/build_out"
        touch "${WORKSPACE}/build_out/kernel_${SOC_NAME}.tar.gz"
        exit 0
    fi
    mkdir -p "${WORKSPACE}/build_out"
    # kernel 产物打包上传, 供 repackage 任务下载
    tar -czf "${WORKSPACE}/build_out/kernel_${SOC_NAME}.tar.gz" -C "${WORKSPACE}/build/binary" "${SOC_NAME}"
    # kernel 任务无 run 包产物, 直接退出以跳过末尾的 run 包检查
    exit 0
else
    if [ "${GE_ST_RT2}X" == "kirinx90X" ]; then
        if [ "${GIT_TARGET_BRANCH}" = "master" ]; then
            ops_names=$(GET_PR_RELATED_OPS)
            if [ -n "${ops_names}" ]; then
                LOG_INFO "kirinx90 incremental build for ops: ${ops_names}"
                LOG_DO bash build.sh --pkg --soc=kirinx90  --PR_PKG ./pr_filelist.txt --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
            else
                ops_names="moe_token_unpermute"
                LOG_INFO "kirinx90 ops_names empty,build default op: ${ops_names}"
                LOG_DO bash build.sh --pkg --soc=kirinx90 --ops="${ops_names}" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
            fi
            DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
        else
            echo "not need build mobile_station"
            mkdir build_out
            touch build_out/cann-ops-transformer-kirinx90_linux-x86_64.run
            exit 0
        fi
    elif [ "${GE_ST_RT2}X" == "kirin9030X" ];then
        if [ "${GIT_TARGET_BRANCH}" = "master" ];then
            wget -nv https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/asc-devkit/package/5396/cann-asc-devkit_linux-x86_64_ubuntu24.run
            chmod +x *.run
            sudo chmod 777 /home/jenkins/Ascend
            yes "y" | sudo bash cann-asc-devkit_linux-x86_64_ubuntu24.run --full --install-path=/home/jenkins/Ascend
            ops_names=$(GET_PR_RELATED_OPS)
            if [ -n "${ops_names}" ]; then
                LOG_INFO "kirin9030 incremental build for ops: ${ops_names}"
                LOG_DO bash build.sh --pkg --soc=kirin9030  --PR_PKG ./pr_filelist.txt --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
            else
                ops_names="moe_token_unpermute"
                LOG_INFO "kirin9030 ops_names empty,build default op: ${ops_names}"
                LOG_DO bash build.sh --pkg --soc=kirin9030 --ops="${ops_names}" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
            fi
            DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
        else
            echo "not need build mobile_station"
            mkdir build_out
            touch build_out/cann-ops-transformer-kirin9030_linux-x86_64.run
            exit 0
        fi
    elif [ "${GE_ST_RT2}X" == "experimentalX" ]; then
        if [ "${GIT_TARGET_BRANCH}" = "master" ]; then
            LOG_DO bash build.sh --experimental --PR_PKG "pr_filelist.txt" --soc=ascend910b -j16 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}
            DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
        else
            echo "not need build experimental"
            mkdir build_out
            touch build_out/cann-ops-math-experimental_linux-${OS_TYPE}.run
            exit 0
        fi
    elif [ "${GE_ST_RT2}X" == "experimental_950X" ]; then
        if [ "${GIT_TARGET_BRANCH}" = "master" ]; then
            LOG_DO bash build.sh --experimental --PR_PKG "pr_filelist.txt" --soc=ascend950 --PR_PKG "pr_filelist.txt" -j16 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}
            BUILD_EXIT_CODE=$?
            DP_ASSERT_CHECK_SKIP "$?" "0" "Build ${REPOSITORY_NAME}"
        else
            echo "not need build experimental_950"
            mkdir build_out
            touch build_out/cann-ops-math-experimental_linux-${OS_TYPE}.run
            exit 0
        fi
    elif [[ "${task_name}" =~ monitor ]]; then
        if [ "${GIT_TARGET_BRANCH}" = "master" ]; then
            if [[ "${task_name}" =~ "910c" ]]; then
                LOG_DO bash build.sh --pkg --jit --PR_PKG "pr_filelist.txt" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16 --soc=ascend910_93
                BUILD_EXIT_CODE=$?
                DP_ASSERT_CHECK_SKIP "$?" "0" "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend910_93]"
            elif [[ "${task_name}" =~ "950" ]]; then
                LOG_DO bash build.sh --pkg --jit --PR_PKG "pr_filelist.txt" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16 --soc=ascend950
                BUILD_EXIT_CODE=$?
                DP_ASSERT_CHECK_SKIP "$?" "0" "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend950]"
            else
                LOG_DO bash build.sh --pkg --jit --PR_PKG "pr_filelist.txt" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16 --soc=ascend910b
                BUILD_EXIT_CODE=$?
                DP_ASSERT_CHECK_SKIP "$?" "0" "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend910b]"
            fi
        else
            echo "not need build monitor"
            mkdir build_out
            touch build_out/cann-ops-transformer_linux-x86_64.run
            exit 0
        fi
    else
        LOG_DO bash build.sh --pkg --jit --PR_PKG "pr_filelist.txt" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
        BUILD_EXIT_CODE=$?
        DP_ASSERT_CHECK_SKIP "$?" "0" "Build ${REPOSITORY_NAME}"
    fi
fi
compile_package_name=$(ls "${WORKSPACE}/build_out/" |grep -E "*.run$"|head -n1)
if [[ -z "${compile_package_name}" ]]; then
    if [[ "${BUILD_EXIT_CODE}" == "200" ]]; then
        echo "not need compile"
        mkdir build_out
        touch build_out/cann-ops-transformer_linux-${OS_TYPE}.run
        echo "api-check=continue" >> "${ATOMGIT_OUTPUT}"
        exit 0
    else
        echo "ERROR: Not find *.run in  ${WORKSPACE}/output/package/!"
        exit 1
    fi
fi
