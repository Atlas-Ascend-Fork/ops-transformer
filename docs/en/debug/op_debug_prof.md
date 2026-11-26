# Operator Debugging and Tuning

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-12T12:12:38.640Z pushedAt=2026-11-18T08:54:34.936Z nl=2 -->

## Debugging and Locating (AI Core Operator)

During operator runtime, if issues such as operator execution failure or precision exceptions occur, print the information of each stage, such as Kernel intermediate results, to analyze and locate the issue.

### 1. Host-Side Log Collection Method

* **plog collection**

    After the program execution ends, view the logs in "$HOME/ascend/log" by default. The host log file storage path is as follows:

    ```bash
    $HOME/ascend/log/debug/plog/plog-pid_*.log
    ```

   Enable the environment variable ASCEND_SLOG_PRINT_TO_STDOUT to print log logs directly to the screen (1: enable screen printing, 0: disable screen printing). The configuration example is as follows:

    ```bash
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    ```

    For log-related introduction, see [*Log Reference*](https://hiascend.com/en/document/redirect/CannCommunitylogref). For environment variable introduction, see [*Environment Variables*](https://hiascend.com/en/document/redirect/CannCommunityEnvRef).

* **aclnn exception error message collection**

    Obtain the exception information during the aclnn API call through the aclGetRecentErrMsg API (see [*Runtime APIs*](https://hiascend.com/en/document/redirect/CannCommunityRuntimeApi)). The usage is as follows:

   ```cpp
   printf("%s",aclGetRecentErrMsg());
   ```

    The printed error message example is as follows:

   ```text
   [PID:646612] 2026-01-24-11:53:44.671.727 AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument addmmTensor.self.
   ```

### 2. Kernel Debugging

Common debugging methods are as follows:

* **printf**

    This API supports printing Scalar type data, such as integers, characters, and Boolean types. For detailed introduction, see "Operator Debugging APIs > printf" in [*Ascend C APIs*](https://hiascend.com/en/document/redirect/CannCommunityAscendCApi).

    ```c++
    blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
    tileNum_ = tilingData->tileNum;
    tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;
    // Print the Block length computed by the current core.
    AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
    ```

* **DumpTensor**

    This API supports dumping the content of a specified Tensor and printing custom additional information, such as the current line number. For detailed introduction, see "Operator Debugging APIs > DumpTensor" in [*Ascend C APIs*](https://hiascend.com/en/document/redirect/CannCommunityAscendCApi).

    ```c++
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    // Print zLocal Tensor information.
    DumpTensor(zLocal, 0, 128);
    AscendC::DataCopy(outputGMZ[progress * tileLength_], zLocal, tileLength_);
    ```

* **msDebug single-step debugging**

  For issue locating in complex scenarios, such as operator hang and GM/UB out-of-bounds access, use [msDebug](https://atomgit.com/Ascend/msdebug/blob/master/docs/en/user_guide/msdebug_user_guide.md) single-step debugging. The specific operation steps are as follows:

  1. **Compile the Kernel with O0 -g**

     Before using msDebug for debugging, compile the Kernel without optimization and with debug information through the `--op_debug_config "ccec_O0,ccec_g"` option:

     ```bash
     bash build.sh --pkg --ops=add_example --soc=ascend910b --op_debug_config "ccec_O0,ccec_g"
     ```

     If the preceding compilation fails, or the operator itself has precision issues when running, it may be because the operator has many tilingKeys (a large operator). Debug the branch code corresponding to the target tilingKey separately. In this case, add the `--tiling_key` option to specify the target tilingKey.

     First, run the operator once with msDebug and observe the Kernel startup information to determine the target tilingKey:

     ```
     [Launch of Kernel RunAicpuKfcResInit on Device 0]
     [Launch of Kernel RunAicpuKfcResInit on Device 1]
     [Launch of Kernel AddExample_c9e4347749ac13425bf7e31c6a5e306a_7 on Device 1]
     [Launch of Kernel AddExample_c9e4347749ac13425bf7e31c6a5e306a_7 on Device 0]
     ```

     Among them, `_c9e4347749ac13425bf7e31c6a5e306a` is the tilingKey. After determining the target tilingKey, the compile command is as follows:

     ```bash
     bash build.sh --pkg --ops=add_example --soc=ascend910b --op_debug_config "ccec_O0,ccec_g" --tiling_key=c9e4347749ac13425bf7e31c6a5e306a
     ```

  2. **Start msDebug debugging**

     After installing the operator package, enter the directory where the operator executable file is located and use msDebug to start debugging:

     ```bash
     msdebug ./test_aclnn_add_example
     ```

     For more msDebug usage, see the [msDebug](https://atomgit.com/Ascend/msdebug/blob/master/docs/en/user_guide/msdebug_user_guide.md) operator debugging tool documentation.

### 3. Kernel Detection

[msSanitizer](https://www.hiascend.com/en/document/redirect/CannCommunityToolMssanitizer) is an Ascend C operator memory detection and race detection tool. It can detect issues such as GM/UB out-of-bounds access, memory leaks, and concurrent races during Kernel runtime. The following uses the `add_example` operator as an example to introduce the usage steps:

1. **Compile the Kernel with detection enabled**

   Compile the Kernel with detection through the `--op_debug_config "sanitizer"` option. Replace `add_example` in the following command with the actual operator name to be detected:

   ```bash
   bash build.sh --pkg --ops=add_example --soc=ascend910b --op_debug_config "sanitizer"
   ```

   For Ascend 950, temporarily modify the opc compile script in the CANN package and add the `--cce-enable-sanitizer`, `-g`, and `-fno-jump-tables` options in the `_gen_compile_cmd_c310` function:

   ```bash
   vim ${ASCEND_HOME_PATH}/cann/python/site-packages/asc_op_compile_base/asc_op_compiler/ascendc_compile_v220.py
   ```

2. **Install the compiled package**

   After compilation, enter the `build_out` directory to install the generated operator package (the file name varies with the SoC version; use 950 as an example):

   ```bash
   cd build_out
   bash cann-ops-transformer-custom_linux-aarch64.run
   cd ..
   ```

3. **Compile and run the test case**

   ```bash
   bash build.sh --run_example add_example eager cust --soc=ascend910b
   ```

4. **Execute msSanitizer detection**

   Use mssanitizer to execute memory detection and race detection simultaneously:

   ```bash
   mssanitizer --tool=memcheck --tool=racecheck \
       --log-level=error \
       --kernel-name=AddExample \
       -- build/test_aclnn_add_example > mssanitizer_result.txt 2>&1
   ```

   The detection result is output to `mssanitizer_result.txt`. For more msSanitizer usage, see the [msSanitizer](https://www.hiascend.com/en/document/redirect/CannCommunityToolMssanitizer) documentation.

## Performance Tuning

During operator runtime, if issues such as execution precision degradation or abnormal memory usage occur, use [msProf](https://www.hiascend.com/en/document/redirect/CannCommunityToolMsprof) to analyze the metric data of each operator runtime stage (such as throughput, memory usage, and time consumption), determine the root cause of the issue, and optimize accordingly.

This chapter uses the `AddExample` custom operator as an example to introduce the commonly used on-board performance collection and pipeline diagram simulation methods in operator tuning.

**Use case differences:**

* **On-board performance collection**: applicable to running operators on real NPU hardware to quickly obtain the overall performance metrics of the operator (such as Kernel time consumption, Block count, and pipeline proportion), helping determine whether the operator has performance issues.
* **Pipeline diagram simulation**: applicable to developers without NPU hardware, or scenarios that require in-depth analysis of instruction-level pipeline bottlenecks within the operator and optimization of instruction scheduling. It provides more detailed instruction-level pipeline analysis than on-board collection.

### Method 1: On-board Performance Collection

* **Prerequisite**

    After completing operator development and compilation, assuming the use of the aclnn API call method, the directory where the generated operator executable file (test_aclnn_add_example) is located is `examples/add_example/examples/build/bin/` in this project.

* **Collect performance data**

    When collecting various pipeline metrics of the operator on-board runtime, enter the directory where the operator executable file is located and execute the following command:

    ```bash
    msprof op ./test_aclnn_add_example
    ```

    The collection results are in the `examples/add_example/examples/build/bin/OPPROF_*` directory in this project. After collection, the following information is printed:

    ```text
    Op Name: AddExample_a1532827238e1555db7b997c7bce2928_high_performance_1
    Op Type: vector
    Task Duration(us): 97.861954
    Block Dim: 8
    Mix Block Dim:
    Device Id: 0
    Pid: 2776181
    Current Freq: 1800
    Rated Freq: 1800
    ```

    Among them, Task Duration is the time consumption of the current operator Kernel, and Block Dim is the number of cores for the current operator execution.

    For detailed metrics of various operator pipelines, refer to the `ArithmeticUtilization` file under `OPPROF_*`, which contains the proportion of various pipelines. For specific introduction, see the "Performance Data File > msprof op > ArithmeticUtilization (time consumption and proportion of cube and vector type instructions)" chapter in [msProf](https://www.hiascend.com/en/document/redirect/CannCommunityToolMsprof).

### Method 2: Pipeline Diagram Simulation Collection

* **Prerequisite**

    After completing operator development and compilation, assuming the use of the aclnn API call method, the directory where the generated operator executable file (test_aclnn_add_example) is located is `examples/add_example/examples/build/bin/` in this project.

* **For Ascend 950PR, use the [NPU Simulator](./npu_sim.md) simulation tool to execute the simulation command and generate simulation data.**

    Execute the simulation command to generate simulation data

    ```bash
    npusim record ./test_aclnn_add_example -s Ascend950 --gen-report
    ```

    The simulation results are in the `examples/add_example/examples/build/bin/npusim_*/report/results/kernel_*/core_*` directory in this project. The pipeline-related files are:

    ```bash
    trace_core0.json
    ```

    Enter "chrome://tracing" in the Chrome browser and drag the generated instruction pipeline diagram file (trace_core0.json) to the blank area to open it. For specific parameter introduction, see the ["Simulation Result Parsing"](./npu_sim.md#simulation-result-parsing) chapter in NPU Simulator.

* **For Atlas A2/A3 products, use [msProf](https://www.hiascend.com/en/document/redirect/CannCommunityToolMsprof) to execute the simulation command and generate simulation data.**

    Before using the msProf tool for operator simulation tuning, execute the following command to configure the environment variables.

    ```bash
    export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
    ```

    Modify the preceding environment variables according to the actual installation path of the CANN software package and the AI processor model.

    Then enter the directory where the operator executable file is located and execute the following command:

    ```bash
    msprof op simulator --output=$PWD/pipeline_auto --kernel-name "AddExample" ./test_aclnn_add_example
    ```

      The collection results are in the `$PWD/pipeline_auto/OPPROF_**` directory in this project.
      Among them, the pipeline-related file path is `OPPROF_*/simulator/visualize_data.bin`. Use the "Basic Operations > Import Data" chapter in [MindStudio Insight](https://www.hiascend.com/en/document/redirect/MindStudioInsight) to view how to import pipeline data.
