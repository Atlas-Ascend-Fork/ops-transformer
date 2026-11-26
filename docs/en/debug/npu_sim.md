# Introduction

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-12T12:14:36.104Z pushedAt=2026-11-18T02:51:06.808Z nl=2 -->

NPU Simulator is a SoC-level chip simulation tool for operator development scenarios. It analyzes the precision and performance data (such as instruction execution status) of AI tasks running on the AI simulator at each stage. This tool helps users perform in-depth performance tuning. It enables developers to obtain verification results and performance feedback almost identical to a real chip even when the chip is unavailable or chip resources are scarce.

# Functions

This tool maintains binary compatibility with on-board execution (the same kernel can run on both the simulator and the AI processor). Its main purposes are as follows:

* Precision simulation: outputs bit-level precision results to assist users in completing operator precision verification.
* Performance simulation: outputs instruction pipeline diagrams to assist users in locating operator performance bottlenecks.

# Before Use

## Constraints

* Recommended environment configuration for the tool: 16-core CPU and 32 GB or more memory.
* For all example paths in this document, ensure that the running user has read or read/write permission.
* For security and least-privilege considerations, run this tool with a regular user account. Avoid using high-privilege accounts such as root.
* This tool depends on the CANN software package. Install the CANN software package before use. Driver and firmware installation is not required. Run the set_env.sh environment variable file of CANN using the source command. For security, do not modify the environment variables in set_env.sh after running the source command.
* Users should follow the least-privilege principle. For example, files input to the tool must not be writable by other users. In some scenarios with stricter security requirements, ensure that input files are not writable by group users.
* This tool is a development tool and is not recommended for production environments.
* The simulation function of the tool only supports single-device scenarios and cannot simulate multi-device environments. Only device 0 can be set in the code. Modifying the visible device ID will cause simulation failure.
* The simulation environment only supports AI Core compute operators (MC2 and HCCL operators are not supported).
* The NPU Simulator tool only supports Ascend 950PR and Ascend 950DT chips. It is recommended to configure the simulator running environment with a 16-core CPU and 32 GB or more memory.
* Simulation on the arm environment is not supported.

```
Name change notice:
Starting from the version released on July 30, 2026, the simulator name cannsim is officially changed to npusim, and all command line tool names are changed accordingly. The old command cannsim is retained as an alias for a period of time. Migrate to npusim as soon as possible.
```

## Environment Preparation

NPU Simulator is integrated in the CANN toolkit package. Refer to [Environment Deployment](../install/quick_install.md) to complete the software package installation.

# Quick Start

The following uses [add_example](../../../examples/add_example/) as an example to describe operator simulation in detail.

## Operator Build

* Refer to [Operator Call](../invocation/quick_op_invocation.md) to complete the operator build and installation of add_example.

```bash
# Note: Enter the project root directory and run the following build command. The command is for reference only. For details, see the operator call description.
bash build.sh --pkg --soc=ascend950 --vendor_name=custom --ops=add_example
# Install the custom operator package.
./build_out/cann-ops-transformer-${vendor_name}_linux-${arch}.run
```

* Refer to [aclnn Call](../invocation/quick_op_invocation.md) to complete the build of test_aclnn_add_example.cpp and generate the executable file test_aclnn_add_example.

## Executing Simulation Command

```bash
npusim record ./test_aclnn_add_example -s Ascend950 --gen-report
```

The simulation tool execution log file is in the examples/add_example/examples/build/bin/npusim_* directory, and the execution log file is npusim.log.

The print information in the example can be seen from the simulation tool log file:

```bash
add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 2.000000
add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 2.000000
add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 2.000000
add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 2.000000
add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 2.000000
add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 2.000000
add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 2.000000
```

## Viewing Performance Pipeline File

The simulation performance pipeline file is in the `examples/add_example/examples/build/bin/npusim_*/report/results/kernel_*/core_*` directory of this project. The pipeline-related files are:

```bash
trace_core0.json
```

Enter "chrome://tracing" in the Chrome browser, and drag the generated instruction pipeline diagram file (trace_core0.json) to the blank area to open it. For parameter details, see the "Simulation Result Parsing" section.

# Simulation Execution

## Command Function

Execute the application in the simulation environment.

## Command Format

npusim record [options] user_app

## Parameters

Table 1 Simulation execution parameters

|Parameter|Required/Optional|Description|
| --- | --- | --- |
|-s or --soc-version | Required | Specifies the target chip version to simulate (for example, Ascend950). |
|-o or --output | Optional | Path where the generated files are stored. It can be configured as an absolute path or a relative path, and the user running the tool must have read/write permission. If the path is not specified, data is saved in the current directory by default. |
|-g or --gen-report | Optional | Enables automatic parsing after simulation is complete and generates an analysis report. Automatic parsing is disabled by default. |
|-u or --user-option | Optional | User-defined operator parameters, passed to the operator program as command line options. |
|-n or --core-id | Optional | AI Core for which logs are enabled during simulation. The format is the same as report -n: 'all', '0-2,12-14', '5'. All cores are enabled by default. When used with -g and not specified, it falls back to core 0. |
|user_app|Required|Operator program or command to run (for example, ./app, python train.py, bash run.sh). |

## Example

1. Complete operator development and compilation.
2. Execute the simulation command. Refer to the following usage example.

    ```bash
    # Method 1: Enable simulation and save the output to the ./output directory. /path/to/app is the operator program.
    $ npusim record /path/to/app -o ./output -s Ascend950

    # Method 2: Enable simulation and generate a report for subsequent performance analysis.
    $ npusim record /path/to/app -o ./output -s Ascend950 --gen-report
    ```

3. After the command is complete, a folder named "npusim_{timestamp}_${user_app}" is generated in the default path or the specified "output" directory. The structure example is as follows:

    ```bash
    ├─npusim_{timestamp}_${user_app}
    ├── npusim.log
    ```

4. Users can obtain the operator execution result and compare the precision. The result is displayed in npusim.log. The example is as follows.

    The following output is only an example of the precision comparison result of Ascend C single-operator direct call. It may vary slightly by version. Use the actual output as the standard.

    ```bash
    INFO:root:[INFO] compare data case[ case001]
    INFO:root:---------------RESULT---------------
    INFO:root:['case_name', 'wrong_num', 'total_num', 'result', 'task_duration']
    INFO:root:[' case001', 0, 65536, 'Success']
    ```

5. View the operator instruction pipeline diagram. Refer to Simulation Result Parsing.

# Simulation Result Parsing

## Command Function

Generate a visualized instruction pipeline diagram.

## Command Format

npusim report [options]

## Parameters

Table 1 Simulation result parsing parameters

|Parameter | Required/Optional | Description|
| --- | --- | --- |
|-e or --export | Required | Simulation execution result directory, specified to the npusim_{timestamp}_${user_app} level. It can be configured as an absolute path or a relative path, and the running user must have read/write permission. |
|-o or --output | Optional | Output directory of the instruction pipeline diagram. It can be configured as an absolute path or a relative path, and the running user must have read/write permission. If the path is not specified, it is the same as the export directory by default. |
|-n or --core-id | Optional | Specifies the core ID for generating the instruction pipeline. Supported formats: 'all', '0-2,12-14', '5'. If not specified, the instruction pipeline of core 0 is generated by default. |
|-f or --object-file | Optional | Device object file path, used to assist in generating the report. |

## Example

1. Refer to Simulation Execution for operator simulation, and compare the output example to ensure the corresponding result is executed correctly.
2. Execute the simulation result parsing command. Refer to the following execution example.

    ```bash
    # Generate a performance analysis report in the current directory (only core 0 is analyzed by default).
    npusim report -e /path/to/npusim_{timestamp}_${user_app}

    # Generate performance analysis reports for core 0, core 1, core 11, and core 12 in the specified directory.
    npusim report -e /path/to/npusim_{timestamp}_${user_app} -o /path/to/report -n '0-1, 11-12'
    ```

3. After the command is executed, the corresponding pipeline file is generated in the directory configured by output. The file format is json. The output result example is as follows:

    ```bash
    trace_core0.json
    trace_core1.json
    ...
    ```

4. View simulation result.
    Enter "chrome://tracing" in the Chrome browser, and drag the generated instruction pipeline diagram file (trace.json) to the blank area to open it. Use keyboard shortcuts (W: zoom in, S: zoom out, A: move left, D: move right) to view it.
    ![Instruction pipeline diagram](../figures/cannsim_pipeline_chart.png)

    Table 2 Key field description

    |Field|Field meaning|
    | --- | --- |
    |VECTOR|Vector operation unit. |
    |SCALAR|Scalar operation unit. |
    |Cube|Matrix multiplication operation unit. |
    |MTE1|Data transfer pipeline. Data transfer direction: L1 -> {L0A/L0B, UBUF}. |
    |MTE2|Data transfer pipeline. Data transfer direction: {DDR/GM, L2} -> {L1, L0A/B, UBUF}. |
    |MTE3|Data transfer pipeline. Data transfer direction: UBUF -> {DDR/GM, L2, L1}, L1 -> {DDR/L2}. |
    |FIXP|Data transfer pipeline. Data transfer direction: FIXPIPE L0C -> OUT/L1. |
    |FLOWCTRL|Control flow instruction. |
    |ICACHELOAD|View the missed ICache. |

# Querying Help Information

## Command Function

Query tool help information.

## Command Format

Query tool help information:

```bash
npusim --help
```

Query the help information of the record subcommand of the tool:

```bash
npusim record --help
```

Query the help information of the report subcommand of the tool:

```bash
npusim report --help
```

## Parameters

None.

## Example

1. Log in to the Host-side server.
2. Run the following command.

    ```bash
    npusim --help
    ```

## Outputs

```bash
usage: npusim [-h] {record,report} ...

Command-line tool for performance simulation analysis on Ascend hardware.

positional arguments:
  {record,report}  Available commands
    record         Run user application in AscendOps simulation environment
    report         Generate performance analysis reports

options:
  -h, --help       show this help message and exit
```
