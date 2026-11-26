# build Parameters

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-12T12:16:23.981Z pushedAt=2026-11-18T02:51:06.825Z nl=2 -->

## Introduction

build.sh is the build script of this project. By default, it is located under the project root directory. It automatically compiles, links, and configures the source code, and finally generates executable files, library files, or other target files that can be installed or run directly. Specifically, the script implements multiple functions by configuring different parameters, including building multiple target libraries (for example, libophost_transformer.so), compiling operator packages, and running unit tests.

## Usage

1. **Configure environment variables**

   Refer to [Environment Deployment](quick_install.md) to complete the environment variable configuration.

   ```bash
   # Install to the default path, using the root user as an example
   source /usr/local/Ascend/cann/set_env.sh
   ```

2. **Build command format**

   Take the command for compiling an operator package as an example. The format is as follows, where `--vendor_name` and `--ops` are optional in this scenario.

   ```bash
   bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
   ```

   For the meanings of all parameters, see the Parameters section below. Select appropriate parameters based on the actual situation.

## Parameters

build.sh supports multiple functions. Run the following command to view all function parameters.

```bash
bash build.sh --help
```

| Name              | Required/Optional  | Description                                                                         |
|------------------|--------|------------------------------------------------------------------------------|
| -j${n}           | Optional     | Specifies the number of compilation threads. ${n} is the specific number of threads, and the default value is 8 (for example, -j8). If the number of threads exceeds the number of CPU cores, it is automatically adjusted to the number of CPU cores.                   |
| -v               | Optional     | Views the CMake compilation configuration information.                                                               |
| -O${n}           | Optional     | Specifies the compilation optimization level. Supports O0/O1/O2/O3 (for example, -O3). ${n} is the optimization level identifier.                                   |
| -u               | Optional     | Enables the unit test (UT) compilation mode and compiles all UT targets.                                                     |
| --help, -h        | Optional     | Prints the script help information.                                                                  |
| --ops            | Optional     | Specifies the operators to be compiled, for example, apply_rotary_pos_emb,rope_quant_kvcache. Multiple operators are separated by English commas ",". Cannot be used together with --ophost, --opapi, or --opgraph. |
| --module         | Optional     | Specifies the module to be compiled, for example, --module=mc2,attention. Multiple modules are separated by English commas "," (supported: mc2,attention,moe,ffn,mhc,posembedding,gmm). |
| --soc            | Optional     | Specifies the NPU model. When `--pkg` is used for packaging, multiple models can be separated by English commas, and the script builds them one by one.                                               |
| --jit            | Optional     | In the static graph scenario, when compiling the full package `cann-${soc_name}-ops-transformer_${cann_version}_linux-${arch}.run`, the operator binary files do not need to be compiled (the graph runtime compiles them online). Configure this option to improve the compilation speed. |
| --static         | Optional     | After configuration, generates static library files, including libcann_transformer_static.a and aclnn API header files. Used together with --pkg to generate a static library package.|
| --vendor_name    | Optional     | Specifies the name of the custom operator package. The default value is custom.                                                      |
| --build-type     | Optional     | Specifies the build type. Supports Release/Debug (for example, --build-type=Debug). The default value is Release. Cannot be used together with --mssanitizer, --oom, or --dump_cce. |
| --version        | Optional     | Specifies the version number (for example, --version=1.0.0).                                                       |
| --cov            | Optional     | Enables unit test code coverage statistics (must be used together with -u). Uses lcov to generate the coverage report.                                           |
| --noexec         | Optional     | Only compiles the unit test binary files and does not automatically run the compiled UT executable files.                                               |
| --valgrind       | Optional     | Uses valgrind to run unit tests for memory detection (must be used together with -u). After it is enabled, ASAN and --noexec are disabled.                                |
| --opkernel       | Optional     | Compiles the binary kernel.                                                                     |
| --opkernel_aicpu | Optional     | Compiles the AI CPU operator kernel.                                                                   |
| --pkg        | Optional     | Generates the installation package. Cannot be used together with -u (UT mode), --ophost, --opapi, or --opgraph.                            |
| --pkg-type       | Optional     | Specifies the packaging type. Values: run/rpm/deb/all. The default value is run. Can be used only together with --pkg.                                  |
| --make_clean     | Optional     | Performs basic cleanup (cleans compilation artifacts). The script exits after execution.                                                    |
| --asan           | Optional     | Enables ASAN (AddressSanitizer) memory detection.                                            |
| --ophost         | Optional     | Compiles the libophost_transformer.so library. Cannot be used together with --pkg or --ops.                                     |
| --opapi          | Optional     | Compiles the libopapi_transformer.so library. Cannot be used together with --pkg or --ops.                                      |
| --opgraph        | Optional     | Compiles the libopgraph_transformer.so library. Cannot be used together with --pkg or --ops.                                    |
| --onnxplugin     | Optional     | Compiles the liboponnx_plugin_transformer.so library.                                                 |
| --ophost_test   | Optional     | Compiles ophost-related unit tests. Equivalent to the combination of -u and --ophost.                                          |
| --opapi_test    | Optional     | Compiles opapi-related unit tests. Equivalent to the combination of -u and --opapi.                                            |
| --opgraph_test  | Optional     | Compiles opgraph-related unit tests. Equivalent to the combination of -u and --opgraph.                                            |
| --opkernel_test | Optional     | Compiles opkernel-related unit tests. Equivalent to the combination of -u and --opkernel.                                      |
| --run_example    | Optional     | Compiles the samples of the specified operators and modes, and runs the compiled executable files.                                                    |
| --simulator      | Optional     | Enables the simulator mode to run the --run_example task. In simulator mode, the corresponding simulation library is linked based on soc_version. Currently, only Ascend950 is supported.          |
| --genop          | Optional     | Creates the initial directory of an AI Core custom operator.                                                       |
| --genop_aicpu    | Optional     | Creates the initial directory of an AI CPU custom operator.                                                       |
| --experimental  | Optional     | Compiles the user operators under the experimental directory.                                                   |
| --oom            | Optional     | Enables kernel-side oom memory detection.                                                   |
| --mssanitizer    | Optional     | Enables kernel-side mssanitizer memory detection. Cannot be used together with --oom or --dump_cce.                              |
| --dump_cce       | Optional     | Enables kernel-side dumping of precompiled files. Cannot be used together with --mssanitizer.                                         |
| --bisheng_flags  | Optional     | Specifies a Bisheng compiler compilation parameter. Cannot be used together with --mssanitizer, --oom, or --dump_cce.     |
| --kernel_template_input     | Optional     | Specifies the tilingKey template for kernel compilation. Only one template can be specified. Template parameters are separated by English commas ",". Used together with --ops, and only one operator can be specified. The binary files of other operators that this operator depends on are not compiled.     |
| --cann_3rd_lib_path           | Optional     | The directory where third-party libraries are stored in offline compilation scenarios.                                                   |
