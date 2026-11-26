# Quick Start: Based on the ops-transformer Repository

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-18T08:14:41.635Z pushedAt=2026-11-18T08:25:31.496Z nl=2 -->

## Precautions

This guide helps developers quickly get started with CANN and the `ops-transformer` operator repository. To quickly understand the complete operator development process, the **AddExample** operator is used as the practice object. The operator source code is located at `ops-transformer/examples/add_example`. The operation process is as follows:

1. **[Prerequisites](../README_en.md)**: See the project README to complete environment preparation and source code download. This is not repeated here. For the quick start task, **CANNLab or Docker deployment is recommended** because it is simple to operate.

   > **Note**: The CANNLab or Docker environment provides the latest CANN package by default. If you need to experience the latest capabilities of the master branch, set up the environment manually.

2. **[Build and Run](#1-build-and-run)**: Compile and install the custom operator package to quickly call the operator.

3. **[Operator Development](#2-operator-development)**: Modify the existing operator Kernel to experience the complete loop of development, compilation, and verification.

4. **[Operator Debugging](#3-operator-debugging)**: Learn the methods of operator printing and performance collection.

5. **[Operator Verification](#4-operator-verification)**: Learn how to modify the operator example sample to verify the functional correctness of the operator under different inputs.

## 1. Build and Run

The purpose of this phase is to **quickly experience the standard project process** and verify whether the environment can successfully build, package, install, and run the operator source code.

> This guide uses the single-operator compilation process as an example. It also supports building all operators in the operator library, offline compilation, and other tasks. For common issues during build, see [*Build from Source Code*](en/install/compile.md).

### 1. Enter the Project Source Code Directory

- CANNLab cloud development environment:

   It provides the project source code supporting the latest CANN package by default. The resources are generally in the `/mnt/workspace/gitCode` directory. Enter the target branch source code directory.

- Non-CANNLab cloud development environment:

  Obtain the version compatibility between the source code and CANN from the release-notes files in each folder of the [release repository](https://atomgit.com/cann/release-management) (the CANN ops combination package supports independently upgrading subpackage compatibility). Run the following command to download the source code. Replace \$\{tag\_version\} with the target branch tag, for example, 9.0.0.

  ```bash
  git clone -b ${tag_version} https://atomgit.com/cann/ops-transformer.git && cd ops-transformer
  ```

> Note: If you need to switch the source code branch version, see the following instructions.
>
> 1. In the source code directory, run `git branch` to query the current source code version.
> 2. In the source code directory, run `git checkout ${tag_version}` to switch to the target branch source code. Ensure that the source code and CANN version compatibility is met. If the source code already exists, run `git pull` to pull the latest source code.

### 2. Build the AddExample Operator

To build a specified operator, the general build command format is: `bash build.sh --pkg --soc=<chip version> --ops=<operator name>`.

> **Note**: Before build, ensure that the CANN environment variables are configured. Otherwise, build may fail because `ASCEND_HOME_PATH` cannot be found. For the default path installation, run:
>
> ```bash
> source /usr/local/Ascend/cann/set_env.sh
> ```

Use the AddExample operator as an example. The build command is as follows:

```bash
bash build.sh --pkg --soc=${soc_version} --ops=add_example -j16
```

<a id="soc-version-mapping"></a>

The ${soc_version} values corresponding to the product name are as follows. Pass parameters according to the actual scenario.

- Atlas A2 products: The value is ascend910b.
- Atlas A3 products: The value is ascend910_93.
- 950 products: The value is ascend950.

If the following information is displayed, the compilation is successful.

```bash
Self-extractable archive "cann-ops-transformer-custom_linux-${arch}.run" successfully created.
```

After successful compilation, the run package is stored in the build_out directory under the project root directory.

### 3. Install the AddExample Operator Package

```bash
./build_out/cann-ops-transformer-*linux*.run
```

`AddExample` is installed in the ```${ASCEND_HOME_PATH}/opp/vendors``` path. ```${ASCEND_HOME_PATH}``` indicates the CANN software installation directory.

> After installation, run `grep load_priority ${ASCEND_HOME_PATH}/opp/vendors/config.ini` to verify. The expected output contains this vendor. If there is no output, the owner or permission of `${ASCEND_HOME_PATH}/opp/vendors` is usually incorrect. Troubleshoot and then reinstall.

### 4. Configure Environment Variables

Add the path of the custom operator package to the environment variable to ensure that it can be found at runtime.

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

### 5. Quick Verification: Run the Operator Sample

The general run command format is: `bash build.sh --run_example <operator name> <run mode> <package mode> --vendor_name=<custom operator package name>`.

Use AddExample as an example. It provides a simple operator sample `add_example/examples/test_aclnn_add_example.cpp`. Run this sample to verify whether the operator function is normal.

```bash
bash build.sh --run_example add_example eager cust --vendor_name=custom --soc=${soc_version}
```

> **Note**: When running the sample, ensure that the `--soc` parameter is consistent with the `--soc` value used when compiling the operator package. Otherwise, `error 161001` may be reported (for example, `aclnnXxxGetWorkspaceSize failed`). If this error occurs, return to [Section 2](#2-build-the-addexample-operator) to check the `--soc` value and then rebuild and install.

- operator name: add_example
- run mode: eager
- package mode: cust
- --vendor_name (optional): vendorname indicates the custom operator package name to build. The default name is custom.

Expected output: prints the addition computation result of the `AddExample` operator, indicating that the operator has been successfully deployed and executed correctly.

```bash
add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 2.000000
add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 2.000000
add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 2.000000
add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 2.000000
add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 2.000000
add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 2.000000
add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 2.000000
add_example first input[7] is: 1.000000, second input[7] is: 1.000000, result[7] is: 2.000000
...
```

## 2. Operator Development

The purpose of this phase is to try to **modify the Kernel code** of the successfully run AddExample operator.

### 1. Modify the Kernel Implementation

Find the core kernel implementation file `ops-transformer/examples/add_example/op_kernel/add_example.h` of the AddExample operator, and try to change the Add operation in the operator to a Mul operation:

```cpp
__aicore__ inline void AddExample<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    // === Replace Add with Mul here ===
    // AscendC::Add(zLocal, xLocal, yLocal, tileLength_);
    AscendC::Mul(zLocal, xLocal, yLocal, tileLength_);
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}
```

### 2. Build and Verify

Repeat the steps in the [Build and Run](#1-build-and-run) section:

1. **Rebuild**:
    First return to the project root directory. The build command is as follows:

    ```bash
    bash build.sh --pkg --soc=${soc_version} --ops=add_example -j16
    ```

    For the ${soc_version} values corresponding to the product name, see the [SoC Value Mapping Table](#soc-version-mapping). Pass parameters according to the actual scenario.

2. **Reinstall**:

    ```bash
    ./build_out/cann-ops-transformer-*linux*.run
    ```

3. **Re-verify**:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom --soc=${soc_version}
    ```

4. **Success indicator**: The output result becomes a multiplication result.

    ```bash
    add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 1.000000
    add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 1.000000
    add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 1.000000
    add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 1.000000
    add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 1.000000
    add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 1.000000
    add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 1.000000
    add_example first input[7] is: 1.000000, second input[7] is: 1.000000, result[7] is: 1.000000
    ...
    ```

## 3. Operator Debugging

In this phase, use AddExample as an example to add printing in the operator and collect operator performance data for subsequent problem analysis and location.

### 1. Operator Printing

If the operator encounters issues such as execution failure or precision exceptions, add printing for problem analysis and location.

Modify the code in `ops-transformer/examples/add_example/op_kernel/add_example.h`.

- **PRINTF**

  This API supports printing Scalar type data, such as integers, characters, and booleans. For details, see "Ascend C Operator Development APIs > SIMD APIs > Basic APIs > Debugging APIs > On-board Printing > printf" in [*Ascend C APIs*](https://hiascend.com/en/document/redirect/CannCommunityAscendCApi).

  ```c++
  blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;
  // Print the Block length computed by the current core.
  AscendC::PRINTF("Tiling blockLength is %ld\n", blockLength_);
  ```

- **DumpTensor**

  This API supports dumping the content of a specified Tensor and printing custom additional information, such as the current line number. For details, see "Ascend C Operator Development APIs > SIMD APIs > Basic APIs > Debugging APIs > On-board Printing > DumpTensor" in [*Ascend C APIs*](https://hiascend.com/en/document/redirect/CannCommunityAscendCApi).

  ```c++
  // In the Compute function, call after Add and before EnQue.
  DumpTensor(zLocal, 0, 128);
  ```

### 2. Performance Collection

After the operator function is verified to be correct, use the `msprof op` command to collect operator-level performance data.

1. **Generate an executable file**

    Call the example sample of the AddExample operator to generate an executable file (test_aclnn_add_example). The file is located in the `ops-transformer/build` directory of the project.

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom --soc=${soc_version}
    ```

2. **Collect performance data**

    Enter the executable file directory `ops-transformer/build/` of the AddExample operator and run the following command:

    ```bash
    msprof op ./test_aclnn_add_example
    ```

    After execution, the basic operator information (such as Op Name, Op Type, Task Duration, and Block Dim) and performance bottleneck prompts are printed directly.

The collection result is stored in the `OPPROF_*` folder under the `ops-transformer/build/` directory of the project. After the command is executed, the performance data file is automatically parsed and exported. If you need to further interpret each performance metric (such as pipeline ratio and bandwidth utilization), see [*msProf User Guide*](https://www.hiascend.com/en/document/redirect/CannCommunityToolMsprof).

## 4. Operator Verification

In this phase, modify the input data in the example sample of the AddExample operator to verify the functional correctness of the operator in multiple scenarios.

### 1. Modify the Test Input

Find and edit `ops-transformer/examples/add_example/examples/test_aclnn_add_example.cpp` of `AddExample`, and modify the shape and values of the input tensor.

**Modify input/output data**: Modify the shape information of the input and output, as well as the initialization data, to construct the corresponding input and output tensors.

```c++
int main() {
    // ... Initialization code ...

    // === ① Modify the input of selfX ===
    // Before modification: shape = {32, 4, 4, 4}, all values are 1
    // After modification: change the input shape to {8, 8, 8, 8} and fill different test data
    std::vector<int64_t> selfXShape = {8, 8, 8, 8};
    std::vector<float> selfXHostData(4096); // 4096 = 8 * 8 * 8 *8
    // Use a loop to fill more distinguishable data, for example, an increasing sequence
    for (int i = 0; i < 4096; ++i) {
        selfXHostData[i] = static_cast<float>(i % 10); // Fill cyclic values 0-9
    }
    // === ② Refer to selfX and similarly modify the shape and data of selfY (the second input) and out (the output) ===

    // ... Subsequent execution code ...
}
```

> **Note**: When modifying the shape, synchronously modify the length of the corresponding host-side data vector (in this example, the vector lengths of `selfX`, `selfY`, and `out` must all be changed from the default 2048 to 4096 = 8×8×8×8), and ensure that the three shapes are consistent. Otherwise, data out-of-bounds or result mismatch may occur.

### 2. Rebuild and Verify

1. Since only the example test code is modified, there is no need to rebuild the operator package.

2. Run the verification command again:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom --soc=${soc_version}
    ```

3. Observe whether the operator output result meets expectations.

## Conclusion

After the above process, you have basically completed the operator development process. If you want to further contribute new operators or learn more advanced development and debugging skills, see the [Advanced Tutorials](../README_en.md#learning-tutorials) and [Contribution Guide](../README_en.md#related-information) in the project README.
