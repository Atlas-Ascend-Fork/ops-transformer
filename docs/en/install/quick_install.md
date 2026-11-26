# Environment Deployment

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-12T12:20:50.901Z pushedAt=2026-11-18T07:30:05.169Z nl=2 -->

Before learning the operations in the [learning tutorials](../../../README_en.md#learning-tutorials), refer to the following steps to complete the basic environment setup and source code download. Ensure that the NPU driver, firmware, and CANN software (`Ascend-cann-toolkit` and `Ascend-cann-ops`) are installed.

## Environment Installation

This project provides multiple methods to set up the Ascend environment. Select one as needed.

> **Note**: The build state and runtime state mentioned in this document are defined as follows. Select one according to the actual situation.
>
> - Build state: For the use case of compiling this project only without running it, install only the CANN Toolkit package.
> - Runtime state: For the use case of running this project (build and run, or run only), install the driver, the CANN Toolkit package, and the CANN ops package.

|  Installation Method  |  Instruction  |  Use Case  |
| ----- | ------ | ------ |
|  CANNLab  | A one-stop development platform that provides an online Ascend environment for direct running, with no manual installation required.<br>Currently provides single-device computing power and **installs the latest version of the CANN package by default**. | Suitable for developers without Ascend devices.|
|  Docker  | A Docker image is an efficient deployment method with the CANN package and essential dependencies pre-integrated.<br>Currently applicable to Atlas A2 and A3 series products, with OS support for ubuntu22.04, openEuler24.03, and others. **Installs the latest version of the CANN package by default**. |Suitable for developers with Ascend devices who need to quickly set up the environment.|
|  Manual installation  | Manually install the CANN package and basic dependencies, with high flexibility. |Suitable for developers with Ascend devices who want to experience manual installation of the CANN package or experience the latest master branch capability.|

### Method 1: CANNLab

For developers without Ascend devices, use the CANNLab cloud development environment directly, that is, the "**one-stop development platform**". This platform provides an online Ascend environment that can run directly. The essential driver, software packages, and dependencies are already installed in the environment, with no manual installation required.

> **Note**: The environment installs the latest version of the CANN package by default. When downloading the source code, ensure it matches the software version. For more information about the development platform, refer to [CANNLab Guide](https://atomgit.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md).

1. Enter the open-source project and click the "`CANNLab`" button. Log in with an authenticated Huawei Cloud account. If you have not registered or authenticated, register and authenticate according to the page prompts.

   <img src="../figures/cloudIDE.png" alt="cloud platform"  width="750px" height="85px">

2. Create an NPU environment and configure the specifications according to the page prompts. After starting the cloud development environment, click "`Connect > WebIDE`" to enter the one-stop development platform. The current open-source project resources are in the `/mnt/workspace/gitCode` directory by default.

   <img src="../figures/webIDE.png" alt="cloud platform"  width="1000px" height="150px">

### Method 2: Docker Deployment

For developers with Ascend devices, if you want to quickly set up the Ascend environment, use Docker image deployment.

> **Note**:
>
> - The image file is large and takes some time to download. Please wait patiently. For the option descriptions of the docker command, query them through `docker --help`.
> - The environment installs the latest version of the CANN package by default. When downloading the source code, ensure it matches the software version.

1. **Install the driver (runtime dependency)**

    The driver is a runtime dependency. If you only compile operators, you can skip the installation. Use `npu-smi info` to check whether there is NPU-related information. If not, refer to *[CANN Installation](https://www.hiascend.com/en/cann/download)* to complete the driver installation.

2. **Download the image**

    - Step 1: Log in to the host machine as the root user. Ensure that the Docker engine (version 1.11.2 or later) is installed on the host machine. Use `docker --version` to check the Docker version. If not, refer to the [Docker installation guide](https://docs.docker.com/engine/install/).
    - Step 2: Pull the image with the CANN software package and the dependencies required by `ops-transformer` pre-integrated from the [Ascend image repository](https://www.hiascend.com/en/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884).

    The example is as follows. Replace the CANN version number, chip series, operating system, python version, and other information on your own. The supported values of each field can be queried on the Ascend image repository page above.

    ```bash
    # Take cann:9.1.0-beta.1 as an example
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-openeuler24.03-py3.12-devel
    ```

    > **Note**: The image tag format is `<CANN version>-<chip series>-<operating system>-<Python version>-devel`. Images with the `-devel` suffix are operator development images and contain operator development compilation dependencies.

3. **Run Docker**

After pulling the image, start the container with specific parameters so that the Ascend devices of the host can be accessed inside the container.

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-openeuler24.03-py3.12-devel bash
```

> **Note**: `--name` is used to specify the container name. Replace it with a custom name for ease of identification. If the name is already in use, the container fails to start.

| Parameter | Description | Notes |
| :--- | :--- | :--- |
| `--name cann_container` | Specifies a name for the container for ease of management. | Customizable. |
| `--device /dev/davinci0` | Core: maps the NPU device of the host machine into the container. Multiple NPU devices can be specified for mapping. | Must be adjusted according to the actual situation: `davinci0` corresponds to NPU device 0 in the system. Run the `npu-smi info` command on the host machine first, and modify this number according to the device numbers displayed in the output (such as `NPU 0`, `NPU 1`).|
| `--device /dev/davinci_manager` | Maps the NPU device management interface. | - |
| `--device /dev/devmm_svm` | Maps the device memory management interface. | - |
| `--device /dev/hisi_hdc` | Maps the communication interface between the host and the device. | - |
| `-v /usr/local/dcmi:/usr/local/dcmi` | Mounts the tools and libraries related to the Device Container Management Interface (DCMI). | - |
| `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | Mounts the `npu-smi` tool. | Enables running this command directly inside the container to query the NPU status and performance information.|
| `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | Key mount: maps the NPU driver libraries of the host machine into the container. | - |
| `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | Mounts the driver version information file. | - |
| `-v /etc/ascend_install.info:/etc/ascend_install.info` | Mounts the CANN software installation information file. | - |
| `-it` | A combined parameter of `-i` (interactive) and `-t` (allocate a pseudo-terminal). | - |
| `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-openeuler24.03-py3.12-devel` | Specifies the Docker image to run. | Ensure that this image name and tag are exactly the same as the image you pulled through `docker pull`. |
| `bash` | The command executed immediately after the container starts. | - |

### Method 3: Manual Installation

For developers with Ascend devices, if you want to manually set up the Ascend environment, refer to the following steps.

#### Install Software

- **Experience branch capability or development based on the CANN stable version**

    Refer to *[CANN Installation](https://www.hiascend.com/en/cann/download)*, select `Stable version` (only CANN 8.5.0 and later versions are supported), and download the corresponding package based on the product series, CPU architecture, and operating system. Finally, refer to the commands provided on the page to complete the installation.

#### Install Basic Dependencies

The basic dependencies of this project are as follows. Ensure that the version number requirements are met.

- Python >= 3.7.0 (recommended version <= 3.10)
- gcc >= 7.3.0
- CMake >= 3.18.4
- pigz (optional, improves packaging speed after installation, recommended version >= 2.4)
- dos2unix
- make
- patch
- GoogleTest (required only when executing UT, recommended version [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0))

The above dependencies can be installed with one click through the project script. The operation steps are as follows:

1. Download the source code.

    Download the branch source code matching the CANN version. The commands are as follows. Replace `${tag_version}` with the branch tag name.

    ```bash
    git clone -b ${tag_version} https://atomgit.com/cann/ops-transformer.git
    ```

2. Install dependencies.

    First, install the above dependencies with one click through `install_deps.sh` in the project root directory. The commands are as follows. If the system is not supported, refer to this file to adapt it on your own.

    ```bash
    bash install_deps.sh
    ```

    Then, continue to install the Python third-party library dependencies through `requirements.txt` in the project root directory. The commands are as follows.

    ```bash
    pip3 install -r requirements.txt
    ```

## Environment Verification

After installing the CANN package, verify whether the environment and the driver are normal.

- **Check the NPU device**

    ```bash
    # Run npu-smi. If the device information is displayed normally, the driver is normal.
    npu-smi info
    ```

- **Check the CANN version**

    ```bash
    # View the CANN toolkit package version information (installed to the default path)
    # Docker and manual installation use cases:
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_toolkit_install.info
    # CANNLab use case:
    cat /home/developer/Ascend/cann/${arch}-linux/ascend_toolkit_install.info

    # View the CANN ops package version information (installed to the default path)
    # Docker and manual installation use cases:
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_ops_install.info
    # CANNLab use case:
    cat /home/developer/Ascend/cann/${arch}-linux/ascend_ops_install.info
    ```

    `${arch}` can be queried through `uname -m` for the current architecture, such as aarch64 and x86_64.

## Environment Variable Configuration

Select the appropriate command as needed to make the environment variables take effect.

```bash
# Installed to the default path, using the root user as an example (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Installed to a specified path
# source ${install_path}/cann/set_env.sh
```
