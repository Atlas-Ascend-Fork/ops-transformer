# Documentation Center

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-18T08:14:36.812Z pushedAt=2026-11-18T08:25:31.499Z nl=2 -->

## Directory Structure

The `docs` directory structure is described as follows:

```text
├── zh
  ├── context                            # Common documents, such as terminology and basic concepts
  ├── debug                              # Operator debugging guide documents
  │   ├── op_debug_prof.md
  │   ├── ...
  ├── develop                            # Operator development guide documents
  │   ├── aicore_develop_guide.md
  │   ├── aicpu_develop_guide.md
  │   ├── metadata_develop_guide.md
  │   ├── graph_develop_guide.md
  │   ├── ...
  ├── figures                            # Image directory
  ├── install                            # Environment installation and compilation guide documents
  │   ├── build.md                       # build parameter description
  │   ├── compile.md                     # Source code build guide (offline compilation, custom operator package compilation, transformer package compilation, etc.)
  │   ├── quick_install.md               # Environment deployment, CANN package installation, etc.
  │   ├── dir_structure.md               # Project directory hierarchy introduction
  │   └── ...
  ├── invocation                         # Operator call guide documents (including aclnn call, graph mode call, etc.)
  │   ├── quick_op_invocation.md
  │   ├── ...
  ├── menu_aclnn_api.md                  # Full aclnn API index file
  ├── menu_torch_api.md                  # Full torch_extension API index file
  ├── ascend950_op_list.md               # Ascend 950 operator list
  ├── op_api_list.md                     # aclnn API list
  ├── op_list.md                         # Full operator list
  └── torch_api_list.md                  # torch_extension API list
├── en                                   # English documentation
  ├── context                            # Common documentation
  ├── debug                              # Operator debugging guides
  ├── develop                            # Operator development guides
  ├── figures                            # Image directory
  ├── install                            # Environment installation and compilation guides
  ├── invocation                         # Operator invocation guides
  ├── menu_aclnn_api.md                  # aclnn API index file
  ├── menu_torch_api.md                  # torch_extension API index file
  ├── op_api_list.md                     # aclnn API list
  ├── op_list.md                         # Operator list
  └── torch_api_list.md                  # torch_extension API list
├── CONTRIBUTING_DOCS.md                 # Document contribution description
├── QUICKSTART.md                        # Quick start
└── README.md
```

## Advanced Tutorials

### Guide Documentation

|   Documentation                   |   Description                 |
| --------------------------- | -------------------- |
| [Build from Source Code](en/install/compile.md)                        | Introduces different source code build methods and verification methods in online/offline scenarios.          |
| [Operator Call](en/invocation/quick_op_invocation.md)         | Introduces the method of calling operator examples and different operator call methods (such as PyTorch/aclnn/graph). |
| [Standard Operator Development](en/develop/aicore_develop_guide.md)       | Introduces how to define operator prototypes and implement Tiling and Kernel based on the standard project. Such operators are called "standard operators".<br>Standard operators support aclnn and graph mode calls. |
| [AI CPU Operator Development Guide](en/develop/aicore_develop_guide.md)     | Describes how to define an AI CPU operator prototype and implement Kernel based on the standard project. Such operators are called "AI CPU operators".<br>AI CPU operators support aclnn and graph mode calls. |
| [Metadata Operator Development Guide](en/develop/metadata_develop_guide.md) | Describes the development method of metadata-type pre-slicing operators (running on AI CPU) and how they work with the main operator (metadata as an optional input). |
| [Simple Operator Development](../examples/fast_kernel_launch_example/README.md) | Introduces how to implement fast_kernel_launch based on the simple project, that is, the `<<<>>>` method. Such operators are called "simple operators".<br>Simple operators support only PyTorch calls. |
| [Operator Debugging and Tuning](en/debug/op_debug_prof.md)                    | Introduces common operator function debugging and performance tuning methods (such as data collection and simulation pipeline). |

### API Documentation

| Documentation        | Description                  |
| ----------------------- | ---------------------- |
| [Operator List](en/op_list.md)                        | Describes the built-in standard operator list of the project. |
| [aclnn API List](en/op_api_list.md)               | Describes the aclnn API list provided by the project. All operators are encapsulated as C language APIs prefixed with `aclnn` to facilitate direct calls on the Host side. |
| [PyTorch API List](en/torch_api_list.md) | Introduces the torch_extension API list provided by the project. It bridges PyTorch and aclnn APIs through JIT compilation, and supports the TorchAir graph mode through GE Converter. |

### Tool Documentation

| Documentation        | Description                  |
| ----------------------- | ---------------------- |
| [Simulator Simulation Tool](en/debug/npu_sim.md) | A SoC-level simulation tool for operator development scenarios, used to analyze the precision and performance data of AI tasks running on the AI simulator at each stage. |

### Sample Practice

To learn high-performance practical cases in the operator domain, see the [cann-samples Performance directory](https://atomgit.com/cann/cann-samples/blob/master/Samples/2_Performance/README.md).

|  Operator Category       |  Sample Operator       | Description                  |
| ----------------------- | ---------------------- | ---------------------- |
|  attention  |   full_quant_fused_infer_attention_score_story    |  Provides a per-block full quantization implementation example for the FIA (Fused Infer Attention Score) operator, including input data generation, operator execution, and result verification.|
|  mc2        | moe_dispatch_and_combine_story | Provides performance optimization practice for the moe dispatch/combine communication operator, including build and run commands, test data generation, and precision verification. |
|  moe        | moe_init_routing_story | Describes the complete performance optimization practice of the MoeInitRoutingV3 operator. It includes optimization strategies such as multi-core parallelism, memory bandwidth optimization, intra-core pipeline layout, SIMT programming, and hardware feature adaptation, serving as an end-to-end tuning guide from theoretical analysis to code practice.|

### More Documentation

Welcome to the [wiki center](https://atomgit.com/cann/ops-transformer/wiki/Home.md) to learn more about the project, including project positioning, supplementary operator development knowledge, performance optimization methods and sample practices, FAQ, and problem locating methods.

## Appendix

| Documentation                                | Description                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| [Operator Basic Concepts](en/context/basic_concept.md) | Introduces basic concepts and terminology in the operator domain, such as quantization/sparsity, data types, and data formats. |
| [build Parameter Description](en/install/build.md)   | Introduces the functions and values of the build.sh parameters in this project, including source code compilation, operator call, and debugging. |
