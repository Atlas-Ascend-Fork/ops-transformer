# Project Directory

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-12T12:16:18.922Z pushedAt=2026-11-18T02:51:06.822Z nl=2 -->

## Detailed Directory Hierarchy

> Some directories listed in this chapter are optional. Refer to the actual deliverables. In particular, the deliverables of the **single-operator directory** differ in different use cases. The details are as follows:
>
> - If the op_host directory is missing, the implementation of another operator's op_host may be called. For the call logic, see the source code implementation in the op_api or op_graph directory of the operator. Alternatively, the Kernel may not have an Ascend C implementation yet. If needed, developers are welcome to refer to the [contribution guide](../../../CONTRIBUTING.md) to contribute the operator.
> - If the op_kernel directory is missing, the implementation of another operator's op_kernel may be called. For the call logic, see the source code implementation in the op_api or op_graph directory of the operator. Alternatively, the Kernel may not have an Ascend C implementation yet. If needed, developers are welcome to refer to the [contribution guide](../../../CONTRIBUTING.md) to contribute the operator.
> - If the op_api directory is missing, the operator does not support aclnn calls yet.
> - If the op_graph directory is missing, the operator does not support graph mode calls yet.

```text
├── cmake                                               # Project compilation directory
│   ├── aclnn_ops_transformer.h.in                       # aclnn aggregate header file template
│   └── ...
├── common                                              # Project public header files and public code
│   ├── CMakeLists.txt
│   ├── inc                                             # Public header file directory
│   └── src                                             # Public code directory
├── experimental                                        # Directory for storing user-defined custom operators
│   ├── attention                                       # Optional. Directory for user-developed attention operators
│   │   └── CMakeLists.txt
│   └── ...
│
├── ${op_class}                                         # Operator category, such as attention, ffn, and gmm operators
│   ├${op_name}                                         # Operator project directory. ${op_name} indicates the operator name (lowercase and underscore form)
│   │   ├── CMakeLists.txt                              # Operator cmakelist entry
│   │   ├── README.md                                   # Operator introduction documentation
│   │   ├── docs                                        # Operator documentation directory
│   │   │   ├── aclnn${OpName}.md                       # Operator aclnn API introduction documentation. ${OpName} indicates the operator name (upper camel case form)
│   │   │   └── torchapi_${op_api}.md                   # Optional. torch_extension API documentation. The main name is consistent with the external Python API (lowercase and underscore)
│   │   ├── examples                                    # Operator call example directory
│   │   │   ├── test_aclnn_${op_name}.cpp               # Example of calling the operator through aclnn
│   │   │   └── test_geir_${op_name}.cpp                # Example of calling the operator through geir
│   │   ├── op_graph                                    # Graph fusion related implementation
│   │   │   ├── CMakeLists.txt                          # op_graph-side cmakelist file
│   │   │   ├── ${op_name}_graph_infer.cpp              # InferDataType file. Implements operator data type derivation
│   │   │   ├── ${op_name}_proto.h                      # Operator prototype definition. Used to identify the operator in the graph optimization and fusion phases
│   │   │   └── fusion_pass                             # Operator fusion rule directory
│   │   ├── op_host                                     # Host-side implementation
│   │   │   ├── config                                  # Optional. Binary configuration file. If not configured, the project is automatically generated
│   │   │   │   ├── ${soc_version}                      # Binary information configured for the operator on the NPU. ${soc_version} indicates the NPU model
│   │   │   │   │   ├── ${op_name}_binary.json          # Operator binary configuration file
│   │   │   │   │   └── ${op_name}_simplified_key.ini   # Operator SimplifiedKey configuration information
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_def.cpp                      # Operator information library. Defines basic operator information, such as name, input, output, and data type
│   │   │   ├── ${op_name}_infershape.cpp               # Optional. InferShape implementation. Derives the output shape according to the operator shape. If not configured, the output shape is the same as the input shape
│   │   │   ├── ${op_name}_tiling_${sub_case}.cpp       # Optional. Tiling optimization in certain sub-scenarios. ${sub_case} indicates the sub-scenario. For example, ${op_name}_tiling_arch35 is the optimization for the arch35 architecture. If the file does not exist, it indicates that the operator has no specific Tiling strategy for the corresponding sub-scenario. Note: The Tiling implementation file name must contain the `_tiling` identifier to be recognized by the compilation system and participate in compilation
│   │   │   ├── ${op_name}_tiling_${sub_case}.h         # Optional. Header file used for the Tiling implementation in the ${sub_case} sub-scenario
│   │   │   ├── ${op_name}_tiling.cpp                   # Optional. If the file does not exist, it indicates that there is no Tiling implementation in the corresponding scenario (dividing the tensor into multiple small blocks and distinguishing data types for parallel computation)
│   │   │   ├── ${op_name}_tiling.h                     # Optional. Header file used for the Tiling implementation
│   │   │   └── CMakeLists.txt                          # Host-side cmakelist file
│   │   ├── op_api                                      # Optional. Operator aclnn implementation file directory. If not configured, the project is automatically generated
│   │   │   ├── aclnn_${op_name}.cpp                    # Operator aclnn API implementation file
│   │   │   ├── aclnn_${op_name}.h                      # Operator aclnn API implementation header file
│   │   │   ├── ${op_name}.cpp                          # Operator l0 API implementation file
│   │   │   ├── ${op_name}.h                            # Operator l0 API implementation header file
│   │   │   └── CMakeLists.txt
│   │   │── op_kernel                                   # AI Core operator Device-side Kernel implementation
│   │   │   ├── ${sub_case}                             # Optional. Directory used in the ${sub_case} sub-scenario
│   │   │   │   ├── ${op_name}_${model}.h               # Operator kernel implementation file. ${model} indicates the user-defined file name suffix, usually the Tiling template name
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_tiling_key.h                 # Optional. TilingKey file. Defines the Key of the Tiling strategy and identifies different division methods. If not configured, it indicates that the operator has no corresponding Tiling strategy
│   │   │   ├── ${op_name}_tiling_data.h                # Optional. TilingData file. Stores Tiling strategy related configuration information, such as block size and parallelism. If not configured, it indicates that the operator has no corresponding Tiling strategy
│   │   │   ├── ${op_name}.cpp                          # Kernel entry file. Contains the main function and scheduling logic
│   │   │   └── ${op_name}.h                            # Kernel implementation file. Defines the Kernel header file, including function declarations, structure definitions, and logic implementation
│   │   └── tests                                       # Operator test case directory
│   │       ├── CMakeLists.txt
│   │       └── ut                                      # Optional. UT test cases. Develop corresponding cases according to the actual situation
│   └── ...
├── docs                                                # Project related documentation directory
├── examples                                            # End-to-end operator development and call examples
│   ├── add_example                                     # AI Core operator example directory
│   │   ├── CMakeLists.txt                              # Operator compilation configuration file
│   │   ├── examples                                    # Operator use example directory
│   │   ├── op_graph                                    # Operator graph construction related directory
│   │   ├── op_host                                     # Operator information library, Tiling, and InferShape related implementation directory
│   │   ├── op_kernel                                   # Operator Kernel directory
│   │   └── tests                                       # Operator test case directory
│   ├── CMakeLists.txt
│   └── README.md                                       # Project example introduction documentation
├── scripts                                             # Script directory. Contains custom operator and Kernel build related configuration files
├── tests                                               # Project-level test directory
├── torch_extension                                     # torch extension directory. Used to expose the torch extension API
│   ├── cann_ops_transformer                            # torch extension module root directory
│       ├── common                                      # Public capability directory used by the torch extension
│       ├── docs                                        # torch_extension development specifications and others (for API documentation, see docs/ of each operator)
│       ├── op_builder                                  # OpBuilder base class. Manages JIT compilation and schema/meta registration
│       ├── ops                                         # torch extension operator API directory
│       │   ├── csrc                                    # C++ implementation file directory of the operator API
│       │   ├── graph_convert                           # Converter directory for adapting the operator API to the torchair graph mode
│       │   ├── ${api_name}.py                          # Python frontend definition and implementation of the operator API
│       │   └── __init__.py                             # ops subpackage entry. Exports the operator API and Converter
│       └── __init__.py                                 # Package root entry. Exports the operator API externally
├── CMakeLists.txt                                      # Project cmakelist entry
├── CONTRIBUTING.md                                     # Project contribution guide file
├── LICENSE                                             # Project open source license information
├── OAT.xml                                             # Configuration script. Used by repository tools to check whether the License is compliant
├── README.md                                           # Project general introduction documentation
├── SECURITY.md                                         # Project security statement file
├── build.sh                                            # Project compilation script
├── classify_rule.yaml                                  # Component division information
├── install_deps.sh                                     # Project installation dependency package script
├── requirements.txt                                    # Third-party dependency packages of the project
└── version.info                                        # Project version information
```
