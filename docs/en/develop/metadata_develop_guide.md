# Metadata Operator Development Guide

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-12T12:14:59.362Z pushedAt=2026-11-18T02:51:06.814Z nl=2 -->

## Overview

> Description:
>
> 1. The Metadata operator is a preceding scheduling operator: it runs before the main operator (such as `flash_attn`) executes, reads variable-length sequence information and shape parameters, completes task split and scheduling computation, and writes the split and scheduling results into an INT32 metadata tensor as the input of the main operator.
> 2. The development method of the Metadata operator is consistent with that of AI CPU operators. This guide only introduces the parts of Metadata-type operators **that differ from ordinary AI CPU operators**. For repeated content such as basic concepts involved in operator development, CpuKernel base class usage, operator registration, and UT verification, refer to [*AI CPU Operator Development Guide*](./aicpu_develop_guide.md). For Tiling/Kernel development of the main operator (AI Core side), refer to [*AI Core Operator Development Guide*](./aicore_develop_guide.md).
> 3. build.sh: The commands involved in operator development can be viewed through `bash build.sh --help`. For the function parameter introduction, refer to [build parameter description](../install/build.md).

This guide uses the `FlashAttnMetadata` operator development as an example to introduce the new operator development process and the involved deliverables. For the complete sample code, refer to the `attention/flash_attn_metadata` directory in the repository.

### Background and Principle

As the computation logic of the main operator becomes increasingly complex, higher requirements are imposed on the organization and scheduling of computation tasks. To address this, the Metadata operator is introduced to complete task split and scheduling decisions before the main operator executes, decoupling scheduling logic from computation logic.

The Metadata operator is a type of preceding operator that runs on the AI CPU. Its execution process is as follows: read variable-length sequence information and shape parameters, execute the task split algorithm, and write the split result into an INT32 metadata tensor. After the main operator starts, it performs computation according to the task ranges described by the metadata.

The core values of this mechanism include:

1. **Data local access**: The actual values of information such as variable-length sequences are stored on the Device side. During the Host-side Tiling stage, only the tensor shape can be obtained, and the values cannot be read. The AI CPU also runs on the Device side and can directly read tensor values to complete precise splitting. The split result (metadata tensor) is also generated on the Device side and consumed by the main operator, requiring no data transfer between Host and Device throughout the process.
2. **Execution stream parallelism**: The Metadata operator can be scheduled and executed in an independent stream, overlapping with the computation of other operators on the AI Core except the main operator. This reduces the occupation of the critical computation path and improves the overall NPU utilization.
3. **Output result reuse**: In scenarios such as multi-layer attention, if the shape, variable length, mask, number of cores, and other information of each layer's input remain unchanged, the output of the Metadata operator can be directly reused without repeating the split computation.

### Guide Overview

1. [Project Creation](#project-creation): Before developing an operator, complete environment deployment and create the operator directory.
2. [Operator Definition](#operator-definition): Determine the operator function and prototype definition.
3. [Kernel Implementation](#kernel-implementation): Implement the per-core computation and metadata tensor write on the Device side (AI CPU).
4. [aclnn Adaptation](#aclnn-adaptation): The two-stage aclnn API and parameter verification of the Metadata operator.
5. [torch Adaptation](#torch-adaptation): Wrap aclnn as a PyTorch operator.
6. [Build and Deployment](#build-and-deployment): Complete operator compilation and installation through the project build script.
7. [Load Balancing](#load-balancing): Introduce the principle and processing flow of load balancing.
8. [Usage](#usage): Introduce the usage of the Metadata operator.
9. [Precautions](#notes): Precautions for Metadata operator development.

## Project Creation

The environment deployment and directory creation process is exactly consistent with that of AI CPU operators. For details, refer to the [Project Creation](./aicpu_develop_guide.md#project-creation) chapter of [*AI CPU Operator Development Guide*](./aicpu_develop_guide.md).

## Operator Definition

The operator definition requirements are consistent with those of AI CPU operators. For details, refer to the [Operator Definition](./aicpu_develop_guide.md#operator-definition) chapter of [*AI CPU Operator Development Guide*](./aicpu_develop_guide.md).

## Kernel Implementation

The Kernel implementation is consistent with that of AI CPU operators. For details, refer to the [Kernel Implementation](./aicpu_develop_guide.md#kernel-implementation) chapter of [*AI CPU Operator Development Guide*](./aicpu_develop_guide.md).

## aclnn Adaptation

The aclnn API follows the two-stage specification and is exactly consistent with that of AI CPU operators. For details, refer to the [aclnn Adaptation](./aicpu_develop_guide.md#aclnn-adaptation) chapter of [*AI CPU Operator Development Guide*](./aicpu_develop_guide.md).

## torch Adaptation

The torch adaptation method is consistent with that of AI CPU operators. As a preceding operator, the Metadata operator is highly bound to the main operator. It is recommended to write the torch-layer adaptation code of the two operators together, usually in the torch adaptation file of the main operator. In terms of invocation, the Metadata operator supports Aclgraph and GE graph integration. For details, refer to [*Graph Mode Adaptation Guide*](./graph_develop_guide.md).

## Build and Deployment

The compilation and deployment is consistent with that of AI CPU operators. For details, refer to the [Build and Deployment](./aicpu_develop_guide.md#build-and-deployment) chapter of [*AI CPU Operator Development Guide*](./aicpu_develop_guide.md).

## Load Balancing

Load balancing is the core responsibility of the Metadata operator: based on the computation cost of each task, the computation tasks are evenly distributed to each core participating in the computation, to maximize hardware utilization and minimize the overall computation latency. The split algorithm can be designed by the developer. This repository provides a mature load balancing implementation for Attention-type operators. For details, see `attention/common/op_kernel/load_balance`.

Taking the TND (variable-length sequence) scenario as an example, the sequence lengths of different batches differ. The following figure assumes that the numbers of Q/K/V heads are all 1, and the task space is divided in three dimensions (Batch, QuerySeq, KVSeq) to illustrate the task split method of load balancing.

![Load balancing schematic](../figures/load_balance_compute_example.png "Load balancing")

The processing flow of load balancing is as follows:

1. Divide the input according to the block size determined by Tiling to obtain each computation sub-block in the Batch dimension shown in the figure.
2. Select an appropriate cost model to compute the load of each sub-block, and aggregate them to obtain the total load.
3. Use the ratio of the total load to the number of cores as the average load of each core, and evenly distribute the sub-blocks to each core so that the load of each core approaches this average value.
4. Task split may cause the computation of a single Batch to span multiple cores. Reduction must be performed after all cores complete their computation. Such computation is called a reduction task. The load of a reduction task is computed based on the QuerySeq and KVSeq scales involved, and is distributed to available AIV cores for execution.

## Usage

As the preceding operator of the main operator, the Metadata operator must follow a strict order when invoked: generate the metadata first, and then the main operator consumes it. The following uses the torch API invocation of `flash_attn` as an example for description.

```python
import torch
import torch_npu
from cann_ops_transformer.ops import flash_attn, flash_attn_metadata

dev = torch.device('npu:0')

dtype = torch.bfloat16
batch_size = 1
q_len = 1024
kv_len = 1024
q_heads = 1
kv_heads = 1
head_dim = 128
mask_mode = 0
win_left = -1
win_right = -1
softmax_scale = 1.0 / (head_dim ** 0.5)

metadata = torch.ops.cann_ops_transformer.flash_attn_metadata(
    num_heads_q=q_heads,
    num_heads_kv=kv_heads,
    head_dim=head_dim,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=batch_size,
    max_seqlen_q=q_len,
    max_seqlen_kv=kv_len,
    mask_mode=mask_mode,
    win_left=win_left,
    win_right=win_right,
    layout_q='BNSD',
    layout_kv='BNSD',
    layout_out='BNSD',
)

q = torch.randn(batch_size, q_heads, q_len, head_dim, dtype=dtype, device=dev)
k = torch.randn(batch_size, kv_heads, kv_len, head_dim, dtype=dtype, device=dev)
v = torch.randn(batch_size, kv_heads, kv_len, head_dim, dtype=dtype, device=dev)

attn_out, softmax_lse = torch.ops.cann_ops_transformer.flash_attn(
    q,
    k,
    v,
    block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=metadata,
    softmax_scale=softmax_scale,
    mask_mode=mask_mode,
    win_left=win_left,
    win_right=win_right,
    max_seqlen_q=q_len,
    max_seqlen_kv=kv_len,
    layout_q='BNSD',
    layout_kv='BNSD',
    layout_out='BNSD',
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
```

## Notes

1. **Input parameter consistency**: The Metadata operator and the main operator are invoked in two stages through two independent APIs. The main operator **cannot verify** whether the metadata matches the parameters of this invocation. When invoking, ensure that the input parameters on both sides are exactly consistent. Otherwise, undefined behavior will be triggered (precision error, illegal memory access, program crash).
2. **Metadata tensor layout consistency**: The layout protocol of the metadata tensor is shared by the write side (Metadata operator) and the parse side (main operator). When modifying the layout, both sides must be updated synchronously. Otherwise, undefined behavior will be triggered (precision error, illegal memory access, program crash).
3. **Dynamic number of cores**: The size of the metadata tensor may depend on the number of AIC/AIV cores. When computing the tensor size, adaptation must be performed according to the actual number of cores.
4. **fallback function**: The Metadata operator may not contain tensor input parameters. In this case, the torch adaptation must register a fallback function to ensure that the dispatcher can correctly route to the NPU implementation. For an implementation example, refer to `attention/flash_attn/torch_extension/flash_attn.py`.
5. **Metadata default fallback**: If the main operator does not receive metadata, it is recommended to preset a simple per-core split algorithm on its Tiling side as a fallback. When metadata is not passed in, fall back to this alternative to ensure that the main operator function remains available.
