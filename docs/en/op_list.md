# Operator List

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-12T12:22:32.071Z pushedAt=2026-11-18T02:51:06.852Z nl=2 -->

> **Instruction**:
>
> - **Operator directory**: The directory name is the lowercase underscore form of the operator name. Each directory contains all deliverables of the operator, including code implementation, examples, and documents. For details about the directory, see [Project Directory](./install/dir_structure.md).
> - **Operator execution hardware unit**: Most operators run on AI Core, and a few run on AI CPU. By default, operators mentioned in the project refer to AI Core operators. For details about AI Core and AI CPU, see [Ascend C Operator Development](https://hiascend.com/en/document/redirect/CannCommunityOpdevAscendC). In version 8.5.0 or later, the corresponding section is "Hardware Implementation". In other versions, the corresponding section is "Concepts and Terms > Hardware Architecture and Data Processing Principles".
> - **Operator API list**: To facilitate operator calls, CANN provides a set of C APIs for executing operators, generally prefixed with aclnn. For the full API list, see [aclnn List](op_api_list.md).
> - **V version evolution**: Some operators have multiple V versions. Select the highest V version when using them. A higher-version operator is compatible with all capabilities of lower-version operators.

All operator categories and operators provided by the project are as follows:

<table><thead>
  <tr>
    <th rowspan="2">Operator Category</th>
    <th rowspan="2">Operator Directory</th>
    <th colspan="2">Operator Implementation</th>
    <th>aclnn Call</th>
    <th>Graph Mode Call</th>
    <th rowspan="2">Operator Execution Hardware Unit</th>
    <th rowspan="2">Description</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
    <th>op_api</th>
    <th>op_graph</th>
  </tr></thead>
<tbody>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/attention_update/README_en.md">attention_update</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Updates the intermediate results lse and localOut output by the PA operator in each SP domain into global results.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/attention_worker_combine/README_en.md">attention_worker_combine</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Fuses the attention token data processed by multiple computation units, weights the result based on expert weights, and outputs the final attention fusion result.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/attention_worker_scheduler/README_en.md">attention_worker_scheduler</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>Attention-side data scanning operator in the scenario where Attention and FFN are deployed separately.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/block_sparse_attention/README_en.md">block_sparse_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Completes attention computation based on the block sparse mode and supports multiple Query and KV Cache layouts.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/bsa_select_block_mask/README_en.md">bsa_select_block_mask</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>A preceding operator of BSA (BlockSparseAttention). It dynamically generates blockSparseMask based on the Query and Key content, changing the BSA call chain from "manually providing a mask" to "adaptively selecting the sparse mode based on Q/K content".</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/block_attn_res_prepare/README_en.md">block_attn_res_prepare</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>BlockAttnRes first-stage preparation operator. Computes inter-block attention for all target slots in parallel, and outputs the softmax weighted numerator and statistics for fusing the current residual in the next stage.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/block_attn_res_update/README_en.md">block_attn_res_update</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>BlockAttnRes second-stage update operator. Updates the partial block and computes the attention output corresponding to the online Softmax state.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/block_sparse_attention_grad/README_en.md">block_sparse_attention_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Computes the backward output of attention in training scenarios, that is, the backward computation of BlockSparseAttention.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/chunk_gated_delta_rule/README_en.md">chunk_gated_delta_rule</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Chunk version of the linear attention Gated Delta Rule in inference scenarios. Used for prefill nodes.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/chunk_kda_fwd/README_en.md">chunk_kda_fwd</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Completes the chunked forward computation of KDA without CP splitting. Outputs the attention result, final state, and optional backward intermediate result.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_grad_kl_loss/README_en.md">dense_lightning_indexer_grad_kl_loss</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Computes the backward gradient of LightningIndexer and fuses the KL divergence loss computation.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_softmax_lse/README_en.md">dense_lightning_indexer_softmax_lse</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Computes the Softmax intermediate result required by DenseLightningIndexerGradKlLoss. Outputs softmax_max and softmax_sum.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_softmax_lse_v2/README_en.md">dense_lightning_indexer_softmax_lse_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>The DenseLightningIndexerGradKlLoss operator computes a branch operator of the Softmax input. Compared with dense_lightning_indexer_softmax_lse, it adds compressed attention support and supports per-core load balancing through the metadata preceding operator.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_softmax_lse_v2_metadata/README_en.md">dense_lightning_indexer_softmax_lse_v2_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>The preceding operator of dense_lightning_indexer_softmax_lse_v2, used to compute the load balancing of dense_lightning_indexer_softmax_lse_v2.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_attn/README_en.md">flash_attn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Implements self-attention computation based on the FlashAttention algorithm. Supports training and inference normalization. Supports only non-quantization scenarios.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_attn_grad/README_en.md">flash_attn_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Computes the backward output of attention in training scenarios, that is, the backward computation of flash_attn. It must be used with the per-core metadata generated by the flash_attn_metadata preceding operator (is_grad_enabled=true).</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_attention_score/README_en.md">flash_attention_score</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Implements self-attention computation using the FlashAttention algorithm.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_attention_score_grad/README_en.md">flash_attention_score_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Computes the backward output of attention in training scenarios, that is, the backward computation of FlashAttentionScore.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/fused_causal_conv1d/README_en.md">fused_causal_conv1d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs causal 1D convolution on the sequence. Uses cached data along the sequence dimension (length is the convolution kernel width minus 1) to pad the head of each sequence, ensuring that the output depends on the current and historical inputs. After the convolution completes, updates the current sequence part data to the cache. Based on the output of causal 1D convolution, adds the original input to the output to implement residual connection. Supports APC (Automatic Prefix Caching), MTP (speculative decoding), residual connection, and other features. Passes in the maxDraftTokens parameter to specify the maximum speculative count.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/fused_floyd_attention/README_en.md">fused_floyd_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>In training scenarios, uses the FloydAttention algorithm to compute multi-dimensional self-attention.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/fused_floyd_attention_grad/README_en.md">fused_floyd_attention_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>In training scenarios, computes the backward output of Floyd attention. Compared with the traditional FA, FloydAttn additionally treats seq as the batch axis when computing qk/pv attention, thereby converting it to batchMatmul.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/fused_infer_attention_score/README_en.md">fused_infer_attention_score</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>FlashAttention operator for decode and prefill scenarios.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/gather_pa_kv_cache/README_en.md">gather_pa_kv_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>According to the blockId values in blockTables and the seqLen of key/value in seqLens, moves and concatenates non-contiguous tokens from keyCache/valueCache into contiguous key/value sequences.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/generic_block_sparse_attention_metadata/README_en.md">generic_block_sparse_attention_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>Preceding API of generic_block_sparse_attention, used to compute the load balancing of generic_block_sparse_attention.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/generic_block_sparse_attention/README_en.md">generic_block_sparse_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Forward attention computation that specifies the sparse block pattern through sparseBlockIdx/sparseBlockCount.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/generic_block_sparse_attention_grad/README_en.md">generic_block_sparse_attention_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Generic block sparse attention backward operator. Computes and propagates gradients on the selected Q blocks according to rsvdBlockIdx/rsvdBlockCount.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/generic_block_sparse_attention_grad_metadata/README_en.md">generic_block_sparse_attention_grad_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>Preceding operator of generic_block_sparse_attention_grad, used to compute the load balancing of generic_block_sparse_attention_grad.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/incre_flash_attention/README_en.md">incre_flash_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>FlashAttention operator for the incremental inference scenario.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/indexer_quant_cache/README_en.md">indexer_quant_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs in-place block-wise dynamic quantization compression and update on the KV Cache in the Epilog stage of the Indexer attention mechanism. Supports four quantization modes: MX-FP8, Normal, HiFloat8, and MX-FP4.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/inplace_fused_causal_conv1d/README_en.md">inplace_fused_causal_conv1d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs causal 1D convolution on the sequence. Uses cached data along the sequence dimension (length is the convolution kernel width minus 1) to pad the head of each sequence, ensuring that the output depends on the current and historical inputs. After the convolution completes, updates the current sequence part of the data to the cache. Based on the output of causal 1D convolution, adds the original input to the output to implement residual connection. Supports APC (Automatic Prefix Caching), MTP (speculative decoding), residual connection, in-place update, and other features. Passes in the maxDraftTokens parameter to specify the maximum speculative count.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/kda_input_proj/README_en.md">kda_input_proj</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Completes the preceding projection of Recurrent KDA and generates qkv, beta, gate, and g.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/kv_compress_epilog/README_en.md">kv_compress_epilog</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs in-place group-wise dynamic quantization compression and update on the cache in the Epilog stage of the KV Cache. Compresses bfloat16 activations to FP8 format and scatters them according to slotMapping.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/kv_quant_sparse_flash_attention/README_en.md">kv_quant_sparse_flash_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Supports [Per-Token-Head-Tile-128 quantization] input based on Sparse Flash Attention.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/kv_quant_sparse_flash_attention_v2/README_en.md">kv_quant_sparse_flash_attention_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Supports the sinks feature and output of softmax_max and softmax_sum based on Kv Quant Sparse Flash Attention.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer/README_en.md">lightning_indexer</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Obtains the Top-k positions corresponding to each token based on a series of operations.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer_grad/README_en.md">lightning_indexer_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Operator for the backward gradient calculation of lightning_indexer.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer_kl_loss/README_en.md">lightning_indexer_kl_loss</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Computes the KL divergence loss between the teacher distribution and the student distribution in Lightning Indexer.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer_v2/README_en.md">lightning_indexer_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Obtains the Top-k positions corresponding to each token based on a series of operations. Supports the KV compression use case.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer_v2_metadata/README_en.md">lightning_indexer_v2_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>Preceding operator of lightning_indexer_v2. Used to compute the load balancing of lightning_indexer_v2.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/masked_causal_conv1d/README_en.md">masked_causal_conv1d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs a masked causal 1D grouped convolution operation on the tokens of the hidden layer.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/masked_causal_conv1d_backward/README_en.md">masked_causal_conv1d_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Performs the backward gradient calculation of the 1D grouped convolution operation on the tokens of the hidden layer.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mixed_quant_sparse_flash_mla/README_en.md">mixed_quant_sparse_flash_mla</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Supports Sliding Window Attention, Compressed Attention, and Sparse Compressed Attention.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mixed_quant_sparse_flash_mla_metadata/README_en.md">mixed_quant_sparse_flash_mla_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>Preceding operator of the mixed_quant_sparse_flash_mla operator, used to compute the load balancing of mixed_quant_sparse_flash_mla.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_preprocess_v2/README_en.md">mla_preprocess_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Completes Multi-Head Latent Attention preprocessing, fusing normalization, matrix multiplication, rotary position encoding, and cache update.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_sparse_flash_mla/README_en.md">quant_sparse_flash_mla</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Supports Sliding Window Attention, Compressed Attention, and Sparse Compressed Attention computation in the full quantization scenario.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_sparse_flash_mla_metadata/README_en.md">quant_sparse_flash_mla_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>Preceding operator of the quant_sparse_flash_mla operator, used to compute the load balancing of quant_sparse_flash_mla.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_preprocess/README_en.md">mla_preprocess</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Inference MlaPreprocess operator.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_prolog/README_en.md">mla_prolog</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Inference MlaProlog operator.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_prolog_v2/README_en.md">mla_prolog_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Inference MlaPrologV2WeightNz operator.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_prolog_v3/README_en.md">mla_prolog_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Inference MlaPrologV3WeightNz operator.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/msa_index_score/README_en.md">msa_index_score</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Computes the block score of the MSA Index Branch. Performs matmul+maxpool on each query token and KV sparse block to obtain the importance score.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress/README_en.md">nsa_compress</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>In training scenarios, uses the NSA Compress algorithm to reduce the attention computation of long-context, implementing compression on the KV sequence dimension.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress_attention/README_en.md">nsa_compress_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Compress attention and select topk index computation in NSA.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress_attention_infer/README_en.md">nsa_compress_attention_infer</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Implements the Compress Attention computation during Native Sparse Attention inference.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress_grad/README_en.md">nsa_compress_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Backward computation of the aclnnNsaCompress operator.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress_with_cache/README_en.md">nsa_compress_with_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Implements KV compression in the Native Sparse Attention inference stage.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_selected_attention_infer/README_en.md">nsa_selected_attention_infer</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Implements the computation of Selected Attention during Native Sparse Attention inference.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_selected_attention/README_en.md">nsa_selected_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>In training scenarios, implements the computation of selected-attention in the NativeSparseAttention algorithm.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/rain_fusion_attention/README_en.md">rain_fusion_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Completes Rain sparse attention computation according to the sparse block selection result.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/scatter_pa_kv_cache/README_en.md">scatter_pa_kv_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Updates the key and value at the specified position in the KV Cache.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_lightning_indexer/README_en.md">sparse_lightning_indexer</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Consumer-side operator of the lightning_indexer_v2 candidate mode (candidateTopkBlocks enabled). It consumes the candidate block indexes output by the source, performs leak-down TopK on reachable positions outside the candidates, and selects the Top-k sparse positions for each token.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_lightning_indexer_grad_kl_loss/README_en.md">sparse_lightning_indexer_grad_kl_loss</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>The SparselightningIndexerGradKlLoss operator is the backward operator of LightningIndexer, and additionally fuses the Loss computation function output.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_selected_attention_grad/README_en.md">nsa_selected_attention_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Selects data of size selectedBlockSize from key and value according to topkIndices and rearranges it, then computes the backward output of attention in training scenarios.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/prompt_flash_attention/README_en.md">prompt_flash_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>FlashAttention operator for the full inference scenario.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_compressor/README_en.md">quant_compressor</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Preprocessing operator for SMLA and QLI in inference scenarios. It is the quantized version of Compressor. It compresses the KV cache of every 4 or 128 tokens into one, then each token performs DSA computation with these compressed KV caches.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_lightning_indexer/README_en.md">quant_lightning_indexer</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Preprocessing calculation of SparseFlashAttention in inference scenarios. It selects key sparse tokens and performs quantization on the input query and key to implement store-8-compute-8.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_lightning_indexer_v2/README_en.md">quant_lightning_indexer_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Preprocessing calculation of sparse attention in inference scenarios. It selects key sparse tokens and supports INT8/FLOAT8_e4m3fn/HIFLOAT8 quantization on the input query and key to implement store-8-compute-8 and obtain the maximum benefit.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_lightning_indexer_v2_metadata/README_en.md">quant_lightning_indexer_v2_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>Preceding operator of the quant_lightning_indexer_v2 operator. It is used to compute the load balancing of quant_lightning_indexer_v2.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/recurrent_gated_delta_rule/README_en.md">recurrent_gated_delta_rule</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Recurrent Gated Delta Rule operator for the incremental inference scenario.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/recurrent_kda/README_en.md">recurrent_kda</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Completes the recursive forward calculation of KDA (Kimi Delta Attention), for decode and MTP short-sequence tasks.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/ring_attention_update/README_en.md">ring_attention_update</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>In training scenarios, updates the results of two FlashAttention computations.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/scatter_pa_cache/README_en.md">scatter_pa_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Updates the key at the specified position in KCache.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_attention/README_en.md">sparse_flash_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>An efficient attention computation module for inference tasks with long sequence lengths.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_attention_grad/README_en.md">sparse_flash_attention_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>In training scenarios, computes the backward output of sparse_flash_attention attention.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla/README_en.md">sparse_flash_mla</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Supports SWA (Sliding Window Attention), CSA (Compressed Sparse Attention), and HCA (Heavily Compressed Attention).</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla_grad/README_en.md">sparse_flash_mla_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>In training scenarios, computes the backward output of sparse_flash_mla attention.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla_grad_metadata/README_en.md">sparse_flash_mla_grad_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>Preceding operator of sparse_flash_mla_grad, used to compute the load balancing of sparse_flash_mla_grad.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla_metadata/README_en.md">sparse_flash_mla_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>Preceding operator of sparse_flash_mla, used to compute the load balancing of sparse_flash_mla.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_lightning_indexer_kl_loss_grad/README_en.md">sparse_lightning_indexer_kl_loss_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>SparselightningIndexerKlLossGrad is the backward operator of LightningIndexer. It supports output of the partial score of the Index required for loss computation.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_lightning_indexer_kl_loss_grad_metadata/README_en.md">sparse_lightning_indexer_kl_loss_grad_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>Preceding operator of sparse_lightning_indexer_kl_loss_grad, used to compute the load balancing of sparse_lightning_indexer_kl_loss_grad.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/compressor/README_en.md">compressor</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Preprocessing operator of SMLA and QLI in inference scenarios. It compresses the KV cache of every 4 or 128 tokens into one, and then each token performs DSA computation with these compressed KV caches.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/compressor_grad/README_en.md">compressor_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Backward operator of Compressor, used to compute the gradients of input X, weights W^KV/W^Gate, and position encoding Ape. When gradEnabled is true, the forward pass exports softmax_score and the kv intermediate result as the input of this operator.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/stem_oam_prep_varlen_q/README_en.md">stem_oam_prep_varlen_q</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Q-side preprocessing operator in Stem OAM block-sparse attention. It converts the variable-length Q tensor from the paged storage format into a flattened qFlat output grouped by stem block.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/scatter_pa_kv_cache_with_k_scale/README_en.md">scatter_pa_kv_cache_with_k_scale</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>In training scenarios, updates the key and value at the specified position in KvCache, and updates the scale value of the key.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_kl_loss_grad/README_en.md">dense_lightning_indexer_kl_loss_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>The DenselightningIndexerKlLossGrad operator is the backward operator of LightningIndexer. It supports output of the partial score of the Index part required for Loss computation.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_kl_loss_grad_metadata/README_en.md">dense_lightning_indexer_kl_loss_grad_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>The preceding operator of the dense_lightning_indexer_kl_loss_grad operator. It is used to compute the load balancing of dense_lightning_indexer_kl_loss_grad.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla_softmax_l1_norm/README_en.md">sparse_flash_mla_softmax_l1_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>In training scenarios, it serves as the preceding API of dense_lightning_indexer_kl_loss_grad and is used to compute attnSoftmaxL1Norm.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_flash_attn_grad/README_en.md">quant_flash_attn_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Computes the backward gradient of quantized attention (QuantFlashAttention).</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla_softmax_l1_norm_metadata/README_en.md">sparse_flash_mla_softmax_l1_norm_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>The preceding operator of the sparse_flash_mla_softmax_l1_norm operator. It is used to compute the load balancing of sparse_flash_mla_softmax_l1_norm.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/stem_oam_prep_paged_kv/README_en.md">stem_oam_prep_paged_kv</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>The preceding scoring module of the dynamic sparse attention mechanism for large model inference. It is the preceding scoring module of block-sparse-attention.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/pool_key_indexer/README_en.md">pool_key_indexer</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Packs multiple consecutive tokens into one pool. Computes the attention relevance score per pool and selects the top-k positions. This reduces index overhead while retaining the advantages of sparse attention.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/key_pool/README_en.md">key_pool</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs K projection and Gate projection on each input token separately. Groups every consecutive `cmpRatio` tokens into one group. Merges the K values within a group into one Key based on the weight determined by Gate and position bias. Writes the intermediate state required for the uncompressed group into the paged `stateCache`.</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/swin_attention_score_quant/docs/aclnnSwinAttentionScoreQuant.md">swin_attention_score_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Completes the attention computation in the Swin Transformer scenario and supports int8 quantization.</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/ffn/README_en.md">ffn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Provides the computation of MoeFFN and FFN.</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/swin_attention_ffn/README_en.md">swin_attention_ffn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>A post-processing fusion operator for the attention output of Swin Transformer. Completes output linear projection (MatMul+Bias), window restoration, reverse cyclic shift, and residual addition.</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../experimental/ffn/swiglu_gated_mlp/README.md">swiglu_gated_mlp</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Completes the fused SwiGLU gated MLP computation, including the first MatMul, SwiGLU activation, and the second MatMul.</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/ffn_worker_scheduler/README_en.md">ffn_worker_scheduler</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>A side data scanning operator on the FFN side in the attention and FFN separation scenario.</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/ffn_worker_batching/README_en.md">ffn_worker_batching</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs side data scanning and token rearrangement on the FFN side in the attention and FFN separation scenario.</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/swin_transformer_ln_qkv/README_en.md">swin_transformer_ln_qkv</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Completes the computation of Q, K, and V for the Swin Transformer network model in the fp16 weight scenario.</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/swin_transformer_ln_qkv_quant/README_en.md">swin_transformer_ln_qkv_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Completes the computation of Q, K, and V for the Swin Transformer network model.</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul/README_en.md">grouped_matmul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Implements grouped matrix multiplication computation.</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_add/README_en.md">grouped_matmul_add</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Implements grouped matrix multiplication computation. The dimension size of each group of matrix multiplication can be different.</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_activation/README_en.md">grouped_matmul_activation</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Fuses A4W4 WeightNZ grouped matrix multiplication with split-half SwiGLU and directly outputs BF16 results.</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_activation_quant/README_en.md">grouped_matmul_activation_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Fuses GMM, activation functions, and quantization operators. Completes grouped matrix multiplication, activation, and quantization computation, and outputs the quantization result and quantization factor.</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_finalize_routing/README_en.md">grouped_matmul_finalize_routing</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>A fusion operator of GroupedMatmul and MoeFinalizeRouting. The output of GroupedMatmul is combined according to the index.</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_swiglu_quant/README_en.md">grouped_matmul_swiglu_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Fuses GroupedMatmul, dequant, swiglu, and quant.</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_swiglu_quant_v2/README_en.md">grouped_matmul_swiglu_quant_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Fuses GroupedMatmul, dequant, swiglu, and quant. Adds the MXFP8 quantization scenario (supported only by the AI processor of Ascend 950PR&950DT products).</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/quant_grouped_matmul_dequant/README_en.md">quant_grouped_matmul_dequant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Fuses input quantization, grouped matrix multiplication, and dequantization computation.</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/quant_grouped_matmul_inplace_add/README_en.md">quant_grouped_matmul_inplace_add</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Implements grouped matrix multiplication computation and addition computation.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/all_gather_matmul/README_en.md">all_gather_matmul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Completes AllGather communication and MatMul computation fusion.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/all_gather_matmul_v2/README_en.md">all_gather_matmul_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Completes AllGather communication and MatMul computation fusion. Extends the capability of all_gather_matmul.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/allto_all_all_gather_batch_mat_mul/README_en.md">allto_all_all_gather_batch_mat_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Completes AllToAll and AllGather collective communication fused and parallel with BatchMatMul computation.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/allto_all_matmul/README_en.md">allto_all_matmul</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Completes AlltoAll communication and MatMul computation fusion.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/allto_allv_grouped_mat_mul/README_en.md">allto_allv_grouped_mat_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Completes routing expert AlltoAllv, Permute, and GroupedMatMul fusion, and implements parallel fusion with shared expert MatMul. Communication first, then computation.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/allto_allv_quant_grouped_mat_mul/README_en.md">allto_allv_quant_grouped_mat_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Fuses the routing expert AlltoAllv, Permute, and quantized GroupedMatMul computation, and runs in parallel with the shared expert quantized MatMul.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/ffn_to_attention/README_en.md">ffn_to_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Within a communication domain, the FFN node sends data to the Attention node and writes a status bit to detect whether the communication link is normal.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/ffn_to_attention_v2/README_en.md">ffn_to_attention_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Within a communication domain, the FFN node sends data to the Attention node and writes a status bit to detect whether the communication link is normal.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/attention_to_ffn/README_en.md">attention_to_ffn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Within a communication domain, the Attention node sends data to the FFN node and writes a status bit to detect whether the communication link is normal.</td>
  </tr>
    <tr>
    <td>mc2</td>
    <td><a href="../../mc2/attention_to_ffn_v2/README_en.md">attention_to_ffn_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Within a communication domain, the Attention node sends data to the FFN node and writes a status bit to detect whether the communication link is normal.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/batch_mat_mul_reduce_scatter_allto_all/README_en.md">batch_mat_mul_reduce_scatter_allto_all</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Implements BatchMatMul computation in parallel with ReduceScatter and AllToAll collective communication.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/distribute_barrier/README_en.md">distribute_barrier</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Completes full-card synchronization within the communication domain. xRef is used only to build the Tensor dependency, and no operation is performed on xRef within the API.</td>
  </tr>
   <tr>
    <td>mc2</td>
    <td><a href="../../mc2/distribute_barrier_extend/README_en.md">distribute_barrier_extend</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Complete full-card synchronization within the communication domain. xRef is used only to build Tensor dependency. No operation is performed on xRef within the API.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/grouped_mat_mul_all_reduce/README_en.md">grouped_mat_mul_all_reduce</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Implement multi-device parallel AllReduce based on fused GroupedMatMul. Implement grouped matrix multiplication computation. The dimension size of each group of matrix multiplication can be different.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/grouped_mat_mul_allto_allv/README_en.md">grouped_mat_mul_allto_allv</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Complete routing expert GroupedMatMul, Unpermute, and AlltoAllv fusion. Implement parallel fusion with shared expert MatMul. Computation first, then communication.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/inplace_matmul_all_reduce_add_rms_norm/README_en.md">inplace_matmul_all_reduce_add_rms_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Complete mm + all_reduce + add + rms_norm computation.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/matmul_allto_all/README_en.md">matmul_allto_all</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Complete MatMul computation and AlltoAll communication fusion.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/matmul_all_reduce/README_en.md">matmul_all_reduce</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Complete MatMul computation and AllReduce communication fusion.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/matmul_all_reduce_add_rms_norm/README_en.md">matmul_all_reduce_add_rms_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td> Complete mm + all_reduce + add + rms_norm computation.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/matmul_reduce_scatter/README_en.md">matmul_reduce_scatter</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Complete mm + reduce_scatter_base computation.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/matmul_reduce_scatter_v2/README_en.md">matmul_reduce_scatter_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Completes the fusion of MatMul computation and ReduceScatter communication, extending the capability of matmul_reduce_scatter.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/mega_moe/README_en.md">mega_moe</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Completes the end-to-end fused computation of dispatch + group_matmul1 + swiglu_quant + group_matmul2 + combine.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine/README_en.md">moe_distribute_combine</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>When there is TP-domain communication, first perform ReduceScatterV communication, then perform AlltoAllV communication, and finally integrate the received data (multiply by weights and then add). When there is no TP-domain communication, perform AlltoAllV communication, and finally integrate the received data (multiply by weights and then add).</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine_add_rms_norm/README_en.md">moe_distribute_combine_add_rms_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>When there is TP-domain communication, first perform ReduceScatterV communication, then perform AlltoAllV communication, and finally integrate the received data (multiply by weights and then add). When there is no TP-domain communication, perform AlltoAllV communication, and finally integrate the received data (multiply by weights and then add), then complete the Add + RmsNorm fusion.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine_setup/README_en.md">moe_distribute_combine_setup</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>The preparation stage of MoE Combine. Sends data and starts communication, used together with teardown.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine_teardown/README_en.md">moe_distribute_combine_teardown</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>The final stage of MoE Combine. Waits for communication to complete and integrates the result, used together with setup.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine_v2/README_en.md">moe_distribute_combine_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>When there is TP-domain communication, first perform ReduceScatterV communication, then perform AlltoAllV communication, and finally integrate the received data. When there is no TP-domain communication, perform AlltoAllV communication, and finally integrate the received data. When expertScales is non-empty, multiply by weights and then add. When it is an empty Tensor, add directly.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch/README_en.md">moe_distribute_dispatch</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Quantize the token data (optional). When TP domain communication exists, first perform AllToAllV communication in the EP (Expert Parallelism) domain, then perform AllGatherV communication in the TP (Tensor Parallelism) domain. When there is no TP domain communication, perform AllToAllV communication in the EP (Expert Parallelism) domain.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch_setup/README_en.md">moe_distribute_dispatch_setup</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>The preparation stage of MoE Dispatch. Sends data and starts communication. Used together with teardown.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch_teardown/README_en.md">moe_distribute_dispatch_teardown</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>The final stage of MoE Dispatch. Waits for communication to complete and integrates the received data. Used together with setup.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch_v2/README_en.md">moe_distribute_dispatch_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Quantize the token data (optional). When TP domain communication exists, first perform AllToAllV communication in the EP (Expert Parallelism) domain, then perform AllGatherV communication in the TP (Tensor Parallelism) domain. When there is no TP domain communication, perform AllToAllV communication in the EP (Expert Parallelism) domain.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch_v3/README_en.md">moe_distribute_dispatch_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Quantize the token data (optional). When TP domain communication exists, first perform AllToAllV communication in the EP (Expert Parallelism) domain, then perform AllGatherV communication in the TP (Tensor Parallelism) domain. When there is no TP domain communication, perform AllToAllV communication in the EP (Expert Parallelism) domain.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_update_expert/README_en.md">moe_update_expert</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Completes the mapping from the topK logical expert IDs of each token to physical device IDs.</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/quant_grouped_mat_mul_allto_allv/README_en.md">quant_grouped_mat_mul_allto_allv</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Fuses the quantized GroupedMatMul, AlltoAllv, and Unpermute computation of the routing expert, and runs in parallel with the quantized MatMul of the shared experts.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_compute_expert_tokens/README_en.md">moe_compute_expert_tokens</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>During MoE computation, finds the position of the last row processed by each expert through binary search.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_finalize_routing/README_en.md">moe_finalize_routing</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>During MoE computation, finally process and merge the output result of the MoE FFN.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_finalize_routing_v2/README_en.md">moe_finalize_routing_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>During MoE computation, finally process and merge the output result of the MoE FFN.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_finalize_routing_v2_grad/README_en.md">moe_finalize_routing_v2_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Backward propagation of aclnnMoeFinalizeRoutingV2.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_fused_topk/README_en.md">moe_fused_topk</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Fuses Sigmoid, expert grouped filtering, and TopK computation to select routing experts and weights.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_gating_top_k/README_en.md">moe_gating_top_k</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>During MoE computation, perform Sigmoid, SoftMax, or SqrtSoftplus computation for input x, sort the computation results by group, and finally select the top k experts according to the grouped sort result.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_gating_top_k_backward/README_en.md">moe_gating_top_k_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Backward operator of MoeGatingTopK.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_gating_top_k_softmax/README_en.md">moe_gating_top_k_softmax</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>During MoE computation, perform Softmax computation on the output of x and take the TopK operation.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_gating_top_k_softmax_v2/README_en.md">moe_gating_top_k_softmax_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>In MoE computation, if renorm=0, perform Softmax on the output of x first, then perform the topk operation. If renorm=1, perform the topk operation on the output of x first, then perform the Softmax operation.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing/README_en.md">moe_init_routing</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE routing computation. Performs routing processing according to the computation result of <a href="../../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md">aclnnMoeGatingTopKSoftmax</a>.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_quant/README_en.md">moe_init_routing_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE routing computation. Performs routing processing according to the computation result of <a href="../../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md">aclnnMoeGatingTopKSoftmax</a>, and performs quantization on the result.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_quant_v2/README_en.md">moe_init_routing_quant_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE routing computation. Performs routing processing according to the computation result of <a href="../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md">aclnnMoeGatingTopKSoftmaxV2</a>.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_v2/README_en.md">moe_init_routing_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Takes the output x and expert_idx of the MoeGatingTopKSoftmax operator as input, and outputs the Routing matrix expanded_x and other results for subsequent computation.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_v2_grad/README_en.md">moe_init_routing_v2_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Backward propagation of <a href="../../moe/moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md">aclnnMoeInitRoutingV2</a>. Completes the weighted summation of tokens.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_v3/README_en.md">moe_init_routing_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE routing computation. Performs routing processing according to the computation result of <a href="../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md">aclnnMoeGatingTopKSoftmaxV2</a>. Supports non-quantization and dynamic quantization modes.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_v4/README_en.md">moe_init_routing_v4</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE routing computation. Performs routing processing according to the computation result of <a href="../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md">aclnnMoeGatingTopKSoftmaxV2</a>. Supports non-quantization, static quantization, and dynamic quantization modes.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_re_routing/README_en.md">moe_re_routing</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>In an MoE network, after the AlltoAll operation obtains the tokens to be computed from other devices, rearrange the tokens according to expert order.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_re_routing_v2/README_en.md">moe_re_routing_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>In an MoE network, after the AlltoAll operation obtains the tokens to be computed from other devices, rearrange the tokens according to expert order. Supports rearrangement of topkWeight.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute/README_en.md">moe_token_permute</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE permute computation. Broadcasts and sorts tokens according to the index indices.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_grad/README_en.md">moe_token_permute_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Backward propagation computation of aclnnMoeTokenPermute.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_ep/README_en.md">moe_token_permute_with_ep</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>MoE permute computation. Broadcasts and sorts tokens and optional probs according to the index indices, then slices them according to the range in rangeOptional.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_ep_grad/README_en.md">moe_token_permute_with_ep_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Backward propagation computation of aclnnMoeTokenPermuteWithEp.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_routing_map/README_en.md">moe_token_permute_with_routing_map</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>MoE permute computation. Passes in the token and expert labels as routingMap, then broadcasts and sorts tokens and optional probsOptional according to routingMap.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_routing_map_grad/README_en.md">moe_token_permute_with_routing_map_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>MoE permute computation. Sorts tokens and optional probs after broadcast according to the index indices, and slices them by the range in rangeOptional.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute/README_en.md">moe_token_unpermute</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Obtains the input data stored in permutedTokens according to the stored subscript of sortedIndices. If probs data exists, permutedTokens is multiplied by probs. Finally performs accumulation and summation, and outputs the computation result.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_grad/README_en.md">moe_token_unpermute_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Backward propagation of aclnnMoeTokenUnpermuteGrad.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_ep/README_en.md">moe_token_unpermute_with_ep</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Obtains the input data in permutedTokens according to the stored subscript position of sortedIndices, multiplies it by probs, and performs merge and accumulation.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_ep_grad/README_en.md">moe_token_unpermute_with_ep_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Backward propagation of aclnnMoeTokenUnpermuteWithEp.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_routing_map/README_en.md">moe_token_unpermute_with_routing_map</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Accumulates permutedTokens processed by aclnnMoeTokenpermuteWithRoutingMap back to the original unpermutedTokens. Obtains the input data stored in permutedTokens according to the stored subscript of sortedIndices. If probs data exists, permutedTokens is multiplied by probs. Finally performs accumulation and summation, and outputs the computation result.</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_routing_map_grad/README_en.md">moe_token_unpermute_with_routing_map_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Backward propagation of aclnnMoeTokenUnpermuteWithRoutingMap.</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mega_mhc/README_en.md">mega_mhc</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Fuses rms_norm_0, MHC Post, optional rms_norm_mid, MHC Pre, and rms_norm_1 to compute the next layer input, new mixing coefficients, and residual state.</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_sinkhorn/README_en.md">mhc_sinkhorn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Projects the mixing matrix of hyper-connections onto the doubly stochastic matrix manifold based on the Sinkhorn-Knopp iterative algorithm. This stabilizes signal propagation in deep networks and solves the vanishing/exploding gradient problem.</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_sinkhorn_backward/README_en.md">mhc_sinkhorn_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Backward operator of mhc_sinkhorn.</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_post/README_en.md">mhc_post</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs Post Mapping on the output of the previous layer in the mHC architecture based on a series of computations, performs Res Mapping on the input of the previous layer, then performs residual connection on the two to obtain the input of the next layer.</td>
  </tr>
   <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_post_backward/README_en.md">mhc_post_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>mhc_post performs Post Mapping on the output of the previous layer in the mHC architecture based on a series of computations, performs Res Mapping on the input of the previous layer, then performs residual connection on the two to obtain the input of the next layer. This operator implements the backward computation of the preceding process.</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_pre/README_en.md">mhc_pre</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Obtains the $H^{res}$ and $H^{post}$ projection matrices of the hidden layer in the MHC architecture and the input matrix $h^{in}$ of the Attention or MLP layer based on a series of computations.</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_pre_backward/README_en.md">mhc_pre_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Backward propagation of the mhc_pre operator. Obtains the gradient of the hidden layer in the MHC architecture based on a series of computations.</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_pre_sinkhorn/README_en.md">mhc_pre_sinkhorn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>The mhc_pre_sinkhorn operator supports the output of the Hin, Hpre, Hres, and Hpost matrices in the MHC architecture.</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_pre_sinkhorn_backward/README_en.md">mhc_pre_sinkhorn_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Backward propagation of the mhc_pre_sinkhorn operator, obtained based on a series of computations to get the gradient of the hidden layer in the MHC architecture.</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/block_attention_residuals/README_en.md">block_attention_residuals</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Concatenate partialBlock and blockRes, then perform RMS, projection scoring, and Softmax weighted fusion, and output hiddenStates.</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/block_attention_residuals_grad/README_en.md">block_attention_residuals_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Backward operator of BlockAttentionResiduals. Computes the gradients of partial_block, block_res, proj_weight, and norm_weight according to inv_norm and probs saved in the forward pass and the upstream gradient grad_hidden_states.</td>
  </tr>
  <tr>
    <td>mamba</td>
    <td><a href="../../mamba/causal_conv1d/README_en.md">causal_conv1d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Completes causal 1D convolution (Causal Conv1d) computation. Supports forward calculation (prefill/chunk-prefill) and state update (decode/update) modes. Supports APC (Automatic Prefix Caching), MTP (speculative decoding), and other features.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/apply_rotary_pos_emb/README_en.md">apply_rotary_pos_emb</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Executes rotary position encoding calculation. The inference network fuses the query and key operators into one path to improve performance.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/apply_rotary_pos_emb_grad/README_en.md">apply_rotary_pos_emb_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Executes the backward computation of the dual-path rotary position encoding ApplyRotaryPosEmb. Computes the rope backward gradients of query and key simultaneously and fuses them into one kernel call.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/dequant_rope_quant_kvcache/README_en.md">dequant_rope_quant_kvcache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>After performing dequant (optional) on the input tensor (x), splits the tail axis according to `sizeSplits` (the split length) into q, k, and vOut. Performs rotary position encoding on q and k to generate qOut and kOut. Then quantizes kOut and vOut and updates them to kCacheRef and vCacheRef according to `indices`.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/interleave_rope/README_en.md">interleave_rope</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs rotary position encoding on a single input x.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/inplace_partial_rotary_mul/README_en.md">inplace_partial_rotary_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs in-place single-path rotary position encoding. Modifies the input tensor directly without producing a new output tensor.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/inplace_partial_rotary_mul_grad/README_en.md">inplace_partial_rotary_mul_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs backward computation of InplacePartialRotaryMul. Performs rotary position encoding gradient calculation on the [start, end) slice of the D dimension of input dy. Writes the result back to dy in place.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/norm_rope_concat/README_en.md">norm_rope_concat</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Fuses normalization, rotary position encoding, and tensor concatenation in multimodal attention.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/norm_rope_concat_grad/README_en.md">norm_rope_concat_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Computes the backward gradient of norm_rope_concat.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/partial_rotary_mul_quant/docs/aclnnPartialRotaryMulQuant.md">partial_rotary_mul_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs partial rotary position encoding on a specified range and performs quantization on the result.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/qkv_rms_norm_rope_cache/README_en.md">qkv_rms_norm_rope_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Inputs a fused qkv tensor. Splits q, k, and v tensors through SplitVD. Performs RmsNorm, ApplyRotaryPosEmb, Quant, and Scatter fusion operations. Outputs q_out, k_cache, v_cache, q_out_before_quant (optional), k_out_before_quant (optional), and v_out_before_quant (optional).</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/qkv_rms_norm_rope_cache_with_k_scale/README_en.md">qkv_rms_norm_rope_cache_with_k_scale</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Fuses Q/K/V split, Q/K RMSNorm, RoPE/M-RoPE, quantization, and KV cache update. Supports the M-RoPE MX scenario.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/rope_quant_kvcache/README_en.md">rope_quant_kvcache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Splits the tail axis of the input tensor according to 'sizeSplits' into q, k, and vOut. Performs rotary position encoding on q and k to generate qOut and kOut. Then quantizes kOut and vOut and updates them to kCacheRef and vCacheRef according to 'indices'.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/rope_with_sin_cos_cache/README_en.md">rope_with_sin_cos_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>To improve performance, the inference network passes sin and cos inputs through cache and performs rotary position encoding calculation.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/rotary_position_embedding/README_en.md">rotary_position_embedding</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs single-path rotary position encoding calculation.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/rotary_position_embedding_grad/README_en.md">rotary_position_embedding_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Performs backward computation of single-path rotary position encoding.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/kv_rms_norm_rope_cache/README_en.md">kv_rms_norm_rope_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Splits the tail axis of the input tensor (kv) into the left half for rms_norm computation and the right half for rope computation. Then scatters the computation results into two caches respectively.</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/und_gen_qkv_rms_norm_rope_cache/README_en.md">und_gen_qkv_rms_norm_rope_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Inputs the fused QKV tensors of the understand and generate paths. Rearranges them according to cat_indices and splits Q, K, and V. Performs RMSNorm on Q and K using their respective weights and performs MRoPE multi-axis rotary position encoding, then outputs q. Scatters K and V according to slot_mapping to update the paged k_cache and v_cache.</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/text_gated_residual/README_en.md">text_gated_residual</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Fuses grouped RMSNorm, low-rank transformation, gated mixing, and optional inject.</td>
  </tr>
</tbody>
</table>
