# AICPU built-in package manifest
# Default is OFF for all AICPU ops; only list ON entries here.
# Key is op name (e.g. "flash_attn_metadata", "attention_worker_scheduler").
# To enable all AICPU ops in built-in package, set FORCE_ALL to ON.

set(AICPU_BUILTIN_PACKAGE_FORCE_ALL OFF)

set(AICPU_BUILTIN_PACKAGE_MANIFEST
    flash_attn_metadata=ON
    quant_flash_attn_metadata=ON
    lightning_indexer_v2_metadata=ON
    quant_lightning_indexer_v2_metadata=ON
    generic_block_sparse_attention_grad_metadata=ON
    sparse_flash_mla_metadata=ON
    mixed_quant_sparse_flash_mla_metadata=ON
    quant_sparse_flash_mla_metadata=ON
    sparse_lightning_indexer_kl_loss_grad_metadata=ON
    sparse_flash_mla_grad_metadata=ON
    dense_lightning_indexer_kl_loss_grad_metadata=ON
    dense_lightning_indexer_softmax_lse_v2_metadata=ON
    sparse_flash_mla_softmax_l1_norm_metadata=ON
    generic_block_sparse_attention_metadata=ON
    flash_mla_metadata=ON
    quant_flash_mla_metadata=ON
    mixed_quant_flash_attn_metadata=ON
)
