# FlashMla Host UT

加载 CANN SDK，在仓库根目录运行，无需 NPU：

```bash
bash build.sh --ophost_test --ops=flash_mla --soc=ascend950 -j25
```

`op_host/arch35/` 的 CSV 覆盖 tiling 正反例；`op_host/` 的 CSV 覆盖 shape/dtype 推导。
