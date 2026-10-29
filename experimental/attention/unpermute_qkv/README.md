# UnpermuteQkv（Experimental）

## 产品支持情况

| 产品                          | 是否支持 |
| ----------------------------- | :------: |
| Atlas A3系列产品              |    √     |
| Atlas A2系列产品              |    √     |

## 功能说明

`UnpermuteQkv` 是 `PermuteQkv` 的逆过程：将 `[split_size, s_c, b, total_head/split_size, h]`
的重排 QKV 还原为 query/key/value 三张张量（SPG->N）。

## 参数说明

| 参数名           | 输入/输出/属性 | 描述                                                   | 数据类型       | 格式 |
| ---------------- | -------------- | ------------------------------------------------------ | -------------- | :--: |
| output           | 输入           | `[split_size, s_c, b, total_head/split_size, h]`       | BF16 / FLOAT16 | ND   |
| q                | 输出           | `[s_c, b, q_head_num, h]`                              | BF16 / FLOAT16 | ND   |
| k                | 输出           | `[s_c, b, kv_head_num, h]`                             | BF16 / FLOAT16 | ND   |
| v                | 输出           | `[s_c, b, kv_head_num, h]`                             | BF16 / FLOAT16 | ND   |

## 构建

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=unpermute_qkv -j4
```
