# PermuteQkv（Experimental）

## 产品支持情况

| 产品                          | 是否支持 |
| ----------------------------- | :------: |
| Atlas A3系列产品              |    √     |
| Atlas A2系列产品              |    √     |

## 功能说明

`PermuteQkv` 将 query/key/value 按 KV group 做头重排（N->SPG），把 `[s_c, b, q_head_num, h]`
与 `[s_c, b, kv_head_num, h]` 的 q/k/v 拼接后重排为 `[total_head, s_c, b, h]`，用于后续 MoE/序列并行分组。

## 参数说明

| 参数名     | 输入/输出/属性 | 描述                                  | 数据类型       | 格式 |
| ---------- | -------------- | ------------------------------------- | -------------- | :--: |
| q          | 输入           | `[s_c, b, q_head_num, h]`             | BF16 / FLOAT16 | ND   |
| k          | 输入           | `[s_c, b, kv_head_num, h]`            | BF16 / FLOAT16 | ND   |
| v          | 输入           | `[s_c, b, kv_head_num, h]`            | BF16 / FLOAT16 | ND   |
| split_size | 必选属性       | 分组数（SPG）                         | INT            | -    |
| output     | 输出           | `[q_head_num + 2*kv_head_num, s_c, b, h]` | BF16 / FLOAT16 | ND |

## 构建

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=permute_qkv -j4
```
