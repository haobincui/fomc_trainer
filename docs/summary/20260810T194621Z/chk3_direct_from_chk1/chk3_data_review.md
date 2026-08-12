# chk1 checkpoint-200 → chk3 direct SFT 数据审阅

审阅时间：2026-08-10 19:46 UTC

## 结论

当前 `chk3_minutes_clean_v3_20260805` 可用于低学习率、短步数、不可晋升的
退化 smoke；不建议在未经 reasoning 清洗的情况下直接进行正式全量训练。

这是一条独立的 `chk1 → Minutes-SFT` 实验分支，不是现有 canonical
`chk2 → chk3` DAG。目标是 DeepSeek 合成的 FOMC Minutes-style paragraph，
不是官方 Minutes 原文。

## 结构与完整性

- release：`dataset/processed/retrain_v2/chk3_minutes_clean_v3_20260805`
- release manifest SHA-256：
  `4c5787945460431179e6b73a23b19a72bced14fc76022ecc1dad52d119abd1d7`
- train / validation / test：`1683 / 199 / 190`，合计 `2072`
- grain：每行一个 chk1 atomic analysis，映射为
  `analysis → reasoning → formal Minutes paragraph`
- 2072/2072 行均只有一个 `\n</think>\n`；boundary 两侧非空；raw target
  不含 opening `<think>`；最终 Minutes 是单段纯文本。
- manifest 记录的文件 bytes、rows 和 SHA-256 均与磁盘一致；sample ID、
  行序、prompt/analysis/reasoning/answer/response hash 均通过。
- exact 与 whitespace-normalized 的 prompt、analysis、reasoning、answer、
  response 以及 pair 均无 split 内或跨 split 重复。
- final Minutes 2072/2072 保持 source analysis 的 canonical numeric
  multiset 与 date set；无新增 attribution；无 strict periodic tail。

## 真实训练 token 路径

使用 cp200 merged chk1 tokenizer、当前 single-BOS renderer、TRL 1.2
completion-only label 路径全量复算：

- prompt tokens：min `300` / p50 `394` / p95 `473` / max `560`
- completion tokens：min `117` / p50 `759` / p95 `1746` / max `2500`
- total tokens：min `432` / p50 `1162` / p95 `2177` / max `2979`
- 2072/2072 精确一个 BOS 与最终 EOS；prompt 全 masked；reasoning、
  `</think>`、Minutes 和 EOS 全部受监督。
- `max_length=4096` 下截断为 `0`，因此继续提高上限不会增加本数据的
  有效监督，只会放宽未来异常行的容忍范围。

## 污染与风险

最终 Minutes 本身没有 `ev-*`：prompt `0`、analysis `0`、final answer `0`。
隐藏 reasoning 中有 9 行、49 个 citation token（37 个完整 ID、12 个缩写）；
它们只是总体风险的很小一部分。

扩展 reasoning 审阅发现：

- 元话语任一命中：`2032/2072`（98.07%）
- source/analysis narration：`1814`
- 第一人称：`1493`
- drafting wording：`1249`
- inventory/ledger heading：`1146`
- bullet reasoning：`1129`
- checking wording：`861`
- quoted draft：`712`
- reasoning 与 final answer 的 8-gram 覆盖至少 50%：`954`
- 覆盖至少 80%：`422`
- 精确重复句：`228`

这些 target 没有直接的严格周期循环，completion 4-gram repetition 也没有
达到 0.5；但监督主体偏长、偏模板化、偏草稿检查流程，和此前 chk1 的
free-running 退化风险方向一致。只删除 9 行 citation 不能解决核心问题。

## 执行决策

1. 以 merged chk1 checkpoint-200 为 base，双 A30 运行 fresh 20-step smoke。
2. 参数参考已稳定的 chk1：QLoRA r32/alpha64、all projections、GA=8、
   batch=1/GPU、LR=`1e-6`、SDPA、max length `4096`。
3. smoke adapter 不晋升、不续接正式 run；正式 run 必须从同一 chk1 fresh start。
4. smoke 后在 validation 的 short/median/long 三个冻结样本上，对 chk1 base
   与 smoke adapter 做相同 greedy 3072-token generation。
5. 任一 EOS/`</think>`/非空单段 Minutes/数字日期保持/周期尾/重复率门禁失败，
   不启动正式训练。
6. 即使 smoke 通过，也只证明短程没有显著行为退化；正式数据仍建议发布
   reasoning-clean v4 后再训练，不能把 v3 smoke 伪称 canonical chk3。

## Smoke 与 generation 结果

20-step 双卡 smoke 于 2026-08-10 19:57:58 UTC 正常结束：

- global step：`20/20`
- train loss：`1.2422389686`
- eval loss：step 10 `1.2117794752`；step 20 `1.2069450617`
- checkpoint：`10 / 18 / 19 / 20`
- checkpoint-20 与根 adapter SHA-256：
  `ad3e5c0aa0eeeab09f130061695466c11371ff6bd1ea2f9b09662ba2678264c1`
- 0 OOM、0 NCCL fatal、0 NaN/Inf；每卡峰值训练显存约 11.85 GiB。

smoke 启动后，Trainer 才新增 standalone release verifier；因此该次 smoke 的
runtime receipt 诚实保留 `legacy_unbound`，不能晋升。后续 full run 使用相同
训练/提示参数，但通过新版 config 显式绑定 sealed v3 manifest，并记录为
`standalone_chk3_direct_non_promotable`。generation 的 `samples_v2` 绑定的是
新增 verifier 后的 config hash；新增字段不改变 system prompt 或 generation。

generation 使用相同的 frozen short/median/long 三样本、greedy、3072-token
上限，并遵守“全部 generation 只用 GPU0”：

| 指标 | chk1 cp200 base | chk3 smoke20 |
|---|---:|---:|
| EOS / 唯一 boundary / 单段 final | 3/3 | 3/3 |
| 数字 multiset / 日期 set 保持 | 3/3 | 3/3 |
| cap / strict periodic tail | 0/3 | 0/3 |
| mean token 4-gram repetition | 0.12572027 | 0.11792738 |
| max token 4-gram repetition | 0.14927769 | 0.13656388 |

comparison 状态为 `passed`；candidate 相对 base 的平均重复率变化为
`-0.00779289`，最大单样本增加仅 `+0.00065105`。因此短程训练没有观察到
此前 chk1 式无限重复退化，但这不解除 v3 reasoning 污染的正式训练阻断。

关键工件：

- `probe/samples_v2.json` SHA-256：
  `112ca6804c5c906b0e3c6bd33e309b196bc2800ee9ee6293e223a3e266c1f9a7`
- `probe/chk1_base_v2/results.jsonl` SHA-256：
  `0fc612b842fad4de8fa8813607189a710b4b351769ecad0f7935f9d595ab27e2`
- `probe/chk3_smoke20_v2/results.jsonl` SHA-256：
  `e182b89555c2fb6c9cf1eea31406b1a866c69aa8f471d1a9ca494fb9363cdea0`
- `probe/comparison_v2.json`

## 已知测试漂移

现有 chk3 数据测试为 `66 passed, 1 failed`。失败不是当前 release 行内容，
而是未来 teacher-repair fixture 仍构造空 `evidence_ids`，与后来加固的共享
teacher contract（要求非空 evidence ID 列表）冲突。当前不调用 DeepSeek API，
因此该失败不阻断本次只读 release 加载和 smoke，但在生成 v4 前必须解耦合同。

## Standalone full run 启动收据

在用户明确要求停止 chk2 并直接训练 chk3 后，已于 2026-08-10 20:07 UTC
启动一个 fresh、双 A30 的 standalone full run。该决定不改变上述数据质量判断：
本次产物被明确标为 `standalone_chk3_direct_non_promotable`，不是 canonical
`chk2 -> chk3` DAG 产物，也不授权作为 chk4 parent。

- base：merged chk1 checkpoint-200
- epochs：`3`
- optimizer steps：`318`
- effective batch：`16`（每卡 1，双卡，gradient accumulation 8）
- learning rate：`1e-6`，cosine，warmup ratio `0.05`
- max length：`4096`，SDPA，QLoRA r32/alpha64
- eval：每 10 step；save：每 step 写入、永久保留所有 10 的倍数及最新 3 个
- sealed v3 release：已由 standalone verifier 全量复核并绑定到 runtime receipt

前两个正式 eval 已完成：step 10 train loss=`1.1789`、eval loss=
`1.2119339705`；step 20 train loss=`1.2274`、eval loss=`1.2058224678`。
验证损失继续下降；所有 loss/gradient 均为有限值，
未出现 OOM、NCCL fatal、NaN 或 Inf。运行中两卡各使用约 11.85 GiB。

正式运行路径：

- adapter：`output/training/retrain_v2/chk3_direct_chk1_cp200_full3ep_lr1e6_20260810/adapters/chk3`
- log：`docs/summary/20260810T194621Z/chk3_direct_from_chk1/full/training.log`
- runtime receipt：上述 adapter 目录中的 `resolved_runtime_config.json`

reasoning-clean v4 仍只允许 dry-run。当前清理合同尚未冻结，最新只读恢复审计
仍有 82 条阻塞（70 条残留污染、12 条低于 64 tokens），其中 79 条没有满足
同 prompt、同 final answer 的本地安全替代；因此不得把 v4 dry-run 发布为完整
release，也不得在本次运行中途换数据。

## checkpoint-170 并发 generation 探针

在不中断双卡训练的条件下，使用 GPU0 的剩余显存对完整 checkpoint-170 运行
冻结的 short/medium/long 三条 greedy 3072-token 探针。探针峰值 reserved
显存约 6.04 GiB；与 rank-0 训练合计约 18.3 GiB，未发生 OOM，训练从
step 180 正常继续。

- 3/3 正常 EOS、未触顶；3/3 恰有一个 `</think>` 且 final 非空单段。
- 3/3 数字 multiset 与日期 set 完全保持。
- 0/3 strict periodic tail；0 个 catastrophic case。
- candidate mean 4-gram repetition：`0.20914911`；max：`0.29129464`。
- chk1 cp200 baseline mean：`0.12572027`；平均增幅：`+0.08342884`，低于
  预注册 `+0.10` 上限。
- 最大单样本增幅（medium）：`+0.14201695`，低于 `+0.20` 上限。
- comparison 状态：`passed`。

这说明 checkpoint-170 尚未出现无限循环或输出合同退化，但重复率相对 chk1
已有可测量上升，尤其是 medium 样本；因此最终 checkpoint 仍需重复同一门禁，
不能只依据持续下降的 eval loss 晋升。

工件：

- `probe/chk3_full_cp170_concurrent_v1/results.jsonl` SHA-256：
  `967c10478c5d2459a0ef1a4f0b7b2f37bb45dd580f3ecda5d2ad15219b16fa18`
- `probe/chk3_full_cp170_concurrent_v1/summary.json` SHA-256：
  `0d30d625c68e47c768d2d40084c0773d37ebb12b3ce7a1a36a552f7812da2926`
- `probe/comparison_full_cp170_vs_chk1_v1.json` SHA-256：
  `2db2489bf683e4153ffd2be5c94cc522be23b994427574b9585c6e236a439918`
