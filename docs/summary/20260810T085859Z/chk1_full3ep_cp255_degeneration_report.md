# chk1 full3ep checkpoint-255 退化检查

检查时间：2026-08-10T09:10:59Z

## 结论

最终 `checkpoint-255` 存在明确的生成退化，未通过与 LR=1e-5 smoke 相同的固定 8-case 门禁。该 checkpoint 不应作为 chk2 的输入模型。

训练 loss 本身没有显示退化：255/255 steps 正常完成，最终 eval loss 为 `1.300019`，step 240 最低值为 `1.300009`。但行为测试出现严重重复、输出显著拉长、格式有效率下降以及 completion 触顶，说明低 eval loss 无法保证 free-running generation 稳定。

## 固定门禁结果

| 指标 | chk0 baseline | LR=1e-5 smoke checkpoint-10 | full3ep checkpoint-255 |
|---|---:|---:|---:|
| EOS rate | 1.000 | 1.000 | 0.875 |
| contract-valid rate | 1.000 | 1.000 | 0.875 |
| completion cap rate | 0.000 | 0.000 | 0.125 |
| catastrophic cases | 0/8 | 0/8 | 3/8 |
| periodic-tail rate | 0.000 | 0.000 | 0.125 |
| mean full 4-gram repetition | 0.0827 | 0.0948 | 0.4086 |
| max full 4-gram repetition | 0.1425 | 0.2244 | 0.8172 |
| max tail 4-gram repetition | 0.1425 | 0.2244 | 0.8815 |

严格 compare 状态为 `failed`。聚合失败项：平均全文重复上升、平均尾部重复上升、输出合同有效率下降、completion cap rate 非零。

## 具体失败

- 一个 greedy case 输出 3072/3072 tokens，未 EOS、合同无效，出现严格周期尾；全文/尾部重复率为 `0.8172/0.8815`。
- 一个 greedy case 虽最终 EOS，但全文重复率为 `0.6178`。
- 一个 sampled case 生成 2444 tokens，全文重复率为 `0.5795`。
- 另一个 sampled case 相对 chk0 的全文重复率增幅为 `0.2257`，超过单-case `0.20` 限制。
- 8 条平均全文重复率相对 chk0 增加 `0.3259`，超过均值增幅 `0.10` 限制。

## 训练侧观测

- 训练成功完成 3 epochs、255 steps，无 OOM、NaN、CUDA/NCCL fatal 或 Traceback。
- eval loss：step 10 `1.834635`，step 100 `1.353610`，step 170 `1.304348`，step 240 `1.300009`，final `1.300019`。
- 三个 epoch 的平均 train loss：`1.58724 → 1.34635 → 1.31135`。
- 因而本次退化属于生成行为退化，不是普通 loss 发散。

## 工件与哈希

- `chk1_full3ep_cp255_probe/results.jsonl`：`9deaf7638088eee3c09810b4ec003df83822b19dd6596ffb2083bdf7c07a799d`
- `chk1_full3ep_cp255_probe/summary.json`：`60ae0bc79a5e3a988da45e153a099f4da2023145bc6c1a699f68c5d6b24de405`
- `chk1_full3ep_cp255_vs_chk0_comparison.json`：`81fc6db71b7ec66ba163a7c5f2abb3636b94699e0b5885e0d565cd3512511030`
- checkpoint-255 adapter：`0a664c18f13ddf1b7f6b96e25b8d68fcb6e61699136c725fcb2f778da7d268c5`

## 建议的下一步

不要用 checkpoint-255 启动 chk2。若要复用本轮训练，应按 `250 → 240 → 200 → 170 → 100 → 50` 回溯运行同一门禁，定位最后一个稳定 checkpoint；选择时必须同时满足行为门禁，不能仅按最低 eval loss。
