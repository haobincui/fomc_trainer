# chk3 可训练数据整理结果

## 结果

- 状态：complete / quality passed
- release：`chk3_minutes_clean_v1_20260805`
- dataset：
  `dataset/processed/retrain_v2/chk3_minutes_clean_v1_20260805/minutes_alignment`
- split：train 1,683、validation 199、test 190，共 2,072；sample ID 全局唯一。
- 训练行 schema：严格为 `{prompt,response}`。

## 清理与重建

- 1,839 条 analysis 原样保留。
- 183 条移除非语义 `ev-` evidence citation；没有改写事实正文。
- 4 条完整 provider JSON envelope 无损投影到内部 `answer`。
- 46 条截断 provider JSON 不能安全复用，已由 `deepseek-v4-pro` 仅根据原 immutable
  point-in-time fact card、topic 和 style guide 重建 analysis。
- 2,020 条 target 在清理后的 analysis 上重新通过原 validator；52 条使用
  `deepseek-v4-pro` 重新生成并通过 validator。
- 所有新增 DeepSeek 响应 identity 均为：
  `deepseek-v4-pro / fp_9954b31ca7_prod0820_fp8_kvcache_20260402`。

## 独立门禁

- 2,072/2,072 target 全量重放 validator 通过。
- transport-wrapped analysis：0；analysis 内 evidence citation：0。
- response boundary 错误：0；每条恰好一个 `</think>`。
- 数字、月份、无来源 attribution、reasoning 元话语门禁全部通过。
- 文件哈希复核：release manifest 中 8 个文件全部一致。
- tokenizer 精确统计：
  - prompt max：615
  - completion max：2,522
  - reasoning max：2,376
  - total max：3,029
- 训练 admission：prompt<=3,072、total<=4,096、completion 无独立上限、禁止截断。
- train+validation 的训练 preflight 共 1,882 条，状态 passed；test 190 条也完成同等逐条
  validator/token 检查。

## DAG 绑定状态

本 release 是 standalone chk3 data release，目前不能冒充完整的 chk2-derived release，
因为当前 run 尚无 sealed chk2 parent，也没有同一 derived release 所需的 chk4
`decision_grpo` 数据。默认 `dag.yaml` 已恢复到现有 v9 run 固定的 SHA256
`a07b9e7a...`，不会破坏正在进行的 chk1/chk2 链路。

后续创建/导入新 run 并绑定 sealed chk2 时，使用：

```text
configs/retrain_v2/dag_chk3_full_completion.yaml
```

该 DAG 只取消 chk3 的独立 1,024 completion ceiling，仍保留 4,096 总长度硬门禁。
