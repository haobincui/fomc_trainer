# chk3 可训练数据整理方案

## 目标

把已完成的 `deepseek_v4_pro_v2` acquisition 整理为独立、不可变、可审计的
chk3 Minutes-SFT 数据 release。训练映射保持为：

```text
chk1 final analysis -> native reasoning -> formal FOMC Minutes paragraph
```

输出训练行只包含 `prompt` 和 `response`；sample ID、split、来源哈希、target
来源和 token 统计只写入 manifest/audit，不进入模型输入。

## 已确认的问题

- acquisition 已完成 2,072/2,072：train 1,683、eval 199、test 190，失败 0。
- 50 条 chk1 final answer 仍带有 provider JSON/截断 transport 外壳；其中 1 条内容仅为
  `{`，不能作为 student analysis。
- 对 49 条可投影记录使用已有 source-only 投影后，37 条现有 provider response 可以在
  新 analysis 上重新通过当前 validator；13 条必须重新请求 DeepSeek target。
- `{` 样本必须先从它的 point-in-time chk1 source prompt/fact card 重建 analysis，再生成
  chk3 target，不能把 chk1 reasoning 或原始 Minutes 作为 chk3 输入。
- 所有样本均满足 4,096 总 token 上限，但 722 条 completion 超过旧 DAG 的 1,024
  单独上限。旧上限与既定的 reasoning<=2,400、总长<=4,096 契约冲突。

## 实现

1. 新增 training-release materializer，读取 immutable chk3 v2 acquisition 和 chk1
   handoff；不覆盖 acquisition、accepted cache 或 teacher response。
2. 对 transport JSON 输入做确定性投影；对唯一空壳样本使用 `deepseek-v4-pro`，只基于
   原 chk1 point-in-time prompt/fact card 重建 final analysis。
3. 对能复用的记录重新运行当前 chk3 validator；只对不能复用的 target 使用
   `deepseek-v4-pro` 定向生成，生成缓存与 acquisition cache 隔离并支持 resume。
4. 原子写入 standalone release：

   ```text
   dataset/processed/retrain_v2/<release_id>/minutes_alignment/
     train.jsonl
     validation.jsonl
     test.jsonl
     manifests/{train,validation,test}.jsonl
     audits/data_quality.json
     release_manifest.json
     handoff.json
   ```

5. eval 映射为 validation；训练 JSONL 恰好为 `{prompt,response}`。
6. chk3 SFT admission 使用 prompt<=3,072、total<=4,096、completion 无独立上限；禁止
   truncate。为避免改写已有 run 固定的默认 DAG 哈希，保留默认 DAG，并新增仅供后续新
   run 使用的 `configs/retrain_v2/dag_chk3_full_completion.yaml`。
7. 因当前没有 sealed chk2 artifact，本 release 明确标记为 standalone、尚未绑定完整
   `chk2_derived` DAG；数据可以完成质量和 tokenizer preflight，但训练前仍需由新 run 绑定
   sealed chk2 parent/model path。

## 最终门禁

- split 数量严格为 1,683/199/190，总计 2,072；sample ID 全局唯一且与 chk1 一一对应。
- prompt/response 非空，schema 固定，无 JSON transport 外壳污染。
- response 恰好一个 `</think>`，reasoning/Minutes 非空并通过现有数字、月份、attribution、
  元话语 validator。
- prompt<=3,072、reasoning<=2,400、完整训练序列<=4,096；不截断。
- manifest 记录每行输入/输出 SHA256、来源类别、模型 identity 和所有文件 SHA256。
- API/validator 失败不发布 release；保留 failure 和 cache 供 `--resume`。
