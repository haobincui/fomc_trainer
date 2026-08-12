# chk2 DeepSeek candidate：首次启动状态

训练已于 2026-08-10 13:47:34 UTC 在物理 GPU1 单进程 fresh 启动。第 1 个 optimizer step 正常完成并保存了完整 `checkpoint-1`；第 2 个 step 的首个 DeepSeek logical request 连续两次返回可重试但不可接受的响应，训练按合同 fail closed，于 13:59:58 UTC 退出。

这不是 OOM、NaN、鉴权失败或 GPU 资源不足。第 1 step 的 8 个 reward 全部有限且位于 `[0,1]`，均值为 `0.829764`，标准差为 `0.161288`，截断率为 `0`。

当前日志只能确定 provider 返回了 `_RetryableProviderResponseError`，不能事后严格区分 `incomplete`、缺失 reasoning usage 或 JSON schema 无效。最强证据指向 16,384 token 总输出预算不足：一个成功请求已经使用 14,343 output tokens，其中 13,639 是隐藏 reasoning；另一个首次失败请求耗时 140.84 秒，比该长成功请求还久。由于 `effort=high` 的隐藏 reasoning 与可见 JSON 共用预算，达到上限后返回 incomplete 是当前最可能原因，但这是基于运行证据的推断，不是 provider 原始错误的直接证明。

不建议原样反复恢复：前 3 个 logical groups 中，已有 2 个组至少出现一次无效响应，而完整训练约需 494 个组。原合同继续运行的中断概率很高。

最低风险修订是保留 policy completion 上限 4,096 和 reward 数学不变，只为 DeepSeek provider 增加脱敏失败 subtype/receipt，并将 Judge 总输出预算从 16,384 提到 32,768；如需提高无人值守可靠性，再把总尝试次数从 2 提到 3。上述变更会形成新的版本化请求合同，不能伪装成当前 run 的无变化恢复。
