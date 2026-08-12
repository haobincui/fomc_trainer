# chk2 DeepSeek Low 600 秒超时切换计划

记录时间：2026-08-10 15:06 UTC

## 目标

保持 `deepseek-v4-flash`、`reasoning.effort=low`、16,384 Judge 输出预算、两次总尝试和现有 reward 数学不变，仅将单次 DeepSeek 响应超时从 420 秒提高到 600 秒，并继续 checkpoint-200 派生的 chk2 训练。

## 执行约束

- 不在正在运行的目录中途改变 Judge 合同；
- 等当前 optimizer step 写出完整 checkpoint 后再停止旧 tmux；
- 若 checkpoint-5 完整落盘，则从 checkpoint-5 恢复；否则只使用已验证完整的 checkpoint-4；
- 新建独立配置、输出目录、tmux session、绑定和启动收据；
- GPU1 单进程运行，GPU0 不启动本地 Qwen Judge；
- 不修改数据、merged chk1、reward 公式、effort、token 上限或 generation 参数；
- API、OOM、上下文或分组错误继续 fail closed，不自动修改同一 run。

## 修改与验证

1. 让 low reward 同时接受历史 420 秒合同和新 600 秒合同；默认值仍为 420，旧配置保持可复现，实际 timeout 继续进入 request contract、cache binding 和脱敏日志。
2. 添加 600 秒 live-contract 单元测试，并保持 high、low-420、v3 回归通过。
3. 创建 `timeout600` 版本化配置，绑定最新完整 checkpoint；除 `judge_timeout`、resume/output 路径外，与当前 low 配置逐项一致。
4. 创建 fail-closed 专用 launcher，校验 checkpoint 完整性、配置/模型/数据/reward 哈希、GPU1、磁盘、API key、fresh 输出目录和唯一 tmux session。
5. 停止旧 tmux 后执行 preflight 并启动新 run；确认 resolved runtime config 显示 `low/600`，完成首个 DeepSeek 请求后核对实际 provider 合同与 usage。

## 可比性说明

这是同一 reward 数学下的运行可靠性参数变更，不是 reward 质量变更。此前 checkpoint-1 使用 high effort，checkpoint-2 起使用 low effort；新的 600 秒 run 仍继承这一历史，因此不会被标记为从 step 0 开始的纯 low 消融实验，也不会自动进入 chk3/chk4。
