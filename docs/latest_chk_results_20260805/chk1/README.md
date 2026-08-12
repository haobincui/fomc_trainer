# chk1 文档包

主报告是 [chk1_summary.html](./chk1_summary.html)。它是单文件离线 HTML，覆盖 chk1 的数据准备、DeepSeek 蒸馏、质量审核、SFT 配置与结果、LoRA 合并、模型血缘、局限和下一步。

## 目录内容

- `chk1_summary.html`：面向阅读的最终技术报告。
- `artifact.json`：HTML 的规范化报告输入、图表数据和来源声明。
- `chk1_audit.py`：只读审计程序；复算关键行数、状态和 SHA 绑定。
- `chk1_audit.ipynb`：已执行 notebook，包含审计输出和一致性断言。
- `source_inventory.md`：报告使用的本地证据清单及口径。

## 当前结论

- DeepSeek 获取：2,117 selected，2,115 落盘，2 条结构无效。
- Canonical 准入：2,072 通过，45 排除，其中 43 条为空 response component。
- chk1 SFT：1,355 train、199 validation；190 test 未进入训练。
- 完成 run：`retrain_v2_full_v7_automated_v5_20260804`，chk1 状态 `sealed`。
- 最终结果：train loss 0.918037，eval loss 0.787729，eval token accuracy 0.794419。
- Merged 模型 SHA-256：`9b355f903b7722f274bca1bc226bf75c1da4ebe449de726174bd1d054ff8948c`。

这些指标确认执行完成，但不替代冻结 test 上的自由生成与人工质量评测。

## 复核方法

在仓库根目录运行：

```bash
conda run -n fomc_trainer python docs/summary/20260805T124140Z/chk1_audit.py
```

重新执行 notebook：

```bash
jupyter nbconvert \
  --execute \
  --to notebook \
  --inplace \
  --ExecutePreprocessor.timeout=120 \
  --ExecutePreprocessor.kernel_name=python3 \
  docs/summary/20260805T124140Z/chk1_audit.ipynb
```

## 验证记录

- 审计程序：通过全部断言。
- Notebook：执行成功，所有代码单元无异常。
- 报告数据合同：通过。
- HTML 打包：通过；18 个 blocks、2 个 charts、1 个 metric strip、4 个 tables。
- HTML 结构验证：通过。
- 本机 Chrome 离线打开与视觉检查：通过，中文、表格和语义 fallback 均可读。
- 增强 reader 的自动交互验证：未完成。报告工具要求专用 Chromium headless-shell；本机只有完整 Google Chrome，二者环境握手不兼容。该限制不影响离线 HTML 的语义内容，但意味着交互和 source dialog 未由自动化 verifier 覆盖。

文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `artifact.json` | `596ed5adf1040c5d782d530f7930932e0be25bcfc68521089e59b379e6389e8d` |
| `chk1_audit.py` | `c95de4a564c4cf5921e149960ea917351a0902b07d96a796a6e098bda2e81a40` |
| `chk1_audit.ipynb` | `0386aec1f352717f9a170fd943287433f33c51d9d09e4608350847350ce39920` |
| `chk1_summary.html` | `c60e5cc303dba7315065ed7a300ee1bf40297216ec4753c91eebbaf270c95a2f` |

