# TOTALSL 数据处理状态

状态：完成  
完成时间：2026-08-10 12:52 UTC

## 已发布数据

`dataset/processed/retrain_v2/analysis_grpo_full_v7_totalsl_billions_v1_20260810`

- overlay manifest SHA-256：`00a825755444b83ea3c3f3d0878b394acbcc2af85ab231ec714aa64f742de410`
- overlay attestation SHA-256：`93285c16f0e5a1f6d698ed6d0032d8695bb533c23a6bf2dd548da0e7fd2186cf`
- output payload SHA-256：`d8587373d3a922300732799b397a1f1294c4b6fb7114948f0309dcfd31546fe2`
- split 数量：train/eval/test = `493/199/190`
- 修改行：`30/12/13`，合计 `55`
- 修改 TOTALSL evidence entries：`46/12/13`，合计 `71`
- 未受影响且逐字节相同的行：`827`
- sample IDs：`882/882` 唯一
- 发布状态：原子 no-overwrite、目录和文件均只读、无软链接

## 验证

- builder 自验证：通过
- 独立逐行/逐字段复核：通过，零错误
- 相关测试：`23 passed`
- Ruff：通过
- 技术报告：validation/package/structural verification 通过；机器未安装 Chromium headless shell，因此浏览器交互验证未运行。

## 本会话边界

按用户最新指令，本会话只处理数据。未执行 checkpoint merge，未启动 chk2，未占用 GPU，也未修改旧父数据。checkpoint-200 merge 由 session `019fd346-c712-72f2-bb1f-7ccb28e26145` 独立负责。
