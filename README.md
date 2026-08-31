# Personal Recall Brain

运行在 Windows 本机、证据可追溯的个人智能第二大脑。它会只读扫描学习资料，回答“我以前学过什么、什么时候学的、在哪个原文件里”，并始终展示原文片段、日期与路径。

## 已实现

- DOCX 正文、标题、表格和内嵌图片发现；PDF、Markdown、TXT 同步索引。
- 独立图片与 Word 图片用 RapidOCR 的 **OpenVINO 推理后端**识别中文；OCR 分批提交、可中断续跑。
- WAV、MP3、M4A、FLAC 用 **OpenVINO GenAI Whisper**按需转写。
- SHA-256 增量扫描：未变化的文件不重复解析，单个坏文件不影响整批。
- SQLite + FTS5 中文全文检索、时间轴、日期来源与置信度。
- 可选 OpenVINO 中文语义检索。
- OpenVINO 本地 Agent 规划工具调用并整理答案；模型不可用时确定性检索仍可用。
- Qwen 视觉模型按需加载、分析结果缓存；不会和主 Agent 长期同时占用内存。
- Streamlit 中文界面：聊天、精确搜索、时间轴、索引状态、一键打开原文件。

## 针对本机的模型方案

默认适配 8GB 内存、Intel CPU/Iris Xe：

| 能力 | 默认模型 | 加载策略 |
|---|---|---|
| 智能问答 | `OpenVINO/Qwen3-1.7B-int4-ov` | 核心模型，回答结束可卸载 |
| 中文 OCR | RapidOCR PP-OCR + OpenVINO backend | 正文扫描后分批加载、逐图保存 |
| 音频转写 | `OpenVINO/whisper-small-int8-ov` | 仅有音频且模型已下载时加载 |
| 语义检索 | `OpenVINO/Qwen3-Embedding-0.6B-int4-cw-ov` | 默认关闭，可选开启 |
| 复杂图片 | `OpenVINO/Qwen3.5-4B-int8-ov` | 默认关闭，只按需加载并缓存结果 |

所有大模型默认从 Intel/OpenVINO 在 Hugging Face 的官方优化仓库下载，直接以 OpenVINO IR 运行。

## 一键使用

1. 双击 `安装与下载模型.bat`。首次安装会准备独立环境并下载约 1.2GB 的核心 INT4 模型。
2. 检查 `config.toml` 中的资料目录。默认已经配置：

   - `D:\本科期间学习\日记`
   - `D:\本科期间学习\考研`

3. 双击 `启动第二大脑.bat`。
4. 浏览器打开后先点左侧“立即扫描资料”，正文扫描完成后即可提问。
5. 需要检索图片里的文字时，点“补充下一批图片文字”，或双击 `补充图片文字.bat`。每张图片完成后都会保存，中断后下次从未完成处继续。

也可以先双击 `扫描学习资料.bat` 做离线扫描。

## 常用问题

- `我以前复习过申论吗？`
- `8.21 那天我学了什么？`
- `我在哪个 Word 里写过公文写作？`
- `我什么时候复习过一致收敛？`

回答会列出原始日期、文件名、证据片段和完整来源路径。关闭所有模型后，搜索与时间轴仍然可用。

## 可选模型

在项目终端中运行：

```powershell
.venv\Scripts\python.exe -m second_brain.cli download-models --profile audio
.venv\Scripts\python.exe -m second_brain.cli download-models --profile semantic
.venv\Scripts\python.exe -m second_brain.cli download-models --profile vision
```

下载语义或视觉模型后，还需要在 `config.toml` 的 `[models]` 中把对应 `*_enabled` 改为 `true`。8GB 机器不建议一次启用全部模型。

## 命令行

```powershell
second-brain scan
second-brain enrich-images --limit 200
second-brain search "申论"
second-brain ask "我以前什么时候复习过申论？"
second-brain status
```

## 数据安全

学习资料是只读源。程序没有删除、移动、重命名或写回资料的功能；数据库、缓存和模型均位于项目目录，且不会提交到 Git。详见 [数据安全说明](docs/数据安全说明.md)。

## 开发验证

```powershell
python -m pytest
```

测试覆盖日期溯源、数据库迁移、DOCX 顺序解析、增量扫描、源文件不变、中文搜索、时间轴、Agent 工具、防幻觉降级与语义召回。
