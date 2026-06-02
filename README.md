# dual-model-router

一个 MCP Server，让 Claude Code 在单个窗口内自动路由任务到 DeepSeek（便宜）或 Anthropic（强）。三档路由：简单任务由 DeepSeek 直接处理，中等任务由 DeepSeek 执行后 Anthropic 审核，复杂任务直接交给 Anthropic。支持图片输入——DeepSeek 不支持多模态，路由会自动将图片交给强模型解析为文本描述再继续。

## 工作原理

```
用户输入（文本 / 文本+图片）
    │
    ├── 包含图片？ ──→ 🖼️ Anthropic 解析图片 → 文本描述合并到任务中
    │                        │
    ▼                        ▼
DeepSeek 判断复杂度（路由，基于纯文本，不含描述）
    │
    ├── weak（简单）→ DeepSeek 直接执行 → 最终结果
    │
    ├── medium（中等）→ DeepSeek 执行 → Anthropic 审核 → 最终结果
    │
    └── strong（复杂）→ Anthropic 直接执行 → 最终结果
```

关键设计：
- **路由判断基于原始文本**，不含图片描述——避免描述文本干扰复杂度判断
- **图片描述只参与执行**，不参与路由——即使有图片，简单问答仍走弱模型
- **三档路由**：weak 不审核（节省强模型调用），medium 必审核，strong 直接用强模型

### 🖼️ 多模态处理

DeepSeek v4 不支持多模态输入。当任务包含图片时，管线自动：

1. **提取图片** — 从 `task` 文本中的 `data:image/...;base64,...` URL 提取（或通过 `images_json` 参数传入）
2. **去重校验** — SHA256 去重，检查数量（上限 8 张）、体积（单张上限 10MB）、base64 合法性，逐张过滤而非全部丢弃
3. **强模型解析** — 图片交给 Anthropic 转为文本描述（文字转录、结构分析、视觉元素）
4. **合并回任务** — 文本描述拼入 task，后续正常路由
5. **截断保护** — 描述被 `max_tokens` 截断时标记 `⚠ 不完整`，保留部分内容

图片理解只消耗强模型一次调用，后续文本处理仍享受弱模型的低成本。

### 🔀 路由判断

```
路由输入：纯文本 task（不含图片描述，避免关键词污染）
    │
    ▼
DeepSeek 分类 → "weak"、"medium" 或 "strong"（支持中英文变体、markdown 格式）
    │
    ├── "weak" / "弱"     → 弱模型直接执行，不审核
    ├── "medium" / "中"   → 弱模型执行 + 强模型审核
    ├── "strong" / "强"   → 强模型直接执行
    └── 无法解析 / 调用失败 → 降级到 strong（fail-safe）
```

## 安装

**环境要求：Python 3.10+**

**1. 克隆到本地**

```bash
git clone git@github.com:Zju-Qibowen/dual-model-router.git ~/.claude/dual-model-router
cd ~/.claude/dual-model-router
```

**2. 安装依赖**

```bash
pip install -r requirements.txt
```

**3. 配置 API Keys**

```bash
cp .env.example .env
```

> **注意**：确认 `.gitignore` 已包含 `.env`，避免将 API key 提交到版本库。

编辑 `.env`，填入你的 API keys：

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_BASE_URL=https://your-relay.example.com   # 使用中转站时填写，否则删除此行
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
ANTHROPIC_MODEL=claude-sonnet-4-6
ANTHROPIC_MAX_TOKENS=32000
ANTHROPIC_WARN_TOKENS=8000
```

> ⚠️ **安全提示**：`ANTHROPIC_BASE_URL` 指向第三方中转站时，请求内容和 API Key 将经过该服务器，请自行评估数据隐私风险。

**4. 注册到 Claude Code（全局，所有项目可用）**

先获取 `server.py` 的绝对路径：

```bash
# macOS / Linux
echo "$(pwd)/server.py"

# Windows PowerShell
Get-Location
```

然后注册（替换为实际路径）：

```bash
# macOS / Linux 示例
claude mcp add dual-model-router python "/Users/<用户名>/.claude/dual-model-router/server.py" --scope user

# Windows 示例 (PowerShell)
claude mcp add dual-model-router python "C:/Users/<用户名>/.claude/dual-model-router/server.py" --scope user
```

**5. 重启 Claude Code**

重启后在任意项目中运行 `/mcp` 确认 `dual-model-router` 出现在列表中。

## 使用方式

注册后无需手动操作，正常描述任务即可。Claude Code 会自动调用 `route_and_answer`，回复中展示详细路由日志：

```
╔══════════════════════════════════════════╗
║  🔀 双模型路由处理中 …                  ║
╠══════════════════════════════════════════╣
║  🖼️ 图片预处理 → strong 解析            ║
║     (1 张图片 → 文本描述) ✅             ║
║  路由判断 → weak（简单任务）             ║
╠══════════════════════════════════════════╣
║  执行模型 → deepseek-v4-pro              ║
║  输出量级 → ~42 tokens                   ║
╠══════════════════════════════════════════╣
║  审核模型 → claude-opus-4-7              ║
╚══════════════════════════════════════════╝

📌 以下为强模型审核后的最终回答：
...
```

**图片传入方式**（Claude Code 中自动处理）：

图片以 `data:image/png;base64,...` 格式内嵌在 task 文本中，`_extract_images_from_task` 的正则自动提取。

**临时切换模型**（对话中直接说）：

- "切换强模型到 claude-opus-4-5"
- "切换弱模型到 deepseek-reasoner"
- "查看当前使用的模型"

切换仅在当前会话生效，重启后恢复 `.env` 中的默认值。

**临时禁用**：

```bash
claude mcp disable dual-model-router
```

## 工具列表

| 工具 | 说明 |
|------|------|
| `route_and_answer` | 自动路由并执行任务（主入口）。task 中可内嵌 `data:image/...;base64,...` URL |
| `ask_weak` | 直接调用 DeepSeek。图片自动预处理为文本描述 |
| `ask_strong` | 直接调用 Anthropic（原生多模态）。可选传入弱模型结果做审核 |
| `review` | 用强模型审核任意内容 |
| `set_weak_model` | 临时切换 DeepSeek 模型 |
| `set_strong_model` | 临时切换 Anthropic 模型 |
| `list_models` | 查看当前使用的模型 |

## 配置说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ANTHROPIC_API_KEY` | Anthropic API Key | 必填 |
| `ANTHROPIC_BASE_URL` | 中转站地址（可选） | 官方端点 |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | 必填 |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 默认弱模型 | `deepseek-v4-pro` |
| `ANTHROPIC_MODEL` | 默认强模型 | `claude-sonnet-4-6` |
| `ANTHROPIC_MAX_TOKENS` | 强模型 API 调用硬上限 | `32000` |
| `ANTHROPIC_WARN_TOKENS` | 强模型输出超过此值追加用量提醒 | `8000` |

## 图片处理参数

以下参数为代码内常量，按需修改：

| 常量 | 说明 | 默认值 |
|------|------|--------|
| `MAX_IMAGES` | 单次请求图片数量上限 | `8` |
| `MAX_IMAGE_BYTES` | 单张图片体积上限 | `10 MB` |
| `MAX_PARTIAL_TEXT_CHARS` | 截断描述最多保留字符数 | `6000` |
| `MAX_IMAGES_PER_REQUEST` | Anthropic API 单次图片上限 | `20` |

## 运行测试

测试使用 mock，无需真实 API key，直接运行即可：

```bash
pytest tests/ -v
```

共 57 个测试，覆盖：模型调用、路由判断、图片提取、去重校验、文本块提取、图片块构建、截断处理、异常降级。

## 架构

```
server.py          MCP 工具层 — 图片管线、路由展示、异常安全
    │
    ├── models.py  模型层 — DeepSeekModel (OpenAI SDK), AnthropicModel
    │               - _extract_text()  健壮的文本块提取（处理 thinking 块）
    │               - _image_to_block() 图片校验 + Anthropic content block
    │               - describe_images() 图片→文本（截断抛异常）
    │               - call_with_images() 原生多模态（截断加 warning）
    │
    └── router.py  路由层 — 复杂度分类，宽松解析 + fail-safe 降级
```

## Troubleshooting

**MCP 未出现在 `/mcp` 列表中**

- 确认已完整执行注册命令，路径无误。
- 重启 Claude Code 后再次确认。
- 检查 `server.py` 路径是否存在，Python 环境是否可用。

**401 认证失败（强模型）**

- 检查 `.env` 中 `ANTHROPIC_API_KEY` 是否正确。
- 如果使用中转站，确认 `ANTHROPIC_BASE_URL` 地址有效。
- 重启 Claude Code 以重新加载 `.env` 配置。

**模型名错误 / 模型不存在**

- 使用 `list_models` 工具查看当前使用的模型。
- 模型名区分大小写，确认与 API 提供方的名称完全一致。
- 切换模型后如报错，可重启会话恢复 `.env` 中的默认值。

**图片未被识别**

- 确认图片以 `data:image/<type>;base64,...` 格式内嵌在 task 文本中。
- 检查图片格式是否为 `image/jpeg`、`image/png`、`image/gif`、`image/webp` 之一。
- 单张图片不超过 10MB，单次不超过 8 张。
- 路由日志中出现 `🖼️ 图片预处理 → strong 解析` 表示图片已被正确处理。
- 出现 `⚠ 图片描述不完整` 表示描述被截断，已保留部分内容继续处理。
- 出现 `⚠ 图片解析失败` 表示图片处理失败，已降级为纯文本路由。
