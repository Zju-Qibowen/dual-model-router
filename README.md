# dual-model-router

一个 MCP Server，让 Claude Code 在单个窗口内自动路由任务到 DeepSeek（便宜）或 Anthropic（强），简单任务由 DeepSeek 执行后由 Anthropic 审核，复杂任务直接交给 Anthropic 处理。

## 工作原理

```
用户输入任务
    │
    ▼
DeepSeek 判断复杂度（路由，成本极低）
    │
    ├── 简单任务 → DeepSeek 执行 → Anthropic 审核弱模型结果 → 最终结果
    │
    └── 复杂任务 → Anthropic 直接执行 → 最终结果
```

审核时只传 diff 或文本差异，并设有 token 上限（默认 2000），超过则跳过审核，避免审核成本超过直接使用强模型。

## 安装

**环境要求：Python 3.10+**

**1. 克隆到本地**

```bash
git clone https://github.com/Zju-Qibowen/dual-model-router.git ~/.claude/dual-model-router
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
REVIEW_TOKEN_LIMIT=2000
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

注册后无需手动操作，正常描述任务即可。Claude Code 会自动调用 `route_and_answer`，在回复末尾显示详细路由日志：

```
---
  路由到: weak (deepseek-v4-pro)
  已审核: True
  弱模型 token 估算: 42
  弱模型原始回答: 你好，你好吗？
```

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
| `route_and_answer` | 自动路由并执行任务（主入口） |
| `ask_weak` | 直接调用 DeepSeek |
| `ask_strong` | 直接调用 Anthropic，可选传入弱模型结果做审核 |
| `set_weak_model` | 临时切换 DeepSeek 模型 |
| `set_strong_model` | 临时切换 Anthropic 模型 |
| `list_models` | 查看当前使用的模型 |
| `list_available_models` | 列出所有可选的弱模型和强模型 |

## 配置说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ANTHROPIC_API_KEY` | Anthropic API Key | 必填 |
| `ANTHROPIC_BASE_URL` | 中转站地址（可选） | 官方端点 |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | 必填 |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 默认弱模型 | `deepseek-v4-pro` |
| `ANTHROPIC_MODEL` | 默认强模型 | `claude-sonnet-4-6` |
| `REVIEW_TOKEN_LIMIT` | 弱模型审核 token 上限，超过则跳过审核 | `2000` |
| `ANTHROPIC_MAX_TOKENS` | 强模型 API 调用硬上限（兜底，不截断） | `32000` |
| `ANTHROPIC_WARN_TOKENS` | 强模型输出超过此值追加用量提醒 | `8000` |
| `WEAK_MODELS` | 可选弱模型列表（逗号分隔），供 `list_available_models` 展示 | 见 `.env` |
| `STRONG_MODELS` | 可选强模型列表（逗号分隔），供 `list_available_models` 展示 | 见 `.env` |

示例：

```
WEAK_MODELS=deepseek-v4-flash,deepseek-v4-pro
STRONG_MODELS=claude-sonnet-4-6,claude-opus-4-6,claude-haiku-4-5
```

## 运行测试

测试使用 mock，无需配置真实 API key，直接运行即可：

```bash
pytest tests/ -v
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

- 使用 `list_available_models` 工具查看当前可用的模型列表。
- 模型名区分大小写，确认与 API 提供方的名称完全一致。
- 切换模型后如报错，可重启会话恢复 `.env` 中的默认值。
