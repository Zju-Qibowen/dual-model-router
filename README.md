# dual-model-router

一个 MCP Server，让 Claude Code 在单个窗口内自动路由任务到 DeepSeek（便宜）或 Anthropic（强），简单任务由 DeepSeek 执行后由 Anthropic 自动审核，复杂任务直接交给 Anthropic 处理。

## 工作原理

```
用户输入任务
    │
    ▼
DeepSeek 判断复杂度（路由，成本极低）
    │
    ├── 简单任务 → DeepSeek 执行 → Anthropic 自动审核 → 最终结果
    │
    └── 复杂任务 → Anthropic 直接执行 → 最终结果
```

审核时只传 diff 或文本差异，并设有 2000 token 硬上限，避免审核成本超过直接使用强模型。

## 工具列表

| 工具 | 说明 |
|------|------|
| `route_and_answer` | 自动路由并执行任务（主入口） |
| `ask_weak` | 直接调用 DeepSeek |
| `ask_strong` | 直接调用 Anthropic，可选传入弱模型结果做审核 |
| `set_weak_model` | 临时切换 DeepSeek 模型 |
| `set_strong_model` | 临时切换 Anthropic 模型 |
| `list_models` | 查看当前使用的模型 |

## 安装

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

编辑 `.env`，填入你的 API keys：

```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_BASE_URL=https://你的中转站地址   # 使用中转站时填写，否则删除此行
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
ANTHROPIC_MODEL=claude-sonnet-4-6
REVIEW_TOKEN_LIMIT=2000
```

**4. 注册到 Claude Code（全局，所有项目可用）**

```bash
claude mcp add dual-model-router python "/absolute/path/to/.claude/dual-model-router/server.py" --scope user
```

Windows 示例：

```powershell
claude mcp add dual-model-router python "C:/Users/<用户名>/.claude/dual-model-router/server.py" --scope user
```

**5. 重启 Claude Code**

重启后在任意项目中运行 `/mcp` 确认 `dual-model-router` 出现在列表中。

## 使用方式

注册后无需手动操作，正常描述任务即可。Claude Code 会自动调用 `route_and_answer`，在回复末尾显示路由信息：

```
[路由到: weak | 已审核: True]
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

## 配置说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ANTHROPIC_API_KEY` | Anthropic API Key | 必填 |
| `ANTHROPIC_BASE_URL` | 中转站地址（可选） | 官方端点 |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | 必填 |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 默认弱模型 | `deepseek-v4-pro` |
| `ANTHROPIC_MODEL` | 默认强模型 | `claude-sonnet-4-6` |
| `REVIEW_TOKEN_LIMIT` | 审核 token 上限，超过则跳过审核 | `2000` |

## 运行测试

```bash
pytest tests/ -v
```
