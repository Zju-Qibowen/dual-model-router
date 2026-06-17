# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_conversation.py -v

# Run a specific test
pytest tests/test_server.py::test_handle_task_weak_with_pre_context -v
```

No build step. Tests use mock models — no real API keys needed.

## Architecture

This is a Python MCP Server (FastMCP) that interposes a three-tier model router between Codex and two external LLMs:

```
Codex (has tools)
  → MCP tool call (route_and_answer / ask_weak / ask_strong / review)
    → Image pipeline (extract → dedup → validate → Anthropic describe → cache)
      → handle_task()
        ├─ Weak:   DeepSeek-only execution
        ├─ Medium: DeepSeek execution + Anthropic review with VERDICT
        └─ Strong: Anthropic direct execution (with smart context)
        → Response with routing header
```

**Routing** (`router.py`): DeepSeek classifies task complexity into `weak`/`medium`/`strong`. The classification is based on the cleaned task text only — never on image descriptions, pre-collected context, or conversation history. Parsing is lenient (supports Chinese/English variants, markdown formatting). Any failure defaults to `strong`.

**Models** (`models.py`): `DeepSeekModel` wraps OpenAI SDK; `AnthropicModel` wraps Anthropic SDK. `DeepSeekModel.summarize_conversation()` is used for progressive summarization. `AnthropicModel.describe_images()` converts images to text for the weak model (which can't do vision). `AnthropicModel.call_with_images()` handles native multimodal calls.

**Conversation** (`conversation.py`): `ConversationHistory` manages sliding-window history with progressive summarization. When old turns are evicted from the window, they're moved to `_pending_pairs` (not discarded). When `_pending_pairs` reaches 3 items, `get_context_for_strong` lazily triggers DeepSeek to generate a cumulative summary. If DeepSeek is unavailable, `_summarize_fallback()` provides the pending pairs as raw text (marked `⚠ 摘要生成失败，使用原文`), and the pairs are retained for retry. Summary is truncated to 2000 chars at store time. `_pending_pairs` is capped at 30 — when exceeded, the oldest pairs are silently dropped with a warning log.

`get_context_for_strong` returns task-first prompt: `[当前任务]` at top, then `---` separator, then reference materials (`[参考材料 · 预收集的上下文]`, `[参考材料 · 工具操作记录]`, etc.). This ensures the model prioritizes the task even with large contexts.

**Logging** (`log.py`): Singleton `RouteLogger` writes to a local SQLite DB (`~/.Codex/dual-model-router/routing_log.db`). Pruning to 1000 rows runs after every successful write. Writes are fault-tolerant — failures are logged but never propagate.

## Key invariants

- **Task ≠ execution_task**: `task` (cleaned, no images, no context) is for routing. `execution_task` (with image descriptions) is for execution. These must stay separate.
- **`pre_context` is an argument, not stored history**: The `pre_context` parameter is passed as a function argument through `get_context_for_strong(pre_context=...)` and `handle_task(pre_context=...)`. It is injected into model prompts but never written to `ConversationHistory` via `add_user`/`add_assistant`. This prevents file contents from bloating the sliding window.
- **Medium path avoids double context injection**: In `handle_task`, `strong_context` (from `get_context_for_strong`) already includes `pre_context` as the `[预收集的上下文]` section. The medium review path checks `if strong_context` first; only falls back to `elif pre_context` when `strong_context` is absent. This prevents the section appearing twice.
- **Summary truncated at store time**: `_maybe_summarize` truncates the summary to `SUMMARY_MAX_CHARS` (2000) before writing to `self.summary`. This prevents unbounded growth and wasted tokens when re-feeding the summary back to the weak model. `get_context_for_strong` keeps a defensive fallback check for summaries migrated from older versions.
- **Routing marker is display-only**: The `🔀 WEAK/MEDIUM/STRONG` header is appended to the MCP tool return string but never enters `ConversationHistory` or any model prompt.
- **Orphan messages preserved**: When `_trim` evicts a lone assistant message (no preceding user), it's preserved in `_pending_pairs` as a single-element list. `_summarize_fallback` and `_maybe_summarize` handle single-element pairs gracefully.
- **Image cache key uses full hashes**: Cache filename is `{SHA256(image_data)[:16]}_{SHA256(task_context)[:8]}.txt`. Both components are hex digests of SHA256 hashes, not raw string prefixes — collision risk is negligible (~4 billion task-context buckets). Cache TTL defaults to 7 days with LRU eviction (mtime updated on each hit). Max 100 files.

## MCP tool docstrings

All four main tools (`route_and_answer`, `ask_weak`, `ask_strong`, `review`) have docstrings instructing Codex to restate routing/model metadata in its own response text. This is necessary because Codex's UI collapses MCP tool results, hiding the routing header. The docstrings follow a specific format with parameter documentation and `【展示规范 · 必须遵守】` sections. When changing tool behavior, update the corresponding docstring so Codex's model knows how to present results.

## ConversationHistory source tracking

Assistant messages carry a `source` field (`"weak"`/`"strong"`/`"user"`). `get_context_for_strong` annotates weak-model messages with `💡 弱模型` prefix so the strong model can distinguish which history came from a cheaper model. `get_messages()` strips internal fields, returning only `role` + `content`.
