"""对话历史管理 —— 为多轮上下文提供滑动窗口存储。

同时支持渐进式摘要：窗口溢出时旧对话不被丢弃，而是用弱模型压缩为摘要，
仅在强模型调用时注入，弥补滑动窗口丢失早期关键信息的缺陷。
"""

from dataclasses import dataclass, field
import logging

logger = logging.getLogger("dual-model-router")

DEFAULT_MAX_TURNS = 10
DEFAULT_MAX_TOKENS = 16000
SUMMARY_MAX_CHARS = 2000          # 摘要渲染上限（字符数），只在此处截断一次
PENDING_MIN_PAIRS = 3             # 攒够几对被裁对话才触发摘要
MAX_PENDING_PAIRS = 30            # pending 上限，防止摘要长期不触发时无限增长
MAX_TOOL_NOTES = 50               # 工具操作记录上限


@dataclass
class ConversationHistory:
    max_turns: int = DEFAULT_MAX_TURNS
    max_tokens: int = DEFAULT_MAX_TOKENS
    turns: list[dict] = field(default_factory=list)

    # ── 渐进式摘要 ──────────────────────────────
    summary: str = ""                  # 累积摘要
    _pending_pairs: list = field(default_factory=list)  # 待摘要的旧 pair（每项为 [user, assistant]）
    _summary_failed: bool = False      # 上次摘要是否失败

    # ── 工具操作记录 ──────────────────────────
    _tool_notes: list[str] = field(default_factory=list)  # Claude Code 工具操作的摘要

    # ── 基础操作 ────────────────────────────────

    def add_user(self, content: str) -> None:
        self.turns.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str, source: str = "strong") -> None:
        """添加 assistant 消息。source 标注产出模型：weak / strong / user。"""
        self.turns.append({"role": "assistant", "content": content, "source": source})
        self._trim()

    def get_messages(self) -> list[dict]:
        """返回不含内部字段的消息列表。"""
        return [{"role": t["role"], "content": t["content"]} for t in self.turns]

    def add_tool_note(self, content: str) -> None:
        """添加一条工具操作记录（不参与对话轮次，不触发裁剪）。

        Claude Code 在调用外部模型之前，可用此方法向 history 注入工具操作摘要，
        让外部模型了解"Claude Code 已经做了什么"（读了哪些文件、跑了什么命令等）。
        记录出现在 get_context_for_strong 的 [工具操作记录] 段落中。

        超上限时丢弃最旧的记录。
        """
        self._tool_notes.append(content)
        if len(self._tool_notes) > MAX_TOOL_NOTES:
            dropped = len(self._tool_notes) - MAX_TOOL_NOTES
            del self._tool_notes[:dropped]

    def clear(self) -> None:
        self.turns.clear()
        self.summary = ""
        self._pending_pairs.clear()
        self._tool_notes.clear()
        self._summary_failed = False

    # ── 状态查询 ────────────────────────────────

    @property
    def turn_count(self) -> int:
        return sum(1 for t in self.turns if t["role"] == "user")

    @property
    def estimated_tokens(self) -> int:
        return self._total_tokens()

    # ── token 估算 ──────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        cjk = sum(1 for c in text if '一' <= c <= '鿿')
        return int(cjk * 1.5 + (len(text) - cjk) * 0.25)

    def _total_tokens(self) -> int:
        return sum(self._estimate_tokens(t["content"]) for t in self.turns)

    # ── 滑动窗口裁剪 ────────────────────────────

    def _trim(self) -> None:
        """裁剪超出窗口的旧对话。

        被裁掉的 pair 不丢弃，追加到 _pending_pairs 供后续摘要使用。
        摘要本身不在 _trim 里触发（懒触发：在 get_context_for_strong 时才调）。
        """
        def _evict_front() -> None:
            """移除 turns 最前面一条/一对，保留到 _pending_pairs。"""
            if len(self.turns) >= 2 and self.turns[0]["role"] == "user" and self.turns[1]["role"] == "assistant":
                self._pending_pairs.append([dict(self.turns[0]), dict(self.turns[1])])
                del self.turns[0:2]
            elif self.turns:
                # 孤儿消息也保留进 pending，避免静默丢失（_summarize_fallback 兼容单元素）
                self._pending_pairs.append([dict(self.turns[0])])
                self.turns.pop(0)
            # pending 上限保护
            if len(self._pending_pairs) > MAX_PENDING_PAIRS:
                dropped = len(self._pending_pairs) - MAX_PENDING_PAIRS
                del self._pending_pairs[:dropped]
                logger.warning("_pending_pairs 超过上限 %d，丢弃最旧 %d 项", MAX_PENDING_PAIRS, dropped)

        # 按轮数裁
        while len(self.turns) > self.max_turns * 2:
            _evict_front()

        # 按 token 数裁
        while self._total_tokens() > self.max_tokens and len(self.turns) > 2:
            _evict_front()

        # 确保开头是 user
        while self.turns and self.turns[0]["role"] == "assistant":
            self.turns.pop(0)

    # ── 渐进式摘要 ──────────────────────────────

    def _maybe_summarize(self, weak_model) -> None:
        """如果积压的旧 pair 达到阈值，用弱模型生成摘要。

        摘要成功 → 合并到 self.summary，清空 _pending_pairs。
        摘要失败 → _pending_pairs 保留，下次重试。不抛异常，不阻塞调用方。
        """
        if len(self._pending_pairs) < PENDING_MIN_PAIRS:
            return

        # 组装待摘要文本（兼容孤儿消息：pair 可能只有 user 一个元素）
        lines = []
        # 先把工具操作记录写进去（它们在对话之前，对理解后续决策有关键作用）
        if self._tool_notes:
            lines.append("[Claude Code 工具操作记录]")
            for note in self._tool_notes:
                lines.append(f"- {note}")
            lines.append("")
        for pair in self._pending_pairs:
            lines.append(f"用户: {pair[0]['content']}")
            if len(pair) > 1:
                lines.append(f"助手: {pair[1]['content']}")
            lines.append("")
        pending_text = "\n".join(lines)

        try:
            new_summary = weak_model.summarize_conversation(pending_text, self.summary)
            if new_summary:
                # 存储时截断，避免摘要无限膨胀 + 下次喂回弱模型时浪费 token
                if len(new_summary) > SUMMARY_MAX_CHARS:
                    new_summary = new_summary[:SUMMARY_MAX_CHARS] + (
                        f"\n...[摘要已截断，原 {len(new_summary)} 字符]"
                    )
                self.summary = new_summary
                self._pending_pairs.clear()
                self._tool_notes.clear()  # 工具操作记录已纳入摘要
                self._summary_failed = False
                logger.debug("摘要成功，summary 长度: %d 字符", len(self.summary))
        except Exception:
            self._summary_failed = True
            logger.warning(
                "摘要生成失败，保留 %d 对 pending，下次重试",
                len(self._pending_pairs), exc_info=True,
            )

    def _summarize_fallback(self) -> str:
        """摘要失败时的降级：将 pending_pairs 中的对话原文拼接返回。

        返回后不清空 _pending_pairs（保留供下次重试）。
        兼容孤儿消息（pair 可能只含 user 一个元素）。
        """
        if not self._pending_pairs:
            return ""
        label = "摘要生成失败，使用原文" if self._summary_failed else "待摘要的早期对话"
        lines = [f"[以下为裁剪的早期对话（{label}）]"]
        for pair in self._pending_pairs:
            lines.append(f"用户: {pair[0]['content']}")
            if len(pair) > 1:
                assistant_msg = pair[1]
                src = assistant_msg.get("source", "")
                tag = "💡 弱模型" if src == "weak" else ""
                lines.append(f"助手{tag}: {assistant_msg['content']}")
            lines.append("")
        return "\n".join(lines)

    # ── 强模型上下文入口 ────────────────────────

    def get_context_for_strong(self, task: str | None = None, weak_model=None,
                               pre_context: str = "", include_last_turn: bool = False) -> str:
        """为强模型调用装配完整上下文。任务在前，参考材料在后。

        include_last_turn: False 时排除 turns 中最新的 user 消息（防止当前任务
                          在 [最近对话] 中重复出现导致模型误认为对话中断）。
        """
        # 懒触发摘要
        if weak_model and len(self._pending_pairs) >= PENDING_MIN_PAIRS:
            self._maybe_summarize(weak_model)

        parts = []

        # —— 当前任务（放在最前面，加明确的行为指令）——
        if task:
            parts.append("请完成以下任务。如有参考材料，请结合材料作答。")
            parts.append("")
            parts.append(task)

        # —— 预收集的上下文 ——
        if pre_context:
            if parts:
                parts.append("")
            parts.append("---")
            parts.append("[参考材料 · 预收集的上下文]")
            parts.append(pre_context)

        # —— 工具操作记录 ——
        if self._tool_notes:
            if parts:
                parts.append("")
            if not pre_context:
                parts.append("---")
            parts.append("[参考材料 · Claude Code 已执行的步骤]")
            for note in self._tool_notes:
                parts.append(f"- {note}")

        # —— 摘要 ——
        # 已在 _maybe_summarize 存储时截断，此处通常无需再截。
        # 保留防御性检查：处理从旧版本迁移过来的未截断摘要。
        if self.summary:
            summary_text = self.summary
            if len(summary_text) > SUMMARY_MAX_CHARS:
                summary_text = summary_text[:SUMMARY_MAX_CHARS] + (
                    f"\n...[摘要过长已截断，原 {len(self.summary)} 字符]"
                )
            if parts:
                parts.append("")
            if not pre_context and not self._tool_notes:
                parts.append("---")
            parts.append("[参考材料 · 对话摘要]")
            parts.append(summary_text)

        # —— 摘要失败的降级原文 ——
        fallback = self._summarize_fallback()
        if fallback:
            if parts:
                parts.append("")
            if not pre_context and not self._tool_notes and not self.summary:
                parts.append("---")
            parts.append(fallback)

        # —— 最近对话 ——
        # 默认排除最新的 user 消息（它已经作为 task 出现在前面，不应重复）
        turns_to_show = list(self.turns)
        if not include_last_turn and turns_to_show and turns_to_show[-1]["role"] == "user":
            turns_to_show.pop()
        if turns_to_show:
            recent_lines = []
            for t in turns_to_show:
                role = t["role"]
                content = t["content"]
                if role == "user":
                    recent_lines.append(f"用户: {content}")
                else:
                    src = t.get("source", "")
                    prefix = "💡 弱模型: " if src == "weak" else "助手: "
                    recent_lines.append(f"{prefix}{content}")
            if parts:
                parts.append("")
            if not pre_context and not self._tool_notes and not self.summary and not fallback:
                parts.append("---")
            parts.append("[参考材料 · 最近对话]")
            parts.append("\n".join(recent_lines))

        return "\n".join(parts) if parts else ""
