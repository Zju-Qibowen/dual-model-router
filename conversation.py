"""对话历史管理 —— 为多轮上下文提供滑动窗口存储。"""

from dataclasses import dataclass, field

DEFAULT_MAX_TURNS = 10
DEFAULT_MAX_TOKENS = 16000


@dataclass
class ConversationHistory:
    max_turns: int = DEFAULT_MAX_TURNS
    max_tokens: int = DEFAULT_MAX_TOKENS
    turns: list[dict] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.turns.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        self.turns.append({"role": "assistant", "content": content})
        self._trim()

    def get_messages(self) -> list[dict]:
        return [{"role": t["role"], "content": t["content"]} for t in self.turns]

    def clear(self) -> None:
        self.turns.clear()

    @property
    def turn_count(self) -> int:
        return sum(1 for t in self.turns if t["role"] == "user")

    @property
    def estimated_tokens(self) -> int:
        return self._total_tokens()

    def _estimate_tokens(self, text: str) -> int:
        cjk = sum(1 for c in text if '一' <= c <= '鿿')
        return int(cjk * 1.5 + (len(text) - cjk) * 0.25)

    def _total_tokens(self) -> int:
        return sum(self._estimate_tokens(t["content"]) for t in self.turns)

    def _trim(self) -> None:
        while len(self.turns) > self.max_turns * 2:
            # 成对弹出，保证 user 在前
            if len(self.turns) >= 2 and self.turns[0]["role"] == "user" and self.turns[1]["role"] == "assistant":
                del self.turns[0:2]
            else:
                self.turns.pop(0)
        while self._total_tokens() > self.max_tokens and len(self.turns) > 2:
            if len(self.turns) >= 2 and self.turns[0]["role"] == "user" and self.turns[1]["role"] == "assistant":
                del self.turns[0:2]
            else:
                self.turns.pop(0)
        # 确保开头是 user（防御异常状态）
        while self.turns and self.turns[0]["role"] == "assistant":
            self.turns.pop(0)
