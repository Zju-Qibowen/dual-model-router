import os
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)


class DeepSeekModel:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def call(self, prompt: str, context: str = None) -> str:
        content = f"{context}\n\n{prompt}" if context else prompt
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
        )
        return response.choices[0].message.content


class AnthropicModel:
    def __init__(self, api_key: str, model: str, base_url: str = None,
                 max_tokens: int = 32000, warn_tokens: int = 8000):
        kwargs = dict(api_key=api_key, timeout=600)
        if base_url:
            kwargs["base_url"] = base_url
        self.client = Anthropic(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.warn_tokens = warn_tokens

    def call(self, prompt: str, context: str = None) -> str:
        content = f"{context}\n\n{prompt}" if context else prompt
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        text = response.content[0].text
        estimated = len(text) // 4

        if response.stop_reason == "max_tokens":
            text += (
                f"\n\n{'─' * 42}\n"
                f"⚠️ 输出被硬截断（已达 max_tokens={self.max_tokens} 上限）\n"
                f"   模型可能还有未输出内容。如需继续，请发送「继续」或「接着输出」。"
            )
        elif estimated > self.warn_tokens:
            text += (
                f"\n\n{'─' * 42}\n"
                f"📊 本次输出较长（约 {estimated} tokens），如需精简可要求「简短一点」。"
            )
        return text


def load_models() -> tuple[DeepSeekModel, AnthropicModel]:
    weak = DeepSeekModel(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    )
    strong = AnthropicModel(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", "32000")),
        warn_tokens=int(os.environ.get("ANTHROPIC_WARN_TOKENS", "8000")),
    )
    return weak, strong
