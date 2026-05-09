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
    def __init__(self, api_key: str, model: str, base_url: str = None):
        self.client = Anthropic(api_key=api_key, base_url=base_url) if base_url else Anthropic(api_key=api_key)
        self.model = model

    def call(self, prompt: str, context: str = None) -> str:
        content = f"{context}\n\n{prompt}" if context else prompt
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text


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
    )
    return weak, strong
