import os
from typing import Optional
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

# Anthropic 支持的图片格式
SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})

# Anthropic 单次请求图片数量上限
MAX_IMAGES_PER_REQUEST = 20


class ImageDescriptionTruncated(Exception):
    """describe_images 返回的文本描述被 max_tokens 截断。"""

    def __init__(self, partial_text: str, max_tokens: int):
        self.partial_text = partial_text
        self.max_tokens = max_tokens
        super().__init__(
            f"图片描述被截断（max_tokens={max_tokens}），已返回 {len(partial_text)} 字符的部分内容"
        )


class DeepSeekModel:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def call(self, prompt: str, context: Optional[str] = None, history: Optional[list[dict]] = None) -> str:
        content = f"{context}\n\n{prompt}" if context else prompt
        messages = list(history) if history else []
        messages.append({"role": "user", "content": content})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content


class AnthropicModel:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        max_tokens: int = 32000,
        warn_tokens: int = 8000,
    ):
        kwargs = dict(api_key=api_key, timeout=600)
        if base_url:
            kwargs["base_url"] = base_url
        self.client = Anthropic(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.warn_tokens = warn_tokens

    # ── 内部工具方法 ────────────────────────────

    @property
    def supports_vision(self) -> bool:
        """当前配置的强模型是否支持多模态（图片输入）。

        Claude 3+ 全系支持 vision，Claude 2.x 已弃用。
        保留此属性以便将来切换到不支持 vision 的模型时返回 False。
        """
        return True

    @staticmethod
    def _extract_text(response) -> str:
        """从 Anthropic API 响应中提取第一个文本块。

        处理 thinking 块在前的情况（extended thinking 时 content[0] 可能是
        thinking 类型），也处理空 content 的极端情况。
        """
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""

    @staticmethod
    def _image_to_block(img: dict) -> dict:
        """将图片 dict 转为 Anthropic API image content block。

        校验：
        - data 是非空字符串
        - data 不含 "data:" 前缀（常见错误）
        - media_type 在支持列表中
        """
        data = img.get("data")
        if not isinstance(data, str) or not data:
            raise ValueError("图片 data 必须是非空 base64 字符串")
        if data.startswith("data:"):
            raise ValueError(
                "图片 data 字段不应包含 'data:image/...;base64,' 前缀，"
                "只需纯 base64 内容"
            )

        media_type = img.get("media_type", "")
        if media_type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(
                f"不支持的图片格式: {media_type}，"
                f"支持的格式: {', '.join(sorted(SUPPORTED_IMAGE_TYPES))}"
            )
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }

    def _post_process(self, text: str, response) -> str:
        """追加截断 / 长度警告。使用 API 返回的精确 output_tokens。

        兼容响应缺少 usage 字段的情况（如测试 mock）。
        """
        tokens = None
        usage = getattr(response, "usage", None)
        if usage is not None:
            tokens = getattr(usage, "output_tokens", None)

        if not isinstance(tokens, int):
            tokens = len(text) // 4  # 近似回退

        if response.stop_reason == "max_tokens":
            text += (
                f"\n\n{'─' * 42}\n"
                f"⚠️ 输出被硬截断（已达 max_tokens={self.max_tokens} 上限）\n"
                f"   模型可能还有未输出内容。如需继续，请发送「继续」或「接着输出」。"
            )
        elif tokens > self.warn_tokens:
            text += (
                f"\n\n{'─' * 42}\n"
                f"📊 本次输出较长（约 {tokens} tokens），如需精简可要求「简短一点」。"
            )
        return text

    @staticmethod
    def _build_describe_prompt(task_context: str) -> str:
        """构建图片描述 prompt。"""
        base = (
            "请详细描述以上图片的内容。"
            "你的描述将用于后续文本模型处理该任务，"
            "因此请尽可能完整地提取图片中的信息。注意：\n"
            "1. 图片中的所有文字内容（逐字转录）\n"
            "2. 图片的结构、布局、图表/表格数据\n"
            "3. 人物、物体、场景、UI 元素等视觉内容"
        )
        if task_context:
            return (
                f"{base}\n"
                f"4. 与以下任务特别相关的细节：\n"
                f"{task_context}"
            )
        return base

    # ── 公开方法 ────────────────────────────────

    def call(self, prompt: str, context: Optional[str] = None, history: Optional[list[dict]] = None) -> str:
        """纯文本调用。"""
        content = f"{context}\n\n{prompt}" if context else prompt
        messages = list(history) if history else []
        messages.append({"role": "user", "content": content})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        text = self._extract_text(response)
        return self._post_process(text, response)

    def describe_images(
        self,
        images: list[dict],
        task_context: str = "",
    ) -> str:
        """将图片交给强模型解析为文本描述。

        用于弱模型（DeepSeek）不支持多模态的场景。
        如果描述被 max_tokens 截断，抛出 ImageDescriptionTruncated。

        Args:
            images: [{"data": "<base64>", "media_type": "image/png"}, ...]
            task_context: 用户原始任务，用于引导描述关注相关细节

        Returns:
            纯文本图片描述

        Raises:
            ValueError: images 为空、格式不支持、或超过数量上限
            ImageDescriptionTruncated: 输出被 max_tokens 截断
        """
        if not images:
            raise ValueError("images 不能为空")
        if len(images) > MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"图片数量 ({len(images)}) 超过单次请求上限 ({MAX_IMAGES_PER_REQUEST})"
            )

        # 图片在前，文本在后（Anthropic 推荐顺序）
        content_blocks = [self._image_to_block(img) for img in images]
        content_blocks.append({
            "type": "text",
            "text": self._build_describe_prompt(task_context),
        })

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content_blocks}],
        )
        text = self._extract_text(response)

        if response.stop_reason == "max_tokens":
            raise ImageDescriptionTruncated(text, self.max_tokens)

        return text

    def call_with_images(self, prompt: str, images: list[dict], history: Optional[list[dict]] = None) -> str:
        """原生多模态调用 —— prompt + 图片直接发给强模型。

        describe_images 截断时抛异常（中间产物，下游不知情），
        而 call_with_images 截断只加 warning（终端输出，用户可见）。

        Args:
            prompt: 文本提示
            images: [{"data": "<base64>", "media_type": "image/png"}, ...]
            history: 对话历史（可选）

        Returns:
            模型回复文本

        Raises:
            ValueError: images 为空、格式不支持、或超过数量上限
        """
        if not images:
            raise ValueError("images 不能为空")
        if len(images) > MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"图片数量 ({len(images)}) 超过单次请求上限 ({MAX_IMAGES_PER_REQUEST})"
            )

        content_blocks = [self._image_to_block(img) for img in images]
        content_blocks.append({"type": "text", "text": prompt})

        messages = list(history) if history else []
        messages.append({"role": "user", "content": content_blocks})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        text = self._extract_text(response)
        return self._post_process(text, response)


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
