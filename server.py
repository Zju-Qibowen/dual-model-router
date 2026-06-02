import os
import re
import json
import hashlib
import logging
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from models import load_models, DeepSeekModel, AnthropicModel, ImageDescriptionTruncated
from router import route_task
from conversation import ConversationHistory

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

mcp = FastMCP("dual-model-router")
logger = logging.getLogger("dual-model-router")

# ── 图片处理常量 ──────────────────────────────────
# DeepSeek v4 不支持多模态输入，因此将图片先交给强模型解析为文本描述，
# 再合并文本描述到原始任务中，后续正常走弱模型路由。

# 匹配 base64 data URL（注意：不含 http(s):// 远端图片链接）
IMAGE_DATA_URL_RE = re.compile(
    r'data:(image/[a-zA-Z0-9.+\-]+);base64,([A-Za-z0-9+/]+={0,2})',
    re.IGNORECASE,
)

MAX_IMAGES = 8
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB per image
MAX_PARTIAL_TEXT_CHARS = 6000         # 截断描述最多保留字符数
DEFAULT_IMAGE_PROMPT = "请根据以上图片内容回答问题。"


REVIEW_PROMPT_CODE = """请审核以下代码修改：指出错误或遗漏，给出最终建议。

<user_input>
{task}
</user_input>

DeepSeek 的修改（diff 格式）：
{weak_response}"""

REVIEW_PROMPT_TEXT = """请审核以下回答：指出错误或遗漏，并给出最终完整答案。

<user_input>
{task}
</user_input>

DeepSeek 的回答：
{weak_response}"""


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _is_code_response(text: str) -> bool:
    return "```" in text or text.count("\n") > 10


def _format_error(step: str, model_name: str, error: Exception, model_type: str) -> str:
    error_type = type(error).__name__
    error_msg = str(error)
    label = "弱模型" if model_type == "weak" else "强模型"
    return (
        f"[错误: {step} | {label}: {model_name}]\n"
        f"{error_type}: {error_msg}\n"
        f"建议: set_weak_model / set_strong_model 切换模型, 或用 ask_weak / ask_strong 逐个调试"
    )


# ═══════════════════════════════════════════════════
# 图片提取 & 预处理
# ═══════════════════════════════════════════════════

def _image_data_hash(img: dict) -> str:
    """计算图片 data 的短哈希，用于去重。"""
    return hashlib.sha256(img.get("data", "").encode()).hexdigest()[:16]


def _validate_images(images: list[dict]) -> tuple[list[dict], list[str]]:
    """逐张校验图片列表。

    返回 (有效图片列表, 警告/错误消息列表)。
    逐张跳过不合规的图片而非全部丢弃。
    """
    if not images:
        return [], []

    warnings: list[str] = []

    if len(images) > MAX_IMAGES:
        warnings.append(f"图片数量超过上限（{MAX_IMAGES}），已保留前 {MAX_IMAGES} 张")
        images = images[:MAX_IMAGES]

    valid: list[dict] = []
    for i, img in enumerate(images):
        data = img.get("data", "")
        decoded_bytes = len(data) * 3 // 4 - data.count("=")  # 减去 padding
        if decoded_bytes > MAX_IMAGE_BYTES:
            warnings.append(f"图片 {i+1} 体积超过上限（{MAX_IMAGE_BYTES // 1024 // 1024} MB），已跳过")
            continue
        if not data or not re.match(r'^[A-Za-z0-9+/]+={0,2}$', data):
            warnings.append(f"图片 {i+1} 的 data 不是有效的 base64 字符串，已跳过")
            continue
        valid.append(img)

    return valid, warnings


def _extract_images_from_task(task: str) -> tuple[str, list[dict]]:
    """从任务文本中提取内嵌的 data:image/...;base64,... 图片。

    替换为 [图片] 占位符（不带编号，因为去重后编号会错位）。
    返回 (清理后的任务文本, 图片列表)。
    """
    images: list[dict] = []

    def _replacer(m: re.Match) -> str:
        images.append({"data": m.group(2), "media_type": m.group(1)})
        return "[图片]"

    cleaned = IMAGE_DATA_URL_RE.sub(_replacer, task)
    return cleaned, images


def _deduplicate_images(images: list[dict]) -> list[dict]:
    """按 base64 data 去重，保留首次出现顺序。"""
    seen: set[str] = set()
    result: list[dict] = []
    for img in images:
        h = _image_data_hash(img)
        if h not in seen:
            seen.add(h)
            result.append(img)
    return result


def _parse_images_param(images_json: str | None) -> list[dict]:
    """解析 images_json 参数为图片列表。

    支持的格式：
    - JSON 数组: [{"data": "<base64>", "media_type": "image/png"}, ...]
    - 兼容缺少 media_type 的情况（默认 image/png）
    """
    if not images_json:
        return []
    try:
        parsed = json.loads(images_json)
        if not isinstance(parsed, list):
            return []
        result = []
        for img in parsed:
            if not isinstance(img, dict):
                continue
            data = img.get("data", "")
            if not data:
                continue
            result.append({
                "data": data,
                "media_type": img.get("media_type", "image/png"),
            })
        return result
    except (json.JSONDecodeError, TypeError):
        return []


def _collect_images(task: str, images_json: str | None) -> tuple[str, list[dict], list[str]]:
    """统一的图片搜集入口。

    返回 (清理后的 task 文本, 有效图片列表, 校验警告列表)。
    警告列表用于展示给用户（如超限、格式错误等）。
    """
    all_images: list[dict] = []
    all_warnings: list[str] = []

    if images_json:
        all_images.extend(_parse_images_param(images_json))

    cleaned_task, embedded = _extract_images_from_task(task)
    if embedded:
        all_images.extend(embedded)
        task = cleaned_task

    if all_images:
        all_images = _deduplicate_images(all_images)
        all_images, warnings = _validate_images(all_images)
        all_warnings.extend(warnings)

    return task, all_images, all_warnings


def _describe_images(
    images: list[dict],
    task_context: str,
    strong: AnthropicModel,
) -> str:
    """用强模型将图片解析为文本描述。

    task_context 应为 cleaned task（含 [图片] 占位符，不含 base64 data URL）。
    失败或截断时返回带警告标记的文本，不抛异常（保证后续路由能继续）。
    """
    if not images:
        return ""

    try:
        descriptions = strong.describe_images(images, task_context=task_context)
    except ImageDescriptionTruncated as e:
        logger.warning("图片描述被截断: max_tokens=%d, chars=%d", e.max_tokens, len(e.partial_text or ""))
        partial = (e.partial_text or "")[:MAX_PARTIAL_TEXT_CHARS]
        if e.partial_text and len(e.partial_text) > MAX_PARTIAL_TEXT_CHARS:
            partial += f"\n...[已截断，原 {len(e.partial_text)} 字符，仅保留前 {MAX_PARTIAL_TEXT_CHARS} 字符]"
        count_info = f"{len(images)} 张图片，" if len(images) > 1 else ""
        return (
            f"\n\n[⚠ 图片描述不完整 —— {count_info}被 max_tokens={e.max_tokens} 截断]\n"
            f"{partial}\n"
            f"[/图片描述]"
        )
    except Exception as e:
        logger.warning("图片解析失败: %s: %s", type(e).__name__, e)
        return f"\n\n[⚠ 图片解析失败: {type(e).__name__}]\n"

    # 正常返回
    if not descriptions or not descriptions.strip():
        logger.warning("图片描述返回空内容")
        return f"\n\n[⚠ 图片描述为空]\n"

    count_info = f"{len(images)} 张图片，" if len(images) > 1 else ""
    return (
        f"\n\n[图片描述 —— {count_info}由强模型 ({strong.model}) 解析]\n"
        f"{descriptions}\n"
        f"[/图片描述]"
    )


# ═══════════════════════════════════════════════════
# 路由过程展示
# ═══════════════════════════════════════════════════

def _build_process_header(
    route_decision: str,
    weak_model_name: str,
    strong_model_name: str,
    images_processed: int = 0,
    images_truncated: bool = False,
    warnings: list[str] | None = None,
) -> str:
    """构建简洁的路由元信息行。"""
    parts = []
    if images_processed > 0:
        status = "⚠不完整" if images_truncated else ""
        parts.append(f"图片: {images_processed}张{status}")
    if warnings:
        parts.append(f"警告: {'; '.join(warnings[:2])}")

    if route_decision == "strong":
        parts.append(f"路由: strong → {strong_model_name}")
    elif route_decision == "medium":
        parts.append(f"路由: medium → {weak_model_name} + {strong_model_name}审核")
    else:
        parts.append(f"路由: weak → {weak_model_name}")

    return "[" + " | ".join(parts) + "]"




# ═══════════════════════════════════════════════════
# 核心路由处理
# ═══════════════════════════════════════════════════

def handle_task(
    task: str,
    weak: DeepSeekModel,
    strong: AnthropicModel,
    execution_task: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """路由 + 执行。

    三档路由：
    - weak: 弱模型执行，不审核
    - medium: 弱模型执行 + 强模型审核
    - strong: 强模型直接执行

    task: 用于路由判断的文本（不含图片描述，避免描述文本干扰路由）
    execution_task: 实际执行的文本（含图片描述）。为 None 时使用 task。
    history: 对话历史（传给执行调用，不传给路由判断和审核）。
    """
    if execution_task is None:
        execution_task = task

    try:
        decision = route_task(task, weak)
    except Exception as e:
        return {"_error": True, "step": "路由判断", "model": weak.model, "model_type": "weak", "error": e}

    if decision == "strong":
        try:
            answer = strong.call(execution_task, history=history)
        except Exception as e:
            return {"_error": True, "step": "强模型直接执行", "model": strong.model, "model_type": "strong", "error": e}
        return {"routed_to": "strong", "final_answer": answer, "reviewed": False, "weak_result": None, "token_count": None}

    # weak 和 medium 都先走弱模型执行
    try:
        weak_result = weak.call(execution_task, history=history)
    except Exception as e:
        return {"_error": True, "step": "弱模型执行", "model": weak.model, "model_type": "weak", "error": e}

    token_count = _estimate_tokens(weak_result)

    # weak: 直接返回，不审核
    if decision == "weak":
        return {
            "routed_to": "weak",
            "final_answer": weak_result,
            "reviewed": False,
            "weak_result": weak_result,
            "token_count": token_count,
        }

    # medium: 强模型审核（审核 prompt 自包含，不传 history）
    if _is_code_response(weak_result):
        review_prompt = REVIEW_PROMPT_CODE.format(task=execution_task, weak_response=weak_result)
    else:
        review_prompt = REVIEW_PROMPT_TEXT.format(task=execution_task, weak_response=weak_result)

    try:
        reviewed = strong.call(review_prompt)
    except Exception as e:
        return {"_error": True, "step": "强模型审核", "model": strong.model, "model_type": "strong", "error": e, "weak_result": weak_result, "token_count": token_count}

    return {"routed_to": "medium", "final_answer": reviewed, "reviewed": True, "weak_result": weak_result, "token_count": token_count}


_weak, _strong = None, None
_history = ConversationHistory()


def _get_models():
    global _weak, _strong
    if _weak is None:
        _weak, _strong = load_models()
    return _weak, _strong


# ═══════════════════════════════════════════════════
# MCP 工具
# ═══════════════════════════════════════════════════

@mcp.tool()
def set_weak_model(model: str) -> str:
    """切换弱模型（DeepSeek）。可用值如 deepseek-chat、deepseek-reasoner。"""
    global _weak
    _get_models()
    _weak.model = model
    return f"弱模型已切换为: {model}"


@mcp.tool()
def set_strong_model(model: str) -> str:
    """切换强模型（Anthropic）。可用值如 claude-haiku-4-5-20251001、claude-sonnet-4-5、claude-opus-4-5。"""
    global _strong
    _get_models()
    _strong.model = model
    return f"强模型已切换为: {model}"


@mcp.tool()
def list_models() -> str:
    """查看当前使用的弱模型和强模型。"""
    weak, strong = _get_models()
    return f"弱模型（DeepSeek）: {weak.model}\n强模型（Anthropic）: {strong.model}"


@mcp.tool()
def route_and_answer(task: str, images_json: str = "") -> str:
    """自动路由任务到合适的模型并返回结果。支持多轮追问（服务端维护对话历史）。
    三档路由：weak→弱模型直接回答，medium→弱模型+强模型审核，strong→强模型直接执行。
    支持图片输入（images_json 或内嵌 data:image URL）。

    展示要求：请按如下格式向用户展示结果——
    > **[元信息行原样保留]**
    > （空行后引用块展示模型原文）"""
    weak, strong = _get_models()

    cleaned_task, all_images, warnings = _collect_images(task, images_json)

    images_truncated = False
    descriptions = ""
    if all_images:
        descriptions = _describe_images(all_images, cleaned_task, strong)
        if descriptions.startswith("\n\n[⚠ 图片描述不完整"):
            images_truncated = True

    execution_task = f"{cleaned_task}{descriptions}" if descriptions else cleaned_task

    # 记录用户输入（用 execution_task 以便后续追问时模型知道图片内容）
    _history.add_user(execution_task)

    result = handle_task(
        task=cleaned_task,
        execution_task=execution_task,
        weak=weak,
        strong=strong,
        history=_history.get_messages()[:-1],  # 不含刚加的当前 user message（由 model.call 自行追加）
    )
    route_decision = result.get("routed_to", "strong")

    # 成功且回答非空时记录 assistant 回答
    if not result.get("_error") and result.get("final_answer"):
        _history.add_assistant(result["final_answer"])

    header = _build_process_header(
        route_decision, weak.model, strong.model,
        images_processed=len(all_images),
        images_truncated=images_truncated,
        warnings=warnings if warnings else None,
    )

    if result.get("_error"):
        # 失败时撤回刚加的 user message
        if _history.turns and _history.turns[-1]["role"] == "user":
            _history.turns.pop()
        prefix = ""
        if result.get("weak_result"):
            prefix = f"弱模型已执行但审核失败，原始回答: {result['weak_result'][:100]}\n"
        return header + "\n" + prefix + _format_error(result["step"], result["model"], result["error"], result["model_type"])

    return f"{header}\n{result['final_answer']}"


@mcp.tool()
def ask_weak(prompt: str, images_json: str = "") -> str:
    """直接调用 DeepSeek（弱模型）。支持多轮对话上下文。如有图片会先用强模型解析为文本描述。

    展示要求：请按如下格式向用户展示结果——
    > **[元信息行原样保留]**
    > （空行后引用块展示模型原文）"""
    weak, strong = _get_models()

    prompt, all_images, warnings = _collect_images(prompt, images_json)

    try:
        if all_images:
            descriptions = _describe_images(all_images, prompt, strong)
            if not prompt.strip():
                prompt = DEFAULT_IMAGE_PROMPT
            prompt = f"{prompt}{descriptions}"
            header = f"[弱模型: {weak.model} | 图片: {len(all_images)}张]"
        else:
            header = f"[弱模型: {weak.model}]"

        _history.add_user(prompt)
        result = weak.call(prompt, history=_history.get_messages()[:-1])
        _history.add_assistant(result)
        return f"{header}\n{result}"
    except Exception as e:
        if _history.turns and _history.turns[-1]["role"] == "user":
            _history.turns.pop()
        return _format_error("弱模型调用", weak.model, e, "weak")


@mcp.tool()
def ask_strong(prompt: str, weak_response: str = "", images_json: str = "") -> str:
    """直接调用 Anthropic（强模型）。支持多轮对话上下文。可选传入弱模型结果作为审核上下文。
    支持多模态图片直接传入。

    展示要求：请按如下格式向用户展示结果——
    > **[元信息行原样保留]**
    > （空行后引用块展示模型原文）"""
    _, strong = _get_models()

    prompt, all_images, warnings = _collect_images(prompt, images_json)

    try:
        if all_images and not strong.supports_vision:
            descriptions = _describe_images(all_images, prompt, strong)
            if not prompt.strip():
                prompt = DEFAULT_IMAGE_PROMPT
            prompt = f"{prompt}{descriptions}"

        _history.add_user(prompt)
        hist = _history.get_messages()[:-1]

        if weak_response:
            context = f"以下是弱模型的回答，请审核并给出最终答案：\n{weak_response}"
            full_prompt = f"{context}\n\n{prompt}"
            header = f"[强模型审核: {strong.model}]"
            if all_images and strong.supports_vision:
                result = strong.call_with_images(full_prompt, all_images, history=hist)
            else:
                result = strong.call(full_prompt, history=hist)
        elif all_images and strong.supports_vision:
            header = f"[强模型: {strong.model} | 多模态]"
            result = strong.call_with_images(prompt, all_images, history=hist)
        else:
            header = f"[强模型: {strong.model}]"
            result = strong.call(prompt, history=hist)

        _history.add_assistant(result)
        return f"{header}\n{result}"
    except Exception as e:
        if _history.turns and _history.turns[-1]["role"] == "user":
            _history.turns.pop()
        return _format_error("强模型调用", strong.model, e, "strong")


@mcp.tool()
def review(content: str, context: str = "") -> str:
    """用强模型审核任意内容，返回审核意见。不计入对话历史。

    展示要求：请按如下格式向用户展示结果——
    > **[元信息行原样保留]**
    > （空行后引用块展示审核原文）"""
    _, strong = _get_models()
    prompt = f"请审核以下内容，指出错误、遗漏或改进点：\n\n{content}"
    if context:
        prompt = f"背景：{context}\n\n{prompt}"
    try:
        result = strong.call(prompt, history=_history.get_messages())
        return f"[审核: {strong.model}]\n{result}"
    except Exception as e:
        return _format_error("审核", strong.model, e, "strong")


@mcp.tool()
def clear_context() -> str:
    """清除对话历史。开始新话题或上下文不再相关时使用。"""
    count = _history.turn_count
    _history.clear()
    return f"已清除对话历史（{count} 轮对话）。后续调用将不包含之前的上下文。"


@mcp.tool()
def get_context_status() -> str:
    """查看当前对话上下文状态：轮数、token 估算。"""
    return (
        f"对话历史状态:\n"
        f"  轮数: {_history.turn_count}/{_history.max_turns}\n"
        f"  估计 tokens: {_history.estimated_tokens}/{_history.max_tokens}\n"
        f"  消息数: {len(_history.turns)}"
    )


if __name__ == "__main__":
    mcp.run()
