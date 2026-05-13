import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from models import load_models, DeepSeekModel, AnthropicModel
from router import route_task

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

mcp = FastMCP("dual-model-router")

REVIEW_TOKEN_LIMIT = int(os.environ.get("REVIEW_TOKEN_LIMIT", "2000"))

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
    return (
        f"❌ 调用失败\n"
        f"  步骤: {step}\n"
        f"  模型: {model_name} ({'弱模型/DeepSeek' if model_type == 'weak' else '强模型/Anthropic'})\n"
        f"  错误: {error_type}: {error_msg}\n"
        f"---\n"
        f"建议: set_weak_model / set_strong_model 切换模型, list_available_models 查看可用模型, "
        f"或用 ask_weak / ask_strong 逐个调试"
    )


def _build_process_header(strong_path: bool, weak_model_name: str, strong_model_name: str) -> str:
    """构建醒目的双路由处理过程头部"""
    bar = "═" * 42
    header = f"╔{bar}╗\n║  🔀 双模型路由处理中 …           ║\n╠{bar}╣"
    if strong_path:
        header += f"\n║  路由判断 → strong（复杂任务）   ║\n║  执行模型 → {strong_model_name:<22}║"
    else:
        header += f"\n║  路由判断 → weak（简单任务）      ║"
    return header


def _build_process_footer(strong_path: bool, reviewed: bool, weak_model_name: str, strong_model_name: str, token_count: int | None, note: str | None) -> str:
    """构建路由处理过程尾部"""
    bar = "═" * 42
    lines = []
    if strong_path:
        lines.extend([
            f"╚{bar}╝",
            "",
            "📌 以下为强模型直接生成的回答：",
        ])
    else:
        lines.append(f"╠{bar}╣")
        lines.append(f"║  执行模型 → {weak_model_name:<22}║")
        if token_count is not None:
            lines.append(f"║  输出量级 → ~{token_count} tokens{'':<16}║")
        if reviewed:
            lines.append(f"╠{bar}╣")
            lines.append(f"║  审核模型 → {strong_model_name:<22}║")
        lines.append(f"╚{bar}╝")
        if not reviewed and note:
            lines.append(f"\n⚠ {note}")
        lines.append("")
        if reviewed:
            lines.append("📌 以下为强模型审核后的最终回答：")
        else:
            lines.append("📌 以下为弱模型的回答：")
    return "\n".join(lines)


def handle_task(
    task: str,
    weak: DeepSeekModel,
    strong: AnthropicModel,
    review_token_limit: int = REVIEW_TOKEN_LIMIT,
) -> dict:
    try:
        decision = route_task(task, weak)
    except Exception as e:
        return {"_error": True, "step": "路由判断", "model": weak.model, "model_type": "weak", "error": e}

    if decision == "strong":
        try:
            answer = strong.call(task)
        except Exception as e:
            return {"_error": True, "step": "强模型直接执行", "model": strong.model, "model_type": "strong", "error": e}
        return {"routed_to": "strong", "final_answer": answer, "reviewed": False, "weak_result": None, "token_count": None}

    try:
        weak_result = weak.call(task)
    except Exception as e:
        return {"_error": True, "step": "弱模型执行", "model": weak.model, "model_type": "weak", "error": e}

    token_count = _estimate_tokens(weak_result)

    if token_count > review_token_limit:
        return {
            "routed_to": "weak",
            "final_answer": weak_result,
            "reviewed": False,
            "weak_result": weak_result,
            "token_count": token_count,
            "note": f"输出超过 {review_token_limit} token 限制（约 {token_count} tokens），已跳过审核",
        }

    if _is_code_response(weak_result):
        review_prompt = REVIEW_PROMPT_CODE.format(task=task, weak_response=weak_result)
    else:
        review_prompt = REVIEW_PROMPT_TEXT.format(task=task, weak_response=weak_result)

    try:
        reviewed = strong.call(review_prompt)
    except Exception as e:
        return {"_error": True, "step": "强模型审核", "model": strong.model, "model_type": "strong", "error": e, "weak_result": weak_result, "token_count": token_count}

    return {"routed_to": "weak", "final_answer": reviewed, "reviewed": True, "weak_result": weak_result, "token_count": token_count}


_weak, _strong = None, None


def _get_models():
    global _weak, _strong
    if _weak is None:
        _weak, _strong = load_models()
    return _weak, _strong


@mcp.tool()
def set_weak_model(model: str) -> str:
    """切换弱模型（DeepSeek）。可用值如 deepseek-chat、deepseek-reasoner。"""
    global _weak
    _get_models()  # 确保已初始化
    _weak.model = model
    return f"弱模型已切换为: {model}"


@mcp.tool()
def set_strong_model(model: str) -> str:
    """切换强模型（Anthropic）。可用值如 claude-haiku-4-5-20251001、claude-sonnet-4-5、claude-opus-4-5。"""
    global _strong
    _get_models()  # 确保已初始化
    _strong.model = model
    return f"强模型已切换为: {model}"


@mcp.tool()
def list_models() -> str:
    """查看当前使用的弱模型和强模型。"""
    weak, strong = _get_models()
    return f"弱模型（DeepSeek）: {weak.model}\n强模型（Anthropic）: {strong.model}"


@mcp.tool()
def route_and_answer(task: str) -> str:
    """自动路由任务到合适的模型并返回结果。简单任务用 DeepSeek，复杂任务用 Anthropic。"""
    weak, strong = _get_models()
    result = handle_task(task, weak, strong)
    is_strong = result.get("routed_to") == "strong"

    header = _build_process_header(is_strong, weak.model, strong.model)

    if result.get("_error"):
        prefix = ""
        if result.get("weak_result"):
            prefix = f"║  ⚠ 弱模型已执行但审核失败         ║\n║  原始回答: {result['weak_result'][:50]:<22}║\n"
        return header + "\n" + prefix + _format_error(result["step"], result["model"], result["error"], result["model_type"])

    footer = _build_process_footer(
        is_strong,
        result["reviewed"],
        weak.model,
        strong.model,
        result.get("token_count"),
        result.get("note"),
    )
    return f"{header}\n{footer}\n{result['final_answer']}"


@mcp.tool()
def ask_weak(prompt: str) -> str:
    """直接调用 DeepSeek（弱模型）。"""
    weak, _ = _get_models()
    try:
        result = weak.call(prompt)
        return f"🔀 双路由 · 弱模型 ({weak.model})\n{'─' * 42}\n{result}"
    except Exception as e:
        return _format_error("弱模型调用", weak.model, e, "weak")


@mcp.tool()
def ask_strong(prompt: str, weak_response: str = "") -> str:
    """直接调用 Anthropic（强模型）。可选传入弱模型结果作为审核上下文。"""
    _, strong = _get_models()
    try:
        if weak_response:
            context = f"以下是弱模型的回答，请审核并给出最终答案：\n{weak_response}"
            result = strong.call(prompt, context=context)
            return f"🔀 双路由 · 强模型审核 ({strong.model})\n{'─' * 42}\n{result}"
        result = strong.call(prompt)
        return f"🔀 双路由 · 强模型 ({strong.model})\n{'─' * 42}\n{result}"
    except Exception as e:
        return _format_error("强模型调用", strong.model, e, "strong")


@mcp.tool()
def review(content: str, context: str = "") -> str:
    """用强模型审核任意内容。传入需要审核的文本，返回审核意见。
    使用场景：审核 Claude Code 刚才的回答/代码修改/分析结论，或检查任意文本质量。
    不需要路由判断，直接交给强模型审核。"""
    _, strong = _get_models()
    prompt = f"请审核以下内容，指出错误、遗漏或改进点：\n\n{content}"
    if context:
        prompt = f"背景：{context}\n\n{prompt}"
    try:
        result = strong.call(prompt)
        return f"🔀 双路由 · 审核 ({strong.model})\n{'─' * 42}\n{result}"
    except Exception as e:
        return _format_error("审核", strong.model, e, "strong")


if __name__ == "__main__":
    mcp.run()
