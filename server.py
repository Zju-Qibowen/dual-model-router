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
    return (
        f"❌ 调用失败\n"
        f"  步骤: {step}\n"
        f"  模型: {model_name} ({'弱模型/DeepSeek' if model_type == 'weak' else '强模型/Anthropic'})\n"
        f"  错误: {error_type}\n"
        f"---\n"
        f"建议: set_weak_model / set_strong_model 切换模型, list_available_models 查看可用模型, "
        f"或用 ask_weak / ask_strong 逐个调试"
    )


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
def list_available_models() -> str:
    """列出所有可选的弱模型和强模型（从 .env 的 WEAK_MODELS / STRONG_MODELS 读取）。"""
    weak_models = os.environ.get("WEAK_MODELS", "").split(",")
    strong_models = os.environ.get("STRONG_MODELS", "").split(",")
    weak, strong = _get_models()
    weak_list = "\n".join(
        f"  {'* ' if m.strip() == weak.model else '  '}{m.strip()}"
        for m in weak_models if m.strip()
    )
    strong_list = "\n".join(
        f"  {'* ' if m.strip() == strong.model else '  '}{m.strip()}"
        for m in strong_models if m.strip()
    )
    return f"弱模型（* 为当前）:\n{weak_list}\n\n强模型（* 为当前）:\n{strong_list}"


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

    if result.get("_error"):
        # 如果审核失败但弱模型结果已拿到，一并展示
        prefix = ""
        if result.get("weak_result"):
            prefix = f"⚠️ 弱模型已执行但审核失败\n弱模型原始回答: {result['weak_result']}\n\n"
        return prefix + _format_error(result["step"], result["model"], result["error"], result["model_type"])

    lines = [
        f"路由到: {result['routed_to']} ({weak.model if result['routed_to'] == 'weak' else strong.model})",
        f"已审核: {result['reviewed']}",
    ]
    if result.get("token_count") is not None:
        lines.append(f"弱模型 token 估算: {result['token_count']}")
    if result.get("weak_result") and result["reviewed"]:
        lines.append(f"弱模型原始回答: {result['weak_result']}")
    if "note" in result:
        lines.append(result["note"])
    log = "\n".join(f"  {l}" for l in lines)
    return result["final_answer"] + f"\n\n---\n{log}"


@mcp.tool()
def ask_weak(prompt: str) -> str:
    """直接调用 DeepSeek（弱模型）。"""
    weak, _ = _get_models()
    try:
        return weak.call(prompt)
    except Exception as e:
        return _format_error("弱模型调用", weak.model, e, "weak")


@mcp.tool()
def ask_strong(prompt: str, weak_response: str = "") -> str:
    """直接调用 Anthropic（强模型）。可选传入弱模型结果作为审核上下文。"""
    _, strong = _get_models()
    try:
        if weak_response:
            context = f"以下是弱模型的回答，请审核并给出最终答案：\n{weak_response}"
            return strong.call(prompt, context=context)
        return strong.call(prompt)
    except Exception as e:
        return _format_error("强模型调用", strong.model, e, "strong")


if __name__ == "__main__":
    mcp.run()
