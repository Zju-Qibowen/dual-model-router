import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from models import load_models, DeepSeekModel, AnthropicModel
from router import route_task

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

mcp = FastMCP("dual-model-router")

REVIEW_TOKEN_LIMIT = int(os.environ.get("REVIEW_TOKEN_LIMIT", "2000"))

REVIEW_PROMPT_CODE = """原始任务：{task}

DeepSeek 的修改（diff 格式）：
{weak_response}

请审核以上修改：指出错误或遗漏，给出最终建议。"""

REVIEW_PROMPT_TEXT = """原始任务：{task}

DeepSeek 的回答：
{weak_response}

请审核以上回答：指出错误或遗漏，并给出最终完整答案。"""


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _is_code_response(text: str) -> bool:
    return "```" in text or text.count("\n") > 10


def handle_task(
    task: str,
    weak: DeepSeekModel,
    strong: AnthropicModel,
    review_token_limit: int = REVIEW_TOKEN_LIMIT,
) -> dict:
    decision = route_task(task, weak)

    if decision == "strong":
        answer = strong.call(task)
        return {"routed_to": "strong", "final_answer": answer, "reviewed": False}

    weak_result = weak.call(task)
    token_count = _estimate_tokens(weak_result)

    if token_count > review_token_limit:
        return {
            "routed_to": "weak",
            "final_answer": weak_result,
            "reviewed": False,
            "note": f"输出超过 {review_token_limit} token 限制（约 {token_count} tokens），已跳过审核",
        }

    if _is_code_response(weak_result):
        review_prompt = REVIEW_PROMPT_CODE.format(task=task, weak_response=weak_result)
    else:
        review_prompt = REVIEW_PROMPT_TEXT.format(task=task, weak_response=weak_result)

    reviewed = strong.call(review_prompt)
    return {"routed_to": "weak", "final_answer": reviewed, "reviewed": True}


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
    note = f"\n\n[路由到: {result['routed_to']} | 已审核: {result['reviewed']}]"
    if "note" in result:
        note += f"\n[{result['note']}]"
    return result["final_answer"] + note


@mcp.tool()
def ask_weak(prompt: str) -> str:
    """直接调用 DeepSeek（弱模型）。"""
    weak, _ = _get_models()
    return weak.call(prompt)


@mcp.tool()
def ask_strong(prompt: str, weak_response: str = "") -> str:
    """直接调用 Anthropic（强模型）。可选传入弱模型结果作为审核上下文。"""
    _, strong = _get_models()
    if weak_response:
        context = f"以下是弱模型的回答，请审核并给出最终答案：\n{weak_response}"
        return strong.call(prompt, context=context)
    return strong.call(prompt)


if __name__ == "__main__":
    mcp.run()
