import logging
from models import DeepSeekModel

logger = logging.getLogger("dual-model-router")

ROUTE_PROMPT = """判断以下任务的复杂度，只回答 "weak" 或 "strong"，不要解释。
weak = 翻译、摘要、格式转换、简单问答、单函数代码
strong = 架构设计、多步推理、代码审查、安全分析、复杂重构

<user_input>
{task}
</user_input>"""


def route_task(task: str, weak_model: DeepSeekModel) -> str:
    prompt = ROUTE_PROMPT.format(task=task)

    try:
        response = weak_model.call(prompt)
    except Exception:
        logger.exception("路由判断失败，降级到 strong")
        return "strong"

    normalized = response.strip().lower()

    # 宽松匹配：兼容 "weak."、"**weak**"、"弱" 等变体
    if "weak" in normalized or "弱" in normalized:
        return "weak"
    if "strong" in normalized or "强" in normalized:
        return "strong"

    logger.warning("路由判断无法解析，降级到 strong。原始响应: %s", response[:100])
    return "strong"
