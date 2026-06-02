import logging
from models import DeepSeekModel

logger = logging.getLogger("dual-model-router")

ROUTE_PROMPT = """判断以下任务的复杂度，只回答 "weak"、"medium" 或 "strong"，不要解释。

weak   = 简单翻译、格式转换、单步问答、查找替换、打招呼
medium = 多段翻译、摘要、单函数代码、解释概念、对比分析、改写润色
strong = 架构设计、多步推理、代码审查、安全分析、复杂重构、多文件改动

示例：
- "把hello翻译成中文" → weak
- "总结这篇文章的要点" → medium
- "写一个快速排序函数" → medium
- "设计一个分布式消息队列的架构" → strong
- "审查这段代码的安全漏洞" → strong

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

    if "weak" in normalized or "弱" in normalized:
        return "weak"
    if "medium" in normalized or "中等" in normalized:
        return "medium"
    if "strong" in normalized or "强" in normalized:
        return "strong"

    logger.warning("路由判断无法解析，降级到 strong。原始响应: %s", response[:100])
    return "strong"
