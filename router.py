from models import DeepSeekModel

ROUTE_PROMPT = """判断以下任务的复杂度，只回答 "weak" 或 "strong"，不要解释。
weak = 翻译、摘要、格式转换、简单问答、单函数代码
strong = 架构设计、多步推理、代码审查、安全分析、复杂重构

<user_input>
{task}
</user_input>"""


def route_task(task: str, weak_model: DeepSeekModel) -> str:
    prompt = ROUTE_PROMPT.format(task=task)
    response = weak_model.call(prompt)
    normalized = response.strip().lower()
    if normalized in ("weak", "strong"):
        return normalized
    return "strong"
