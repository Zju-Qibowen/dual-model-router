import os
import re
import json
import time
import hashlib
import logging
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from models import load_models, DeepSeekModel, AnthropicModel, ImageDescriptionTruncated
from router import route_task
from conversation import ConversationHistory
from log import _route_logger, task_hash as log_task_hash

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

# ── 图片描述缓存 ──────────────────────────────
IMAGE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".claude", "dual-model-router", "image_cache")
IMAGE_CACHE_MAX_FILES = 100
IMAGE_CACHE_TTL_HOURS = int(os.environ.get("IMAGE_CACHE_TTL_HOURS", "168"))  # 默认 7 天


# ── 审核增强：VERDICT 正则 ──────────────────────
VERDICT_RE = re.compile(r'\[VERDICT:\s*(pass|revise|overturn)\]', re.IGNORECASE)

REVIEW_PROMPT_CODE = """请审核以下代码修改。

关注：正确性（逻辑/边界条件）、安全性（注入/越权/泄露）、可维护性（命名/结构/重复）。

<任务>
{task}
</任务>

<弱模型修改>
{weak_response}
</弱模型修改>

请在审核结论最前面用以下标签之一标注你的判定：
[VERDICT: pass]     弱模型修改正确，无需改动
[VERDICT: revise]   有错误或遗漏，已在下方修正
[VERDICT: overturn] 弱模型修改完全不可用，已重写"""

REVIEW_PROMPT_TEXT = """请审核以下回答。

关注：事实准确性、逻辑完整性、表达清晰度。

<任务>
{task}
</任务>

<弱模型回答>
{weak_response}
</弱模型回答>

请在审核结论最前面用以下标签之一标注你的判定：
[VERDICT: pass]     弱模型回答正确，无需修改
[VERDICT: revise]   有错误或遗漏，已在下方修正
[VERDICT: overturn] 弱模型回答完全不可用，已重写"""

REVIEW_PROMPT_REASONING = """请审核以下推理/分析。

关注：推理链完整性、逻辑跳步、假设缺失、结论是否与前提一致。

<任务>
{task}
</任务>

<弱模型分析>
{weak_response}
</弱模型分析>

请在审核结论最前面用以下标签之一标注你的判定：
[VERDICT: pass]     弱模型分析正确，无需修改
[VERDICT: revise]   有错误或遗漏，已在下方修正
[VERDICT: overturn] 弱模型分析完全不可用，已重写"""


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _is_code_response(text: str) -> bool:
    return "```" in text or text.count("\n") > 10


def _classify_task_type(task: str, weak_result: str) -> str:
    """判断审核类型：code / reasoning / text。"""
    if "```" in weak_result:
        return "code"
    reasoning_keywords = ["分析", "为什么", "推理", "原因", "逻辑", "判断", "论证", "得出结论"]
    if any(kw in task for kw in reasoning_keywords):
        return "reasoning"
    return "text"


def _build_review_prompt(task: str, weak_result: str) -> str:
    """按任务类型构建审核 prompt，末尾含 VERDICT 指令。"""
    task_type = _classify_task_type(task, weak_result)
    if task_type == "code":
        return REVIEW_PROMPT_CODE.format(task=task, weak_response=weak_result)
    elif task_type == "reasoning":
        return REVIEW_PROMPT_REASONING.format(task=task, weak_response=weak_result)
    else:
        return REVIEW_PROMPT_TEXT.format(task=task, weak_response=weak_result)


def _parse_verdict(reviewed: str) -> tuple[str, str]:
    """从审核结果中提取 VERDICT 标签。

    返回 (verdict, cleaned_text)。
    verdict: "pass" / "revise" / "overturn" / "unknown"（未匹配时默认 revise）。
    cleaned_text: 去掉 VERDICT 行后的审核文本。
    """
    m = VERDICT_RE.search(reviewed)
    if m:
        verdict = m.group(1).lower()
        # 去掉标签行（可能带前缀标记如 ** 或 #）
        cleaned = VERDICT_RE.sub("", reviewed).strip()
        # 清理残留空行
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return verdict, cleaned

    # 未匹配：标记 unknown，header 中显示警告
    return "unknown", reviewed


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


def _describe_images_cached(
    images: list[dict],
    task_context: str,
    strong: AnthropicModel,
) -> tuple[str, int]:
    """带缓存的图片描述。返回 (描述文本, 缓存命中数)。

    cache key = SHA256(base64_data + task_context 前 8 位)，不同任务侧重点不同。
    LRU 上限 100 文件，超量删最旧。
    """
    if not images:
        return "", 0

    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    hits = 0
    results: list[str] = []

    ctx_fingerprint = hashlib.sha256(task_context.encode()).hexdigest()[:8] if task_context else "notask"

    for img in images:
        img_hash = _image_data_hash(img)
        cache_key = f"{img_hash}_{ctx_fingerprint}"
        cache_path = os.path.join(IMAGE_CACHE_DIR, f"{cache_key}.txt")

        # 检查缓存
        if os.path.isfile(cache_path):
            try:
                mtime = os.path.getmtime(cache_path)
                age_hours = (time.time() - mtime) / 3600
                if IMAGE_CACHE_TTL_HOURS == 0 or age_hours < IMAGE_CACHE_TTL_HOURS:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    # 文件格式：第一行是 timestamp，之后是描述
                    desc_start = content.find("\n")
                    if desc_start > 0:
                        results.append(content[desc_start + 1:])
                        hits += 1
                        # 更新 mtime 实现 LRU
                        try:
                            os.utime(cache_path)
                        except OSError:
                            pass
                        continue
            except Exception:
                # 缓存文件损坏，删除并重新解析
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        # 未命中：调强模型解析
        try:
            desc = strong.describe_images([img], task_context=task_context)
            results.append(desc)
            # 写入缓存
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(f"{time.time()}\n{desc}")
            except OSError:
                pass
        except Exception:
            results.append(f"[图片解析失败]")

    # LRU 清理
    _cleanup_image_cache()

    # 组装结果
    if not results:
        return "", hits

    count_info = f"{len(images)} 张图片" if len(images) > 1 else ""
    cache_info = f"({hits}缓存命中)" if hits > 0 else ""
    header = f"\n\n[图片描述 —— {count_info}{'，' if count_info else ''}由强模型 ({strong.model}) 解析{cache_info}]\n"
    return header + "\n".join(results) + "\n[/图片描述]", hits


def _cleanup_image_cache() -> None:
    """删除超出 LRU 上限的最旧缓存文件。"""
    try:
        files = []
        for name in os.listdir(IMAGE_CACHE_DIR):
            path = os.path.join(IMAGE_CACHE_DIR, name)
            if os.path.isfile(path) and name.endswith(".txt"):
                files.append((os.path.getmtime(path), path))
        if len(files) > IMAGE_CACHE_MAX_FILES:
            files.sort(key=lambda x: x[0])
            for _, path in files[:len(files) - IMAGE_CACHE_MAX_FILES]:
                try:
                    os.remove(path)
                except OSError:
                    pass
    except Exception:
        pass


# ═══════════════════════════════════════════════════
# 路由过程展示
# ═══════════════════════════════════════════════════

def _build_process_header(
    route_decision: str,
    weak_model_name: str,
    strong_model_name: str,
    images_processed: int = 0,
    images_truncated: bool = False,
    cache_hits: int = 0,
    warnings: list[str] | None = None,
    verdict: str | None = None,
) -> str:
    """构建单行增强路由标记。

    格式示例：
      🔀 WEAK · deepseek-chat · 🖼️ 2张(1缓存)
      🔀 MEDIUM · deepseek-chat · 🔍 claude-sonnet-4-6 审核 · ✅ 通过
      🔀 STRONG · claude-sonnet-4-6 · 🖼️ 多模态
    """
    parts = []

    if route_decision == "strong":
        parts.append(f"🔀 STRONG · {strong_model_name}")
    elif route_decision == "medium":
        parts.append(f"🔀 MEDIUM · {weak_model_name}")
        parts.append(f"🔍 {strong_model_name} 审核")
        if verdict == "pass":
            parts.append("✅ 通过")
        elif verdict == "revise":
            parts.append("⚠️ 修正")
        elif verdict == "overturn":
            parts.append("❌ 推翻")
        elif verdict == "unknown":
            parts.append("⚠ 审核未返回结构化判定")
    else:
        parts.append(f"🔀 WEAK · {weak_model_name}")

    if images_processed > 0:
        truncated = "⚠截断 " if images_truncated else ""
        cache = f" {cache_hits}缓存" if cache_hits > 0 else ""
        parts.append(f"🖼️ {truncated}{images_processed}张{cache}")

    if warnings:
        parts.append(f"⚠ {'; '.join(warnings[:2])}")

    return " · ".join(parts)




# ═══════════════════════════════════════════════════
# 核心路由处理
# ═══════════════════════════════════════════════════

def handle_task(
    task: str,
    weak: DeepSeekModel,
    strong: AnthropicModel,
    execution_task: str | None = None,
    history: list[dict] | None = None,
    strong_context: str | None = None,
    pre_context: str = "",
) -> dict:
    """路由 + 执行。

    三档路由：
    - weak: 弱模型执行，不审核
    - medium: 弱模型执行 + 强模型审核
    - strong: 强模型直接执行

    task: 用于路由判断的文本（不含图片描述，避免描述文本干扰路由）
    execution_task: 实际执行的文本（含图片描述）。为 None 时使用 task。
    history: 对话历史（传给弱模型执行调用，不传给路由判断）。
    strong_context: 强模型上下文（含摘要 + 历史 + 当前任务，来自 get_context_for_strong）。
                     strong/medium 路径使用，weak 路径忽略。
    pre_context: 预收集的上下文（如文件内容）。注入到弱模型执行 prompt 的最前面，
                 也会通过 strong_context 注入强模型。不参与路由判断。

    INVARIANT: pre_context 以两种形式存在——原始字符串 (pre_context) 和已拼接进
               strong_context 的段落。strong/medium 路径优先用 strong_context（避免
               重复注入），weak 路径用原始 pre_context 拼接。不要在已有 strong_context
               时再额外 prepend pre_context，否则会出现双重 [预收集的上下文]。
    """
    if execution_task is None:
        execution_task = task

    # 弱模型执行时使用的 prompt（含预收集上下文）
    if pre_context:
        weak_execution_task = f"[预收集的上下文]\n{pre_context}\n\n{execution_task}"
        if len(pre_context) > 16000:
            logger.warning("pre_context 较长 (%d 字符 ≈ %d tokens)，可能导致外部模型上下文超限",
                           len(pre_context), len(pre_context) // 4)
    else:
        weak_execution_task = execution_task

    try:
        decision = route_task(task, weak)
    except Exception as e:
        return {"_error": True, "step": "路由判断", "model": weak.model, "model_type": "weak", "error": e}

    if decision == "strong":
        try:
            prompt = strong_context if strong_context else weak_execution_task
            answer = strong.call(prompt)
        except Exception as e:
            return {"_error": True, "step": "强模型直接执行", "model": strong.model, "model_type": "strong", "error": e}
        return {"routed_to": "strong", "final_answer": answer, "reviewed": False, "weak_result": None, "token_count": None}

    # weak 和 medium 都先走弱模型执行
    try:
        weak_result = weak.call(weak_execution_task, history=history)
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

    # medium: 强模型审核
    review_prompt = _build_review_prompt(execution_task, weak_result)

    # 拼入智能上下文（strong_context 已含预收集上下文，无需重复注入）
    if strong_context:
        review_prompt = f"{strong_context}\n\n{review_prompt}"
    elif pre_context:
        # strong_context 为空时的 fallback
        review_prompt = f"[预收集的上下文]\n{pre_context}\n\n{review_prompt}"

    try:
        reviewed_raw = strong.call(review_prompt)
        verdict, reviewed = _parse_verdict(reviewed_raw)
    except Exception as e:
        return {"_error": True, "step": "强模型审核", "model": strong.model, "model_type": "strong", "error": e, "weak_result": weak_result, "token_count": token_count}

    # overturn: 审核结果末尾保留弱模型原文
    if verdict == "overturn" and weak_result:
        trimmed = weak_result[:200]
        if len(weak_result) > 200:
            trimmed += "..."
        reviewed += f"\n\n---\n🤝 弱模型原结论（已被审核推翻，供参考）：\n{trimmed}"

    # 审核未返回结构化判定时标注
    if verdict == "unknown":
        reviewed = f"{reviewed}"  # 文本不变，header 里会标

    return {
        "routed_to": "medium",
        "final_answer": reviewed,
        "reviewed": True,
        "weak_result": weak_result,
        "token_count": token_count,
        "verdict": verdict,
    }


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
def route_and_answer(task: str, images_json: str = "", context: str = "") -> str:
    """自动路由任务到合适模型并返回结果。支持多轮追问。

    【调用时机 · 先判断再调用】
    外部模型没有任何工具，只能靠你传入的文本作答。调用前先判断：
    ✅ 适合卸载 — 纯文本任务、代码分析/解释、代码生成（接口/上下文已明确）、
       翻译/摘要/润色。核心特征：已知要读哪些文件，读完打包发出去就能搞定。
       注意：能快速定位到 1-2 个文件的搜索算"有界搜集"，仍适合卸载。
    ❌ 不适合卸载 — 需要"边看边定位"的探索性搜索、执行命令/运行测试、
       修改多个有依赖关系的文件、边看结果边调整的调试。
       核心特征：搜索本身就是任务的一部分，外部模型替你找不了。
       判据是搜索"是否有界"，不是"有没有搜索"。

    【与 ask_strong / ask_weak 的关系】
    - route_and_answer 自动按难度路由到强/弱模型，适合"整个任务可卸载、
      你不想自己判断难度"的场景。
    - 已确定需要深度推理 → 用 ask_strong；已确定是简单任务 → 用 ask_weak。
    - 混用模式（先搜集信息再让模型分析）：三个工具都支持，选哪个取决于难度。
    - 专门的"审核"需求 → 用 review 工具，它有完整的 VERDICT 流程。

    参数:
    - task: 用户任务文本
    - images_json: 图片 JSON 数组（可选）
    - context: 预收集的上下文。外部模型无法访问你的文件系统——光给路径没用，必须粘贴
      **文件的相关部分原文**（不需要贴整个文件，贴用户问题涉及的关键代码段即可）。
      路径行只是辅助标注。格式: "[文件: path]\\n相关代码...\\n[/文件]"。
      可串联多个文件。不需要预读的情况: 简单翻译、打招呼、纯知识问答。
      context 不写入对话历史（避免 token 溢出），每次调用需重新传入。

    返回首行是元信息（🔀 WEAK/MEDIUM/STRONG · 模型名），其后是模型原文。

    【展示规范 · 必须遵守】
    工具结果会被折叠，用户默认看不到。你必须在自己的回复正文里完成以下两步：

    1. 先用自己的话说明本次路由情况。包含路由档位和执行模型，例如：
       "本次由 deepseek-chat（弱模型）作答，路由判定为简单问题（WEAK）。"
       - WEAK → 弱模型直接回答（简单任务）
       - MEDIUM → 弱模型回答 + 强模型审核
       - STRONG → 强模型直接回答（复杂任务）
       若首行含 ✅/⚠️/❌，是审核结论（通过/修正/推翻），需一并说明。
       若有图片处理信息（🖼️），也需说明。

    2. 用引用块（>）展示模型原文，不要改写。

    元信息必须出现在你自己的回复正文中，不能只藏在工具结果里。"""
    weak, strong = _get_models()
    t0 = time.perf_counter()

    cleaned_task, all_images, warnings = _collect_images(task, images_json)

    images_truncated = False
    cache_hits = 0
    descriptions = ""
    if all_images:
        descriptions, cache_hits = _describe_images_cached(all_images, cleaned_task, strong)
        if descriptions.startswith("\n\n[⚠ 图片描述不完整"):
            images_truncated = True

    execution_task = f"{cleaned_task}{descriptions}" if descriptions else cleaned_task

    # 记录用户输入（用 execution_task 以便后续追问时模型知道图片内容）
    _history.add_user(execution_task)

    # 准备强模型上下文（含预收集上下文 + 摘要 + 最近对话 + 任务）
    strong_context = _history.get_context_for_strong(task=execution_task, weak_model=weak, pre_context=context)

    result = handle_task(
        task=cleaned_task,
        execution_task=execution_task,
        weak=weak,
        strong=strong,
        history=_history.get_messages()[:-1],  # 不含刚加的当前 user message（由 model.call 自行追加）
        strong_context=strong_context,
        pre_context=context,
    )
    route_decision = result.get("routed_to", "strong")

    # 成功且回答非空时记录 assistant 回答，标注产出模型
    if not result.get("_error") and result.get("final_answer"):
        source = "strong" if route_decision in ("strong", "medium") else "weak"
        _history.add_assistant(result["final_answer"], source=source)

    header = _build_process_header(
        route_decision, weak.model, strong.model,
        images_processed=len(all_images),
        images_truncated=images_truncated,
        cache_hits=cache_hits,
        warnings=warnings if warnings else None,
        verdict=result.get("verdict"),
    )

    if result.get("_error"):
        # 失败时撤回刚加的 user message
        if _history.turns and _history.turns[-1]["role"] == "user":
            _history.turns.pop()
        _route_logger.log(
            task_hash=log_task_hash(task),
            task_len=len(cleaned_task),
            has_images=1 if all_images else 0,
            image_count=len(all_images),
            route_decision=route_decision,
            route_method="deepseek",
            exec_model=result.get("model", ""),
            input_tokens=0,
            output_tokens=0,
            success=0,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        prefix = ""
        if result.get("weak_result"):
            prefix = f"弱模型已执行但审核失败，原始回答: {result['weak_result'][:100]}\n"
        return header + "\n" + prefix + _format_error(result["step"], result["model"], result["error"], result["model_type"])

    exec_model = weak.model if route_decision == "weak" else strong.model
    if route_decision == "medium":
        exec_model = f"{weak.model} + {strong.model}"
    _route_logger.log(
        task_hash=log_task_hash(task),
        task_len=len(cleaned_task),
        has_images=1 if all_images else 0,
        image_count=len(all_images),
        route_decision=route_decision,
        route_method="deepseek",
        exec_model=exec_model,
        input_tokens=result.get("token_count", 0) or 0,
        output_tokens=result.get("token_count", 0) or 0,
        success=1,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
    return f"{header}\n{result['final_answer']}"


@mcp.tool()
def ask_weak(prompt: str, images_json: str = "", context: str = "") -> str:
    """直接调用弱模型（低成本）作答，不经过路由判定。

    【调用时机】弱模型成本低，适合简单翻译、格式转换、概念解释等低难度任务。
    适用边界同 route_and_answer（有界、可打包的一次性任务）。
    也支持混用模式：先用工具链读出内容，通过 context 传入，让弱模型做简单转换/提取。
    需要深度推理时改用 ask_strong。

    参数:
    - prompt: 用户提示
    - images_json: 图片 JSON 数组（可选）
    - context: 预收集的上下文。外部模型无法访问你的文件系统——光给路径没用，必须粘贴
      **文件的相关部分原文**（不需要贴整个文件，贴用户问题涉及的关键代码段即可）。
      路径行只是辅助标注。格式: "[文件: path]\\n相关代码...\\n[/文件]"。
      可串联多个文件。context 不写入对话历史，每次调用需重新传入。

    返回首行是元信息（📡 模型名），其后是模型原文。

    【展示规范 · 必须遵守】
    工具结果会被折叠。你必须在自己的回复正文里：

    1. 先用一句话说明："本次直接使用了 <模型名>（弱模型），未经过自动路由。"
       若有图片，补充"并对 N 张图片做了视觉理解。"

    2. 用引用块（>）展示模型原文。

    元信息必须出现在你自己的回复正文中，不能只藏在工具结果里。"""
    weak, strong = _get_models()

    prompt, all_images, warnings = _collect_images(prompt, images_json)

    try:
        if all_images:
            if not prompt.strip():
                prompt = DEFAULT_IMAGE_PROMPT
            descriptions, cache_hits = _describe_images_cached(all_images, prompt, strong)
            prompt = f"{prompt}{descriptions}"
            cache = f"({cache_hits}缓存)" if cache_hits > 0 else ""
            header = f"📡 {weak.model} · 🖼️ {len(all_images)}张{cache}"
        else:
            header = f"📡 {weak.model}"

        # 构建完整 prompt：context（如有）+ 原始 prompt
        full_prompt = f"[预收集的上下文]\n{context}\n\n{prompt}" if context else prompt

        _history.add_user(prompt)  # history 记录不含 context
        result = weak.call(full_prompt, history=_history.get_messages()[:-1])
        _history.add_assistant(result, source="weak")
        return f"{header}\n{result}"
    except Exception as e:
        if _history.turns and _history.turns[-1]["role"] == "user":
            _history.turns.pop()
        return _format_error("弱模型调用", weak.model, e, "weak")


@mcp.tool()
def ask_strong(prompt: str, weak_response: str = "", images_json: str = "", context: str = "") -> str:
    """直接调用强模型（高能力）作答，不经过路由判定。支持多模态图片。

    【调用时机】强模型适合复杂推理、架构分析等高难度任务。两种用法：
    1. 混用模式（推荐）：你用工具链搜集信息（读文件、搜索、理解上下文），
       通过 context 参数传入，让强模型对搜集结果做深度分析。拿到结果后
       你继续改代码、跑测试。典型链路：
         搜寻/读文件 → ask_strong(分析) → 改代码 → review(审核) → 提交
    2. 独立模式：任务本身就是纯分析/推理/生成，不需要工具链参与。
    （混用是通用模式，ask_weak、route_and_answer 同样支持先搜集再调用；
     选 ask_strong 是因为你判断这个分析需要强模型。）
    审核类需求请用 review 工具，它有完善的 VERDICT 审核流程。

    参数:
    - prompt: 用户提示
    - weak_response: 弱模型回答（用于审核场景，可选）
    - images_json: 图片 JSON 数组（可选）
    - context: 预收集的上下文。外部模型无法访问你的文件系统——光给路径没用，必须粘贴
      **文件的相关部分原文**（不需要贴整个文件，贴用户问题涉及的关键代码段即可）。
      路径行只是辅助标注。格式: "[文件: path]\\n相关代码...\\n[/文件]"。
      可串联多个文件。context 不写入对话历史，每次调用需重新传入。

    返回首行是元信息（📡 模型名），其后是模型原文。

    【展示规范 · 必须遵守】
    工具结果会被折叠。你必须在自己的回复正文里：

    1. 先用一句话说明："本次直接使用了 <模型名>（强模型），未经过自动路由。"
       若传入 weak_response 参数，说明"并对弱模型回答进行了审核"。
       若有图片，补充"并对 N 张图片做了视觉理解"。

    2. 用引用块（>）展示模型原文。

    元信息必须出现在你自己的回复正文中，不能只藏在工具结果里。"""
    weak, strong = _get_models()

    prompt, all_images, warnings = _collect_images(prompt, images_json)

    try:
        if all_images and not strong.supports_vision:
            descriptions, _ = _describe_images_cached(all_images, prompt, strong)
            if not prompt.strip():
                prompt = DEFAULT_IMAGE_PROMPT
            prompt = f"{prompt}{descriptions}"

        _history.add_user(prompt)  # history 记录不含 context

        # 装配智能上下文（预收集上下文 + 摘要 + 历史 + 任务）
        strong_context = _history.get_context_for_strong(task=prompt, weak_model=weak, pre_context=context)

        if weak_response:
            full_prompt = f"以下是弱模型的回答，请审核并给出最终答案：\n{weak_response}\n\n{strong_context}"
            header = f"📡 {strong.model} · 🔍 审核"
        else:
            full_prompt = strong_context
            header = f"📡 {strong.model}"

        if all_images and strong.supports_vision:
            result = strong.call_with_images(full_prompt, all_images)
        else:
            result = strong.call(full_prompt)

        _history.add_assistant(result, source="strong")
        return f"{header}\n{result}"
    except Exception as e:
        if _history.turns and _history.turns[-1]["role"] == "user":
            _history.turns.pop()
        return _format_error("强模型调用", strong.model, e, "strong")


@mcp.tool()
def review(content: str, context: str = "") -> str:
    """用强模型审核任意内容，返回审核意见。不计入对话历史。

    返回首行是元信息（📡 审核模型名 · 🔍 审核），其后是审核正文。

    【展示规范 · 必须遵守】
    工具结果会被折叠。你必须在自己的回复正文里：

    1. 先用一句话说明："本次由 <模型名>（强模型）进行了独立审核。"
       若审核正文含 VERDICT 标签（pass/revise/overturn），需说明结论：
       - 通过 → 内容正确，可直接采用
       - 修正 → 有补充或纠错
       - 推翻 → 原内容不可用，已重写

    2. 用引用块（>）展示审核结果正文。

    元信息和审核结论必须出现在你自己的回复正文中，不能只藏在工具结果里。"""
    weak, strong = _get_models()
    base = (
        "请审核以下内容，指出错误、遗漏或改进点。\n\n"
        + f"{content}\n\n"
        + "请在审核结论最前面用以下标签之一标注你的判定：\n"
        + "[VERDICT: pass]   内容正确，无需修改\n"
        + "[VERDICT: revise] 有错误或遗漏，已在下方修正\n"
        + "[VERDICT: overturn] 内容完全不可用，已重写"
    )
    if context:
        base = f"背景：{context}\n\n{base}"
    # 装配智能上下文（摘要 + 历史）
    strong_context = _history.get_context_for_strong(weak_model=weak)
    if strong_context:
        base = f"{strong_context}\n\n{base}"
    try:
        result = strong.call(base)
        return f"📡 {strong.model} · 🔍 审核\n{result}"
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


@mcp.tool()
def add_note(note: str) -> str:
    """向对话历史注入一条工具操作记录，不参与对话轮次，不触发裁剪。

    当你在调用外部模型之前执行了工具操作（读文件、搜索、运行命令），
    用此工具记录摘要，让外部模型了解"你已经做了什么"。

    记录会出现在 get_context_for_strong 的 [工具操作记录] 段落中，
    也会被纳入渐进式摘要。帮助外部模型在混用模式下保持上下文连贯。

    示例:
    - "已读取 server.py (857行) 的全部内容"
    - "git diff 显示 handlers/auth.py 和 models/user.py 被修改"
    - "运行 pytest 发现 test_login 和 test_logout 失败"
    - "搜索 'DeprecatedAPI' 找到 5 处引用"
    """
    _history.add_tool_note(note)
    short = note[:80] + ("..." if len(note) > 80 else "")
    return f"已记录工具操作: {short}"


if __name__ == "__main__":
    mcp.run()
