import pytest
from unittest.mock import MagicMock, patch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_mocks(weak_response="weak", weak_result="deepseek answer", strong_result="anthropic answer"):
    weak = MagicMock()
    strong = MagicMock()
    weak.call.side_effect = [weak_response, weak_result]
    strong.call.return_value = strong_result
    return weak, strong


def test_handle_task_routes_weak_no_review():
    from server import handle_task
    weak, strong = make_mocks(weak_response="weak", weak_result="simple answer")

    result = handle_task("把hello翻译成中文", weak, strong)

    assert result["routed_to"] == "weak"
    assert result["final_answer"] == "simple answer"
    assert result["reviewed"] is False
    strong.call.assert_not_called()


def test_handle_task_passes_history_to_execution():
    from server import handle_task
    weak, strong = make_mocks(weak_response="weak", weak_result="answer with context")
    history = [{"role": "user", "content": "prev q"}, {"role": "assistant", "content": "prev a"}]

    result = handle_task("follow up question", weak, strong, history=history)

    # weak.call is called twice: first for routing (no history), then for execution (with history)
    exec_call = weak.call.call_args_list[1]
    assert exec_call.kwargs.get("history") == history


def test_handle_task_strong_path_uses_strong_context():
    from server import handle_task
    weak = MagicMock()
    strong = MagicMock()
    weak.call.return_value = "strong"
    strong.call.return_value = "complex answer"

    handle_task("complex task", weak, strong, strong_context="summary + history + task")

    strong.call.assert_called_once()
    assert strong.call.call_args.args[0] == "summary + history + task"


def test_handle_task_strong_path_falls_back_to_task_without_context():
    from server import handle_task
    weak = MagicMock()
    strong = MagicMock()
    weak.call.return_value = "strong"
    strong.call.return_value = "complex answer"

    handle_task("complex task", weak, strong)

    strong.call.assert_called_once()
    assert strong.call.call_args.args[0] == "complex task"


def test_handle_task_routes_medium_with_review():
    from server import handle_task
    weak, strong = make_mocks(weak_response="medium", weak_result="summary here")
    strong.call.return_value = "reviewed summary"

    result = handle_task("总结这篇文章", weak, strong)

    assert result["routed_to"] == "medium"
    assert result["final_answer"] == "reviewed summary"
    assert result["reviewed"] is True


def test_handle_task_routes_to_strong_directly():
    from server import handle_task
    weak = MagicMock()
    strong = MagicMock()
    weak.call.return_value = "strong"
    strong.call.return_value = "complex answer"

    result = handle_task("设计分布式架构", weak, strong)

    assert result["routed_to"] == "strong"
    assert result["final_answer"] == "complex answer"
    assert result["reviewed"] is False


def test_set_weak_model_changes_model():
    import server
    from unittest.mock import MagicMock
    mock_weak = MagicMock()
    mock_weak.model = "deepseek-chat"
    mock_strong = MagicMock()
    server._weak = mock_weak
    server._strong = mock_strong

    result = server.set_weak_model("deepseek-reasoner")

    assert mock_weak.model == "deepseek-reasoner"
    assert "deepseek-reasoner" in result


def test_set_strong_model_changes_model():
    import server
    from unittest.mock import MagicMock
    mock_weak = MagicMock()
    mock_strong = MagicMock()
    mock_strong.model = "claude-haiku-4-5-20251001"
    server._weak = mock_weak
    server._strong = mock_strong

    result = server.set_strong_model("claude-sonnet-4-5")

    assert mock_strong.model == "claude-sonnet-4-5"
    assert "claude-sonnet-4-5" in result


def test_list_models_returns_current_models():
    import server
    from unittest.mock import MagicMock
    mock_weak = MagicMock()
    mock_weak.model = "deepseek-chat"
    mock_strong = MagicMock()
    mock_strong.model = "claude-haiku-4-5-20251001"
    server._weak = mock_weak
    server._strong = mock_strong

    result = server.list_models()

    assert "deepseek-chat" in result
    assert "claude-haiku-4-5-20251001" in result


def test_clear_context_resets_history():
    import server
    server._history.add_user("test question")
    server._history.add_assistant("test answer")
    assert server._history.turn_count == 1

    result = server.clear_context()

    assert "1 轮" in result
    assert server._history.turn_count == 0


def test_get_context_status_shows_info():
    import server
    server._history.clear()
    server._history.add_user("hello")
    server._history.add_assistant("hi")

    result = server.get_context_status()

    assert "1/" in result
    assert "轮数" in result


# ── pre_context 测试 ──────────────────────────────


def test_handle_task_weak_with_pre_context():
    """weak 路径下 weak.call 应收到含 pre_context 的 prompt。"""
    from server import handle_task
    weak, strong = make_mocks(weak_response="weak", weak_result="simple answer")

    result = handle_task("simple task", weak, strong, pre_context="[文件: foo.py]\ncode")

    assert result["routed_to"] == "weak"
    # weak.call 被调用两次：第一次路由判断，第二次执行
    exec_call = weak.call.call_args_list[1]
    prompt_text = exec_call.args[0]
    assert "[预收集的上下文]" in prompt_text
    assert "[文件: foo.py]" in prompt_text
    assert "simple task" in prompt_text


def test_handle_task_strong_with_pre_context():
    """strong 路径且无 strong_context 时，fallback 应含 pre_context。"""
    from server import handle_task
    weak = MagicMock()
    strong = MagicMock()
    weak.call.return_value = "strong"
    strong.call.return_value = "complex answer"

    result = handle_task("complex task", weak, strong, pre_context="[文件: bar.py]\ncode")

    assert result["routed_to"] == "strong"
    strong.call.assert_called_once()
    prompt = strong.call.call_args.args[0]
    assert "[预收集的上下文]" in prompt
    assert "[文件: bar.py]" in prompt


def test_handle_task_medium_with_pre_context():
    """medium 路径下 weak 执行和 strong 审核都应含 pre_context。"""
    from server import handle_task
    weak, strong = make_mocks(weak_response="medium", weak_result="summary")
    strong.call.return_value = "[VERDICT: pass] reviewed"

    result = handle_task("analyze this", weak, strong, pre_context="[文件: baz.py]\ncode")

    assert result["routed_to"] == "medium"
    # weak 执行调用（第 2 次 call）
    weak_exec = weak.call.call_args_list[1]
    weak_prompt = weak_exec.args[0]
    assert "[预收集的上下文]" in weak_prompt
    assert "[文件: baz.py]" in weak_prompt

    # strong 审核调用（fallback 路径：strong_context 未传，由 elif 注入）
    strong_prompt = strong.call.call_args.args[0]
    assert "[预收集的上下文]" in strong_prompt


def test_handle_task_weak_without_pre_context():
    """pre_context 为空时不应添加 [预收集的上下文]。"""
    from server import handle_task
    weak, strong = make_mocks(weak_response="weak", weak_result="simple answer")

    result = handle_task("simple task", weak, strong)  # no pre_context

    assert result["routed_to"] == "weak"
    exec_call = weak.call.call_args_list[1]
    prompt_text = exec_call.args[0]
    assert "[预收集的上下文]" not in prompt_text


def test_handle_task_medium_no_duplicate_pre_context():
    """strong_context 已含 pre_context 时，review_prompt 不应重复注入。"""
    from server import handle_task
    weak, strong = make_mocks(weak_response="medium", weak_result="summary")
    strong.call.return_value = "[VERDICT: pass] reviewed"

    # strong_context 已内含一次 [预收集的上下文]（模拟 get_context_for_strong 输出）
    strong_ctx = (
        "[预收集的上下文]\n[文件: baz.py]\ncode\n\n"
        "[当前任务]\nanalyze this"
    )

    result = handle_task(
        "analyze this", weak, strong,
        strong_context=strong_ctx,
        pre_context="[文件: baz.py]\ncode",
    )
    assert result["routed_to"] == "medium"

    strong_prompt = strong.call.call_args.args[0]
    # 核心断言：只出现一次，不应重复
    assert strong_prompt.count("[预收集的上下文]") == 1


def test_handle_task_strong_without_pre_context():
    """strong 路径无 pre_context 时不应添加该段落。"""
    from server import handle_task
    weak = MagicMock()
    strong = MagicMock()
    weak.call.return_value = "strong"
    strong.call.return_value = "complex answer"

    result = handle_task("complex task", weak, strong)

    assert result["routed_to"] == "strong"
    prompt = strong.call.call_args.args[0]
    assert "[预收集的上下文]" not in prompt


# ── system prompt & self-intro 检测 ──────────────────


def test_check_self_intro_detects_claude_intro():
    """检测经典的 Claude 自我介绍。"""
    from server import _check_self_intro
    result = _check_self_intro("我是 Claude，由 Anthropic 开发的 AI 助手。")
    assert result is not None
    assert "⚠️" in result


def test_check_self_intro_detects_model_id_intro():
    """检测带模型名的元信息式介绍（中转站常见模式）。"""
    from server import _check_self_intro
    result = _check_self_intro("我是 Claude，由 Anthropic 开发的 AI 助手，当前请求的模型是 claude-opus-4-8。")
    assert result is not None


def test_check_self_intro_passes_normal_response():
    """正常任务执行结果不应触发检测。"""
    from server import _check_self_intro
    result = _check_self_intro("根据分析，NPC 系统方案存在以下问题：1. ...")
    assert result is None


def test_check_self_intro_ignores_long_response():
    """长度超过 300 字符的响应即使含'我是'也不触发（可能是任务内容）。"""
    from server import _check_self_intro
    long_text = "我是谁？这是一个哲学问题。" + "x" * 300
    assert _check_self_intro(long_text) is None


def test_handle_task_strong_passes_system_prompt():
    """strong 路径应将 DEFAULT_SYSTEM_PROMPT 作为 system 参数传入。"""
    from server import handle_task, DEFAULT_SYSTEM_PROMPT
    weak = MagicMock()
    strong = MagicMock()
    weak.call.return_value = "strong"
    strong.call.return_value = "complex answer"

    handle_task("complex task", weak, strong)

    strong.call.assert_called_once()
    assert strong.call.call_args.kwargs.get("system") == DEFAULT_SYSTEM_PROMPT


def test_handle_task_strong_adds_self_intro_warning():
    """当 strong 模型返回自我介绍时，应在结果前追加警告。"""
    from server import handle_task
    weak = MagicMock()
    strong = MagicMock()
    weak.call.return_value = "strong"
    strong.call.return_value = "我是 Claude，由 Anthropic 开发的 AI 助手。"

    result = handle_task("complex task", weak, strong)

    assert result["routed_to"] == "strong"
    assert "⚠️" in result["final_answer"]
    assert "自我介绍" in result["final_answer"]


def test_handle_task_medium_review_passes_system_prompt():
    """medium 审核路径也应传入 system prompt。"""
    from server import handle_task, DEFAULT_SYSTEM_PROMPT
    weak, strong = make_mocks(weak_response="medium", weak_result="summary")
    strong.call.return_value = "[VERDICT: pass] reviewed"

    handle_task("analyze this", weak, strong)

    assert strong.call.call_args.kwargs.get("system") == DEFAULT_SYSTEM_PROMPT
