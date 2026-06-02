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


def test_handle_task_strong_path_passes_history():
    from server import handle_task
    weak = MagicMock()
    strong = MagicMock()
    weak.call.return_value = "strong"
    strong.call.return_value = "complex answer"
    history = [{"role": "user", "content": "prev"}]

    handle_task("complex task", weak, strong, history=history)

    strong.call.assert_called_once()
    assert strong.call.call_args.kwargs.get("history") == history


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
