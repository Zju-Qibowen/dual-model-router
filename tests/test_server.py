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


def test_handle_task_routes_to_weak_and_reviews():
    from server import handle_task
    weak, strong = make_mocks(weak_response="weak", weak_result="simple answer")
    strong.call.return_value = "reviewed answer"

    result = handle_task("翻译这段话", weak, strong, review_token_limit=2000)

    assert result["routed_to"] == "weak"
    assert result["final_answer"] == "reviewed answer"
    assert result["reviewed"] is True


def test_handle_task_routes_to_strong_directly():
    from server import handle_task
    weak = MagicMock()
    strong = MagicMock()
    weak.call.return_value = "strong"
    strong.call.return_value = "complex answer"

    result = handle_task("设计分布式架构", weak, strong, review_token_limit=2000)

    assert result["routed_to"] == "strong"
    assert result["final_answer"] == "complex answer"
    assert result["reviewed"] is False


def test_handle_task_skips_review_when_over_token_limit():
    from server import handle_task
    weak, strong = make_mocks(weak_response="weak", weak_result="x" * 10000)

    result = handle_task("写代码", weak, strong, review_token_limit=100)

    assert result["routed_to"] == "weak"
    assert result["reviewed"] is False
    assert "超过" in result.get("note", "") or result["final_answer"] == "x" * 10000


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
