import pytest
from unittest.mock import MagicMock
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router import route_task
from models import DeepSeekModel


def make_mock_weak(response: str) -> DeepSeekModel:
    model = MagicMock(spec=DeepSeekModel)
    model.call.return_value = response
    return model


def test_route_returns_weak_for_simple_task():
    weak = make_mock_weak("weak")
    result = route_task("把hello翻译成中文", weak)
    assert result == "weak"


def test_route_returns_medium_for_moderate_task():
    weak = make_mock_weak("medium")
    result = route_task("总结这篇文章的要点", weak)
    assert result == "medium"


def test_route_returns_strong_for_complex_task():
    weak = make_mock_weak("strong")
    result = route_task("设计一个分布式缓存架构", weak)
    assert result == "strong"


def test_route_strips_whitespace_and_lowercases():
    weak = make_mock_weak("  MEDIUM  ")
    result = route_task("写个快速排序", weak)
    assert result == "medium"


def test_route_matches_chinese_variant():
    weak = make_mock_weak("中等")
    result = route_task("解释一下这个概念", weak)
    assert result == "medium"


def test_route_does_not_match_partial_chinese():
    weak = make_mock_weak("中文翻译很简单")
    result = route_task("翻译这段话", weak)
    assert result == "strong"  # 无法解析，降级到 strong


def test_route_defaults_to_strong_on_unexpected_response():
    weak = make_mock_weak("不知道")
    result = route_task("做点什么", weak)
    assert result == "strong"
