import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import DeepSeekModel, AnthropicModel


def test_deepseek_call_returns_string():
    model = DeepSeekModel(api_key="fake-key", base_url="https://api.deepseek.com", model="deepseek-chat")
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "hello from deepseek"
    with patch.object(model.client.chat.completions, "create", return_value=mock_response):
        result = model.call("say hello")
    assert result == "hello from deepseek"


def test_anthropic_call_returns_string():
    model = AnthropicModel(api_key="fake-key", model="claude-haiku-4-5-20251001", base_url="https://fake.proxy.com")
    mock_response = MagicMock()
    mock_response.content[0].text = "hello from anthropic"
    with patch.object(model.client.messages, "create", return_value=mock_response):
        result = model.call("say hello")
    assert result == "hello from anthropic"


def test_anthropic_call_with_context():
    model = AnthropicModel(api_key="fake-key", model="claude-haiku-4-5-20251001", base_url="https://fake.proxy.com")
    mock_response = MagicMock()
    mock_response.content[0].text = "reviewed"
    with patch.object(model.client.messages, "create", return_value=mock_response) as mock_create:
        result = model.call("review this", context="original content")
    assert result == "reviewed"
    call_args = mock_create.call_args
    assert "original content" in call_args.kwargs["messages"][0]["content"]
