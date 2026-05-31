"""测试图片处理管线：提取、校验、描述、路由集成。"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    AnthropicModel,
    DeepSeekModel,
    ImageDescriptionTruncated,
    SUPPORTED_IMAGE_TYPES,
    MAX_IMAGES_PER_REQUEST,
)


# ═══════════════════════════════════════════════════
# server.py 图片提取函数
# ═══════════════════════════════════════════════════

class TestExtractImagesFromTask:
    """从 task 文本提取 data:image URL。"""

    def _extract(self, task):
        from server import _extract_images_from_task
        return _extract_images_from_task(task)

    def test_no_images(self):
        task = "你好，请翻译这段话"
        cleaned, images = self._extract(task)
        assert cleaned == task
        assert images == []

    def test_single_png_image(self):
        task = "描述这张图 data:image/png;base64,iVBORw0KGgo="
        cleaned, images = self._extract(task)
        assert cleaned == "描述这张图 [图片]"
        assert len(images) == 1
        assert images[0]["media_type"] == "image/png"
        assert images[0]["data"] == "iVBORw0KGgo="

    def test_multiple_images(self):
        task = (
            "图1 data:image/png;base64,AAA= 图2 data:image/jpeg;base64,BBB="
        )
        cleaned, images = self._extract(task)
        assert cleaned == "图1 [图片] 图2 [图片]"
        assert len(images) == 2
        assert images[0]["media_type"] == "image/png"
        assert images[1]["media_type"] == "image/jpeg"

    def test_webp_and_gif(self):
        task = "data:image/webp;base64,UklGRg== 和 data:image/gif;base64,R0lGODlh"
        cleaned, images = self._extract(task)
        assert len(images) == 2
        assert images[0]["media_type"] == "image/webp"
        assert images[1]["media_type"] == "image/gif"

    def test_image_with_dots_in_mime(self):
        """image/svg+xml 等带特殊字符的 MIME。"""
        task = "data:image/svg+xml;base64,PHN2Zw=="
        cleaned, images = self._extract(task)
        assert len(images) == 1
        assert images[0]["media_type"] == "image/svg+xml"


class TestParseImagesParam:
    """解析 images_json 参数。"""

    def _parse(self, json_str):
        from server import _parse_images_param
        return _parse_images_param(json_str)

    def test_empty_and_none(self):
        assert self._parse("") == []
        assert self._parse(None) == []

    def test_valid_json_array(self):
        result = self._parse(
            '[{"data": "AAA=", "media_type": "image/png"},'
            ' {"data": "BBB=", "media_type": "image/jpeg"}]'
        )
        assert len(result) == 2
        assert result[0]["data"] == "AAA="
        assert result[0]["media_type"] == "image/png"
        assert result[1]["data"] == "BBB="

    def test_missing_media_type_defaults_to_png(self):
        result = self._parse('[{"data": "AAA="}]')
        assert len(result) == 1
        assert result[0]["media_type"] == "image/png"

    def test_skips_non_dict_and_empty_data(self):
        result = self._parse(
            '[{"data": "AAA="}, "string", {"data": ""}, {"no_data": 1}]'
        )
        assert len(result) == 1

    def test_invalid_json_returns_empty(self):
        assert self._parse("not json") == []
        assert self._parse('"not array"') == []


class TestDeduplicateImages:
    """图片去重。"""

    def _dedup(self, images):
        from server import _deduplicate_images
        return _deduplicate_images(images)

    def test_no_duplicates(self):
        images = [
            {"data": "AAA=", "media_type": "image/png"},
            {"data": "BBB=", "media_type": "image/jpeg"},
        ]
        assert len(self._dedup(images)) == 2

    def test_removes_duplicates_keeps_first(self):
        images = [
            {"data": "AAA=", "media_type": "image/png"},
            {"data": "BBB=", "media_type": "image/jpeg"},
            {"data": "AAA=", "media_type": "image/png"},  # 重复
        ]
        result = self._dedup(images)
        assert len(result) == 2
        assert result[0]["data"] == "AAA="
        assert result[1]["data"] == "BBB="

    def test_empty_list(self):
        assert self._dedup([]) == []


class TestValidateImages:
    """图片校验。"""

    def _validate(self, images):
        from server import _validate_images
        return _validate_images(images)

    def test_valid_images_pass(self):
        images = [{"data": "AAAA", "media_type": "image/png"}]
        valid, warnings = self._validate(images)
        assert len(valid) == 1
        assert warnings == []

    def test_empty_returns_empty(self):
        valid, warnings = self._validate([])
        assert valid == []
        assert warnings == []

    def test_count_exceeds_max(self):
        images = [{"data": "A" * 10, "media_type": "image/png"}] * 10  # exceeds MAX_IMAGES=8
        valid, warnings = self._validate(images)
        assert len(valid) == 8  # capped at MAX_IMAGES
        assert len(warnings) == 1
        assert "超过上限" in warnings[0]

    def test_oversized_image_skipped(self):
        # base64 长度对应 >10MB 的图片
        huge = "A" * (10 * 1024 * 1024 * 4 // 3 + 100)  # >10MB when decoded
        images = [
            {"data": "AAA=", "media_type": "image/png"},
            {"data": huge, "media_type": "image/png"},
            {"data": "BBB=", "media_type": "image/jpeg"},
        ]
        valid, warnings = self._validate(images)
        assert len(valid) == 2  # huge one skipped
        assert valid[0]["data"] == "AAA="
        assert valid[1]["data"] == "BBB="
        assert len(warnings) == 1
        assert "体积超过" in warnings[0]

    def test_invalid_base64_skipped(self):
        images = [
            {"data": "AAA=", "media_type": "image/png"},
            {"data": "!!!not base64!!!", "media_type": "image/png"},
        ]
        valid, warnings = self._validate(images)
        assert len(valid) == 1
        assert "base64" in warnings[0]


class TestCollectImages:
    """统一搜集入口。"""

    def _collect(self, task, images_json=None):
        from server import _collect_images
        return _collect_images(task, images_json)

    def test_no_images(self):
        task, images, warnings = self._collect("hello")
        assert task == "hello"
        assert images == []
        assert warnings == []

    def test_task_embedded_only(self):
        task = "看图 data:image/png;base64,AAA="
        cleaned, images, warnings = self._collect(task)
        assert "data:image" not in cleaned
        assert "[图片]" in cleaned
        assert len(images) == 1

    def test_images_json_only(self):
        task, images, warnings = self._collect(
            "hello", '[{"data": "AAA=", "media_type": "image/png"}]'
        )
        assert task == "hello"
        assert len(images) == 1

    def test_both_sources_merged_and_deduped(self):
        # 同一张图从两边传入 → 去重
        task, images, warnings = self._collect(
            "看图 data:image/png;base64,AAA=",
            '[{"data": "AAA=", "media_type": "image/png"}]',
        )
        assert len(images) == 1  # deduped


# ═══════════════════════════════════════════════════
# models.py 图片处理
# ═══════════════════════════════════════════════════

class TestExtractText:
    """_extract_text 健壮性测试。"""

    def _make_response(self, blocks):
        """构造仿真的 response 对象。"""
        resp = MagicMock()
        resp.content = blocks
        return resp

    def test_extracts_text_block(self):
        from models import AnthropicModel
        resp = self._make_response([
            MagicMock(type="text", text="hello"),
        ])
        result = AnthropicModel._extract_text(resp)
        assert result == "hello"

    def test_skips_thinking_block(self):
        """extended thinking 时 content[0] 可能是 thinking 类型。"""
        from models import AnthropicModel
        resp = self._make_response([
            MagicMock(type="thinking", thinking="..."),
            MagicMock(type="text", text="answer"),
        ])
        result = AnthropicModel._extract_text(resp)
        assert result == "answer"

    def test_returns_empty_for_no_text_blocks(self):
        from models import AnthropicModel
        resp = self._make_response([
            MagicMock(type="tool_use", name="fn", input={}),
        ])
        result = AnthropicModel._extract_text(resp)
        assert result == ""

    def test_returns_empty_for_empty_content(self):
        from models import AnthropicModel
        resp = self._make_response([])
        result = AnthropicModel._extract_text(resp)
        assert result == ""


class TestImageToBlock:
    """图片 → content block 转换和校验。"""

    def test_valid_png(self):
        block = AnthropicModel._image_to_block({
            "data": "iVBORw0KGgo=", "media_type": "image/png"
        })
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"

    def test_rejects_unsupported_media_type(self):
        with pytest.raises(ValueError, match="不支持的图片格式"):
            AnthropicModel._image_to_block({
                "data": "AAA=", "media_type": "image/bmp"
            })

    def test_rejects_empty_data(self):
        with pytest.raises(ValueError, match="非空 base64"):
            AnthropicModel._image_to_block({
                "data": "", "media_type": "image/png"
            })

    def test_rejects_data_with_prefix(self):
        """常见错误：data 字段包含完整的 data: URL。"""
        with pytest.raises(ValueError, match="不应包含"):
            AnthropicModel._image_to_block({
                "data": "data:image/png;base64,iVBORw0KGgo=",
                "media_type": "image/png",
            })


class TestDescribeImages:
    """describe_images 正常和截断路径。"""

    def test_normal_return(self):
        strong = AnthropicModel(api_key="fk", model="claude-haiku-4-5")
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(type="text", text="图片描述内容")]
        mock_resp.stop_reason = "end_turn"
        with patch.object(strong.client.messages, "create", return_value=mock_resp):
            result = strong.describe_images(
                [{"data": "AAA=", "media_type": "image/png"}],
                task_context="测试",
            )
        assert result == "图片描述内容"

    def test_truncation_raises(self):
        strong = AnthropicModel(api_key="fk", model="claude-haiku-4-5")
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(type="text", text="部分内容")]
        mock_resp.stop_reason = "max_tokens"
        with patch.object(strong.client.messages, "create", return_value=mock_resp):
            with pytest.raises(ImageDescriptionTruncated) as exc:
                strong.describe_images(
                    [{"data": "AAA=", "media_type": "image/png"}],
                )
        assert exc.value.max_tokens == strong.max_tokens
        assert exc.value.partial_text == "部分内容"

    def test_rejects_empty_images(self):
        strong = AnthropicModel(api_key="fk", model="claude-haiku-4-5")
        with pytest.raises(ValueError, match="不能为空"):
            strong.describe_images([])

    def test_rejects_too_many_images(self):
        strong = AnthropicModel(api_key="fk", model="claude-haiku-4-5")
        images = [{"data": "AAA=", "media_type": "image/png"}] * (MAX_IMAGES_PER_REQUEST + 1)
        with pytest.raises(ValueError, match="超过"):
            strong.describe_images(images)


class TestCallWithImages:
    """call_with_images 正常路径。"""

    def test_normal_multimodal_call(self):
        strong = AnthropicModel(api_key="fk", model="claude-opus-4-5")
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(type="text", text="分析结果")]
        mock_resp.stop_reason = "end_turn"
        # mock usage
        mock_usage = MagicMock()
        mock_usage.output_tokens = 100
        mock_resp.usage = mock_usage
        with patch.object(strong.client.messages, "create", return_value=mock_resp):
            result = strong.call_with_images(
                "分析这张图",
                [{"data": "AAA=", "media_type": "image/png"}],
            )
        assert "分析结果" in result

    def test_truncation_adds_warning(self):
        """call_with_images 截断不抛异常，加 warning。"""
        strong = AnthropicModel(api_key="fk", model="claude-opus-4-5")
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(type="text", text="结果")]
        mock_resp.stop_reason = "max_tokens"
        mock_usage = MagicMock()
        mock_usage.output_tokens = 32000
        mock_resp.usage = mock_usage
        with patch.object(strong.client.messages, "create", return_value=mock_resp):
            result = strong.call_with_images(
                "分析",
                [{"data": "AAA=", "media_type": "image/png"}],
            )
        assert "截断" in result


# ═══════════════════════════════════════════════════
# router.py
# ═══════════════════════════════════════════════════

class TestRouteTask:
    """路由判断：宽松解析 + 降级。"""

    def _route(self, response):
        from router import route_task
        weak = MagicMock()
        weak.call.return_value = response
        return route_task("test", weak)

    def test_weak(self):
        assert self._route("weak") == "weak"

    def test_strong(self):
        assert self._route("strong") == "strong"

    def test_weak_with_period(self):
        assert self._route("weak.") == "weak"

    def test_strong_with_markdown(self):
        assert self._route("**strong**") == "strong"

    def test_chinese_ruo(self):
        assert self._route("弱") == "weak"

    def test_chinese_qiang(self):
        assert self._route("强") == "strong"

    def test_default_to_strong_on_unexpected(self):
        assert self._route("unexpected response") == "strong"

    def test_fallback_to_strong_on_call_failure(self):
        from router import route_task
        weak = MagicMock()
        weak.call.side_effect = RuntimeError("API error")
        result = route_task("test", weak)
        assert result == "strong"


# ═══════════════════════════════════════════════════
# server.py _describe_images 集成测试
# ═══════════════════════════════════════════════════

class TestDescribeImages:
    """server 层的 _describe_images 包装。"""

    def _describe(self, images, task="", strong=None):
        from server import _describe_images
        if strong is None:
            strong = MagicMock()
            strong.model = "claude-test"
            strong.describe_images.return_value = "描述内容"
        return _describe_images(images, task, strong)

    def test_normal_return_wrapped(self):
        result = self._describe(
            [{"data": "AAA=", "media_type": "image/png"}],
            task="分析图片",
        )
        assert "[图片描述" in result
        assert "描述内容" in result
        assert "[/图片描述]" in result

    def test_truncation_graceful(self):
        strong = MagicMock()
        strong.model = "claude-test"
        strong.describe_images.side_effect = ImageDescriptionTruncated(
            "部分文本" * 100, 1000
        )
        result = self._describe(
            [{"data": "AAA=", "media_type": "image/png"}],
            strong=strong,
        )
        assert "⚠ 图片描述不完整" in result
        assert "被 max_tokens=1000 截断" in result

    def test_exception_graceful(self):
        strong = MagicMock()
        strong.model = "claude-test"
        strong.describe_images.side_effect = RuntimeError("boom")
        result = self._describe(
            [{"data": "AAA=", "media_type": "image/png"}],
            strong=strong,
        )
        assert "⚠ 图片解析失败" in result
        assert "RuntimeError" in result
        # 不包含完整的异常消息（安全）
        assert "boom" not in result

    def test_empty_images_returns_empty(self):
        assert self._describe([]) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
