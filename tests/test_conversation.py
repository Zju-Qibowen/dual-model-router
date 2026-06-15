import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation import ConversationHistory


def test_add_and_get_messages():
    h = ConversationHistory()
    h.add_user("hello")
    h.add_assistant("hi there")
    msgs = h.get_messages()
    assert msgs == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_turn_count():
    h = ConversationHistory()
    h.add_user("q1")
    h.add_assistant("a1")
    h.add_user("q2")
    h.add_assistant("a2")
    assert h.turn_count == 2


def test_clear():
    h = ConversationHistory()
    h.add_user("q1")
    h.add_assistant("a1")
    h.clear()
    assert h.turns == []
    assert h.turn_count == 0


def test_trim_by_max_turns():
    h = ConversationHistory(max_turns=2, max_tokens=100000)
    for i in range(5):
        h.add_user(f"q{i}")
        h.add_assistant(f"a{i}")
    # max_turns=2 means max 4 messages (2 pairs)
    assert len(h.turns) == 4
    assert h.turns[0]["content"] == "q3"
    assert h.turns[1]["content"] == "a3"


def test_trim_by_max_tokens():
    h = ConversationHistory(max_turns=100, max_tokens=50)
    # Each "x" * 200 is ~50 tokens (200/4), so two messages exceed budget
    h.add_user("x" * 200)
    h.add_assistant("y" * 200)
    h.add_user("z" * 200)
    h.add_assistant("w" * 200)
    # Should have trimmed old messages to stay under budget
    assert h.estimated_tokens <= 50 or len(h.turns) == 2


def test_trim_removes_orphan_assistant():
    h = ConversationHistory(max_turns=1, max_tokens=100000)
    h.add_user("q1")
    h.add_assistant("a1")
    h.add_user("q2")
    h.add_assistant("a2")
    # Should not start with assistant
    if h.turns:
        assert h.turns[0]["role"] == "user"


def test_estimated_tokens_cjk():
    h = ConversationHistory()
    h.add_user("你好世界")  # 4 CJK chars -> ~6 tokens
    assert h.estimated_tokens > 0


def test_get_messages_excludes_internal_fields():
    h = ConversationHistory()
    h.add_user("test")
    msgs = h.get_messages()
    assert list(msgs[0].keys()) == ["role", "content"]


# ── pre_context 测试 ──────────────────────────────


def test_get_context_for_strong_with_pre_context():
    """任务指令应出现在输出最前面，参考材料在后。"""
    h = ConversationHistory()
    h.add_user("hello")
    h.add_assistant("hi", source="weak")

    result = h.get_context_for_strong(task="current task", pre_context="[文件: foo.py]\nprint('hello')")

    assert "请完成以下任务" in result
    assert "current task" in result
    assert "[参考材料 · 预收集的上下文]" in result
    assert "[文件: foo.py]" in result
    assert "print('hello')" in result
    # 指令+任务应在参考材料之前
    instr_pos = result.index("请完成以下任务")
    ref_pos = result.index("[参考材料 · 预收集的上下文]")
    assert instr_pos < ref_pos


def test_get_context_for_strong_empty_pre_context():
    """pre_context 为空时不应添加参考材料段落。对话历史现在由 API 原生传递。"""
    h = ConversationHistory()
    h.add_user("hello")
    h.add_assistant("hi")

    result = h.get_context_for_strong(task="current", pre_context="")

    assert "[参考材料 · 预收集的上下文]" not in result
    assert "[参考材料 · 最近对话]" not in result  # 不再嵌入文本，通过 API history 参数传递


def test_get_context_for_strong_pre_context_only():
    """只有 pre_context 没有 task 时，以 --- 分隔符开头。"""
    h = ConversationHistory()

    result = h.get_context_for_strong(pre_context="some context")

    assert "[参考材料 · 预收集的上下文]" in result
    assert "some context" in result


# ── tool_note 测试 ──────────────────────────────


def test_add_tool_note():
    h = ConversationHistory()
    h.add_tool_note("已读取 server.py")
    h.add_tool_note("git diff 显示 3 个文件被修改")

    result = h.get_context_for_strong(task="current task")

    assert "[参考材料 · Claude Code 已执行的步骤]" in result
    assert "已读取 server.py" in result
    assert "git diff 显示 3 个文件被修改" in result
    # 指令应在工具操作记录之前
    instr_pos = result.index("请完成以下任务")
    notes_pos = result.index("[参考材料 · Claude Code 已执行的步骤]")
    assert instr_pos < notes_pos


def test_tool_note_does_not_affect_turn_count():
    h = ConversationHistory()
    h.add_user("hello")
    h.add_assistant("hi")
    h.add_tool_note("some note")

    assert h.turn_count == 1  # tool notes are not turns
    assert len(h.get_messages()) == 2  # tool notes excluded from messages


def test_tool_note_included_in_pre_context_order():
    """当前任务在前，参考材料在后。"""
    h = ConversationHistory()
    h.add_user("hello")
    h.add_assistant("hi")
    h.add_tool_note("read something")

    result = h.get_context_for_strong(task="task", pre_context="file content")

    instr_pos = result.index("请完成以下任务")
    pre_pos = result.index("[参考材料 · 预收集的上下文]")
    notes_pos = result.index("[参考材料 · Claude Code 已执行的步骤]")
    assert instr_pos < pre_pos < notes_pos


def test_tool_note_cleared_with_history():
    h = ConversationHistory()
    h.add_tool_note("some note")
    h.clear()

    result = h.get_context_for_strong(task="task")
    assert "[工具操作记录" not in result


def test_tool_note_capped():
    """工具操作记录超上限时应丢弃最旧的。"""
    from conversation import MAX_TOOL_NOTES
    h = ConversationHistory()
    for i in range(MAX_TOOL_NOTES + 10):
        h.add_tool_note(f"note {i}")

    result = h.get_context_for_strong(task="task")
    # note 0-9 应该被丢弃
    assert "note 0" not in result
    assert "note 9" not in result
    # 保留的是最后 MAX_TOOL_NOTES 条
    assert "note 59" in result
