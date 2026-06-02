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
