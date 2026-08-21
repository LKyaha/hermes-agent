"""Tests for local repeated-skill compaction at the Chat Completions boundary."""

from agent.transports.local_sticky_skills import (
    _enabled_for_endpoint,
    compact_repeated_skill_messages,
)


def _skill(name="work", body="Follow the workflow.", instruction="", runtime=""):
    activation = (
        f'[IMPORTANT: The user has invoked the "{name}" skill, indicating they want '
        "you to follow its instructions. The full skill content is loaded below.]"
    )
    parts = [activation, "", body, "", f"[Skill directory: /skills/{name}]", "Use absolute paths."]
    if instruction:
        parts.extend([
            "",
            "The user has provided the following instruction alongside the skill invocation: "
            + instruction,
        ])
    if runtime:
        parts.extend(["", f"[Runtime note: {runtime}]"])
    return "\n".join(parts)


def test_first_copy_stays_full_and_repeat_compacts():
    first = _skill(instruction="first task")
    second = _skill(instruction="second task")
    messages = [
        {"role": "user", "content": first},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": second},
    ]

    out, count, saved = compact_repeated_skill_messages(messages)

    assert out[0]["content"] == first
    assert "full content was loaded earlier" in out[2]["content"]
    assert "second task" in out[2]["content"]
    assert "Follow the workflow." not in out[2]["content"]
    assert count == 1
    assert saved > 0


def test_transform_is_deterministic_and_append_only_across_turns():
    first = _skill(body="X" * 20_000, instruction="one")
    second = _skill(body="X" * 20_000, instruction="two")
    base = [
        {"role": "user", "content": first},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": second},
    ]
    wire2, _, _ = compact_repeated_skill_messages(base)
    extended = [
        *base,
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": _skill(body="X" * 20_000, instruction="three")},
    ]
    wire3, _, _ = compact_repeated_skill_messages(extended)

    assert wire3[: len(wire2)] == wire2
    assert compact_repeated_skill_messages(base)[0] == wire2


def test_changed_skill_version_gets_new_full_anchor():
    old = _skill(body="version one", instruction="one")
    new = _skill(body="version two", instruction="two")
    new_repeat = _skill(body="version two", instruction="three")
    out, count, _ = compact_repeated_skill_messages([
        {"role": "user", "content": old},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": new},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": new_repeat},
    ])

    assert "version one" in out[0]["content"]
    assert "version two" in out[2]["content"]
    assert "version two" not in out[4]["content"]
    assert count == 1


def test_compression_or_truncation_reanchors_first_surviving_copy():
    surviving_first = _skill(body="large body", instruction="after compression")
    surviving_second = _skill(body="large body", instruction="again")
    out, count, _ = compact_repeated_skill_messages([
        {"role": "system", "content": "summary of older turns"},
        {"role": "user", "content": surviving_first},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": surviving_second},
    ])

    assert out[1]["content"] == surviving_first
    assert "large body" not in out[3]["content"]
    assert count == 1


def test_runtime_note_is_preserved_on_compacted_repeat():
    out, count, _ = compact_repeated_skill_messages([
        {"role": "user", "content": _skill(runtime="turn=1")},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": _skill(runtime="turn=2")},
    ])

    assert count == 1
    assert out[2]["content"].endswith("\n\n[Runtime note: turn=2]")


def test_normal_user_messages_and_bundles_are_untouched():
    normal = {"role": "user", "content": "hello"}
    bundle = {
        "role": "user",
        "content": '[IMPORTANT: The user has invoked the "/a /b" stacked skill bundle, loading 2 skills together.]',
    }
    messages = [normal, bundle]
    out, count, saved = compact_repeated_skill_messages(messages)

    assert out is messages
    assert count == 0
    assert saved == 0


def test_message_metadata_is_preserved_when_copying():
    repeated = {
        "role": "user",
        "content": _skill(instruction="two"),
        "name": "alice",
        "custom": 7,
    }
    out, _, _ = compact_repeated_skill_messages([
        {"role": "user", "content": _skill(instruction="one")},
        repeated,
    ])

    assert out[1]["name"] == "alice"
    assert out[1]["custom"] == 7
    assert repeated["content"] != out[1]["content"]


def test_local_endpoint_auto_enable_and_public_disable(monkeypatch):
    monkeypatch.delenv("HERMES_LOCAL_STICKY_SKILLS", raising=False)
    assert _enabled_for_endpoint("http://127.0.0.1:8000/v1") is True
    assert _enabled_for_endpoint("https://api.openai.com/v1") is False


def test_env_override_can_disable_or_force(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_STICKY_SKILLS", "0")
    assert _enabled_for_endpoint("http://127.0.0.1:8000/v1") is False
    monkeypatch.setenv("HERMES_LOCAL_STICKY_SKILLS", "1")
    assert _enabled_for_endpoint("https://example.com/v1") is True
