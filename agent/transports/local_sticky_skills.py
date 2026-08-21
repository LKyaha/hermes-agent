"""Local KV-friendly compaction for repeated single-skill invocations.

Local OpenAI-compatible backends such as Ollama, llama.cpp, vLLM and oMLX can
reuse a warm KV prefix, but they cannot reuse a large SKILL.md body that is
repeated later in the prompt. Hermes persists the full expanded skill on
every slash-skill invocation, so invoking the same skill again can add tens of
thousands of newly-prefilled tokens even when the earlier copy is still in
context.

This transport shim leaves persisted history untouched. At the
chat-completions wire boundary, the first surviving copy of each exact skill
version stays full and later copies become a short reminder plus the new user
instruction. The transform is deterministic from the current message list,
so retry/resume/rebuild yields the same wire prefix. If compression removes
the old anchor, the first surviving full invocation naturally becomes the new
anchor. If the skill/config/path changes, its fingerprint changes and the new
version is sent in full once.

Only local/private endpoints are enabled by default. Set
``HERMES_LOCAL_STICKY_SKILLS=0`` to disable, or ``=1`` to force-enable for a
custom endpoint during testing. ``HERMES_LOCAL_STICKY_SKILLS_DEBUG=1`` logs
how many characters were removed from each request.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any

from agent.transports.chat_completions import ChatCompletionsTransport

logger = logging.getLogger(__name__)

_SKILL_INVOCATION_PREFIX = "[IMPORTANT: The user has invoked the "
_SINGLE_SKILL_MARKER = "The full skill content is loaded below.]"
_SINGLE_SKILL_INSTRUCTION = (
    "The user has provided the following instruction alongside the skill invocation: "
)
_RUNTIME_NOTE = "\n\n[Runtime note:"
_BUNDLE_MARKER = " skill bundle,"
_SKILL_NAME_RE = re.compile(re.escape(_SKILL_INVOCATION_PREFIX) + r'"([^"]*)"')

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    return None


def _split_full_single_skill(
    content: Any,
) -> tuple[str, str, str, str] | None:
    """Return ``(name, fingerprint, instruction, runtime_tail)`` for a full skill.

    The fingerprint deliberately excludes only the volatile user instruction
    and runtime note. Skill body, resolved config, template substitutions,
    setup notes, supporting-file paths and the activation scaffold remain in
    the hash, so a real change gets a fresh full anchor automatically.
    """
    if not isinstance(content, str):
        return None
    if not content.startswith(_SKILL_INVOCATION_PREFIX):
        return None
    if _BUNDLE_MARKER in content or _SINGLE_SKILL_MARKER not in content:
        return None

    match = _SKILL_NAME_RE.match(content)
    if not match:
        return None
    skill_name = match.group(1).strip()
    if not skill_name:
        return None

    instruction = ""
    runtime_tail = ""
    marker_idx = content.rfind(_SINGLE_SKILL_INSTRUCTION)
    if marker_idx >= 0:
        stable_scaffold = content[:marker_idx].rstrip()
        volatile = content[marker_idx + len(_SINGLE_SKILL_INSTRUCTION):]
        runtime_idx = volatile.find(_RUNTIME_NOTE)
        if runtime_idx >= 0:
            instruction = volatile[:runtime_idx].strip()
            runtime_tail = volatile[runtime_idx:]
        else:
            instruction = volatile.strip()
    else:
        runtime_idx = content.rfind(_RUNTIME_NOTE)
        if runtime_idx >= 0:
            stable_scaffold = content[:runtime_idx].rstrip()
            runtime_tail = content[runtime_idx:]
        else:
            stable_scaffold = content.rstrip()

    fingerprint = hashlib.sha256(stable_scaffold.encode("utf-8")).hexdigest()
    return skill_name, fingerprint, instruction, runtime_tail


def _compact_repeat(skill_name: str, instruction: str, runtime_tail: str) -> str:
    parts = [
        f'[IMPORTANT: The user has invoked the "{skill_name}" skill again. '
        "Its full content was loaded earlier in this conversation and remains "
        "active; continue following those instructions.]"
    ]
    if instruction:
        parts.extend(["", f"{_SINGLE_SKILL_INSTRUCTION}{instruction}"])
    content = "\n".join(parts)
    if runtime_tail:
        content += runtime_tail
    return content


def compact_repeated_skill_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Compact repeated full single-skill user messages deterministically.

    Returns ``(messages, compacted_count, removed_chars)``. When no change is
    needed, the original list object is returned so the normal fast path keeps
    its existing allocation behaviour.
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] | None = None
    compacted = 0
    removed_chars = 0

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        parsed = _split_full_single_skill(msg.get("content"))
        if parsed is None:
            continue

        skill_name, fingerprint, instruction, runtime_tail = parsed
        key = (skill_name, fingerprint)
        if key not in seen:
            seen.add(key)
            continue

        replacement = _compact_repeat(skill_name, instruction, runtime_tail)
        original = msg.get("content")
        if replacement == original:
            continue

        if out is None:
            out = list(messages)
        copied = dict(msg)
        copied["content"] = replacement
        out[idx] = copied
        compacted += 1
        if isinstance(original, str):
            removed_chars += max(0, len(original) - len(replacement))

    return (out if out is not None else messages), compacted, removed_chars


def _enabled_for_endpoint(base_url: Any) -> bool:
    explicit = _env_bool("HERMES_LOCAL_STICKY_SKILLS")
    if explicit is False:
        return False
    if explicit is True:
        return True
    try:
        from agent.model_metadata import is_local_endpoint

        return is_local_endpoint(str(base_url or ""))
    except Exception:
        return False


class LocalStickySkillsChatCompletionsTransport(ChatCompletionsTransport):
    """Chat Completions transport with local-only repeated-skill compaction."""

    def build_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params,
    ) -> dict[str, Any]:
        if _enabled_for_endpoint(params.get("base_url")):
            messages, compacted, removed_chars = compact_repeated_skill_messages(messages)
            if compacted and _env_bool("HERMES_LOCAL_STICKY_SKILLS_DEBUG") is True:
                logger.info(
                    "Local sticky skills compacted %d repeat(s), removed %d prompt chars",
                    compacted,
                    removed_chars,
                )
        return super().build_kwargs(model, messages, tools, **params)


# Imported after the stock chat_completions transport during registry discovery,
# so this deliberately replaces only that registry entry. Direct imports of
# ChatCompletionsTransport remain unchanged for existing unit tests/callers.
from agent.transports import register_transport  # noqa: E402

register_transport("chat_completions", LocalStickySkillsChatCompletionsTransport)
