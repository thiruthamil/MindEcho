# app/prompts.py — text and LLM prompt templates

from typing import List


def build_context(today: str, past_snippets: List[tuple]) -> str:
    lines = ["Today:\n" + today.strip(), "\nRelevant past notes:"]
    for date, preview in past_snippets:
        lines.append(f"- [{date}] {preview}")
    return "\n".join(lines)


FALLBACK_TEMPLATES = [
    "Noted — saved today’s entry. A similar moment on {date_hint}. One tiny step you used then: {tip}.",
    "Noted — saved today’s entry. You handled a similar day on {date_hint} by {tip}. Try a 10-minute version now.",
]


def simple_nudge(today: str, past_snippets: List[tuple]) -> str:
    if not past_snippets:
        return "Noted — saved today’s entry. Keep it light: pick one tiny action for tomorrow and commit to it."
    date_hint = past_snippets[0][0]
    frag = past_snippets[0][1]
    tip = frag.split(".")[0][:120]
    tmpl = FALLBACK_TEMPLATES[0]
    return tmpl.format(date_hint=date_hint, tip=tip)


CHAT_SYSTEM = """You are MindEcho, a calm journaling companion.
Use the user's past reflections only as optional context.
Be supportive, concrete, and concise (2–3 sentences).
Avoid therapy or medical claims. Suggest one small next step only if it fits naturally.
Do not end with a question unless the user explicitly asks for one."""


CHAT_USER = """Today's note:
{user_text}

Relevant past reflections (optional):
{context_block}

Tasks (concise, 2–3 sentences):
- Reflect back what you understood in clear, simple terms.
- Gently note any genuine pattern if you notice one.
- Offer one small next step, only if it feels natural.
"""
