"""Tool that lets the agent answer questions about this project itself.

It returns the detailed project description (PROJECT.md) as its result; the
model then answers the user's actual question from it — "how does barge-in
work?", "where are my notes stored?", "what tools do you have?", "what model
are you using and how do I switch it?". Keeping the doc as the single source
means the agent's self-knowledge stays in sync with the doc, not with a
prompt that drifts.
"""

import config as cfg
from tools import tool

# Cache the doc across calls within a session; it doesn't change while running.
_CACHE = {"text": None}


def _load_doc() -> str:
    if _CACHE["text"] is None:
        try:
            _CACHE["text"] = cfg.PROJECT_DOC_PATH.read_text(encoding="utf-8")
        except OSError as e:
            return f"(Project description unavailable: {e})"
    return _CACHE["text"]


@tool({
    "name": "describe_project",
    "description": (
        "Answer questions about THIS project — the voice agent itself: its "
        "architecture, modules, features, how something works (barge-in, memory, "
        "notetaking, the headset button), where data is stored, what tools exist, "
        "or how to extend it. Call this whenever the user asks how you work, what "
        "you're built from, why something behaves the way it does, or 'tell me "
        "about yourself/this project'. Returns the project's design document; use "
        "it to answer, then reply in a sentence or two spoken aloud — don't read "
        "the whole document out."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": (
                    "Optional: what the user is asking about (e.g. 'barge-in', "
                    "'memory', 'storage'). For logging/context only — the full "
                    "description is returned regardless."
                ),
            }
        },
    },
})
def describe_project(ctx, args):
    return _load_doc()
