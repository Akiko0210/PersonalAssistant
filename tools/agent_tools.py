"""Agent switching: hand the conversation to another persona by tool call.

The regex router in agents.match_address catches explicit addressing ("Bob,
..." / "switch to Cobe"); this tool covers every other phrasing ("can you put
Bob on?", "this sounds like a job for Cobe") — the active model recognises the
intent and calls it, which costs nothing extra since that model call was
already happening. The switch itself is deferred via ctx.pending_switch (same
pattern as save_conversation_note -> pending_note): the agent performs it
after the current reply is spoken, so the goodbye comes out in the old voice
and the next words in the new one.
"""

import agents
from tools import tool


@tool({
    "name": "switch_agent",
    "description": (
        "Hand the conversation over to another assistant so the user talks to "
        "them directly from now on. Use when the user asks for them by name "
        "or when the request clearly belongs to their specialty and they "
        "should own the conversation. Pass the user's question as `forward` "
        "so the new assistant answers it immediately — omit it only when the "
        "user just wants to switch."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "enum": sorted(agents.AGENTS),
                "description": "Which assistant to hand the conversation to.",
            },
            "forward": {
                "type": "string",
                "description": ("Optional: the user's pending question, "
                                "restated self-contained, so the new "
                                "assistant answers it right away."),
            },
        },
        "required": ["agent"],
    },
})
def switch_agent(ctx, args):
    key = agents.resolve(args.get("agent"))
    if key is None:
        known = ", ".join(a["name"] for a in agents.AGENTS.values())
        return f"I don't know that assistant. Available: {known}."
    name = agents.AGENTS[key]["name"]
    if key == ctx.active_agent:
        return f"The user is already talking to {name}."
    ctx.pending_switch = (key, (args.get("forward") or "").strip())
    return (f"Okay — after this reply the conversation moves to {name}. "
            "Keep your reply to a brief handover; the system announces "
            f"{name} and they speak next.")
