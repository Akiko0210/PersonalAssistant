"""Persona hand-offs and background delegation.

Two distinct moves, and the distinction is the whole design:

- switch_agent: the USER moves — they talk to the other persona from now on.
- ask_agent: a TASK moves — the other persona does a job in the background
  while the user stays right here, and speaks up with the result at the next
  pause (see voice_agent's interjection queue).

ask_agent exists because routing a mid-conversation job (like a note save)
through switch_agent meant bouncing the user to Bob and back, and the save's
folder question then straddled turn boundaries — where it once swallowed a
"switch me back to Tom" and stranded the user on the wrong persona for 16
minutes (session_2026-07-31.log, 11:31-11:48). A delegated task never moves
the user, so there is nothing to bounce back from.

Both are deferred via the ToolContext (pending_switch / pending_delegations),
same pattern as save_conversation_note -> pending_note: the agent acts after
the current reply is spoken, so the acknowledgement comes out in the current
voice and the machinery never runs mid-tool-loop.

The regex router in agents.match_address catches explicit addressing ("Bob,
..." / "switch to Tom"); switch_agent covers every other phrasing ("can you
put Bob on?") — the active model recognises the intent and calls it, which
costs nothing extra since that model call was already happening.
"""

import agents
from tools import tool


@tool({
    "name": "switch_agent",
    "description": (
        "Hand the conversation over to another assistant so the user talks to "
        "them directly from now on. Use it ONLY when the user wants to talk "
        "to them — asks for them by name, or wants to continue in their "
        "specialty conversationally. For a one-off job in their specialty "
        "(saving a note, looking something up) use ask_agent instead, so the "
        "user never has to leave you. Pass the user's question as `forward` "
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


@tool({
    "name": "ask_agent",
    "description": (
        "Hand a task to another assistant to do IN THE BACKGROUND while the "
        "user stays in this conversation with you. They work out of earshot "
        "and speak up with the result at the next pause; the user never "
        "leaves you. Use this whenever the user wants something done in "
        "another specialty without going there themselves — e.g. 'have Bob "
        "save this as a note' in the middle of a trading discussion. Write "
        "the task fully self-contained, including ALL content it needs (the "
        "complete note text, the exact question): the assistant cannot see "
        "this conversation. To actually hand the user over, use switch_agent "
        "instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "enum": sorted(agents.AGENTS),
                "description": "Which assistant does the task.",
            },
            "task": {
                "type": "string",
                "description": ("The task, fully self-contained — every "
                                "detail and all content included, since the "
                                "assistant cannot see this conversation."),
            },
        },
        "required": ["agent", "task"],
    },
})
def ask_agent(ctx, args):
    key = agents.resolve(args.get("agent"))
    if key is None:
        known = ", ".join(a["name"] for a in agents.AGENTS.values())
        return f"I don't know that assistant. Available: {known}."
    task = (args.get("task") or "").strip()
    if not task:
        return "No task given — pass the full, self-contained task text."
    name = agents.AGENTS[key]["name"]
    if key == ctx.active_agent:
        return (f"That's you — the user is talking to {name} right now. "
                "Do the task yourself with your own tools.")
    ctx.pending_delegations.append((key, task))
    return (f"Queued for {name}, who will work on it in the background and "
            f"speak up when done. Tell the user briefly that you've asked "
            f"{name} and that you can keep talking meanwhile.")
