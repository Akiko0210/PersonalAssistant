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

A third, create_agent, makes a new persona: a fresh-context copy of the
caller under a new name (agents.create_agent).

Both moves are deferred via the ToolContext (pending_switch / pending_delegations),
same pattern as save_conversation_note -> pending_note: the agent acts after
the current reply is spoken, so the acknowledgement comes out in the current
voice and the machinery never runs mid-tool-loop.

The regex router in agents.match_address catches explicit addressing ("Bob,
..." / "switch to Tom"); switch_agent covers every other phrasing ("can you
put Bob on?") — the active model recognises the intent and calls it, which
costs nothing extra since that model call was already happening.
"""

import re

from brain import agents
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
                # No enum: the schema is built at import, before load_agents
                # adds the assistants the user created (create_agent).
                "description": "Which assistant to hand the conversation to, by name.",
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
        "leaves you. Use it ONLY when the user's current message asks for "
        "that assistant by name — e.g. 'ask Bob to save this as a note' — "
        "never on your own initiative; the call is refused otherwise. Write "
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
                "description": "Which assistant does the task, by name.",
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
    if not agents.named_in(ctx.user_text, key):
        return (f"Not sent: the user didn't ask for {name}. Answer from your "
                "own record and memory, or ask the user whether they want "
                f"{name} to do it.")
    ctx.pending_delegations.append((key, task))
    return (f"Queued for {name}, who will work on it in the background and "
            f"speak up when done. Tell the user briefly that you've asked "
            f"{name} and that you can keep talking meanwhile.")


@tool({
    "name": "create_agent",
    "description": (
        "Create a new assistant: a copy of you — same role, instructions, "
        "tools, accounts and voice — under a new name, starting with a fresh "
        "conversation and none of your memory. Use it ONLY when the user's "
        "current message asks for a new assistant by name ('create an agent "
        "called Noah'); the call is refused otherwise. Assistants can't be "
        "deleted, so never create one the user didn't ask for."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The new assistant's name: one word, as the user said it.",
            },
        },
        "required": ["name"],
    },
})
def create_agent(ctx, args):
    name = (args.get("name") or "").strip()
    # The user_text gate, as on ask_agent: creation is permanent, so it
    # happens only when the user's own words carry the name.
    if not name or not re.search(rf"\b{re.escape(name)}\b", ctx.user_text or "", re.I):
        return ("Not created: the user didn't say that name. Ask them what to "
                "call the new assistant.")
    try:
        key = agents.create_agent(name, ctx.active_agent)
    except ValueError as e:
        return f"Not created: {e} Ask the user for another name."
    new = agents.AGENTS[key]["name"]
    return (f"Created {new}, a copy of you with a fresh conversation. Tell the "
            f"user in one line that {new} is ready and that saying '{new}' or "
            f"'switch to {new}' brings them over.")
