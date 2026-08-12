"""Focus mode: temporarily narrow retrieval to one strategy/underlying.

Focus is session state on ToolContext — every retrieval tool folds it into
its search (hard on private collections, soft on common knowledge), and
llm.converse surfaces it in the system prompt so the model KNOWS its own
retrieval is narrowed and can say so instead of confidently reporting
nothing found. It survives persona switches (deliberate — TODO §1) and dies
with the session (also deliberate: a stale focus makes retrieval look
broken)."""

from tools import tool

# The metadata keys focus may filter on. A registry, so adding a dimension is
# one entry here + tagging entries at write time — not a schema hunt.
FOCUS_KEYS = ("strategy", "underlying")


def describe_focus(focus) -> str:
    """'strategy double_diagonal; underlying SPX' — one line for prompts and
    spoken confirmations."""
    parts = []
    for key in FOCUS_KEYS:
        value = (focus or {}).get(key)
        if value:
            parts.append(f"{key} " + (", ".join(map(str, value))
                                      if isinstance(value, list) else str(value)))
    return "; ".join(parts)


def focus_prompt_block(focus) -> str:
    """System-prompt line for an active focus, or "". Visibility is the
    point: a silently narrowed retrieval makes the model confidently report
    nothing found (TODO §1's stated failure mode)."""
    if not focus:
        return ""
    return ("\n\nFocus: retrieval is narrowed to " + describe_focus(focus)
            + ". If results seem thin, say the focus may be hiding material; "
              "clear_focus widens it.")


@tool({
    "name": "set_focus",
    "description": (
        "Narrow all knowledge/journal retrieval to one trading focus — a "
        "strategy (e.g. double_diagonal, butterfly, credit_spread) and/or an "
        "underlying (e.g. SPX, RUT) — until clear_focus. Use when the user "
        "says to focus on, concentrate on, or only look at certain trades. "
        "Each field accepts one value or a list."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "strategy": {
                "type": ["string", "array"], "items": {"type": "string"},
                "description": "Strategy name(s) to focus on"},
            "underlying": {
                "type": ["string", "array"], "items": {"type": "string"},
                "description": "Underlying symbol(s) to focus on, e.g. SPX"},
        },
    },
})
def set_focus(ctx, args):
    focus = {k: args[k] for k in FOCUS_KEYS if args.get(k)}
    if not focus:
        return "Nothing to focus on — give a strategy and/or an underlying."
    ctx.focus = focus
    return f"Focused on {describe_focus(focus)}. Retrieval is narrowed until clear_focus."


@tool({
    "name": "clear_focus",
    "description": "Clear the current retrieval focus, restoring full-width search.",
    "input_schema": {"type": "object", "properties": {}},
})
def clear_focus(ctx, args):
    if ctx.focus is None:
        return "No focus was set."
    was = describe_focus(ctx.focus)
    ctx.focus = None
    return f"Cleared the focus on {was}. Searching everything again."


@tool({
    "name": "get_focus",
    "description": "Report the current retrieval focus, if any.",
    "input_schema": {"type": "object", "properties": {}},
})
def get_focus(ctx, args):
    if ctx.focus is None:
        return "No focus is set — retrieval is full-width."
    return f"Focused on {describe_focus(ctx.focus)}."
