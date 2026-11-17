"""Tools that keep a persona's working record (stores/working_state.py).

The record is shown to its persona every turn, so there is no "search my
record" tool — only writes, plus read_record for the history the block had
no room for. Every handler passes ctx.active_agent and nothing else as the
owner: a persona can neither write nor name another persona's record. Refused
writes (a body over the cap, someone else's id) come back as sentences the
model can act on.

When each persona writes is its record_rules in brain/agents.py, not here:
these tools are the same for Tom's campaigns, Linda's notebook and Alice's
list.
"""

from stores import working_state
from tools import tool
from tools.model_tools import live_model


def _rid(args):
    try:
        return int(args.get("id"))
    except (TypeError, ValueError):
        raise ValueError("Which record? Pass its id number (the #n in your "
                         "working record).") from None


@tool({
    "name": "update_record",
    "description": (
        "Create or change a record in your working record — the things you "
        "track for the user across sessions, shown to you every turn. Without "
        "an id it creates a new open record (title required); with an id it "
        "replaces whichever of title, body, kind you pass. The body is the "
        "record's CURRENT state — rules, levels, status — in at most 2,000 "
        "characters; dated happenings go in add_entry. Returns the record as "
        "saved: read it back to the user when they need to confirm it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "Existing record to change; omit to create."},
            "kind": {"type": "string", "description": "Short label, e.g. campaign, pattern, item."},
            "title": {"type": "string", "description": "Short, stable name."},
            "body": {"type": "string", "description": "The full current state (replaces the old body)."},
        },
    },
})
def update_record(ctx, args):
    owner = ctx.active_agent
    try:
        if args.get("id") is None:
            # Refused, not redirected: this call's body would replace the
            # project's. On 2026-11-01 an id-less call made a duplicate of a
            # strategy that already had a record, under another name, and the
            # next day's talk followed the wrong one.
            if in_focus := working_state.focused(owner):
                return (f"You are focused on #{in_focus['id']} {in_focus['title']}: "
                        f"pass id {in_focus['id']} to change it. If the user is "
                        "starting something separate, leave focus first with "
                        "focus_project and say so.")
            rid = working_state.create(owner, args.get("title") or "",
                                       kind=args.get("kind") or "",
                                       body=args.get("body") or "")
        else:
            rid = _rid(args)
            working_state.update(owner, rid, title=args.get("title"),
                                 body=args.get("body"), kind=args.get("kind"))
    except ValueError as e:
        return str(e)
    return "Saved. It now reads:\n" + working_state.render(owner, rid)


@tool({
    "name": "add_entry",
    "description": (
        "Add a dated entry under one of your open records: a decision and its "
        "reason, an occurrence of a pattern, a check-in, a review. One or two "
        "sentences, at most 500 characters."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "The record's id."},
            "text": {"type": "string", "description": "What happened, in a sentence or two."},
        },
        "required": ["id", "text"],
    },
})
def add_entry(ctx, args):
    try:
        rid = _rid(args)
        working_state.add_entry(ctx.active_agent, rid, args.get("text") or "",
                                model=live_model(ctx))
    except ValueError as e:
        return str(e)
    return f"Added to #{rid}."


@tool({
    "name": "close_record",
    "description": (
        "Close one of your records when it is finished — a campaign exited, a "
        "list item done or dropped, a pattern resolved — with how it ended. "
        "Closed records leave your working record but stay readable. This is "
        "also how a project is deleted: it can't be focused again, and its "
        "memory stays searchable."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "The record's id."},
            "outcome": {"type": "string", "description": "How it ended, in a sentence."},
        },
        "required": ["id", "outcome"],
    },
})
def close_record(ctx, args):
    try:
        rid = _rid(args)
        was = working_state.focused(ctx.active_agent)
        working_state.close(ctx.active_agent, rid, args.get("outcome") or "",
                            model=live_model(ctx))
    except ValueError as e:
        return str(e)
    if was and was["id"] == rid:
        return f"Closed #{rid}. Focus on it has ended: tell the user."
    return f"Closed #{rid}."


@tool({
    "name": "focus_project",
    "description": (
        "Work inside one project, because the user asked to: a project is one "
        "of your open records. Until the user ends it, even across restarts, "
        "that project's conversations and facts come first in your Background "
        "and each exchange is labeled with it; the rest of your memory stays "
        "searchable. Pass a different id to move to another project, or no id "
        "to leave focus. Never call it on your own, and always tell the user "
        "aloud what changed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "integer",
                   "description": "The project's record id; omit to leave focus."},
        },
    },
})
def focus_project(ctx, args):
    owner = ctx.active_agent
    if args.get("id") is None:
        was = working_state.leave_focus(owner)
        return (f"Left focus on {was['title']}: tell the user. Memory is back to "
                "normal." if was else "No project was in focus.")
    try:
        rid = _rid(args)
        working_state.focus_on(owner, rid)
    except ValueError as e:
        return str(e)
    return ("Now in focus: tell the user. This turn's Background was chosen "
            "before the change, so read this and call recall for anything "
            "more:\n" + working_state.render(owner, rid))


@tool({
    "name": "read_record",
    "description": (
        "Read a record's full history, including entries your working record "
        "had no room to show; or, without an id, list your open records and "
        "your recently closed ones."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "The record to read; omit for the list."},
        },
    },
})
def read_record(ctx, args):
    if args.get("id") is None:
        lines = working_state.listing(ctx.active_agent)
        return "\n".join(lines) if lines else "Your working record is empty."
    try:
        rid = _rid(args)
    except ValueError as e:
        return str(e)
    return (working_state.render(ctx.active_agent, rid)
            or f"You have no record #{rid}.")
