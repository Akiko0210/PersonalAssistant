"""The recall tool: the one way past the Background.

Every turn already retrieves the exchanges, facts and reference passages that
bear on the message (brain/context.py). What that one shot cannot do is the
second question the model only knows to ask after reading the first answer —
"the list mentions a dentist; when did we last discuss the dentist?" — the
exact words behind a fact (its `from` ids), a stretch of time read in order,
or more depth than the budget holds. One tool covers all four, over the
caller's own exchange index and the reference material it may read.
"""

from lib.dates import PERIODS, window_epochs
from tools import tool


@tool({
    "name": "recall",
    "description": (
        "Look further into your own memory than the Background shows: your "
        "past exchanges with the user (back to the beginning) and the user's "
        "ingested reference material (books, PDFs, course videos — cited by "
        "page or timestamp). Use it when the Background raises a second "
        "question, when the user asks for more ('what exactly did I say', "
        "'everything we discussed about X'), to read the exact exchanges "
        "behind a fact (pass its `from` ids), or when the user names a time "
        "('this morning', 'yesterday around 9:20', 'last week'): time words "
        "carry no weight in a search, so pass period or since/until — the "
        "window then replaces topic relevance and the exchanges come back in "
        "order; with a window alone you get that stretch of conversation. You "
        "know the current time from the stamp on each user message. This "
        "never sees another assistant's conversations — ask them with "
        "ask_agent."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Topic to look for. Optional when a time "
                                     "window or ids are given."},
            "period": {"type": "string", "enum": list(PERIODS),
                       "description": "Named day range, local time — for "
                                      "'yesterday', 'last week'."},
            "since": {"type": "string",
                      "description": "ISO local date or datetime (2026-09-17 or "
                                     "2026-09-17T09:00): start of a custom window."},
            "until": {"type": "string",
                      "description": "ISO local date or datetime: end of the window. "
                                     "A bare date means the end of that day."},
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "Exchange ids to read in full — the `from` "
                                   "attribute of a Background fact."},
        },
    },
})
def recall(ctx, args):
    if args.get("ids"):
        return ctx.memory.fetch(args["ids"], caller=ctx.active_agent)
    query = (args.get("query") or "").strip()
    window = window_epochs(args.get("period"), args.get("since"), args.get("until"))
    if not query and window is None:
        return "Give recall a topic, a time window, or exchange ids."
    parts = [ctx.memory.search(query, caller=ctx.active_agent, window=window,
                               exclude_ids=ctx.tail_ids)]
    if query and ctx.kb is not None:
        if passages := ctx.kb.search(query, caller=ctx.active_agent, focus=ctx.focus):
            parts.append("From the reference material:\n" + passages)
    return "\n\n".join(parts)
