"""Tools over the persona's conversation memory (ConversationMemory)."""

from lib.dates import PERIODS, window_epochs
from tools import tool


@tool({
    "name": "search_past_conversations",
    "description": (
        "Search YOUR OWN past exchanges with the user — every conversation you "
        "have had, back to the beginning — for more detail than the Background "
        "section already shows. The most relevant past exchanges are retrieved "
        "into your Background automatically each turn, so reach for this when "
        "the user asks for more ('what exactly did I say', 'list everything we "
        "discussed about X', 'when did we last talk about Y') or when the "
        "Background has nothing on a topic you would expect to remember. Time "
        "words carry no weight in the search itself: when the user refers to a "
        "time ('this morning', 'yesterday around 9:20', 'last week'), pass "
        "period or since/until — the window then replaces topic relevance and "
        "the exchanges come back in order. You know the current time from the "
        "stamp on each user message. This never sees another assistant's "
        "conversations — for those, ask them with ask_agent."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Topic to look for in past conversations"},
            "period": {"type": "string", "enum": list(PERIODS),
                       "description": "Named day range, local time — for "
                                      "'yesterday', 'last week'."},
            "since": {"type": "string",
                      "description": "ISO local date or datetime (2026-09-17 or "
                                     "2026-09-17T09:00): start of a custom window."},
            "until": {"type": "string",
                      "description": "ISO local date or datetime: end of the window. "
                                     "A bare date means the end of that day."},
        },
        "required": ["query"],
    },
})
def search_past_conversations(ctx, args):
    window = window_epochs(args.get("period"), args.get("since"), args.get("until"))
    return ctx.memory.search(args["query"], caller=ctx.active_agent,
                             window=window, exclude_ids=ctx.tail_ids)
