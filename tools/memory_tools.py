"""Tools over the persona's conversation memory (ConversationMemory)."""

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
        "Background has nothing on a topic you would expect to remember. This "
        "never sees another assistant's conversations — for those, ask them "
        "with ask_agent."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Topic to look for in past conversations"}
        },
        "required": ["query"],
    },
})
def search_past_conversations(ctx, args):
    return ctx.memory.search(args["query"], caller=ctx.active_agent)
