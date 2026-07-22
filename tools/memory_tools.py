"""Tools over archived long-term conversation memory (ConversationMemory)."""

from tools import tool


@tool({
    "name": "search_past_conversations",
    "description": (
        "Search everything said outside the current chat window: earlier in "
        "THIS session (verbatim, not yet archived) and archived summaries of "
        "older conversations. Use for 'what did we talk about last week', "
        "'didn't we discuss X before', 'review your memory', or whenever the "
        "user refers to something you don't see in the current history — "
        "including from earlier today."
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
    return ctx.memory.search(args["query"])
