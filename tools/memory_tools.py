"""Tools over archived long-term conversation memory (ConversationMemory)."""

from tools import tool


@tool({
    "name": "search_past_conversations",
    "description": (
        "Search archived summaries of older conversations — ones that have "
        "aged out of the current chat window. Use for 'what did we talk about "
        "last week/month', 'didn't we discuss X before', or any question about "
        "a past conversation you don't see in the current history."
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
