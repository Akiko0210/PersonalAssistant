"""Tools over archived long-term conversation memory (ConversationMemory)."""

from tools import tool


@tool({
    "name": "search_past_conversations",
    "description": (
        "Search YOUR OWN past conversations with the user, outside the "
        "current chat window: earlier in THIS session (verbatim, not yet "
        "archived), your archived summaries of older conversations, and the "
        "pre-isolation shared archive. Use for 'what did we talk about last "
        "week', 'didn't we discuss X before', 'review your memory', or "
        "whenever the user refers to something you don't see in the current "
        "history — including from earlier today. This never sees another "
        "assistant's conversations — for those, ask them with ask_agent."
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
    return ctx.memory.search(args["query"], client=ctx.client,
                             caller=ctx.active_agent)
