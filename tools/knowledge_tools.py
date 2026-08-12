"""Tools over ingested reference material (KnowledgeStore)."""

from tools import tool


@tool({
    "name": "search_knowledge",
    "description": (
        "Search the user's ingested trading reference material — books, PDFs, text "
        "files, and transcribed course videos — for relevant passages. Use for "
        "questions about trading concepts, strategies, definitions, or 'what does "
        "my trading book/course say about X'. Returns excerpts with their source "
        "and location so you can cite them: a page number for PDFs, a timestamp "
        "like 14:32 for video."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look up in the trading material"}
        },
        "required": ["query"],
    },
})
def search_knowledge(ctx, args):
    return ctx.kb.search(args["query"], caller=ctx.active_agent,
                         focus=ctx.focus)
