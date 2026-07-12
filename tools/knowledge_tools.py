"""Tools over ingested reference material (KnowledgeStore)."""

from tools import tool


@tool({
    "name": "search_knowledge",
    "description": (
        "Search the user's ingested trading reference material (PDFs and text "
        "files) for relevant passages. Use for questions about trading concepts, "
        "strategies, definitions, or 'what does my trading book say about X'. "
        "Returns excerpts with their source (and page, for PDFs) so you can cite them."
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
    return ctx.kb.search(args["query"])
