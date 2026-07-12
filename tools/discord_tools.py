"""Tools over captured Discord notifications and trade alerts (DiscordData)."""

from tools import tool


@tool({
    "name": "get_recent_discord_messages",
    "description": (
        "List recent Discord notifications the user has captured (newest at the "
        "end). Use for 'what came in today', 'latest Discord messages', or any "
        "time-based question. trades.txt has no timestamps, so use this with a "
        "date to answer 'what trades came in today'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": "How many to list (default 10)"},
            "date": {
                "type": "string",
                "description": "Optional day filter: 'today', 'yesterday', or YYYY-MM-DD",
            },
        },
    },
})
def get_recent_discord_messages(ctx, args):
    return ctx.discord.recent_messages(int(args.get("n", 10)), args.get("date"))


@tool({
    "name": "search_discord_messages",
    "description": (
        "Search captured Discord notifications by keyword/topic and/or sender. "
        "Use for 'what was said about SPX', 'any iron condors', or "
        "'what did Dan Sheridan say'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keyword or topic to search for"},
            "sender": {
                "type": "string",
                "description": "Optional: only messages from this sender (substring match)",
            },
        },
        "required": ["query"],
    },
})
def search_discord_messages(ctx, args):
    return ctx.discord.search_messages(args["query"], args.get("sender"))


@tool({
    "name": "get_recent_trades",
    "description": (
        "List the most recent trade-ready lines captured from Discord trade "
        "alerts. Use for 'what are the latest trades' or 'recent Discord trades'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": "How many to list (default 10)"}
        },
    },
})
def get_recent_trades(ctx, args):
    return ctx.discord.recent_trades(int(args.get("n", 10)))
