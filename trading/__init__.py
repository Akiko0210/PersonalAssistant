"""Real trading for the voice agent, via the tastytrade Open API.

Self-contained package: multi-leg option strategy building, real-time
bid/ask over DXLink, order review/submit/cancel, and realized/unrealized
P&L. Nothing here is imported by the rest of the app except through
`tools/trading_tools.py` (the voice surface) and the dashboard's lazily
imported trading routes — see TRADING_PLAN.md for the architecture and
TRADING_RESEARCH.md for the API research it is built on.

Module map:
    config.py        env/credentials, base URLs, paths, futures constants
    models.py        dataclasses shared across the package
    symbols.py       OCC option symbols, futures symbology, spoken forms
    tasty_client.py  authenticated REST client (OAuth2 refresh-token)
    chains.py        option-chain fetch + normalization + cache
    strategies.py    the 19 strategy leg templates + strike math
    ticket.py        the working order ticket (validate, price, persist)
    quotes.py        DXLink streaming quotes + REST snapshot fallback
    orders.py        dry-run/review -> submit -> cancel, with audit log
    pnl.py           realized (transactions, FIFO) + unrealized (marks)
    engine.py        lazy singleton wiring the above together
    web_api.py       JSON adapters for the dashboard's trading routes

Deliberately no imports at package level: importing `trading` must stay
free of side effects (and of the `requests`/`websocket` dependencies) so
the dashboard can import submodules lazily.
"""
