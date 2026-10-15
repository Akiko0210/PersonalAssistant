"""Real trading for the voice agent, on two brokers.

Self-contained variant package (CLAUDE.md's convention): multi-leg option
strategy building, live quotes, order review/submit/cancel, and
realized/unrealized P&L, on Charles Schwab (Tom) and tastytrade (Bonny).
Nothing here is imported by the rest of the app except through
`tools/trading_tools.py` (the voice surface) and the dashboard's lazily
imported trading routes — see TRADING_PLAN.md for the architecture and
TRADING_RESEARCH.md for the API research it is built on.

Module map:
    main.py          the broker registry: key -> variant module
    tastytrade.py    VARIANT: OAuth2 refresh-token client, DXLink wiring,
                     chain/order/position/transaction parsing, order payload
    schwab.py        VARIANT: three-legged OAuth2 with the 7-day sign-in,
                     REST quotes (no stream), its own parsing and payload
    config.py        env/credentials, base URLs, per-broker paths, constants
    models.py        dataclasses + BrokerError shared across the package
    symbols.py       OCC option symbols, futures symbology, spoken forms
    chains.py        the per-broker option-chain cache
    strategies.py    the 19 strategy leg templates + strike math
    ticket.py        the working order ticket (validate, price, persist)
    quotes.py        the DXLink streaming service tastytrade.py wires in
    orders.py        dry-run/review -> submit -> cancel, with audit log
    pnl.py           realized (transactions, FIFO) + unrealized (marks)
    engine.py        one lazy engine per broker wiring the above together
    trade_log.py     SQLite log of thinkorswim/Schwab fills + fill-line parser
    statement.py     Schwab account-statement CSV -> orders, with tie-out
    trade_positions.py  minimal zero-net position grouping + P&L (pure)
    web_api.py       JSON adapters for the dashboard's trading routes

Deliberately no imports at package level: importing `trading` must stay
free of side effects (and of the `requests`/`websocket` dependencies) so
the dashboard can import submodules lazily.
"""
