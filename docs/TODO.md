# TODO — planned work

Future work, not yet started. Each item is written so it can be picked up cold:
what it should do, why, where it lands in the existing code, and the steps in
order. Check boxes off as they land; delete a section once it ships and is
documented in `PROJECT.md`.

Current state this plan builds on:

- One Chroma database at `data/chroma`, two collections — `knowledge`
  (ingested books/PDFs/course videos, `knowledge.py`) and `conversations`
  (archived conversation summaries, `memory.py`). Collection names live in
  `config.py:93` and `config.py:105`.
- Three personas in `agents.py` — Alice (general), Bob (notes/memory), Tom
  (trading). Per-turn hats over one shared history, not separate agents.
- Tools are one decorated function each under `tools/`, registered by import
  order in `tools/__init__.py`; per-persona allowlists filter them via
  `api_tools(include=...)`.
- The dashboard is a stdlib `BaseHTTPRequestHandler` server (`dashboard.py`)
  serving `dashboard/index.html`, `app.js`, `style.css` plus `/api/*` JSON
  routes.

---

## 1. A trading-journal collection for Tom, plus focus mode

### Goal

Two things, related:

1. **Separate Chroma collections for Tom** — in particular a *trading journal*
   collection holding recent trading information (fills, adjustments, thesis
   notes, what happened and why), kept apart from the reference-material
   `knowledge` collection. Reference material is stable and impersonal; the
   journal is recent, personal, and changes daily. Mixing them makes a search
   for "how did my last diagonal go" return textbook definitions.
2. **Focus mode** — the ability to tell Tom "focus on my double diagonals" (or
   butterflies, or ES credit spreads) and have every subsequent retrieval be
   scoped to that strategy until focus is cleared. It works like switching
   memories: fewer distractions in the context, better answers.

### Design sketch

**Collections.** Add to `config.py` alongside the existing two:

```python
JOURNAL_COLLECTION = "trading_journal"   # recent trades, adjustments, outcomes
```

Build `trading/journal.py` as a `TradingJournal` store modelled on
`KnowledgeStore` — lazy `_ensure_chroma()`, same `PersistentClient(CHROMA_DIR)`,
same `SentenceTransformerEmbeddingFunction(cfg.EMBED_MODEL)`, so there is one
embedding model and one database, just another collection. Do **not** fork the
Chroma client per store; share it (a small `chroma_client()` helper in
`knowledge.py` or a new `vectors.py` both stores import) so the embedding model
is loaded once, not three times.

**Entry shape.** Every journal entry carries metadata that focus mode can filter
on. Chroma `where=` clauses do the scoping — no separate collection per
strategy:

| field | example | source |
|---|---|---|
| `strategy` | `double_diagonal`, `butterfly`, `credit_spread` | classified at write time |
| `underlying` | `SPX`, `RUT`, `/CL` | from the ticket / trade line |
| `date` | `2026-08-06` | write time |
| `kind` | `fill`, `adjustment`, `note`, `outcome` | which writer produced it |
| `ticket_id` | opaque id | links entries of one trade together |

Use a canonical strategy vocabulary — reuse or extend the names already in
`trading/strategies.py` rather than inventing a second taxonomy, so a strategy
built by voice and a journal entry about it use the same word.

**Focus.** Focus is per-session state, not persisted config. Store it on
`ToolContext` in `tools/__init__.py` next to `active_agent` and `convo_model`:

```python
focus: dict = None   # e.g. {"strategy": "double_diagonal", "underlying": "SPX"}
```

Every retrieval tool reads `ctx.focus` and folds it into its `where=` filter.
Focus must also be *visible*: fold a line like "Focused on: double diagonals
(SPX)" into Tom's system prompt each turn, the way the persona text is folded in
now, so the model knows its own retrieval is narrowed and can say so. Without
that the model silently misses material and confidently reports nothing found.

Design decision to make before coding: **does focus narrow `search_knowledge`
too, or only the journal?** Recommendation — yes, both, but as a *soft* filter
on `knowledge` (rank-boost / fall back to unfiltered when the filtered search
returns nothing) and a *hard* filter on the journal. Reference chunks are not
reliably tagged by strategy; journal entries are, because we write them.

### Steps

- [ ] Extract a shared `chroma_client()` + embedding-function helper so
      `knowledge`, `conversations`, and the new journal share one client and one
      loaded embedding model.
- [ ] Add `JOURNAL_COLLECTION` to `config.py`.
- [ ] Write `trading/journal.py`: `add(text, *, strategy, underlying, kind,
      ticket_id, date)`, `search(query, *, focus=None, n=...)`,
      `recent(days=..., focus=None)`, `forget(ticket_id)`.
- [ ] Decide the strategy vocabulary; add a `classify_strategy(text)` helper
      (deterministic mapping first, model call only as fallback).
- [ ] Wire the store into `ToolContext` as `journal`, constructed where `kb` and
      `memory` are constructed in `llm.py:Claude.__init__`.
- [ ] New `tools/journal_tools.py`:
      - `log_trade_note` — write an entry (Tom calls this when the user says
        "note that I rolled the short put up a week").
      - `search_trading_journal` — semantic search, focus-aware, cites date and
        underlying.
      - `recent_trading_activity` — time-scoped list, focus-aware, for "what
        have I been doing in RUT this month".
      - `set_focus` / `clear_focus` / `get_focus`.
- [ ] Import the module in `tools/__init__.py` and add the new names to Tom's
      allowlist in `agents.py`. Keep them **out** of Alice's and Bob's.
- [ ] Fold the active focus into Tom's system prompt each turn.
- [ ] Auto-journal on submitted orders: when `submit_order` succeeds
      (`tools/trading_tools.py` / `trading/orders.py`), write a `kind="fill"`
      entry with the ticket's strategy and underlying. This is what makes the
      journal fill itself instead of depending on the user remembering to
      dictate notes.
- [ ] Backfill: a one-shot script under `scripts/` that reads existing Discord
      trade lines (`discord_data.py`) and the `Trading` note folder into the
      journal, so focus mode has history on day one.
- [ ] Dashboard: a Journal panel — recent entries, filter by strategy, and a
      delete for a mis-logged entry. Follow the existing narrow-column,
      large-type layout; no wide tables.
- [ ] Tests in `tests/test_journal.py` and `tests/test_focus.py`: metadata
      filtering actually excludes off-strategy entries, focus survives across
      turns, `clear_focus` restores full retrieval, a persona switch away from
      Tom and back does the intended thing with focus (decide: keep or clear —
      recommend keep, and say so aloud on switch back).

### Open questions

- Should focus persist to `data/agent_state.json` and survive a restart? Leaning
  no — a stale focus that outlives the session is exactly the failure mode that
  makes retrieval look broken.
- Multiple simultaneous focuses ("diagonals and butterflies")? The `where=`
  clause supports `$in`, so the store should accept a list from the start even
  if the tool schema initially takes one value.

---

## 2. A trading coach persona (fourth hat)

### Goal

A fourth persona whose job is not to execute or look things up but to *coach*:
review how trades were actually managed against the written plan, name repeated
mistakes, ask the questions a coach asks before a trade goes on, and hold the
user to their own rules.

Distinct from Tom on purpose. Tom is hands-on — quotes, tickets, orders, "what
does the book say". The coach is reflective and has no order tools at all. That
separation is also a safety property: a persona that reviews and second-guesses
should not be able to submit anything.

### Design sketch

Adding a persona is one dict entry in `agents.py` — that part is cheap. The
work is in what it can reach and what it says.

- **Name and aliases.** Pick a name Whisper transcribes reliably and that does
  not collide with `alice`/`bob`/`tom` aliases — `alias_map()` raises on a
  duplicate, so a collision fails at import/test time. Avoid names that rhyme
  with the existing three.
- **Model.** `sonnet`, like Tom. Coaching is the analysis-heavy case.
- **Voice.** Only Zira and David are installed on this machine (see Tom's
  comment in `agents.py`). A fourth persona therefore *must* reuse one at a
  distinct rate, or the install of another SAPI voice becomes a prerequisite.
  Note it in the entry either way so the next reader isn't confused.
- **Tools.** Read-only:
  `search_trading_journal`, `recent_trading_activity`, `get_focus`,
  `search_knowledge`, `get_positions`, `get_pnl`, `get_recent_trades`,
  `search_past_conversations`, `get_current_time`, `get_current_model`,
  `set_conversation_model`, `switch_agent`, `ask_agent`.
  **Explicitly not** `build_strategy`, `adjust_leg`, `set_order_terms`,
  `review_order`, `submit_order`, `cancel_order`, `clear_ticket`.
- **Persona text.** The coach's leverage is the gap between `TRADING_PLAN.md`
  plus the ingested plan PDFs (`knowledge/Diagonal trading plan V4.pdf`, `Amit
  ES Futures credit spread protocol V2.pdf`) and what the journal says actually
  happened. Write the persona to always ground a critique in a retrieved
  citation — plan says X, journal shows Y on this date — never in generalities.
- **Interaction with delegation.** `ask_agent` already exists, so Tom can ask
  the coach for a pre-trade check without moving the user, and the answer comes
  back as a spoken interjection. That is the most valuable path: coaching that
  arrives *before* the order, not only in review.

### Steps

- [ ] Choose the name; check it against `alias_map()`.
- [ ] Add the registry entry in `agents.py` (role, persona, tools, model,
      tts_voice, tts_rate).
- [ ] Extend Tom's persona text to say when to delegate to the coach.
- [ ] Confirm the dashboard's agents panel (`api_agents` / `save_agents`,
      `dashboard.py:455`) renders four personas without layout changes, and that
      `_OVERLAYABLE` editing works for the new one.
- [ ] Add a "pre-trade check" path: before `submit_order`, optionally consult
      the coach against the plan. Decide whether this is opt-in by voice
      ("check this against my plan first") or automatic above a size threshold.
      Recommend opt-in first — automatic consultation on every ticket adds
      latency to the one flow where latency is most annoying.
- [ ] Tests: extend `tests/test_agents.py` and `tests/test_hats.py` for the
      fourth hat; add an explicit assertion that no order-mutating tool appears
      in the coach's allowlist, so a future refactor can't quietly grant it
      execution.

### Open questions

- Should the coach get a scheduled/proactive mode — an end-of-week review that
  speaks unprompted? Out of scope for the first pass; note it and revisit once
  the journal has real data in it.

---

## 3. Web chat UI — test the agent by typing

### Goal

A chat page in the dashboard for exercising the agent without speaking. Voice is
the product, but voice is a slow and lossy way to test: Whisper mishears, TTS
takes real seconds, and reproducing a bug means saying the same sentence five
times. Typing makes iteration fast and makes transcripts exact.

### Design sketch

**Server.** `dashboard.py` is a stdlib `BaseHTTPRequestHandler`, so add:

- `GET /chat` (or a tab within `index.html` — see below) serving the UI.
- `POST /api/chat` — body `{"text": "...", "agent": "tom"}`, returns
  `{"reply": ..., "agent": ..., "tools": [...], "model": ...}`.
- `GET /api/chat/history` — the shared history, so the page reflects voice turns
  too.

The hard part is not the HTTP; it's **who owns the conversation**. Two options:

1. **Talk to the running voice agent.** The dashboard already detects it
   (`agent_running()`, `data/agent.lock`) and reads `data/agent_state.json` and
   `data/history.json`. A chat turn would have to be injected into the running
   process's loop — a request queue file the agent drains, or a small local
   socket. Upside: one true conversation, typed and spoken turns interleaved,
   and everything (personas, delegation, tickets) behaves exactly as in
   production. Downside: real IPC work, and turns must not race the audio loop.
2. **Run a headless `Claude` in the dashboard process.** Construct `llm.Claude`
   with the same stores and call `converse()` directly. Far less work, and it
   exercises the whole model/tool/persona path. Downside: a second history
   writer against `data/history.json` — must be either read-only or pointed at a
   separate history file, or two processes will clobber each other.

**Recommendation:** start with (2), pointed at a separate `data/chat_history.json`
and clearly labelled in the UI as a test conversation, because it delivers the
testing value in a fraction of the time. Keep (1) as a follow-up once the chat
page has proved itself. Whichever is chosen, `atomic_io.write_json_atomic` is
already there for the write side — use it, and never let two writers share one
history path.

**Client.** `dashboard/app.js` + `style.css`, matching the existing dashboard —
narrow single column, large type, large tap targets, no wide tables (the
dashboard is used with tunnel vision; the chat log must stay in one comfortable
reading column and must not require horizontal scanning).

Show per turn:

- who answered (persona name and colour), and the model used;
- the tool calls made, with arguments and results, collapsed by default —
  this is the main reason to build the thing, since tool behaviour is invisible
  over voice;
- the active focus (once §1 lands) and any pending ticket;
- errors verbatim, via `explain_error`.

**Safety.** The chat page can reach Tom, and Tom can submit real orders. Before
this ships, decide and implement one of: the chat context is hard-limited to
`TASTY_ENV=sandbox`; or order-submitting tools are excluded from chat turns; or
submission from chat requires a typed confirmation phrase. Do not ship a text
box that can place live trades as a side effect of "testing the agent". The
existing rule — `submit_order` only after a spoken `review_order` of the exact
current ticket — needs an explicit typed-path equivalent, not an accident.

### Steps

- [ ] Decide owner-of-conversation: headless `Claude` in the dashboard
      (recommended first pass) vs. IPC into the running voice agent.
- [ ] Decide and implement the trading guard for chat turns (above) — before
      any chat turn can reach Tom's tools, not after.
- [ ] Add `POST /api/chat` and `GET /api/chat/history` to `dashboard.py`'s
      route tables.
- [ ] Build the chat tab in `dashboard/index.html` / `app.js` / `style.css`:
      message list, input box, persona selector, tool-call disclosure.
- [ ] Stream or poll? Poll first — a tool-heavy turn can take many seconds and
      a silent page looks hung; a simple "thinking…" state plus polling is
      enough, and avoids adding SSE to the stdlib server.
- [ ] Show tool events (`ctx.events`, `flush_tool_events`) in the transcript.
- [ ] Add a "clear chat" control that resets only the chat history.
- [ ] Tests in `tests/test_chat_api.py`: a turn round-trips, persona selection
      is honoured, tool calls are reported, the trading guard actually blocks
      what it claims to block, and the chat history writer never touches
      `data/history.json`.

### Open questions

- Should typed turns appear in the voice agent's memory (i.e. be archived into
  the `conversations` collection)? Recommend no while it's a test surface —
  test chatter polluting real recall would be worse than the inconvenience.
- Is the chat page also intended as a real everyday input method (not just
  testing)? If yes, option (1) becomes the right architecture and the separate
  history is wrong. Worth answering before building, since it changes the
  design.

---

## Cross-cutting

- [ ] Update `PROJECT.md` and `README.md` as each item lands — they are the
      files `describe_project` answers from, so stale docs mean the agent
      misdescribes itself.
- [ ] Watch startup cost: a third Chroma collection and a fourth persona both
      add to boot. The shared-client change in §1 should keep the embedding
      model at one load; verify rather than assume.
- [ ] Ordering: §1 first (the journal and focus are what the coach reads and
      what the chat UI displays), then §2, then §3. §3 is independently useful
      and could be pulled forward if testing pain is the bigger problem right
      now.
