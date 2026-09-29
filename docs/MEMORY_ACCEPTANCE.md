# Memory redesign — what changed, what to expect, what to test

Branch `feat/retrieval-context`, 2026-09-22. Written for the acceptance pass
before this is trusted or merged. `main` still runs the old design; `git
checkout main` is the rollback (the new `facts_<persona>` collections are
simply ignored there).

## 1. What changed (in user terms)

| Before | Now |
|---|---|
| Each turn retrieved past *exchanges* by similarity. A list built over six turns had to be found six times. | Each finished exchange is also read by a small model that keeps **facts** current: a list is one fact holding all its items; adding item four *updates* it. Facts about the thing you named come first in the model's Background, whole. |
| Your words went straight to the search. "The other one?" found nothing; "this morning at 9:41" was a similarity, not a time. | A small model first **reads your words** against the last four exchanges: resolves pronouns, turns a named time into a filter (a clock time = ±30 min), names the tracked thing you mean. Falls back to your raw words on any failure. |
| Indexing ran before the reply was spoken. | Indexing and fact extraction run on a **background thread** after the reply is returned. |
| Two tools: `search_past_conversations`, `search_knowledge`. | One tool, **`recall`** (topic / time window / exchange ids). |
| — | Facts render as `<item source="fact" entity=… as_of=… from=…>`: the model is told they are derived and dated, and can read the exchanges behind one. |

Models: both new calls run on **DeepSeek Flash** (`QUERY_MODEL`, `FACTS_MODEL`,
Config page). Measured on the offline harness: 7/7 cases, 0 extraction
failures, same as the Sonnet baseline.

## 2. What to expect

**First boot on the branch.** One-time pass over every stored exchange (~592):
`facts: alice has 320 exchange(s) the extractor has not read; reading them in
the background`, then `facts: alice 25/320` … `facts: alice caught up — 320
read, 0 failed`. Roughly 1 s per exchange on Flash. The agent is usable
throughout. Later boots log nothing here unless something was left unread.

**Every turn adds one small call before retrieval** (~0.5–1 s). Log line:
`query 'and the other one?' -> 'the SPX butterfly's other leg' window=no
entities=['trade:spx-butterfly'] (123 ms)`. If it says `(raw)` the rewrite
failed or timed out and the turn used your words as spoken — that is the
designed fallback, not an error, but it should be rare.

**After every reply**, off the speaking path: `remembered exchange for alice:
facts +1 ~0 -0 (…ms)` — added / updated / superseded. Most turns are
`+0 ~0 -0` (nothing durable). A list item added should read `~1`.

**The Background line** now reads `context pull …: conversation 4/16 kept,
facts 1/9 kept (1 pinned), knowledge 0/0 kept`. `pinned` > 0 means the
rewrite named an entity and its fact went in first, uncut.

**Timing gap to know about.** Extraction takes ~1–3 s after a reply. If you
add an item and ask "what's on my list" within that gap, the fact may not
hold the item yet — but the exchange you just spoke is in the verbatim tail,
so the model still sees it. If it ever *loses* an item in that situation,
that is a bug.

**Cost.** Per turn: one Flash call. Per exchange: one Flash call. First boot:
~600 Flash calls once.

## 3. What to test — by voice, with the log open

Filter the session log for `query `, `context pull`, `remembered exchange`,
`facts:`. Each test names the line that proves the mechanism worked, not just
the answer.

| # | Say | Expect | Log proof |
|---|---|---|---|
| 1 | *(Alice)* "What was on my list for September 18th?" | All six items (dashboard/Mike Douglas, transcripts, Tasty Live/Tom Preston, Costa 2525/Thunderbird, meditation session). This is the original incident. | `entities=['list:2026-09-18']`, `facts … (1 pinned)` |
| 2 | Build a new list today, one item per turn, six items, with one correction mid-way ("no — that should be *allocate*, not duplicate"). Then: "Read me my list." | All six, with the correction applied, in order. | Each item turn: `remembered … ~1` (an update, not `+1`). The read-back: `(1 pinned)` |
| 3 | Immediately after adding an item (within 2 s): "What's on my list now?" | The new item is included. | Either `~1` landed first, or the item is visible in the verbatim tail — the answer must include it either way |
| 4 | "Drop item two." Then "What's on my list?" | The list without item two. | `remembered … ~1`; the fact's value no longer has it |
| 5 | Tomorrow: "What was on yesterday's list?" | Yesterday's list, not today's. | `entities=['list:<yesterday>']` — a dated key, not `list:today` |
| 6 | "What did we talk about this morning around 9:40?" | The exchanges from ~9:10–10:10, read back in order. | `window=09/23 09:10-09/23 10:10`; `context pull` conversation count small |
| 7 | *(Tom)* "How did the SPX butterfly do?" … next turn: "And the other one?" | The second answer is about the other trade, not a re-answer of the first. | Second `query` line shows a rewritten query naming the trade |
| 8 | "Read me exactly what I said when I added the Costa email item." | The verbatim exchange, with its time. | `tool_use recall` with `ids=[…]` or a topic |
| 9 | "Save my list as a note" (after test 2). | Bob's note contains every item — the 9/18 failure was exactly this. | `(1 pinned)` on the turn that delegates |
| 10 | *(Tom)* "What's on my list?" | Tom does not know Alice's list and offers to ask her. | `facts 0/…`, no Alice entity keys in Tom's `query` line |
| 11 | Ask about something never discussed ("what did I decide about the garage?"). | "Nothing on that" — no invented fact. | `facts 0/0 kept`, `(0 pinned)` |
| 12 | Let Alice say something wrong, correct her, then ask again later. | Your correction wins. | The `remembered` line after the correction shows `~1` or `-1`, not `+0 ~0 -0` |
| 13 | Restart the agent. | Boot logs `facts: … caught up — 0 read` or nothing; no re-reading. | — |

Tests 1, 2, 9 are the ones that decide whether this shipped. 5 and 12 are
where the extractor is most likely to disappoint.

## 4. Known soft spots (seen on the harness, not fixed)

- The extractor sometimes records the **assistant's own claims** as facts
  (a `problem:memory-loss` fact quoting Alice's wrong 9/18 diagnosis). The
  prompt tells it not to; it does not always obey. Such facts are dated and
  labelled derived, and the model is told your word stands.
- **Entity keys vary** between runs (`todo:` vs `list:`, `person:amar` vs
  `person:amarjargal`). A list can fork into two keys if the extractor names
  a new one. The rewrite sees the keys dated and is told to prefer the most
  recent; a fork would show as a stale answer.
- **Superseded facts are not retrievable** — "what was on my list before I
  removed X" is answered from the exchanges (recall by time), not from
  facts.
- Facts about *you* live in the persona that learned them; Alice's do not
  reach Tom (the same isolation rule as everything else).
- Old July summaries still lose to recent chatter (the `open_old_summary`
  stretch case; recency policy, unchanged).

## 5. If it goes wrong

- A bad or forked fact: fix the prompt in `brain/facts.py`, then (agent off)
  `python -m scripts.rebuild_facts --owner alice` — facts are regenerated from
  the exchanges, never edited by hand.
- To measure a change before trusting it: `python -m scripts.eval_retrieval
  --backfill --extract --understand --owner alice` runs the whole pipeline on
  a copy and replays the 9/17–9/18 questions; add a case to
  `scripts/eval_cases.json` for anything that fails by voice.
- To go back: `git checkout main`. Nothing in `data/` needs undoing.
