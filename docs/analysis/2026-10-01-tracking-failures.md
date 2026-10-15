# Tracking failures across Tom, Linda and Alice — baseline (2026-10-01)

**Purpose.** A baseline taken before fixing the "agents don't keep track"
problem. After the fix has been in daily use for **at least two days**,
re-run the measurements below on the new logs and fill in the *After*
columns.

**Window.** `logs/session_2026-09-22.log`, `_09-24`, `_09-27`, `_09-29`.
That covers 2026-09-22 12:34 → 2026-10-01 14:57: 8 active days, 1,178 user
turns. Every persona ran on DeepSeek (Tom, Linda and Bob on V4 Pro, Alice on
V4 Flash), except a few Opus turns the user asked for.

**Reproduce.** Same rules, any window:

```
python -m scripts.audit_sessions logs/session_2026-09-2*.log --to "2026-10-01 14:57:39"   # reproduces this baseline exactly
python -m scripts.audit_sessions logs/session_*.log --from <first day after the fix>
```

The 9/29 log kept growing after this baseline was taken, so a day-only `--to`
over-counts; bounds accept a full timestamp.

The script ([scripts/audit_sessions.py](../../scripts/audit_sessions.py)) is
read-only. Its correction regex is frozen at this baseline; changing it
breaks the comparison.

---

## Summary

The user's hypothesis is confirmed. Tom (campaigns and strategies), Linda
(behaviour patterns across sessions) and Alice (the daily list) are each
expected to track something ongoing. None of them has a persistent working
record of its own. All three depend on the same two mechanisms:

- relevance-ranked retrieval, which has to win a place again every turn
- asking Bob to save a note

Two-thirds of the corrected answers came from whatever happened to be in the
prompt, with no lookup. Saving notes through Bob interrupted the user, put
notes in the wrong folders, created duplicates, and produced notes that
couldn't be renamed or found. The stored facts fragment and contradict each
other.

Linda's agreed fix, the "bracket" (read a record at the start of each
session, write it at the end), was designed 2026-09-22 and is still not in
her prompt: `brain/agents.py` last changed her persona 2026-08-20. Her
morning check-in, which *is* in the prompt, happened on 5 of 5 session
days. That shows a standing instruction plus a known place to read and write
works.

Two outages made things worse:
- **9/29:** DeepSeek ran out of credit.
- **10/01:** the question-rewriting step failed 78 times, and fact saving
  stalled for about two hours.

## Baseline metrics

| Metric | Before (9/22–10/01) | After |
|---|---|---|
| Tom: user corrections / turns | 97 / 682 (**14.2%**) | |
| Linda: user corrections / turns | 38 / 261 (**14.6%**) | |
| Alice: user corrections / turns | 27 / 170 (**15.9%**) | |
| Corrected answers given with no tool call (Tom / Linda / Alice) | 55% / 74% / 74% | |
| Note-save interruptions ("your note is ready to file") | 40 (**5.0/day**) | |
| Question-rewrite failures | 91 (78 of them on 10/01) | |
| Turns with pinned facts when the rewrite succeeded / failed | 64% / **0%** | |
| Fact-extraction failures | 33 | |
| Turns failed outright | 3 (DeepSeek 402, 9/29) | |
| Working-record writes per day (Tom / Linda / Alice) | 0 / 0 / 0 (did not exist) | |
| Working-record blocks truncated | — | |

Counted by hand (re-check manually):

| Metric | Before | After |
|---|---|---|
| Days with an Alice list vs. lists actually saved | 4 days (9/22, 9/23, 9/27, 9/28) / 2 saved (the 9/28 list never saved) | |
| Linda opened with the check-in | 5 / 5 days | |
| Linda read the previous session's record at session start | 0 days | |
| Linda wrote an end-of-session record under a consistent title | 2 on 9/22, then titles drifted | |
| Separate `plan:campaign-diagonal-*` facts in Tom's store | ~12, two of them contradicting each other on the long leg | |
| Misfiled or duplicate notes (spotted in the logs) | ≥ 6 | |

## Incidents to re-test after the fix

Each one should *not* recur. Note any repeat, with its timestamp.

**Tom**
- [ ] 9/29 17:21: the user had to remind Tom of the day's planned work ("why didn't you remind me?").
- [ ] 9/30 09:58–10:08: Tom couldn't produce the campaign-diagonal rules and gave three different long-leg placements in three minutes.
- [ ] 9/30 16:14: Tom reviewed the user's trades without the plan and made up a reason for them.
- [ ] 10/01 12:00: an expiring position was spotted by the user, not Tom.
- [ ] 10/01 13:58: Tom said he "can't remember to do this every day."
- [ ] 9/27, 9/28: Tom got the market day or open/closed status wrong.

**Linda**
- [ ] 9/22 15:51, 9/24 08:49, 10/01 14:55: the user had to ask why Linda wasn't keeping session continuity.
- [ ] 9/22 15:53: Linda misremembered which sessions had taken place.
- [ ] 9/28 13:27: a Bob note handoff interrupted the coaching session.

**Alice**
- [ ] 9/23 15:58, 22:29: Alice sent a single list item to Bob mid-day against the user's wishes.
- [ ] 9/28: the day's list was built and never saved.
- [ ] 9/27 13:16–14:10: Alice kept sending the user to Tom for an answer neither of them had.

## Root causes, ranked

1. **No persistent working state per persona.** Ongoing context exists
   only through per-turn retrieval or a note, and nothing is read at the
   start of a session or written at the end.
2. **The only way to write is through Bob.** Each save waits on a spoken
   folder dialogue, so it interrupts the conversation, can land in the wrong
   folder or be duplicated, can't be renamed, and isn't searchable until the
   user answers.
3. **Facts fragment and contradict.** About 12 plan facts with no single
   current record. Facts are extracted from the assistant's own wrong
   statements. No link from a trade to the plan it follows.
4. **The question rewrite is a single point of failure.** When it fails,
   nothing gets pinned.
5. **Memory steps depend on one provider with no backup and no time
   limit.** The 9/29 credit outage; the ~900 s extraction hangs on 10/01.
6. **Time and market status are left to the model.** Time stamps carry no
   timezone, and `get_quote` gives no "market closed" signal.

Not a cause: the `archive search … hnsw` error (1,193 lines). It comes from
the legacy 27-row `conversations` collection; the per-persona memories work.

## Planned fixes (to be measured)

- [x] A per-persona working-state store (2026-10-06, branch
      `feat/working-record`). The persona writes to it directly on concrete
      triggers, it's shown in the prompt every turn regardless of ranking
      (cached, ≤ 8,000 chars), and it's authoritative over the Background.
      Tom's instance is the campaign journal, Linda's the Coaches Notebook
      (her dashboard persona merged into code), Alice's the daily list.
      Tom's campaigns link to trade-log fills by contract (`link_trades`;
      `log_trade` names the campaign a new fill may belong to). Linda's
      background safety-net pass ships only if her writes fall short.
- [x] Direct note saves, with the folder supplied by the caller (no spoken
      folder dialogue for background saves) (2026-10-06). Same day:
      `ask_agent` goes through only when the user names the other persona,
      so nobody hands work to Bob on their own initiative. Still open: a
      rename tool.
- [x] A backup model and hard time limits for the question rewrite and fact
      extraction (2026-10-06): Haiku / Sonnet standbys, a 60 s extraction
      timeout, automatic return after a 2-minute cooldown, one spoken notice
      per switch; the conversation's balance failover now returns too.
      Still open: logging tool results.
- [ ] A market-hours helper: `get_quote` reports the index value plus
      "market closed", and the time stamp carries the timezone.

## Comparison (fill in ≥ 2 days after the fixes land)

- Fix landed on: ____ Window measured: ____ → ____
- Script output pasted below.
- Verdict per root cause: fixed / partly fixed / not fixed, with the evidence.
