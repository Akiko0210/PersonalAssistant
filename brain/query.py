"""Query understanding: the one small model call before retrieval.

Embeddings cannot resolve "the other one", "that list" or "this morning at
9:41" — the first two need the recent turns, the third is a filter, not a
similarity (the exchanges it means sit at cosine 0.02-0.09 against it,
2026-09-17). So before the stores are searched, a fast model reads the last
few exchanges and the utterance and returns what the stores can act on: a
self-contained query and a time window when a time was named.

Never raises and never holds a turn for long: a failure, a timeout or an
unparseable reply falls back to the raw utterance, which is what every turn
retrieved on before this existed.
"""

import logging
import time
from dataclasses import dataclass

import config as cfg
from brain.llm.plain import ask
from lib.dates import window_epochs
from lib.llm_json import first_object

log = logging.getLogger("query")

REWRITE_MAX_TOKENS = 400  # a JSON object with one sentence in it


def rewriter(client, model=None):
    """The rewrite's model call, bound to `client`: QUERY_MODEL (or the
    failover's standby), under the timeout that keeps a slow answer from
    holding the turn."""
    return lambda system, prompt: ask(client, model or cfg.QUERY_MODEL, system,
                                      prompt, max_tokens=REWRITE_MAX_TOKENS,
                                      timeout=cfg.QUERY_TIMEOUT_S)


@dataclass
class Query:
    text: str              # what the stores are searched with
    window: tuple = None   # (lo, hi) epoch seconds when a time was named
    source: str = "raw"    # "model" when the rewrite was used; "raw" on fallback


SYSTEM = (
    "You prepare a voice assistant's memory lookup. You are given the recent "
    "exchanges between the user and the assistant, the current local time "
    "and the user's latest utterance (a speech transcript; it may contain "
    "recognition errors). Reply with JSON only, no prose:\n"
    '{"query": "...", "since": null, "until": null}\n'
    "- query: the utterance rewritten to stand alone, so that a search over "
    "past exchanges finds what it refers to — resolve pronouns and references "
    "(\"the other one\", \"that\", \"it\") from the recent exchanges, and keep "
    "the user's own specific words (names, tickers, numbers, item text). If "
    "the utterance refers to nothing earlier, return it unchanged. Never add "
    "topics the user did not raise.\n"
    "- since / until: ISO local date or datetime (2026-09-17 or "
    "2026-09-17T09:00) ONLY when the user refers to a time — 'this morning', "
    "'yesterday around 9:20', 'last week', 'earlier today'; otherwise null. "
    "Resolve relative words against the current time given. A clock time "
    "('around 9:41', 'at 3') means from half an hour before it to half an "
    "hour after; a part of a day ('this morning', 'last night') means that "
    "part of that day only, never past the current time; a day means the "
    "whole day. A date that merely names a thing ('today's list', 'Friday's "
    "trade') is not a time reference — leave since/until null and let the "
    "query carry it."
)


def understand(ask, utterance, recent, *, now) -> Query:
    """`ask(system, prompt) -> str` is the model call (rewriter() bound to a
    client; tests bind a string). `recent` is the last few exchanges as
    'role: text' lines; `now` the current stamp as the history writes it."""
    prompt = (f"Current time: {now}\n\n"
              "Recent exchanges (oldest first):\n"
              + ("\n".join(recent) if recent else "(none)")
              + f"\n\nLatest utterance:\n{utterance}")
    t0 = time.monotonic()
    try:
        data = first_object(ask(SYSTEM, prompt))
    except Exception as e:  # noqa: BLE001 - the turn retrieves on the raw words
        log.warning("query understanding failed (%s); using the utterance", e)
        return Query(text=utterance)
    text = " ".join(str(data.get("query") or "").split()) or utterance
    try:
        window = window_epochs(since=data.get("since") or None,
                               until=data.get("until") or None)
    except (TypeError, ValueError):
        window = None  # a malformed date is no date
    q = Query(text=text, window=window, source="model" if data else "raw")
    log.info("query %r -> %r window=%s (%.0f ms)",
             utterance[:60], text[:80], "yes" if window else "no",
             (time.monotonic() - t0) * 1000)
    return q
