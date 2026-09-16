"""Per-turn Background: what the model gets to see about earlier
conversations and reference material, chosen by relevance and recency.

Every turn retrieves candidates from the persona's exchange index (dense +
BM25, brain/memory.py) and the knowledge base (dense, stores/knowledge.py),
fuses them into one ranking, and fills a character budget with the best. The
scorer follows the field's baseline rather than inventing one: relative-score
fusion of the dense and lexical lists (Weaviate's default), plus an additive
recency term with exponential decay (Park et al. 2023, the Generative Agents
memory stream). Round-level records with time labels are what LongMemEval
found reads best.

The block is the LAST system block: after the frozen system prompt, so the
cached static prefix survives it, and before the conversation, so the user's
question stays last (Anthropic's long-context guidance — documents at the
top, query at the end). Every pull is logged: the INFO line always, the
per-candidate table and the full block at DEBUG (cfg.CONTEXT_DEBUG_LOG),
because the thresholds in config.py are tuned from those lines, not guessed.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import config as cfg
from brain import history as hist
from stores.knowledge import cite

log = logging.getLogger("context")

HEADER = (
    "Background retrieved for this turn — earlier exchanges with this user and "
    "reference material, ranked by relevance and recency. It may not apply: "
    "ignore what doesn't. Past exchanges record what was said at the time, not "
    "current fact. Call search_past_conversations or search_knowledge only when "
    "you need more than this shows."
)


@dataclass
class Candidate:
    source: str            # "conversation" | "knowledge"
    id: str
    doc: str
    meta: dict
    sim: float = 0.0       # cosine from the dense list (0 when lexical-only)
    lex: float = 0.0       # BM25 relative to the turn's best lexical hit
    age_h: float = None    # hours since the record's time; None when unknown
    score: float = 0.0
    gate: bool = False     # passed a relevance gate; eligible for the budget
    text: str = ""         # cleaned, capped text as rendered
    chars: int = 0
    kept: bool = False
    note: str = ""         # why not kept: "gate" | "budget"


@dataclass
class ContextPull:
    query: str = ""
    block: str = ""
    candidates: list = field(default_factory=list)
    ms: float = 0.0


# --- scoring ----------------------------------------------------------------
def similarity(dist) -> float:
    """Cosine similarity from a Chroma distance. The collections use Chroma's
    default squared-L2 space and all-MiniLM-L6-v2 emits unit-norm vectors, so
    d = 2 - 2cos; clamped because HNSW distances drift past 2 by a hair."""
    return max(0.0, 1.0 - float(dist) / 2.0)


def recency(age_h) -> float:
    """Exponential decay of recency credit: 1 now, half at the half-life, 0
    for a record whose time is unknown (never boost what can't be dated)."""
    if age_h is None:
        return 0.0
    return 0.5 ** (max(0.0, age_h) / cfg.CONTEXT_RECENCY_HALF_LIFE_H)


def age_hours(meta, now=None):
    """Hours between `now` (epoch seconds) and a record's time: `epoch` on
    exchange records; the `date` of a legacy summary as the END of that local
    day (a summary covers the day; "A to B" ranges take B); None when neither
    parses."""
    now = time.time() if now is None else now
    epoch = meta.get("epoch")
    if isinstance(epoch, (int, float)):
        return (now - epoch) / 3600.0
    try:
        day = datetime.fromisoformat(str(meta.get("date") or "")[-10:])
    except ValueError:
        return None
    end = day.replace(hour=23, minute=59, second=59)
    return (now - end.timestamp()) / 3600.0


def when(meta) -> str:
    """Human label for a record's time: the exchange stamp as "1:47pm
    8/20/2026", a legacy summary's bare date, or "unknown date"."""
    return (hist.time_label(meta.get("ts"))
            or str(meta.get("date") or "unknown date"))


def age_label(age_h):
    if age_h is None:
        return None
    if age_h < 1:
        return "under an hour ago"
    n = int(age_h) if age_h < 48 else int(age_h // 24)
    unit = "hour" if age_h < 48 else "day"
    return f"{n} {unit}{'s' if n != 1 else ''} ago"


def fuse(dense, lexical, *, now=None, exclude_ids=()):
    """One ranking from a dense list (Chroma distances) and a lexical list
    (BM25 scores) over the same records — relative-score fusion. The dense
    term is the cosine itself (already 0..1, so the absolute gate means the
    same thing every turn); the lexical term is BM25 relative to the turn's
    best lexical hit (a small personal corpus has no stable absolute scale);
    plus the recency term. A record passes if it clears EITHER gate: an exact
    ticker match is a hit even when the embeddings disagree, which is the
    whole point of the lexical list. Every candidate is returned, gated or
    not, so the log can show what was dropped and why; excluded ids (the
    exchanges already in the verbatim tail) are left out entirely."""
    exclude = set(exclude_ids)
    by_id = {}
    for h in dense:
        if h.id in exclude:
            continue
        c = by_id.setdefault(h.id, Candidate("conversation", h.id, h.doc, h.meta))
        c.sim = max(c.sim, similarity(h.score))
    best = max((h.score for h in lexical), default=0.0)
    for h in lexical:
        if h.id in exclude or best <= 0:
            continue
        c = by_id.setdefault(h.id, Candidate("conversation", h.id, h.doc, h.meta))
        c.lex = max(c.lex, h.score / best)
    for c in by_id.values():
        c.age_h = age_hours(c.meta, now)
        c.gate = (c.sim >= cfg.CONTEXT_MIN_SIMILARITY
                  or c.lex >= cfg.CONTEXT_MIN_LEXICAL)
        c.score = (c.sim + cfg.CONTEXT_LEXICAL_WEIGHT * c.lex
                   + cfg.CONTEXT_RECENCY_WEIGHT * recency(c.age_h))
    return sorted(by_id.values(), key=lambda c: -c.score)


# --- assembly ---------------------------------------------------------------
def _clean(doc, cap=None):
    """Collapse runs of whitespace within lines but keep the line structure
    ('user: …' / 'assistant: …' stay on their own lines)."""
    text = "\n".join(" ".join(line.split()) for line in str(doc).splitlines()
                     if line.strip())
    return text[:cap] if cap else text


def _fill(cands, budget, cap=None):
    """Greedy whole-document fill by score order; an oversize document is
    skipped, not truncated, so the next one that fits still gets in.
    Returns the unused budget."""
    for c in cands:
        if not c.gate:
            c.note = "gate"
            continue
        c.text = _clean(c.doc, cap)
        c.chars = len(c.text)
        if c.chars > budget:
            c.note = "budget"
            continue
        c.kept = True
        budget -= c.chars
    return budget


def render(candidates) -> str:
    """The Background block, or "" when nothing was kept. Each item carries
    what the model needs to weigh it: when an exchange happened and how long
    ago, or where a chunk came from."""
    kept = [c for c in candidates if c.kept]
    if not kept:
        return ""
    items = []
    for c in kept:
        if c.source == "conversation":
            attrs = f'source="conversation" when="{when(c.meta)}"'
            if (label := age_label(c.age_h)):
                attrs += f' age="{label}"'
            if c.meta.get("legacy"):
                attrs += ' note="summary from the shared archive, before per-persona memory"'
            if c.meta.get("tools"):
                attrs += f' tools="{c.meta["tools"]}"'
        else:
            attrs = f'source="knowledge" cite="{cite(c.meta)}"'
        items.append(f"<item {attrs}>\n{c.text}\n</item>")
    return HEADER + "\n\n" + "\n".join(items)


def build_context(query, *, owner, memory, kb, exclude_ids=(), focus=None,
                  now=None) -> ContextPull:
    """The Background for one turn. Never raises and always logs: a turn must
    not die on its own memory, so each store is queried under its own guard
    and a failure just means a smaller (or empty) block. Conversation
    candidates fill CONTEXT_CONVO_CHARS first; knowledge takes
    CONTEXT_KB_CHARS plus whatever conversation left over."""
    t0 = time.monotonic()
    pull = ContextPull(query=query)
    convo, chunks = [], []
    try:
        rows = memory.query_rows(query, cfg.CONTEXT_CANDIDATES + len(exclude_ids),
                                 owner)
        convo = fuse(rows.dense, rows.lexical, now=now, exclude_ids=exclude_ids)
    except Exception as e:  # noqa: BLE001 - retrieval must never cost the turn
        log.warning("conversation retrieval failed: %s", e)
    try:
        for h in kb.query_rows(query, cfg.CONTEXT_CANDIDATES, owner, focus):
            c = Candidate("knowledge", h.id, h.doc, h.meta, sim=similarity(h.score))
            c.score = c.sim
            c.gate = c.sim >= cfg.CONTEXT_MIN_SIMILARITY
            chunks.append(c)
    except Exception as e:  # noqa: BLE001
        log.warning("knowledge retrieval failed: %s", e)
    try:
        left = _fill(convo, cfg.CONTEXT_CONVO_CHARS, cap=cfg.CONTEXT_HIT_CHARS)
        _fill(chunks, cfg.CONTEXT_KB_CHARS + left)
        pull.candidates = convo + chunks
        pull.block = render(pull.candidates)
    except Exception as e:  # noqa: BLE001
        log.warning("context assembly failed; answering without Background: %s", e)
        pull.block = ""
    pull.ms = (time.monotonic() - t0) * 1000
    log_pull(pull)
    return pull


def log_pull(pull):
    """One INFO line per turn; at DEBUG, one row per candidate and the block
    itself. This is the evidence the thresholds get tuned from."""
    conv = [c for c in pull.candidates if c.source == "conversation"]
    chunks = [c for c in pull.candidates if c.source == "knowledge"]
    log.info("context pull %r: conversation %d/%d kept (%d chars), knowledge "
             "%d/%d kept (%d chars), %.0f ms",
             pull.query[:80],
             sum(c.kept for c in conv), len(conv),
             sum(c.chars for c in conv if c.kept),
             sum(c.kept for c in chunks), len(chunks),
             sum(c.chars for c in chunks if c.kept), pull.ms)
    if not log.isEnabledFor(logging.DEBUG):
        return
    for c in pull.candidates:
        age = "?" if c.age_h is None else f"{c.age_h:.0f}h"
        log.debug("  %-12s %-32s sim=%.2f lex=%.2f age=%-7s score=%.2f "
                  "chars=%-4d %-11s %r",
                  c.source, str(c.id)[:32], c.sim, c.lex, age, c.score, c.chars,
                  "KEEP" if c.kept else f"drop:{c.note or '-'}",
                  " ".join(str(c.doc).split())[:80])
    if pull.block:
        log.debug("background block:\n%s", pull.block)
