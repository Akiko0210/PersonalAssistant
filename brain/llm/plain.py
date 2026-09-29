"""One plain model call, text in, text out.

The seam behind query understanding (brain/query.py) and fact extraction
(brain/facts.py): each binds its own model and limits here and hands the
bound call down, so neither module knows a client, and the agent and the
offline scripts (the eval harness, the fact rebuild) make the very same call.
Lives beside the providers because the client is theirs; imports none of
them — it is handed one.
"""

import config as cfg


def ask(client, model, system, prompt, *, max_tokens, effort=None, timeout=None) -> str:
    """The text of one messages.create. A timeout gets no retries: on the
    turn's critical path a slow answer is worse than none, and the caller
    falls back to what it had. A reply cut at max_tokens is an error, not a
    text: both callers parse JSON, and a cut object read as "no answer" in
    a log line that blamed the model's format (harness, 2026-09-22)."""
    if timeout:
        client = client.with_options(timeout=timeout, max_retries=0)
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": prompt}],
        **cfg.thinking_kwargs(model, effort))
    if getattr(resp, "stop_reason", None) == "max_tokens":
        raise RuntimeError(f"{model} reply cut at max_tokens={max_tokens} "
                           "(thinking shares the budget)")
    return "".join(b.text for b in resp.content if b.type == "text").strip()
