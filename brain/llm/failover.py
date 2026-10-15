"""Fail over to a standby model, say so once, and come back on our own.

Three calls depend on one provider each — the conversation, the query
rewrite before retrieval, and the fact extraction after the reply — and on
2026-09-29 and 2026-10-01 the last two failed for hours in silence: 90
rewrite failures (timeouts and an empty balance) and extraction hangs of up
to 901 s, with no word to the user and no way back short of a restart. The
conversation failed over, but stuck on the standby until restart.

One Failover per call site. While the default's provider is cooling down
(FAILOVER_COOLDOWN_S after its last failure) calls go straight to the
standby; then the default is probed with the next real call, and a success
moves back. Only a change of state is announced — the switch, the return —
never each retry; a failed probe restarts the cooldown silently.
Thread-safe: extraction runs on the memory worker, delegations on their own
threads.
"""

import logging
import threading
import time

import anthropic

import config as cfg

log = logging.getLogger("llm")


def is_balance_error(e) -> bool:
    """True when an API error means the account is out of money. The two
    providers say it differently — DeepSeek a 402 "Insufficient Balance",
    Anthropic a 400 "credit balance is too low" — and the SDK types neither,
    so the status/message is all there is to go on."""
    return (isinstance(e, anthropic.APIStatusError)
            and (e.status_code == 402
                 or "credit balance is too low" in str(e).lower()))


def _reason(e):
    """Why a provider failed, in words the user hears."""
    if is_balance_error(e):
        return "has insufficient balance"
    if isinstance(e, anthropic.APIConnectionError):  # includes timeouts
        return "isn't responding"
    if isinstance(e, RuntimeError) and "API_KEY" in str(e):
        return "isn't available"
    return "is failing"


def other_provider(model):
    """The cheap model of the provider `model` is not on (FAILOVER_MODELS)."""
    mine = cfg.model_provider(model)
    return next((m for p, m in cfg.FAILOVER_MODELS.items() if p != mine), None)


class Failover:
    def __init__(self, what, standby, notify, *, clock=time.monotonic):
        self.what = what            # "the conversation", "memory lookups", ...
        self._standby = standby     # model id, or (model) -> model id
        self._notify = notify       # (text) -> None; the caller speaks it
        self._clock = clock
        self._down = {}             # provider -> clock() of its last failure
        self._lock = threading.Lock()

    def standby(self, model):
        """The model that takes over from `model`, or None."""
        alt = self._standby(model) if callable(self._standby) else self._standby
        return alt if alt and cfg.model_provider(alt) != cfg.model_provider(model) else None

    def pick(self, model):
        """`model`, or its standby while `model`'s provider cools down."""
        with self._lock:
            since = self._down.get(cfg.model_provider(model))
        if since is not None and self._clock() - since < cfg.FAILOVER_COOLDOWN_S:
            return self.standby(model) or model
        return model

    def failed(self, model, err):
        """`model` failed: stamp its provider down and announce the switch if
        this is news. Returns the standby, or None when there is none."""
        alt = self.standby(model)
        if alt is None:
            return None
        provider = cfg.model_provider(model)
        with self._lock:
            news = provider not in self._down
            self._down[provider] = self._clock()
        log.warning("%s: %s failed (%s); using %s", self.what, model, err, alt)
        if news:
            self._notify(f"{cfg.provider_label(model)} {_reason(err)}, so "
                         f"{self.what} switched to {cfg.provider_label(alt)}.")
        return alt

    def ok(self, model):
        """A call on `model` worked: if its provider was down, it is back."""
        with self._lock:
            was_down = self._down.pop(cfg.model_provider(model), None) is not None
        if was_down:
            log.info("%s: %s answering again", self.what, model)
            self._notify(f"{cfg.provider_label(model)} is answering again, so "
                         f"{self.what} moved back to it.")

    def clear(self):
        """Forget every failure: the user picked a model by hand."""
        with self._lock:
            self._down.clear()

    def wrap(self, make_ask, default):
        """A (system, prompt) -> str call on `default` with this failover:
        `make_ask(model)` binds the call to one model. A failure on the
        standby (or with no standby) is re-raised for the caller to absorb,
        as it already does for any failure."""
        def call(system, prompt):
            model = self.pick(default)
            try:
                text = make_ask(model)(system, prompt)
            except Exception as e:  # noqa: BLE001 - any failure is a reason to switch
                alt = self.failed(model, e) if model == default else None
                if alt is None:
                    raise
                return make_ask(alt)(system, prompt)
            self.ok(model)
            return text
        return call
