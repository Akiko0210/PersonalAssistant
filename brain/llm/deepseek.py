"""DeepSeek provider variant: the Anthropic-compatible endpoint.

DeepSeek models ride the engine's one code path — their endpoint speaks the
Messages API through the same anthropic SDK — so this file owns only what is
DeepSeek's alone: the base URL, the API key, and the known behavioural
differences of the compatible endpoint:
  - `cache_control` breakpoints are accepted but ignored (no prompt caching);
  - thinking `budget_tokens` is accepted but ignored (reasoning is per-model);
  - image content blocks are rejected (text only).
None of these need code branches today — the engine's requests degrade
gracefully — but when one does, the branch belongs here.
"""

import os

import anthropic

API_KEY_ENV = "DEEPSEEK_API_KEY"
BASE_URL = "https://api.deepseek.com/anthropic"


def available() -> bool:
    """Whether a switch to DeepSeek can succeed right now (the key is set)."""
    return bool(os.environ.get(API_KEY_ENV))


def make_client():
    """An anthropic-SDK client aimed at DeepSeek. Raises RuntimeError when the
    key is missing: the set_conversation_model tool refuses first (spoken, via
    available()), so this covers config/dashboard edits that sidestep it."""
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set — add it to .env to use "
            "a DeepSeek model, or switch back to a Claude model.")
    return anthropic.Anthropic(base_url=BASE_URL, api_key=key)
