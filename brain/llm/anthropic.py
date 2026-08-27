"""Anthropic provider variant.

Deliberately thin: the whole engine already speaks the anthropic SDK — that IS
the common part (DeepSeek rides the same wire format) — so this file holds
only Anthropic *account/endpoint* specifics, which today is default client
construction. Future Anthropic-only wiring (a base_url override, default
headers) lands here.

Name note: this module shadows nothing at runtime — under Python 3 absolute
imports, `import anthropic` (here and in main.py) always resolves to the SDK,
never to this sibling file. The one way to break that is running a file in
brain/llm/ directly as a script (which puts this directory first on sys.path);
don't add a __main__ block to any of them.
"""

import anthropic


def make_client():
    return anthropic.Anthropic()  # ANTHROPIC_API_KEY from the environment
