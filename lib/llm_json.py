"""The one JSON object in a model's reply. Models asked for "JSON only" still
wrap it in prose or a code fence now and then; the two callers (query
understanding, fact extraction) both need the object and nothing else, and a
parse failure must read as "no answer", never as an exception on the turn.
Leaf: stdlib only.
"""

import json


def first_object(text) -> dict:
    """The first {...} in `text` as a dict, or {} when there is none or it
    does not parse. Brackets inside strings are not special-cased: the
    outermost braces of a well-formed object are what a model emits, and a
    malformed one is a miss either way. Parsed non-strictly: a model writing
    a list as a value puts real line breaks between the items, which strict
    JSON rejects — six of 303 extractions failed that way (2026-09-22)."""
    text = str(text or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1], strict=False)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}
