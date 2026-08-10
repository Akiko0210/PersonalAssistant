"""Parser for the YAML-ish note frontmatter (--- key: value lines ---).

Its own stdlib-only module because BOTH notes.py and the dashboard need it,
and the dashboard must never import notes.py (which drags in chromadb + the
embedding stack); the two used to carry byte-identical copies instead.
"""

import re


def parse_frontmatter(text):
    """Parse the frontmatter block save_summary writes. Returns
    ({fields}, body). Tolerant: missing or malformed frontmatter yields
    ({}, whole text)."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not m:
        return {}, text
    fields = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields, text[m.end():]
