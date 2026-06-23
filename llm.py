"""Claude integration: conversation (with note-access tools) and summarisation."""

import logging

import anthropic

import config as cfg

log = logging.getLogger("llm")

TOOLS = [
    {
        "name": "search_notes",
        "description": (
            "Semantic search across all saved notes. Use for questions like "
            "'what did I say about X' or to find notes on a topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_recent_notes",
        "description": (
            "List the most recent notes, newest first. Use for 'what's my latest "
            "note' or 'what have I recorded recently'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "How many to list (default 5)"}
            },
        },
    },
    {
        "name": "read_note",
        "description": "Read the full saved summary for a note id (e.g. note_2026-06-22_141500).",
        "input_schema": {
            "type": "object",
            "properties": {
                "note_id": {"type": "string", "description": "The note id to read"}
            },
            "required": ["note_id"],
        },
    },
]

SUMMARY_PROMPT = """You are summarising a spoken notetaking session. The text below is an \
automatic transcript and may contain disfluencies or recognition errors — clean it up \
sensibly without inventing content.

Respond in EXACTLY this format:

TITLE: <a short descriptive title, max ~8 words>
SPOKEN: <a 2-3 sentence spoken recap that will be read aloud to the user; plain sentences, no markdown>
---
## Summary
<a tight prose summary>

## Key Points
- <point>

## Action Items
- <action item, or "None">

Transcript:
"""


class Claude:
    def __init__(self, store):
        self.client = anthropic.Anthropic()
        self.store = store
        self.history = []

    # --- conversation --------------------------------------------------------
    def _dispatch(self, name, args):
        try:
            if name == "search_notes":
                return self.store.search_notes(args["query"])
            if name == "list_recent_notes":
                return self.store.list_recent_notes(int(args.get("n", 5)))
            if name == "read_note":
                return self.store.read_note(args["note_id"])
        except Exception as e:  # surface tool errors back to the model
            return f"Tool error: {e}"
        return f"Unknown tool: {name}"

    def converse(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        while True:
            resp = self.client.messages.create(
                model=cfg.CONVO_MODEL,
                max_tokens=cfg.CONVO_MAX_TOKENS,
                system=cfg.CONVO_SYSTEM,
                tools=TOOLS,
                thinking={"type": "disabled"},
                messages=self.history,
            )
            self.history.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "tool_use":
                results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        log.info("tool_use %s %s", block.name, block.input)
                        out = self._dispatch(block.name, block.input)
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": out,
                        })
                self.history.append({"role": "user", "content": results})
                continue

            return "".join(b.text for b in resp.content if b.type == "text").strip()

    # --- summarisation -------------------------------------------------------
    def summarize(self, transcript: str):
        resp = self.client.messages.create(
            model=cfg.SUMMARY_MODEL,
            max_tokens=cfg.SUMMARY_MAX_TOKENS,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": SUMMARY_PROMPT + transcript}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return self._parse_summary(text)

    @staticmethod
    def _parse_summary(text: str):
        title = "Untitled note"
        spoken = ""
        full = text
        if "---" in text:
            head, full = text.split("---", 1)
            full = full.strip()
        else:
            head = text
        for line in head.splitlines():
            if line.upper().startswith("TITLE:"):
                title = line.split(":", 1)[1].strip() or title
            elif line.upper().startswith("SPOKEN:"):
                spoken = line.split(":", 1)[1].strip()
        if not spoken:
            spoken = "I've saved your note."
        if not full:
            full = f"## Summary\n{spoken}"
        return title, spoken, full
