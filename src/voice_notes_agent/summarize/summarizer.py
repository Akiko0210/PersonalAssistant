"""Transcript -> Claude -> spoken + full summary (§5.4).

Only the transcript **text** is sent to the cloud (FR-S1) — never note audio (C4/NFR-4).
A single Claude call returns a structured result via the Messages API's structured-output
support, from which we derive:

  * a concise **spoken** summary — headline + action items, read back via TTS (FR-S2/S3)
  * a fuller **stored** summary rendered to ``summary.md`` (FR-S2)

If the session contained negligible speech, the full summary is skipped and only the
transcript is kept (FR-S5) — that decision is made by the caller via
:meth:`should_summarize`.

Model: ``claude-opus-4-8`` with adaptive thinking (the default for non-trivial calls).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Structured-output schema for the summary call. Keeps the response parseable (no
# free-form prose to scrape) — see the Anthropic structured-outputs docs.
_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "headline": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "action_items": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
        "spoken_summary": {
            "type": "string",
            "description": "One to three sentences: the headline plus the most important "
            "action items, phrased for text-to-speech read-back.",
        },
    },
    "required": [
        "title",
        "headline",
        "key_points",
        "action_items",
        "questions",
        "spoken_summary",
    ],
    "additionalProperties": False,
}


@dataclass
class SummaryResult:
    title: str
    headline: str
    key_points: list[str]
    action_items: list[str]
    questions: list[str]
    spoken_summary: str

    def to_markdown(self, template: str) -> str:
        def bullets(items: list[str]) -> str:
            return "\n".join(f"- {i}" for i in items) if items else "- (none)"

        return template.format(
            title=self.title,
            headline=self.headline,
            key_points=bullets(self.key_points),
            action_items=bullets(self.action_items),
            questions=bullets(self.questions),
        )


class Summarizer:
    """Wraps the Anthropic client for the summary call. Client injectable for tests."""

    def __init__(self, cfg, client=None) -> None:
        self._cfg = cfg
        self._model = cfg.providers.llm.model
        self._summary_cfg = cfg.summary
        self._client = client  # lazily created if None

    def _get_client(self):  # pragma: no cover - requires anthropic + key
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def should_summarize(self, total_speech_sec: float) -> bool:
        """FR-S5: skip the full summary when speech is negligible."""
        return total_speech_sec >= self._summary_cfg.min_speech_sec

    def summarize(self, transcript: str) -> SummaryResult:
        """Call Claude and return the structured summary."""
        client = self._get_client()
        max_spoken = self._summary_cfg.spoken_max_sentences
        prompt = (
            "Summarize the following voice note transcript. Produce a short title, a "
            "one-line headline, key points, explicit action items, and any open "
            f"questions. The spoken_summary must be at most {max_spoken} sentences and "
            "read naturally aloud.\n\nTRANSCRIPT:\n" + transcript
        )
        # Structured outputs (output_config.format) guarantee parseable JSON. Adaptive
        # thinking is the recommended default for non-trivial calls.
        resp = client.messages.create(  # pragma: no cover - network
            model=self._model,
            max_tokens=2000,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": _SUMMARY_SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(b.text for b in resp.content if b.type == "text")  # pragma: no cover
        data = json.loads(text)  # pragma: no cover
        return SummaryResult(  # pragma: no cover
            title=data["title"],
            headline=data["headline"],
            key_points=data.get("key_points", []),
            action_items=data.get("action_items", []),
            questions=data.get("questions", []),
            spoken_summary=data["spoken_summary"],
        )
