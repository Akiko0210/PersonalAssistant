"""The actions the dashboard can ask a live agent to perform.

One method per action, registered in `actions`; controller.py dispatches to
them generically. To add a control: write a method here that validates its
payload (raise ValueError for a bad one — the server turns that into a 400),
add it to `actions`, and give the dashboard a button that POSTs
/api/control/<name>. Neither controller.py nor dashboard.py routing changes.

Built on injected callables rather than the Agent object so tests exercise it
with plain fakes, and so this file states exactly what it touches.
"""

MAX_MESSAGE_CHARS = 4000  # sanity bound; a "message" is a question, not a document


class ControllerService:
    def __init__(self, read_state, set_mute, send_message):
        """read_state: () -> (muted: bool, mode: str)
        set_mute:     (muted: bool) -> None   — idempotent, announces on change
        send_message: (text: str) -> None     — queue a typed message for the
                                                turn loop to answer aloud"""
        self._read_state = read_state
        self._set_mute = set_mute
        self._send_message = send_message
        self.actions = {"mute": self.mute, "send_message": self.send_message}

    def _state(self, **extra):
        muted, mode = self._read_state()
        return {"muted": bool(muted), "mode": mode, **extra}

    def status(self):
        return self._state()

    def mute(self, payload):
        """{"muted": true|false} -> the resulting state. A state, not a toggle,
        so a stale page or double-tap can't land on the opposite of what the
        user asked for."""
        muted = payload.get("muted") if isinstance(payload, dict) else None
        if not isinstance(muted, bool):
            raise ValueError('expected {"muted": true|false}')
        self._set_mute(muted)
        return self._state(ok=True)

    def send_message(self, payload):
        """{"text": "..."} -> queued ack. The reply is spoken by the agent at
        the next utterance boundary — seconds later, so it can't ride back on
        this response; the Conversation page shows the exchange."""
        text = payload.get("text") if isinstance(payload, dict) else None
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            raise ValueError("message text may not be empty")
        if len(text) > MAX_MESSAGE_CHARS:
            raise ValueError(f"message is over {MAX_MESSAGE_CHARS} characters")
        self._send_message(text)
        return {"ok": True, "queued": True}
