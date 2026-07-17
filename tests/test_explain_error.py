"""Tests for explain_error: a turn failure must be spoken with its real cause.

Two guarantees under test:
1. When the API fails for an identifiable reason (empty credit balance, bad
   key, rate limit, network down), the agent names that cause aloud instead of
   the generic "let's try that again" — which once sent the user chasing a
   phantom database deadlock for a whole session while the actual problem was
   an empty credit balance.
2. The reverse misdirection is just as bad: a LOCAL fault (mic, CUDA, SQLite)
   whose message merely contains "timeout" or "connection" must NOT be
   announced as an Anthropic network problem. Classification dispatches on the
   SDK's typed exceptions, so only genuine API errors get API explanations.
"""

import unittest

import anthropic
import httpx

from voice_agent import explain_error


def _status_error(cls, status, message):
    """Build a real SDK status error the way the client would raise it."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return cls(message, response=response, body={"error": {"message": message}})


class TestApiErrors(unittest.TestCase):
    def test_credit_balance(self):
        # Shape of the 400 seen in session_2026-07-13.log (07-14 outage).
        e = _status_error(
            anthropic.BadRequestError, 400,
            "Your credit balance is too low to access the Anthropic API. "
            "Please go to Plans & Billing to upgrade or purchase credits.",
        )
        msg = explain_error(e)
        self.assertIn("credit balance", msg)
        self.assertIn("won't help", msg)

    def test_bad_api_key(self):
        e = _status_error(anthropic.AuthenticationError, 401, "invalid x-api-key")
        self.assertIn("API key", explain_error(e))

    def test_rate_limit(self):
        e = _status_error(anthropic.RateLimitError, 429,
                          "Number of requests has exceeded your rate limit.")
        self.assertIn("rate limited", explain_error(e))

    def test_overloaded(self):
        e = _status_error(anthropic.InternalServerError, 529, "Overloaded")
        self.assertIn("overloaded", explain_error(e))

    def test_server_error(self):
        e = _status_error(anthropic.InternalServerError, 500,
                          "Internal server error")
        self.assertIn("server error", explain_error(e))

    def test_model_not_found(self):
        e = _status_error(anthropic.NotFoundError, 404,
                          "model: claude-nonexistent")
        self.assertIn("model", explain_error(e).lower())

    def test_network_down(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        e = anthropic.APIConnectionError(request=request)
        self.assertIn("network", explain_error(e))
        e = anthropic.APITimeoutError(request=request)
        self.assertIn("network", explain_error(e))


class TestLocalErrorsStayLocal(unittest.TestCase):
    """Local faults must never be blamed on the API, whatever their message."""

    def test_audio_timeout_is_not_a_network_problem(self):
        msg = explain_error(OSError("PaErrorCode -9987: input timed out"))
        self.assertNotIn("network", msg)
        self.assertNotIn("Anthropic", msg)

    def test_sqlite_connection_is_not_a_network_problem(self):
        msg = explain_error(RuntimeError("unable to open database connection"))
        self.assertNotIn("network", msg)
        self.assertNotIn("Anthropic", msg)

    def test_unknown_falls_back_to_generic(self):
        msg = explain_error(ValueError("something unexpected"))
        self.assertIn("try that again", msg)


if __name__ == "__main__":
    unittest.main()
