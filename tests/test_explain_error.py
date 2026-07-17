"""Tests for explain_error: a turn failure must be spoken with its real cause.

The guarantee under test: when the API fails for an identifiable reason (empty
credit balance, bad key, rate limit, network down), the agent names that cause
aloud instead of the generic "let's try that again" — which once sent the user
chasing a phantom database deadlock for a whole session while the actual
problem was an empty credit balance.
"""

import unittest

from voice_agent import explain_error


class TestExplainError(unittest.TestCase):
    def test_credit_balance(self):
        # Exact shape of the 400 seen in session_2026-07-13.log (07-14 outage).
        e = Exception(
            "Error code: 400 - {'type': 'error', 'error': {'type': "
            "'invalid_request_error', 'message': 'Your credit balance is too "
            "low to access the Anthropic API. Please go to Plans & Billing "
            "to upgrade or purchase credits.'}}"
        )
        msg = explain_error(e)
        self.assertIn("credit balance", msg)
        self.assertIn("won't help", msg)

    def test_bad_api_key(self):
        e = Exception(
            "Error code: 401 - {'type': 'error', 'error': {'type': "
            "'authentication_error', 'message': 'invalid x-api-key'}}"
        )
        self.assertIn("API key", explain_error(e))

    def test_rate_limit(self):
        e = Exception(
            "Error code: 429 - {'type': 'error', 'error': {'type': "
            "'rate_limit_error', 'message': 'Number of requests has exceeded "
            "your rate limit.'}}"
        )
        self.assertIn("rate limited", explain_error(e))

    def test_overloaded(self):
        e = Exception(
            "Error code: 529 - {'type': 'error', 'error': {'type': "
            "'overloaded_error', 'message': 'Overloaded'}}"
        )
        self.assertIn("overloaded", explain_error(e))

    def test_network_down(self):
        e = Exception("Connection error.")
        self.assertIn("network", explain_error(e))
        e = Exception("[Errno 11001] getaddrinfo failed")
        self.assertIn("network", explain_error(e))

    def test_server_error(self):
        e = Exception(
            "Error code: 500 - {'type': 'error', 'error': {'type': "
            "'api_error', 'message': 'Internal server error'}}"
        )
        self.assertIn("server error", explain_error(e))

    def test_unknown_falls_back_to_generic(self):
        msg = explain_error(ValueError("something unexpected"))
        self.assertIn("try that again", msg)


if __name__ == "__main__":
    unittest.main()
