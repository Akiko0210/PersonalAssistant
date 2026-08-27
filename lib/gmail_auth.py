"""Google OAuth for the Gmail tools (ported from the mcp-test project).

The rest of the app needs exactly one thing from here:

    from lib.gmail_auth import get_access_token
    token = get_access_token()   # -> "ya29.a0Af..."

Split from tools/gmail_tools.py so the google-auth imports stay out of the
tool module (which the registry imports at startup on machines that may not
have the deps installed) — gmail_tools imports this lazily, per call.

Inside the agent this NEVER opens a browser: a missing/revoked token raises
with instructions instead, because a blocking consent flow in the middle of
the voice loop would look like a hang. The one-time browser consent is run
standalone:

    python -m lib.gmail_auth
"""

import os

import config as cfg
from lib.atomic_io import write_text_atomic

# Must match the scopes the Cloud-console consent screen offers, or Google
# returns "invalid_scope".
#   gmail.readonly -> read + search messages, threads, drafts, labels
#   gmail.compose  -> create/update drafts (also allows send; we expose drafts only)
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]


def _save(creds):
    # Atomic write: data/ is Dropbox-synced, and a torn token file means a
    # mysterious re-consent later.
    write_text_atomic(cfg.GMAIL_TOKEN_PATH, creds.to_json())
    try:
        os.chmod(cfg.GMAIL_TOKEN_PATH, 0o600)  # best-effort; no-op on Windows
    except OSError:
        pass


def _login():
    """Full browser consent flow. Interactive only — see module docstring."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not cfg.GMAIL_CLIENT_SECRET_PATH.exists():
        raise RuntimeError(
            f"Missing {cfg.GMAIL_CLIENT_SECRET_PATH.name}. Download it from the "
            "Google Cloud console (Google Auth Platform -> Clients -> your "
            f"client -> Download JSON) and save it as {cfg.GMAIL_CLIENT_SECRET_PATH}."
        )
    flow = InstalledAppFlow.from_client_secrets_file(
        str(cfg.GMAIL_CLIENT_SECRET_PATH), SCOPES)
    # access_type="offline" earns the refresh token (silent renewal forever);
    # prompt="consent" forces re-consent so repeat logins still return one.
    return flow.run_local_server(
        port=8765,
        access_type="offline",
        prompt="consent",
        authorization_prompt_message="Opening your browser to approve Gmail access...",
        success_message="Done. You can close this tab and return to the terminal.",
    )


def get_credentials(interactive=False):
    """Valid credentials, cheapest path first: reuse token.json, else refresh
    silently, else (interactive only) browser consent — otherwise raise with
    the command to run."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = None
    if cfg.GMAIL_TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(
                str(cfg.GMAIL_TOKEN_PATH), SCOPES)
        except ValueError:
            creds = None  # malformed file, or saved scopes no longer match

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save(creds)
            return creds
        except Exception:
            if not interactive:
                raise RuntimeError(
                    "the saved Gmail token could not be refreshed — run "
                    "`python -m lib.gmail_auth` to re-authorize")

    if not interactive:
        raise RuntimeError(
            "no usable Gmail token — run `python -m lib.gmail_auth` once to "
            "authorize")
    creds = _login()
    _save(creds)
    return creds


def get_access_token():
    """The string that goes into `Authorization: Bearer <...>`."""
    return get_credentials().token


if __name__ == "__main__":
    had_token = cfg.GMAIL_TOKEN_PATH.exists()
    creds = get_credentials(interactive=True)
    print()
    print("Google OAuth OK")
    print(f"  path taken   : {'reused/refreshed token' if had_token else 'browser consent'}")
    print(f"  refresh token: {'yes' if creds.refresh_token else 'NO - delete the token file and re-run'}")
    print(f"  expires at   : {creds.expiry} UTC")
    print(f"  saved to     : {cfg.GMAIL_TOKEN_PATH}")
