"""Gmail tools: search threads, read a thread, save a draft.

Ported from the mcp-test project's gmail_tools.py (async httpx behind MCP ->
sync requests behind the tool registry; bodies otherwise the same).

send_email is gated by instruction only (its description demands an explicit
user go-ahead for the exact recipient/content), not by a review fingerprint
like trading's submit_order — the user chose the lean version. Drafts remain
the default the descriptions steer toward.
"""

import base64
import json
from email.message import EmailMessage

import requests

from tools import tool

GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
TIMEOUT = 30.0


# --- helpers -----------------------------------------------------------------
def _auth():
    """(headers, None) or (None, spoken-problem). Imports inside the call so
    missing google-auth deps or a missing token degrade to a sentence instead
    of crashing agent startup or the conversation loop."""
    try:
        from lib.gmail_auth import get_access_token
    except ImportError:
        return None, ("Gmail isn't available on this machine: the google-auth "
                      "packages are not installed (pip install -r requirements.txt).")
    try:
        return {"Authorization": f"Bearer {get_access_token()}"}, None
    except Exception as e:
        return None, f"Gmail authorization failed: {e}."


def _mime(to, subject, body, cc=None):
    """Build an RFC-2822 message and base64url it — the format Gmail wants."""
    m = EmailMessage()
    m["To"] = to
    m["Subject"] = subject
    if cc:
        m["Cc"] = cc
    m.set_content(body)
    return base64.urlsafe_b64encode(m.as_bytes()).decode()


def _headers_of(payload):
    return {h["name"].lower(): h["value"] for h in payload.get("headers", [])}


def _plain_text(payload):
    """Walk the MIME tree for text/plain; fall back to text/html."""
    if payload.get("mimeType") == "text/plain":
        if data := payload.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(data).decode("utf-8", "replace")
    for part in payload.get("parts", []):
        if text := _plain_text(part):
            return text
    if payload.get("mimeType") == "text/html":
        if data := payload.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(data).decode("utf-8", "replace")
    return ""


# --- tools -------------------------------------------------------------------
@tool({
    "name": "search_email_threads",
    "description": (
        "Search the user's Gmail and return matching threads (a conversation "
        "with 8 replies counts as one result). Uses Gmail search syntax, e.g. "
        "'is:unread', 'from:dana@x.com', 'subject:invoice newer_than:7d'. "
        "Note 'in:inbox' excludes Sent, Archive, Spam and Trash; use "
        "'in:anywhere' to search everything, or 'in:sent' for sent mail."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Gmail search query (default 'in:inbox')"},
            "max_results": {"type": "integer",
                            "description": "How many threads to return (1-50, default 10)"},
        },
    },
})
def search_email_threads(ctx, args):
    headers, problem = _auth()
    if problem:
        return problem
    query = args.get("query") or "in:inbox"
    max_results = max(1, min(int(args.get("max_results") or 10), 50))

    r = requests.get(f"{GMAIL_API}/threads", headers=headers, timeout=TIMEOUT,
                     params={"q": query, "maxResults": max_results})
    r.raise_for_status()
    out = []
    for t in r.json().get("threads", []):
        d = requests.get(f"{GMAIL_API}/threads/{t['id']}", headers=headers,
                         timeout=TIMEOUT,
                         params={"format": "metadata",
                                 "metadataHeaders": ["Subject", "From", "Date"]})
        if d.status_code != 200:
            continue
        msgs = d.json().get("messages", [])
        h = _headers_of(msgs[0].get("payload", {})) if msgs else {}
        out.append({
            "thread_id": t["id"],
            "subject": h.get("subject", "(no subject)"),
            "from": h.get("from", "?"),
            "date": h.get("date", "?"),
            "messages": len(msgs),
            "snippet": t.get("snippet", "")[:200],
        })
    return json.dumps({"query": query, "count": len(out), "threads": out}, indent=2)


@tool({
    "name": "get_email_thread",
    "description": (
        "Read a full Gmail thread, including every message body. Use after "
        "search_email_threads to read the actual emails."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "thread_id": {"type": "string",
                          "description": "Thread id from search_email_threads"},
        },
        "required": ["thread_id"],
    },
})
def get_email_thread(ctx, args):
    headers, problem = _auth()
    if problem:
        return problem
    thread_id = args["thread_id"]
    r = requests.get(f"{GMAIL_API}/threads/{thread_id}", headers=headers,
                     timeout=TIMEOUT, params={"format": "full"})
    r.raise_for_status()

    messages = []
    for m in r.json().get("messages", []):
        h = _headers_of(m.get("payload", {}))
        messages.append({
            "message_id": m.get("id"),
            "from": h.get("from", "?"),
            "to": h.get("to", "?"),
            "date": h.get("date", "?"),
            "subject": h.get("subject", ""),
            "body": _plain_text(m.get("payload", {}))[:4000],
        })
    return json.dumps({"thread_id": thread_id, "messages": messages}, indent=2)


@tool({
    "name": "create_email_draft",
    "description": (
        "Save a draft email in the user's Gmail. Does NOT send anything — the "
        "user reviews and sends it from Gmail. Prefer this over send_email "
        "unless the user explicitly asked you to send. Pass thread_id to "
        "attach the draft as a reply to an existing thread."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {"type": "string",
                   "description": "Recipient address(es), comma-separated"},
            "subject": {"type": "string", "description": "Subject line"},
            "body": {"type": "string", "description": "Plain-text body"},
            "cc": {"type": "string", "description": "Optional CC address(es)"},
            "thread_id": {"type": "string",
                          "description": "Optional thread to reply to"},
        },
        "required": ["to", "subject", "body"],
    },
})
def create_email_draft(ctx, args):
    headers, problem = _auth()
    if problem:
        return problem
    message = {"raw": _mime(args["to"], args["subject"], args["body"],
                            args.get("cc"))}
    if args.get("thread_id"):
        message["threadId"] = args["thread_id"]

    r = requests.post(f"{GMAIL_API}/drafts", headers=headers, timeout=TIMEOUT,
                      json={"message": message})
    r.raise_for_status()
    return json.dumps({"status": "draft saved", "draft_id": r.json().get("id"),
                       "to": args["to"], "subject": args["subject"]}, indent=2)


@tool({
    "name": "send_email",
    "description": (
        "Send an email from the user's Gmail, immediately and irreversibly. "
        "Only call this after the user has heard the recipient, subject, and "
        "gist of THIS exact message and explicitly said to send it — never on "
        "your own initiative. If the user merely wants an email written, use "
        "create_email_draft instead. Pass thread_id to send as a reply on an "
        "existing thread."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {"type": "string",
                   "description": "Recipient address(es), comma-separated"},
            "subject": {"type": "string", "description": "Subject line"},
            "body": {"type": "string", "description": "Plain-text body"},
            "cc": {"type": "string", "description": "Optional CC address(es)"},
            "thread_id": {"type": "string",
                          "description": "Optional thread to reply on"},
        },
        "required": ["to", "subject", "body"],
    },
})
def send_email(ctx, args):
    headers, problem = _auth()
    if problem:
        return problem
    message = {"raw": _mime(args["to"], args["subject"], args["body"],
                            args.get("cc"))}
    if args.get("thread_id"):
        message["threadId"] = args["thread_id"]

    r = requests.post(f"{GMAIL_API}/messages/send", headers=headers,
                      timeout=TIMEOUT, json=message)
    r.raise_for_status()
    m = r.json()
    return json.dumps({"status": "sent", "message_id": m.get("id"),
                       "thread_id": m.get("threadId"), "to": args["to"]},
                      indent=2)
