"""Which account a persona works on, and the user's names for accounts.

A broker login can share several accounts (the user's Schwab login shares
an IRA and "red"). The accounts themselves are discovered from the broker
(Client.account_numbers); this module keeps only what the broker can't
know, in data/trading/<broker>/accounts.json:

    {"names": {"699": ["ira"], "868": ["red"]},     # key: digits the user gave
     "focus": {"tom": "12345699"}}                   # persona -> account number

Names are keyed by whatever digits the user said ("699" — statements and
alerts mask all but the last three), and match any account number ending in
them. Focus is per persona and per broker, and persisted on purpose: an
account is a standing choice — "move to my IRA" should still hold tomorrow —
unlike the retrieval focus (set_focus), which dies with the session.

Pure helpers + one small JSON file; no broker calls here.
"""

from lib.atomic_io import read_json, write_json_atomic
from trading import config as tcfg


def _path(broker):
    return tcfg.broker_dir(broker) / "accounts.json"


def _load(broker):
    data = read_json(_path(broker), {}, expect=dict)
    return {"names": dict(data.get("names") or {}), "focus": dict(data.get("focus") or {})}


def _save(broker, data):
    tcfg.ensure_dirs(broker)
    write_json_atomic(_path(broker), data)


def _digits(text):
    return "".join(c for c in str(text or "") if c.isdigit())


def names(broker, number):
    """The user's names for `number` (an account number or masked id)."""
    d = _digits(number)
    return [n for key, ns in _load(broker)["names"].items()
            if key and d.endswith(key) for n in ns]


def digits_for(broker, nickname):
    """The digits the user named `nickname` by ('699'), or None — what the
    trade log's masked ids ('*****699SCHW') can be matched on."""
    nick = (nickname or "").strip().lower()
    for key, ns in _load(broker)["names"].items():
        if nick in (n.lower() for n in ns):
            return key
    return None


def label(broker, number):
    """Spoken label: 'IRA (…699)' with a name, 'account …699' without."""
    tail = f"…{_digits(number)[-3:]}"
    ns = names(broker, number)
    return f"{ns[0]} ({tail})" if ns else f"account {tail}"


def set_name(broker, number, nickname):
    """Name an account, kept as the user said it ('IRA'). The key is the
    digits given ('699'); the name moves off any account that had it, so one
    name never means two."""
    nick = " ".join(nickname.split())
    key = _digits(number)
    if not nick or not key:
        raise ValueError("Say the account (its last digits) and the name for it.")
    data = _load(broker)
    for k in list(data["names"]):
        data["names"][k] = [n for n in data["names"][k] if n.lower() != nick.lower()]
        if not data["names"][k]:
            del data["names"][k]
    data["names"].setdefault(key, []).append(nick)
    _save(broker, data)


def focus(broker, persona):
    return _load(broker)["focus"].get(persona)


def set_focus(broker, persona, number):
    data = _load(broker)
    data["focus"][persona] = str(number)
    _save(broker, data)


def resolve(broker, text, numbers):
    """(account_number, None) or (None, problem) for what the user said: a
    name ('ira', 'my red account'), trailing digits ('699', '5699'), or the
    full number. `numbers` are the broker's live account numbers."""
    said = f" {str(text or '').strip().lower()} "
    for filler in (" my ", " the ", " account "):
        said = said.replace(filler, " ")
    said = said.strip()
    if _digits(said) and _digits(said) == said.replace(" ", ""):
        hits = [n for n in numbers if _digits(n).endswith(_digits(said))]
    else:
        hits = [n for n in numbers if said in (x.lower() for x in names(broker, n))]
    if len(hits) == 1:
        return hits[0], None
    listing = ", ".join(label(broker, n) for n in numbers)
    if hits:
        return None, f"'{text}' matches several accounts: {listing}. Which one?"
    return None, (f"No account called '{text}'. The accounts are: {listing}. "
                  "Name one with set_account (account plus nickname).")


def default(broker, persona, numbers, configured=None):
    """The account a persona works on when the user didn't say one: its
    focus if that account still exists, the only account, or the .env
    default — else None, and the caller asks."""
    f = focus(broker, persona)
    if f in numbers:
        return f
    if len(numbers) == 1:
        return numbers[0]
    if configured:
        hit = [n for n in numbers if _digits(n).endswith(_digits(configured))]
        if len(hit) == 1:
            return hit[0]
    return None
