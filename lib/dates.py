"""Date words shared by tools — a named period or ISO bounds to a day range,
and the same to an epoch window for metadata filters. Lifted from
tools/trading_tools (get_pnl) so search_past_conversations can take the same
arguments without importing the trading package. Leaf: stdlib only.
"""

from datetime import date, datetime, timedelta

PERIODS = ("today", "yesterday", "this_week", "last_week", "this_month",
           "last_month", "this_year")


def period_range(period=None, start=None, end=None, today=None):
    """(start, end) as ISO dates: the named period, else the explicit bounds,
    else today — an unknown name also means today, as get_pnl always did.
    `today` is injectable so tests don't depend on the calendar."""
    today = today or date.today()
    if not period:
        return (start or today.isoformat(), end or today.isoformat())
    monday = today - timedelta(days=today.weekday())
    first = today.replace(day=1)
    if period == "today":
        return today.isoformat(), today.isoformat()
    if period == "yesterday":
        d = today - timedelta(days=1)
        return d.isoformat(), d.isoformat()
    if period == "this_week":
        return monday.isoformat(), today.isoformat()
    if period == "last_week":
        return ((monday - timedelta(days=7)).isoformat(),
                (monday - timedelta(days=1)).isoformat())
    if period == "this_month":
        return first.isoformat(), today.isoformat()
    if period == "last_month":
        last_end = first - timedelta(days=1)
        return last_end.replace(day=1).isoformat(), last_end.isoformat()
    if period == "this_year":
        return today.replace(month=1, day=1).isoformat(), today.isoformat()
    return today.isoformat(), today.isoformat()


def _epoch(text, *, end):
    """Local epoch seconds for an ISO date or datetime. A bare date is a whole
    day, so its `end` is 23:59:59; a naive datetime is local time, which is
    what history.now_iso stamps (it just writes the offset out)."""
    dt = datetime.fromisoformat(text)
    if len(text) == 10 and end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.timestamp()


def window_epochs(period=None, since=None, until=None, today=None):
    """(lo, hi) epoch bounds for a named period or ISO since/until, or None
    when no time was given at all. A missing `since` is the beginning of
    time; a missing `until` is the end of today."""
    if not (period or since or until):
        return None
    if period:
        since, until = period_range(period, today=today)
    today = today or date.today()
    lo = _epoch(since, end=False) if since else 0.0
    hi = _epoch(until or today.isoformat(), end=True)
    return lo, hi
