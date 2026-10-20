"""Clock/date tools."""

from datetime import datetime

from lib.dates import zone_name
from tools import tool


@tool({
    "name": "get_current_time",
    "description": (
        "Get the current date, time, and day of the week. Use whenever the user "
        "asks what time or day it is, or when you need today's date to reason "
        "about recency (e.g. 'today', 'this week', 'how long ago')."
    ),
    "input_schema": {"type": "object", "properties": {}},
})
def get_current_time(ctx, args):
    now = datetime.now().astimezone()
    return now.strftime("%A, %B %d, %Y at %I:%M %p ") + zone_name(now)
