"""CSV exports for the user's own spreadsheets: the one writer behind
query_trade_log's trade export and every display tool's table export
(tools.dispatch). Files land in config.EXPORTS_DIR."""

import csv
from datetime import datetime

import config as cfg


def write(prefix, header, rows):
    """A new file `<prefix>_<date>_<time>.csv`; returns its path. Mode "x":
    a name already taken fails loudly rather than overwrite an export."""
    cfg.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.EXPORTS_DIR / f"{prefix}_{datetime.now():%Y-%m-%d_%H%M%S}.csv"
    # utf-8-sig so Excel opens it as UTF-8 rather than guessing a code page.
    with path.open("x", newline="", encoding="utf-8-sig") as f:
        out = csv.writer(f)
        out.writerow(header)
        out.writerows(rows)
    return path
