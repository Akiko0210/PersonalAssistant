"""Order lifecycle: dry-run review -> confirmed submit -> cancel.

The safety gate lives here, not in the UIs: submit() refuses unless the
*current* ticket was dry-run reviewed (fingerprint match, review fresh)
and the caller passes confirmed=True — which the voice tool only sets
after the user has heard the review and said yes, and the dashboard only
sets from the confirm dialog. Cancelling never needs ceremony.

Every dry-run, submit, cancel, and rejection is appended to
data/trading/orders_log.json (atomic, capped) — the audit trail the
dashboard shows.

Status checks are on-demand only (get/list); nothing here polls
/orders/live — see TRADING_PLAN.md §1.4.
"""

import json
import time
from datetime import datetime, timezone

from lib.atomic_io import write_json_atomic
from trading import config as tcfg
from trading import ticket as tticket
from trading.models import DryRunResult, PlacedOrder
from trading.tasty_client import TastyError

REVIEW_MAX_AGE_S = 10 * 60   # a stale review means stale prices — redo it
LOG_CAP = 1000


def _money(d, key):
    """Signed float from tastytrade's value/effect pairs
    (Credit = +, Debit = -, None = missing)."""
    try:
        v = float(d.get(key))
    except (TypeError, ValueError):
        return None
    return -v if d.get(f"{key}-effect") == "Debit" else v


def _parse_order(o):
    return PlacedOrder(
        id=str(o.get("id", "")),
        status=o.get("status", "Unknown"),
        underlying=o.get("underlying-symbol", ""),
        price=float(o["price"]) if o.get("price") else None,
        price_effect=o.get("price-effect"),
        time_in_force=o.get("time-in-force"),
        legs=[{"symbol": l.get("symbol"), "action": l.get("action"),
               "quantity": l.get("quantity"),
               "remaining": l.get("remaining-quantity")}
              for l in o.get("legs", [])],
        created_at=o.get("received-at") or o.get("created-at"),
        raw=o)


class OrderManager:
    def __init__(self, client):
        self._client = client

    # --- review ----------------------------------------------------------

    def review(self, tk, chain):
        """Dry-run the ticket. On success stamps tk.review (persisted) so
        submit() will accept this exact ticket."""
        try:
            payload = tk.payload(chain)
        except ValueError as e:
            return DryRunResult(ok=False, errors=[str(e)])
        try:
            data = self._client.dry_run_order(payload)
        except TastyError as e:
            self._log("dry-run-rejected", ticket=tk.describe(),
                      errors=e.details())
            return DryRunResult(ok=False, errors=[e.details()], raw={})
        bp = data.get("buying-power-effect") or {}
        fees = data.get("fee-calculation") or {}
        fee_total = _money(fees, "total-fees")
        result = DryRunResult(
            ok=True,
            warnings=[w.get("message", str(w)) for w in data.get("warnings", [])],
            buying_power_change=_money(bp, "change-in-buying-power"),
            new_buying_power=_money(bp, "new-buying-power"),
            margin_requirement=_money(bp, "isolated-order-margin-requirement"),
            # Fees are always a debit; keep the magnitude (spoken as "fees
            # about 3.40", not "-3.40").
            total_fees=abs(fee_total) if fee_total is not None else None,
            raw=data)
        tk.review = {"fingerprint": tk.fingerprint(), "at": time.time(),
                     "buying_power_change": result.buying_power_change,
                     "total_fees": result.total_fees,
                     "warnings": result.warnings}
        tticket.save(tk)
        self._log("dry-run", ticket=tk.describe(),
                  buying_power_change=result.buying_power_change,
                  total_fees=result.total_fees, warnings=result.warnings)
        return result

    # --- submit ----------------------------------------------------------

    def submit_blockers(self, tk):
        """Why this ticket may not be submitted right now (empty = go)."""
        problems = []
        if not tk.review:
            problems.append("The order has not been reviewed — run a "
                            "review (dry run) first.")
            return problems
        if tk.review.get("fingerprint") != tk.fingerprint():
            problems.append("The ticket changed after its review — review "
                            "it again before submitting.")
        elif time.time() - tk.review.get("at", 0) > REVIEW_MAX_AGE_S:
            problems.append("The review is stale — prices have had time to "
                            "move. Review again before submitting.")
        return problems

    def submit(self, tk, chain, confirmed=False):
        """Place the reviewed ticket. Returns (PlacedOrder, None) or
        (None, reason)."""
        if not confirmed:
            return None, ("Not submitted: needs explicit confirmation "
                          "after the review.")
        blockers = self.submit_blockers(tk)
        if blockers:
            return None, " ".join(blockers)
        try:
            payload = tk.payload(chain)
            data = self._client.place_order(payload)
        except (ValueError, TastyError) as e:
            reason = e.details() if isinstance(e, TastyError) else str(e)
            self._log("submit-rejected", ticket=tk.describe(), error=reason)
            return None, f"Order rejected: {reason}"
        order = _parse_order(data.get("order", data))
        self._log("submitted", order_id=order.id, status=order.status,
                  ticket=tk.describe())
        tk.review = None       # a review authorizes exactly one submit
        tticket.save(tk)
        return order, None

    # --- status / cancel -------------------------------------------------

    def get(self, order_id):
        return _parse_order(self._client.get_order(order_id))

    def list_orders(self, start_date=None):
        data = self._client.list_orders(start_date=start_date)
        items = data.get("items", []) if isinstance(data, dict) else data
        return [_parse_order(o) for o in items]

    def working_orders(self, start_date=None):
        return [o for o in self.list_orders(start_date) if o.is_working]

    def cancel(self, order_id):
        """Returns (PlacedOrder, None) or (None, reason)."""
        try:
            data = self._client.cancel_order(order_id)
        except TastyError as e:
            self._log("cancel-failed", order_id=order_id, error=e.details())
            return None, e.details()
        order = _parse_order(data)
        self._log("cancelled", order_id=order.id, status=order.status)
        return order, None

    # --- audit log -------------------------------------------------------

    def _log(self, kind, **detail):
        tcfg.ensure_dirs()
        try:
            with open(tcfg.ORDERS_LOG_PATH, encoding="utf-8") as f:
                entries = json.load(f)
        except (OSError, ValueError):
            entries = []
        entries.append({
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "env": "live" if tcfg.is_live() else "sandbox",
            "kind": kind, **detail})
        write_json_atomic(tcfg.ORDERS_LOG_PATH, entries[-LOG_CAP:])


def read_log(limit=100):
    try:
        with open(tcfg.ORDERS_LOG_PATH, encoding="utf-8") as f:
            return json.load(f)[-limit:]
    except (OSError, ValueError):
        return []
