"""Order lifecycle: dry-run review -> confirmed submit -> cancel.

The safety gate lives here, not in the UIs: submit() refuses unless the
*current* ticket was dry-run reviewed (fingerprint match, review fresh)
and the caller passes confirmed=True — which the voice tool only sets
after the user has heard the review and said yes, and the dashboard only
sets from the confirm dialog. Cancelling never needs ceremony.

The broker variant (trading/main.py) supplies the payload and the parsers;
the gate and the audit trail are the same for both. Every dry-run, submit,
cancel, and rejection is appended to data/trading/<broker>/orders_log.json
(atomic, capped) — the audit trail the dashboard shows.

Status checks are on-demand only (get/list); nothing here polls
/orders/live — see TRADING_PLAN.md §1.4.
"""

import json
import time
from datetime import datetime, timezone

from lib.atomic_io import write_json_atomic
from trading import config as tcfg
from trading import ticket as tticket
from trading.models import BrokerError, DryRunResult

REVIEW_MAX_AGE_S = 10 * 60   # a stale review means stale prices — redo it
LOG_CAP = 1000


class OrderManager:
    def __init__(self, client, broker):
        self._client = client
        self._broker = broker        # a variant module (tastytrade / schwab)

    # --- review ----------------------------------------------------------

    def review(self, tk, chain):
        """Dry-run the ticket. On success stamps tk.review (persisted) so
        submit() will accept this exact ticket."""
        try:
            payload = self._broker.order_payload(tk, chain)
        except ValueError as e:
            return DryRunResult(ok=False, errors=[str(e)])
        try:
            data = self._client.dry_run_order(tk.account, payload)
        except BrokerError as e:
            self._log("dry-run-rejected", ticket=tk.describe(),
                      errors=e.details())
            return DryRunResult(ok=False, errors=[e.details()], raw={})
        result = self._broker.parse_dry_run(data)
        if not result.ok:
            self._log("dry-run-rejected", ticket=tk.describe(),
                      errors="; ".join(result.errors))
            return result
        tk.review = {"fingerprint": tk.fingerprint(), "at": time.time(),
                     "buying_power_change": result.buying_power_change,
                     "total_fees": result.total_fees,
                     "warnings": result.warnings}
        tticket.save(tk, self._broker.KEY)
        self._log("dry-run", ticket=tk.describe(),
                  buying_power_change=result.buying_power_change,
                  total_fees=result.total_fees, warnings=result.warnings)
        return result

    # --- submit ----------------------------------------------------------

    def submit_blockers(self, tk):
        """Why this ticket may not be submitted right now (empty = go)."""
        problems = []
        if not tk.account:
            problems.append("The ticket has no account — review it first.")
            return problems
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
            payload = self._broker.order_payload(tk, chain)
            data = self._client.place_order(tk.account, payload)
        except (ValueError, BrokerError) as e:
            reason = e.details() if isinstance(e, BrokerError) else str(e)
            self._log("submit-rejected", ticket=tk.describe(), error=reason)
            return None, f"Order rejected: {reason}"
        order = self._broker.parse_order(data)
        self._log("submitted", order_id=order.id, status=order.status,
                  account=tk.account, ticket=tk.describe())
        tk.review = None       # a review authorizes exactly one submit
        tticket.save(tk, self._broker.KEY)
        return order, None

    # --- status / cancel -------------------------------------------------

    def get(self, account, order_id):
        return self._broker.parse_order(self._client.get_order(account, order_id))

    def list_orders(self, account, start_date=None):
        return self._broker.parse_order_list(
            self._client.list_orders(account, start_date=start_date))

    def working_orders(self, account, start_date=None):
        return [o for o in self.list_orders(account, start_date) if o.is_working]

    def cancel(self, account, order_id):
        """Returns (PlacedOrder, None) or (None, reason)."""
        try:
            data = self._client.cancel_order(account, order_id)
        except BrokerError as e:
            self._log("cancel-failed", order_id=order_id, account=account,
                      error=e.details())
            return None, e.details()
        order = self._broker.parse_order(data)
        self._log("cancelled", order_id=order.id, account=account, status=order.status)
        return order, None

    # --- audit log -------------------------------------------------------

    def _log(self, kind, **detail):
        key = self._broker.KEY
        tcfg.ensure_dirs(key)
        entries = read_log(key, limit=LOG_CAP)
        entries.append({
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "env": self._broker.env_label(), "broker": key,
            "kind": kind, **detail})
        write_json_atomic(tcfg.orders_log_path(key), entries[-LOG_CAP:])


def read_log(broker, limit=100):
    try:
        with open(tcfg.orders_log_path(broker), encoding="utf-8") as f:
            return json.load(f)[-limit:]
    except (OSError, ValueError):
        return []
