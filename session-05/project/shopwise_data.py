"""ShopWise's records — the mock backend, loaded as five plain dictionaries.

WHAT THIS IS
    Customers, products, orders, shipments and refunds. From v2 the assistant
    answers from these and from nothing else, so every fact it states about an
    order is either in here or invented.

    Poke it in a cell. It is dicts:

        >>> from shopwise_data import CUSTOMERS, ORDERS
        >>> CUSTOMERS["ayesha@example.com"]["tier"]
        'gold'
        >>> ORDERS["ORD-5001"]["status"]
        'delivered'

WHY IT IS A LOADER AND NOT THE DATA
    The obvious version of this file is the data itself — five dict literals,
    two hundred lines, nothing to read from disk. That is what PROJECT-SPINE §2
    specified, and it has one real virtue: nothing stands between the reader and
    the values.

    It loses to arithmetic. Snapshots are copied forward, never edited in place
    (ADR-0001), so a module holding the data would be twelve identical copies by
    session 16, and correcting one fact in session 11 would mean seven edits and
    six chances to miss one. The data therefore lives once, in
    data/backend.yml, beside tickets.yml and prices.yml — which already
    cross-reference it — and this file reads it. ADR-0006 records the trade.

    The seam turns out to be the point. In a real system this *is* your database
    or your API, and a file that opens a connection and hands back rows is
    exactly the shape of the thing you would write. The lesson survives the
    change; it gets a little more honest.

WHY IT IS FIVE LINES
    Because anything more would be a layer, and a layer here would be the wrong
    lesson at the wrong moment. There is no ORM, no cache, no query object, no
    class — the point of section 1 is that the backend is *not mysterious*
    before we wrap tools around it. Read it in ten seconds, then move on.

    Everything that dresses this data up for a customer — expanding an order's
    SKUs into product names, matching "the kettle" to SKU-300, returning
    {"error": ...} when a lookup finds nothing — belongs in tools.py, where the
    model can see it. Keep this file boring.
"""

from __future__ import annotations

import yaml

from config import settings

# Read once at import. The fixture is a few kilobytes and never changes while
# the program runs, so re-reading it per lookup would buy nothing; and doing it
# here means a malformed fixture fails at import, with the filename in the
# traceback, rather than halfway through an agent run.
_BACKEND = yaml.safe_load((settings.DATA_DIR / "backend.yml").read_text(encoding="utf-8"))

CUSTOMERS: dict[str, dict] = _BACKEND["customers"]   # keyed by email — what a customer types
PRODUCTS: dict[str, dict] = _BACKEND["products"]     # keyed by SKU
ORDERS: dict[str, dict] = _BACKEND["orders"]         # keyed by order id
SHIPMENTS: dict[str, dict] = _BACKEND["shipments"]   # keyed by ORDER id, not tracking id
REFUNDS: dict[str, dict] = _BACKEND["refunds"]       # keyed by refund id


if __name__ == "__main__":
    # `uv run python shopwise_data.py` — the whole store, in one screen. Worth
    # running before you write a single tool: you cannot ground an assistant in
    # facts you have not looked at.
    for name, table in (("CUSTOMERS", CUSTOMERS), ("PRODUCTS", PRODUCTS),
                        ("ORDERS", ORDERS), ("SHIPMENTS", SHIPMENTS), ("REFUNDS", REFUNDS)):
        print(f"\n{name}  ({len(table)})")
        for key, row in table.items():
            print(f"  {key:24} {row}")
