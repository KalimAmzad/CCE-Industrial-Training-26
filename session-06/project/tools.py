"""The seven things the assistant may ask ShopWise's backend to do.

WHAT A TOOL IS, IN ONE SENTENCE
    A Python function the model may ask your code to run. It does not run it.
    It reads the four things below, decides one would help, and emits a
    *request*; the loop in agent.py executes that request and hands back the
    result. Every line in this file is written for that reader.

THE FOUR PARTS THE MODEL ACTUALLY READS
    1. THE NAME        a verb. `get_order`, not `order_handler`. It is the
                       first thing the model matches a question against.
    2. THE ARGUMENTS   with type hints, which are not documentation here: the
                       @tool decorator turns them into the JSON schema the
                       model is given, and an untyped argument arrives at the
                       model as an untyped blank.
    3. THE DOCSTRING   written for the model, not for a colleague. Say WHEN to
                       use this and WHAT comes back. It is the routing: the
                       model picks between seven tools on the strength of seven
                       paragraphs, and a vague one loses to a precise one.
    4. THE RETURN      plain data — a dict or a list of dicts — that survives
                       being serialised to text, because text is what the model
                       receives.

WHY THE RETURNS ARE PLAIN DICTS AND NOT PYDANTIC MODELS
    Session 3's rule was: validate at the boundary. This looks like a boundary
    and it bends the rule, deliberately.

    The rule is about data coming IN — a model's JSON answer entering your
    program, where a wrong type becomes a crash three functions later. A tool's
    return goes the other way. It leaves your program, is serialised to a
    string, and lands in a message the model reads. Nothing downstream indexes
    it; nothing type-checks it. A Pydantic model here buys validation of data
    you just produced yourself from a fixture you control, and costs a
    `.model_dump()` at every call site.

    Where it does not bend: what comes back from the *model*. v2 still parses
    that through schemas.py, and section 10 uses the very same TriageResult.

THE ERROR CONTRACT — RETURN, DO NOT RAISE
    Every lookup below returns `{"error": "..."}` when it finds nothing, rather
    than raising. That is a choice with a cost, and section 9 measures both
    halves of it. In short: "no such order" is a fact the model can act on — it
    can ask for a correct number, or say honestly that it cannot find it. An
    exception is not a fact, it is the end of the turn.

    The cost is that a caller who is NOT a model — a script, a test, another
    function — has to remember to check for the key, and Python will not remind
    it. `if "error" in result` is a habit; `except KeyError` is enforced.

THE SIGNATURES ARE FIXED HERE FOR THE REST OF THE COURSE
    PROJECT-SPINE §3. Sessions 6 to 16 add tools — search_policy in S09,
    web_search in S12 — and never change these seven, because eleven sessions
    of notebooks, prompts and exercises name them.
"""

from __future__ import annotations

from langchain.tools import tool

from shopwise_data import CUSTOMERS, ORDERS, PRODUCTS, REFUNDS, SHIPMENTS


@tool
def lookup_customer(email: str) -> dict:
    """Find a ShopWise customer by their email address.

    Use this FIRST whenever a customer identifies themselves by email, because
    every other customer-level tool needs the customer_id this returns.

    Returns customer_id, name, tier and the date they joined, or
    {"error": "..."} if no customer has that address.
    """
    customer = CUSTOMERS.get(email.strip().lower())
    if customer is None:
        return {"error": f"no customer found with email {email}"}
    return {"email": email.strip().lower(), **customer}


@tool
def get_order(order_id: str) -> dict:
    """Get one order by its ID, including what was in it.

    Use this when the customer names an order, e.g. "ORD-5001". Order IDs look
    like ORD-NNNN.

    Returns the order's status, dates, total, and its items with product names
    and prices filled in — or {"error": "..."} if there is no such order.
    """
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        return {"error": f"no order found with id {order_id}"}

    # The stored order holds SKUs and quantities. A customer cannot read
    # "1 x SKU-100", so the names are joined on here rather than left for the
    # model to guess — a model asked to expand a SKU will happily invent a
    # product name, and it will sound right.
    items = [
        {
            "sku": item["sku"],
            "qty": item["qty"],
            "name": PRODUCTS.get(item["sku"], {}).get("name", "unknown product"),
            "unit_price": PRODUCTS.get(item["sku"], {}).get("price"),
        }
        for item in order["items"]
    ]
    return {"order_id": order_id.strip().upper(), **order, "items": items}


@tool
def list_orders(customer_id: str) -> list[dict]:
    """List every order a customer has placed, most recent first.

    Takes a customer_id like "C-1001", NOT an email — call lookup_customer
    first to turn an email into a customer_id.

    Returns one entry per order with its id, status, date and total. Returns an
    empty list if the customer has never ordered anything.
    """
    found = [
        {"order_id": oid, "status": o["status"], "order_date": o["order_date"], "total": o["total"]}
        for oid, o in ORDERS.items()
        if o["customer_id"] == customer_id.strip().upper()
    ]
    return sorted(found, key=lambda o: o["order_date"], reverse=True)


@tool
def get_shipment(order_id: str) -> dict:
    """Get tracking information for an order that has shipped.

    Use this for "where is my parcel" questions, after get_order confirms the
    order exists.

    Returns carrier, tracking_id, status and estimated delivery date — or
    {"error": "..."} if the order has not shipped yet, which is a normal answer
    and not a failure.
    """
    shipment = SHIPMENTS.get(order_id.strip().upper())
    if shipment is None:
        return {"error": f"no shipment yet for order {order_id}"}
    return {"order_id": order_id.strip().upper(), **shipment}


@tool
def get_product(sku_or_name: str) -> dict:
    """Look up a product by SKU or by name.

    Accepts either a SKU like "SKU-300" or part of a product name as the
    customer said it — "kettle", "the backpack". Use it for questions about
    price, availability or warranty.

    Returns sku, name, product_type, price, stock and warranty_months. stock 0
    means out of stock; warranty_months 0 means the product has no warranty at
    all. Returns {"error": "...", "available": [...]} listing what IS stocked if
    nothing matches, and {"error": "...", "candidates": [...]} if the name is
    ambiguous.
    """
    query = sku_or_name.strip()

    if query.upper() in PRODUCTS:
        return {"sku": query.upper(), **PRODUCTS[query.upper()]}

    # Fuzzy half. Customers say "the kettle"; the catalogue says "BrewMaster
    # Electric Kettle". Substring on a lowercased name is the whole algorithm —
    # deliberately, because four products do not need an index, and a reader
    # who can see the matching rule can predict what the tool will do.
    needle = query.lower()
    matches = [sku for sku, p in PRODUCTS.items() if needle in p["name"].lower()]
    if len(matches) == 1:
        return {"sku": matches[0], **PRODUCTS[matches[0]]}
    if len(matches) > 1:
        # Ambiguity is a fact, and a fact the model can act on: it can ask which
        # one the customer meant. Picking the first would be a guess presented
        # as an answer.
        return {"error": f"{query!r} matches more than one product",
                "candidates": [{"sku": s, "name": PRODUCTS[s]["name"]} for s in matches]}
    return {"error": f"no product matches {query!r}",
            "available": [{"sku": s, "name": p["name"]} for s, p in PRODUCTS.items()]}


@tool
def check_refund_status(refund_id: str) -> dict:
    """Check where a refund has got to.

    Use this when a customer asks about a refund they have already requested
    and gives a refund ID like "REF-9001".

    Returns the refund's status, amount, the order it belongs to and the reason
    recorded — or {"error": "..."} if there is no such refund. Status
    "requested" means it has been logged but no money has moved yet.
    """
    refund = REFUNDS.get(refund_id.strip().upper())
    if refund is None:
        return {"error": f"no refund found with id {refund_id}"}
    return {"refund_id": refund_id.strip().upper(), **refund}


@tool
def issue_refund(order_id: str, amount: float, reason: str) -> dict:
    """Refund money to a customer for an order. This moves real money.

    Only use when a refund has been explicitly approved. Returns the created
    refund record, or {"error": "..."} if the order does not exist or the
    amount exceeds what was paid.
    """
    # Everything below is ordinary defensive code, and it is here for a reason
    # worth saying out loud: these checks are NOT the model's job. The model
    # advises; code decides. A refund that must not exceed the order total is a
    # rule, and a rule enforced by a paragraph of English in a prompt is a rule
    # you have chosen to enforce probabilistically.
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        return {"error": f"no order found with id {order_id}"}
    if amount <= 0:
        return {"error": "refund amount must be positive"}
    if amount > order["total"]:
        return {"error": f"refund of {amount} exceeds order total of {order['total']}"}

    refund_id = f"REF-{9001 + len(REFUNDS)}"
    REFUNDS[refund_id] = {
        "order_id": order_id.strip().upper(),
        "amount": amount,
        "status": "processed",
        "reason": reason,
        "requested_date": order["order_date"],
    }
    return {"refund_id": refund_id, **REFUNDS[refund_id]}


# The list the agent is given — and issue_refund is not in it.
#
# It is not commented out, not unfinished, and not broken: section 5.2 of the
# notebook calls it directly and watches it work. It is *withheld*. The
# assistant can look anything up and can move no money, because nothing here
# decides whether a refund is deserved, and the model is not allowed to decide
# that either.
#
# We do not give the new intern the company chequebook on their first day. In
# session 7 the chequebook comes out and a human signs for it.
TOOLS = [
    lookup_customer,
    get_order,
    list_orders,
    get_shipment,
    get_product,
    check_refund_status,
]


if __name__ == "__main__":
    # `uv run python tools.py` — every tool called directly, as the plain
    # functions they still are. No model, no agent, no key required. If a tool
    # is wrong, find out here, where the traceback is yours.
    print("TOOLS given to the agent:", [t.name for t in TOOLS])
    print("Defined but withheld:    ", issue_refund.name, "\n")

    print(lookup_customer.invoke({"email": "ayesha@example.com"}))
    print(get_order.invoke({"order_id": "ORD-5001"}))
    print(list_orders.invoke({"customer_id": "C-1001"}))
    print(get_shipment.invoke({"order_id": "ORD-5003"}))
    print(get_shipment.invoke({"order_id": "ORD-5001"}))   # the honest "not yet"
    print(get_product.invoke({"sku_or_name": "kettle"}))
    print(check_refund_status.invoke({"refund_id": "REF-9001"}))
    print(issue_refund.invoke({"order_id": "ORD-5001", "amount": 129.0,
                               "reason": "left earcup crackles"}))
