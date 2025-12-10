import json
import re
from typing import Any, Dict, List, Optional, Tuple, Callable

from openai import OpenAI

SENSITIVE_TOOLS = {"get_customer", "list_orders", "get_order", "create_order"}
VERIFY_TOOL = "verify_customer_pin"

UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

SYSTEM = """
You are a helpful customer support chatbot for a company selling computer products (monitors, printers, etc).
You can use tools to look up products, customers, and orders.

Rules:

## PRODUCT LOOKUPS
- If the user asks to "show", "list", "display", "see", "find", "browse", or asks "do you have", or "what ... do you have", ALWAYS call list_products.
- If the user references a product category (e.g., "monitors", "printers", "computers"), map it directly to the `category` argument of list_products.
- Normalize the category name to Title Case (e.g., "monitors" → "Monitors").
- Never answer product availability manually; ALWAYS call list_products or get_product.

Examples:
User: "show me monitors" → call list_products({"category": "Monitors"})
User: "any printers available?" → call list_products({"category": "Printers"})
User: "list computers" → call list_products({"category": "Computers"})

## SENSITIVE ACTIONS
- For get_customer, list_orders, get_order, and create_order, the user must be verified via verify_customer_pin(email, pin).

## ORDER CREATION
- Before create_order: ALWAYS call get_product for each sku to confirm price and stock; use that exact unit_price in create_order.

## GENERAL
- Never invent SKUs, prices, or inventory; always use the tools.
- Keep answers concise and action-oriented.
"""


def openai_tools_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_products",
                "description": "List products with optional filters: category, is_active",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": ["string", "null"]},
                        "is_active": {"type": ["boolean", "null"]},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": "Search products by name/description (query).",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_product",
                "description": "Get detailed product information by SKU.",
                "parameters": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verify_customer_pin",
                "description": "Verify customer identity with email + 4-digit PIN.",
                "parameters": {
                    "type": "object",
                    "properties": {"email": {"type": "string"}, "pin": {"type": "string"}},
                    "required": ["email", "pin"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_customer",
                "description": "Get customer info by customer_id (UUID).",
                "parameters": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "string"}},
                    "required": ["customer_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_orders",
                "description": "List orders with optional filters: customer_id, status.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": ["string", "null"]},
                        "status": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": "Get order details by order_id (UUID).",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_order",
                "description": "Create a new order for a customer_id with items [{sku, quantity, unit_price, currency}].",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"},
                        "items": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["customer_id", "items"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _extract_uuid(text: str) -> Optional[str]:
    m = UUID_RE.search(text or "")
    return m.group(0) if m else None


def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


def _autofill_customer_id(tool_name: str, args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    If verified and we have customer_id, auto-inject it for tools that support/require it.
    This makes "track my order" work reliably.
    """
    cid = state.get("customer_id")
    if not cid or not state.get("verified"):
        return args

    args = dict(args or {})
    if tool_name == "list_orders":
        if args.get("customer_id") in (None, "", "null"):
            args["customer_id"] = cid
    elif tool_name == "get_customer":
        if args.get("customer_id") in (None, "", "null"):
            args["customer_id"] = cid
    elif tool_name == "create_order":
        if args.get("customer_id") in (None, "", "null"):
            args["customer_id"] = cid

    return args


async def run_agent_turn(
    user_text: str,
    chat_history: List[Dict[str, Any]],
    state: Dict[str, Any],
    mcp_call_fn: Callable[[str, Dict[str, Any]], Any],
    model: str,
    api_key: str,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    client = OpenAI(api_key=api_key)

    state = state or {}
    state.setdefault("verified", False)
    state.setdefault("customer_id", None)
    state.setdefault("customer_summary", None)

    # IMPORTANT: chat_history here is the full internal LLM history (including tools)
    messages: List[Dict[str, Any]] = (
        [{"role": "system", "content": SYSTEM}]
        + (chat_history or [])
        + [{"role": "user", "content": user_text}]
    )

    tools = openai_tools_schema()

    for _ in range(8):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg_obj = resp.choices[0].message
        msg = msg_obj.model_dump(exclude_none=True)

        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            messages.append(msg)

            for tc in tool_calls:
                name = tc["function"]["name"]
                args = _safe_json_loads(tc["function"].get("arguments", ""))

                # Gate sensitive tools until verified
                if name in SENSITIVE_TOOLS and not state.get("verified"):
                    tool_text = (
                        "ERROR: Customer not verified. Ask the user for email + 4-digit PIN, "
                        "then call verify_customer_pin."
                    )
                else:
                    args = _autofill_customer_id(name, args, state)
                    tool_text = await mcp_call_fn(name, args)

                if name == VERIFY_TOOL:
                    maybe_id = _extract_uuid(str(tool_text))
                    if maybe_id:
                        state["verified"] = True
                        state["customer_id"] = maybe_id
                        state["customer_summary"] = str(tool_text)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(tool_text),
                    }
                )
            continue

        assistant_text = (msg.get("content") or "").strip()
        messages.append({"role": "assistant", "content": assistant_text})

        new_history = [m for m in messages if m.get("role") != "system"]
        return assistant_text, new_history, state

    new_history = [m for m in messages if m.get("role") != "system"]
    return "I hit a tool loop. Can you rephrase your request in one sentence?", new_history, state
