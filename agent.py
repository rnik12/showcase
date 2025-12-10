import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

SENSITIVE_TOOLS = {"get_customer", "list_orders", "get_order", "create_order"}
VERIFY_TOOL = "verify_customer_pin"

UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)

SYSTEM = """You are a helpful customer support chatbot for a company selling computer products (monitors, printers, etc).
You can use tools to look up products, customers, and orders.

Rules:
- For product questions: use search_products/list_products/get_product as needed.
- For any customer/order actions (get_customer, list_orders, get_order, create_order), the user must be verified first via verify_customer_pin(email, pin).
- Before create_order: ALWAYS call get_product for each sku to confirm price and stock; use that unit_price in create_order.
- Never invent SKUs, prices, or inventory. If unsure, call tools.
- Keep answers concise and action-oriented. Ask only necessary questions.
"""

def openai_tools_schema() -> List[Dict[str, Any]]:
    # Manually mirror the MCP tool schemas you posted (enough for a prototype).
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

async def run_agent_turn(
    user_text: str,
    chat_history: List[Dict[str, Any]],
    state: Dict[str, Any],
    mcp_call_fn,
    model: str,
    api_key: str,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    chat_history is OpenAI-style messages WITHOUT the system.
    state holds:
      - verified: bool
      - customer_id: optional uuid
      - customer_summary: optional string
    """
    client = OpenAI(api_key=api_key)

    messages = [{"role": "system", "content": SYSTEM}] + chat_history + [{"role": "user", "content": user_text}]
    tools = openai_tools_schema()

    for _ in range(8):  # tool loop safety cap
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # If model wants to call tools:
        if msg.tool_calls:
            messages.append(msg)  # assistant message with tool_calls

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")

                # Gate sensitive tools until verified
                if name in SENSITIVE_TOOLS and not state.get("verified"):
                    tool_text = (
                        "ERROR: Customer not verified. Ask the user for email + 4-digit PIN, "
                        "then call verify_customer_pin."
                    )
                else:
                    tool_text = await mcp_call_fn(name, args)

                # If verification succeeded, store customer_id if present
                if name == VERIFY_TOOL:
                    maybe_id = _extract_uuid(tool_text)
                    if maybe_id:
                        state["verified"] = True
                        state["customer_id"] = maybe_id
                        state["customer_summary"] = tool_text

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_text,
                    }
                )
            continue

        # Otherwise final assistant text
        assistant_text = msg.content or ""
        # Persist history without system
        new_history = [m for m in messages if m["role"] != "system"]
        return assistant_text, new_history, state

    # If we hit the loop cap:
    new_history = [m for m in messages if m["role"] != "system"]
    return "I hit a tool loop. Can you rephrase your request in one sentence?", new_history, state
