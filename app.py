import os
import ast
import json
from typing import Any, Dict, List, Optional, Union

import gradio as gr
from dotenv import load_dotenv

from mcp_client import MCPToolClient
from agent import run_agent_turn

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://vipfapwm3x.us-east-1.awsapprunner.com/mcp")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

mcp = MCPToolClient(MCP_SERVER_URL)


# -------------------------
# Gradio 6 messages helpers
# -------------------------

ContentBlock = Dict[str, Any]
Message = Dict[str, Any]


def _text_block(text: str) -> ContentBlock:
    # Gradio 6 structured content block (OpenAI-style)
    return {"type": "text", "text": text}


def _maybe_parse_literal(s: str) -> Optional[Union[dict, list]]:
    """
    Try to parse strings like:
      "[{'text': 'hi', 'type': 'text'}]"
    or JSON strings. Returns parsed object or None.
    """
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t or t[0] not in "[{":
        return None

    # Try JSON first
    try:
        return json.loads(t)
    except Exception:
        pass

    # Then safe Python literal (handles single quotes)
    try:
        return ast.literal_eval(t)
    except Exception:
        return None


def _coerce_to_blocks(value: Any, *, _depth: int = 0, _max_depth: int = 6) -> List[ContentBlock]:
    """
    Convert whatever we have into Gradio 6 structured content blocks:
      [{"type":"text","text":"..."}]
    Also repairs the "nested string of list-of-dicts" corruption by repeatedly unwrapping.
    """
    if _depth > _max_depth:
        return [_text_block(str(value))]

    # Already correct: list of content blocks
    if isinstance(value, list) and value and all(isinstance(x, dict) and "type" in x for x in value):
        blocks: List[ContentBlock] = []
        for b in value:
            btype = b.get("type")
            if btype == "text":
                txt = b.get("text", "")
                # txt itself might be a stringified list; unwrap it
                parsed = _maybe_parse_literal(txt) if isinstance(txt, str) else None
                if parsed is not None:
                    blocks.extend(_coerce_to_blocks(parsed, _depth=_depth + 1))
                else:
                    blocks.append(_text_block(str(txt)))
            else:
                # file/image/audio/etc blocks: pass through as-is
                blocks.append(b)
        return blocks or [_text_block("")]

    # Single content block dict
    if isinstance(value, dict):
        # Sometimes you may have old-style dicts like {"text": "...", "type": "text"}
        if value.get("type") == "text":
            txt = value.get("text", value.get("content", ""))
            if isinstance(txt, str):
                parsed = _maybe_parse_literal(txt)
                if parsed is not None:
                    return _coerce_to_blocks(parsed, _depth=_depth + 1)
            return [_text_block(str(txt))]

        # Unknown dict → render as pretty JSON text
        return [_text_block(json.dumps(value, indent=2, default=str))]

    # Plain string
    if isinstance(value, str):
        parsed = _maybe_parse_literal(value)
        if parsed is not None:
            return _coerce_to_blocks(parsed, _depth=_depth + 1)
        return [_text_block(value)]

    # Fallback
    return [_text_block(str(value))]


def _normalize_ui_history(history: Any) -> List[Message]:
    """
    Ensure Chatbot value is always:
      [{"role":"user|assistant", "content":[{"type":"text","text":"..."}]}, ...]
    Handles:
      - correct messages format
      - tuples format (legacy)
      - already-corrupted content strings (repairs them)
    """
    if not history:
        return []

    cleaned: List[Message] = []

    # Legacy: list of (user, assistant) tuples/lists
    if isinstance(history, list) and history and isinstance(history[0], (tuple, list)) and len(history[0]) == 2:
        for u, a in history:
            if u not in (None, ""):
                cleaned.append({"role": "user", "content": _coerce_to_blocks(u)})
            if a not in (None, ""):
                cleaned.append({"role": "assistant", "content": _coerce_to_blocks(a)})
        return cleaned

    # Messages format
    if isinstance(history, list):
        for m in history:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            content = m.get("content", "")
            cleaned.append({"role": role, "content": _coerce_to_blocks(content)})

    return cleaned


async def respond(user_message: str, history: Any, state: Dict[str, Any]):
    state = state or {}
    state.setdefault("verified", False)
    state.setdefault("customer_id", None)
    state.setdefault("customer_summary", None)
    state.setdefault("ui_history", [])
    state.setdefault("llm_history", [])

    # Repair / normalize whatever Gradio gives us (and whatever we stored earlier)
    ui_history = _normalize_ui_history(history) or _normalize_ui_history(state.get("ui_history", []))

    llm_history = state.get("llm_history", [])
    if not isinstance(llm_history, list):
        llm_history = []

    async def mcp_call(name, args):
        return await mcp.call_tool(name, args)

    assistant_text, new_llm_history, new_state = await run_agent_turn(
        user_text=user_message,
        chat_history=llm_history,
        state=state,
        mcp_call_fn=mcp_call,
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
    )

    assistant_text = (assistant_text or "").strip() or "Okay — what would you like to do next?"

    # Append in the *correct* Gradio 6 structured content format (NO str() on content!)
    ui_history.append({"role": "user", "content": _coerce_to_blocks(user_message)})
    ui_history.append({"role": "assistant", "content": _coerce_to_blocks(assistant_text)})

    new_state["ui_history"] = ui_history
    new_state["llm_history"] = new_llm_history

    return "", ui_history, new_state


with gr.Blocks(title="Computer Products Support Bot") as demo:
    gr.Markdown(
        "## 🛠️ Customer Support Chatbot (MCP Tools + Mini LLM)\n"
        "Ask about monitors/printers, product details, or order status (verification required)."
    )

    state = gr.State(
        {
            "verified": False,
            "customer_id": None,
            "customer_summary": None,
            "ui_history": [],
            "llm_history": [],
        }
    )

    # IMPORTANT: type="messages" so we can pass OpenAI-style message dicts. :contentReference[oaicite:1]{index=1}
    chatbot = gr.Chatbot(
        height=420,
        type="messages",
        render_markdown=True,
        sanitize_html=True,
        line_breaks=True,
        layout="bubble",
    )

    msg = gr.Textbox(
        label="Message",
        placeholder="E.g., 'Show me monitors' or 'Track my order'",
    )

    with gr.Row():
        send = gr.Button("Send", variant="primary")
        reset = gr.Button("Reset")

    with gr.Accordion("🔐 Verification status", open=False):
        verified_box = gr.Markdown("❌ Not verified")
        customer_box = gr.Markdown("")

    def update_status(st):
        if st and st.get("verified"):
            return "✅ Verified", f"**Customer**:\n\n{st.get('customer_summary','')}"
        return "❌ Not verified", ""

    def on_reset():
        fresh = {
            "verified": False,
            "customer_id": None,
            "customer_summary": None,
            "ui_history": [],
            "llm_history": [],
        }
        return [], fresh, "❌ Not verified", ""

    send.click(respond, inputs=[msg, chatbot, state], outputs=[msg, chatbot, state]).then(
        update_status, inputs=[state], outputs=[verified_box, customer_box]
    )

    reset.click(on_reset, outputs=[chatbot, state, verified_box, customer_box])

demo.queue()
demo.launch()
