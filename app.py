# showcase/app.py
import os
import ast
import json
import html
import gradio as gr
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional

from mcp_client import MCPToolClient
from agent import run_agent_turn

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://vipfapwm3x.us-east-1.awsapprunner.com/mcp")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

mcp = MCPToolClient(MCP_SERVER_URL)


def _to_content_list(obj) -> List[Dict[str, str]]:
    """
    Convert a variety of shapes into Gradio chat content list:
      [{"type":"text", "text":"..."}]
    If obj is a list of content-like dicts, normalize the keys.
    If obj is a string, return single text content.
    """
    if obj is None:
        return []

    # If it's already a list-like of content dicts
    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, dict):
                # possible shapes: {"type":"text","text":"..."}, or {"text":"...","type":"text"}, or {"type":"file",...}
                if "type" in item and "text" in item:
                    out.append({"type": item["type"], "text": str(item["text"])})
                elif "text" in item:
                    out.append({"type": "text", "text": str(item["text"])})
                elif "file" in item:
                    # represent file as a text placeholder (frontend can handle file dicts too if you want)
                    out.append({"type": "file", "text": str(item.get("file", item.get("file_path", "")))})
                else:
                    out.append({"type": "text", "text": json.dumps(item, default=str)})
            else:
                out.append({"type": "text", "text": str(item)})
        return out

    # If obj is a dict with "text"
    if isinstance(obj, dict):
        if "text" in obj:
            return [{"type": obj.get("type", "text"), "text": str(obj["text"])}]
        # otherwise string-ify
        return [{"type": "text", "text": json.dumps(obj, default=str)}]

    # If obj is a plain string
    return [{"type": "text", "text": str(obj)}]


def _deep_parse_string(s: str):
    """
    Try to decode nested stringified JSON / Python-literal structures.
    Attempts json.loads, ast.literal_eval, and repeats a few times.
    Returns a Python object (list/dict/str) or original string on failure.
    """
    if not isinstance(s, str):
        return s

    candidate = s
    # unescape HTML entities (sometimes frontend stores escaped)
    candidate = html.unescape(candidate)

    for _ in range(6):
        try:
            parsed = json.loads(candidate)
            # If we parsed to a primitive string but it still contains JSON-like, continue
            if isinstance(parsed, (dict, list)):
                return parsed
            # If parsed to str and it's different, try again
            if isinstance(parsed, str) and parsed != candidate:
                candidate = parsed
                continue
            return parsed
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, (dict, list)):
                    return parsed
                if isinstance(parsed, str) and parsed != candidate:
                    candidate = parsed
                    continue
                return parsed
            except Exception:
                # cannot parse further
                break
    # fallback: return original string
    return s


def _normalize_ui_history(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Normalize many possible history shapes into:
      [{"role":"user","content":[{"type":"text","text":"..."}]}, ...]
    Be tolerant of:
    - existing correct shape
    - shapes where content is a stringified list/dict (python repr or json)
    - content as a single text string
    - older openai-style lists with {"text": "...", "type": "text"} items
    """
    if not history:
        return []

    normalized = []
    for m in history:
        # safety: skip malformed entries
        if not isinstance(m, dict):
            continue

        role = m.get("role") or m.get("sender") or "user"
        raw_content = m.get("content")

        # If content is already the expected list format, attempt to normalize items
        if isinstance(raw_content, list):
            content_list = _to_content_list(raw_content)
        elif isinstance(raw_content, dict):
            # maybe {"type":"text","text":"..."} or old {"text":"..."}
            content_list = _to_content_list(raw_content)
        elif isinstance(raw_content, str):
            # Try deep parse (unpack nested encodings)
            parsed = _deep_parse_string(raw_content)
            if isinstance(parsed, (list, dict)):
                content_list = _to_content_list(parsed)
            else:
                # treat the original string as plain text
                content_list = _to_content_list(raw_content)
        else:
            # anything else -> string-ify
            content_list = _to_content_list(str(raw_content))

        # final guard: if empty, push a placeholder
        if not content_list:
            content_list = [{"type": "text", "text": ""}]

        normalized.append({"role": role, "content": content_list})

    return normalized


async def respond(user_message: str, history: List[Dict[str, Any]], state: Dict[str, Any]):
    """
    Main responder used by Gradio click event.
    Returns: (cleared textbox value, updated_chatbot_value, new_state)
    """
    state = state or {}
    # ensure structured defaults
    state.setdefault("verified", False)
    state.setdefault("customer_id", None)
    state.setdefault("customer_summary", None)
    state.setdefault("ui_history", [])
    state.setdefault("llm_history", [])

    # Normalize any existing UI history (tolerant)
    ui_history = _normalize_ui_history(history or state.get("ui_history", []))

    # LLM/internal history remains as-is (it's a list of message dicts)
    llm_history = state.get("llm_history", []) or []
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

    # Append structured message entries (do NOT store stringified lists)
    user_content = _to_content_list(user_message)
    assistant_content = _to_content_list(assistant_text)

    ui_history.append({"role": "user", "content": user_content})
    ui_history.append({"role": "assistant", "content": assistant_content})

    # Persist both histories in state
    new_state["ui_history"] = ui_history
    new_state["llm_history"] = new_llm_history

    # Clear the textbox return value, update chatbot and state
    return "", ui_history, new_state


with gr.Blocks(title="Computer Products Support Bot") as demo:
    gr.Markdown(
        "## 🛠️ Customer Support Chatbot (MCP Tools + Mini LLM)\n"
        "Ask about monitors/printers, product details, or order status (verification required)."
    )

    # state stores structured histories
    state = gr.State(
        {
            "verified": False,
            "customer_id": None,
            "customer_summary": None,
            "ui_history": [],
            "llm_history": [],
        }
    )

    # Chatbot expects messages in OpenAI-style role/content list format
    chatbot = gr.Chatbot(height=420)

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

    # hook up interaction (respond is async)
    send.click(respond, inputs=[msg, chatbot, state], outputs=[msg, chatbot, state]).then(
        update_status, inputs=[state], outputs=[verified_box, customer_box]
    )

    reset.click(on_reset, outputs=[chatbot, state, verified_box, customer_box])

demo.queue()
demo.launch()
