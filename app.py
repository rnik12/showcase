import os
import gradio as gr
from dotenv import load_dotenv

from mcp_client import MCPToolClient
from agent import run_agent_turn

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://vipfapwm3x.us-east-1.awsapprunner.com/mcp")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

mcp = MCPToolClient(MCP_SERVER_URL)

def _normalize_history(history):
    """
    Gradio 6.x Chatbot expects messages format:
      [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    But to be robust, handle old tuple history too.
    """
    if not history:
        return []

    # messages format
    if isinstance(history, list) and len(history) > 0 and isinstance(history[0], dict):
        cleaned = []
        for m in history:
            role = m.get("role")
            content = m.get("content", "")
            if role in ("user", "assistant") and isinstance(content, str):
                cleaned.append({"role": role, "content": content})
        return cleaned

    # tuple/list pair format fallback: [(user, assistant), ...]
    cleaned = []
    for pair in history:
        if not pair or len(pair) != 2:
            continue
        u, a = pair
        if u is not None:
            cleaned.append({"role": "user", "content": str(u)})
        if a is not None:
            cleaned.append({"role": "assistant", "content": str(a)})
    return cleaned

def _ui_history_from_openai(new_openai_history):
    """
    Convert OpenAI-style history (may include tool messages or assistant tool_call stubs)
    into Gradio messages format for display.
    """
    out = []
    for m in new_openai_history or []:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            # Skip assistant "tool_calls" messages that have no content
            continue
        out.append({"role": role, "content": content})
    return out

async def respond(user_message, history, state):
    openai_history = _normalize_history(history)

    state = state or {"verified": False, "customer_id": None, "customer_summary": None}

    async def mcp_call(name, args):
        return await mcp.call_tool(name, args)

    answer, new_openai_history, new_state = await run_agent_turn(
        user_text=user_message,
        chat_history=openai_history,   # already messages format
        state=state,
        mcp_call_fn=mcp_call,
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
    )

    ui_history = _ui_history_from_openai(new_openai_history)

    # Clear textbox, update chatbot, update state
    return "", ui_history, new_state

with gr.Blocks(title="Computer Products Support Bot") as demo:
    gr.Markdown(
        "## 🛠️ Customer Support Chatbot (MCP Tools + Mini LLM)\n"
        "Ask about monitors/printers, product details, or order status (verification required)."
    )

    state = gr.State({"verified": False, "customer_id": None, "customer_summary": None})

    # Gradio 6.x expects messages format by default
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
        return [], {"verified": False, "customer_id": None, "customer_summary": None}, "❌ Not verified", ""

    send.click(respond, inputs=[msg, chatbot, state], outputs=[msg, chatbot, state]).then(
        update_status, inputs=[state], outputs=[verified_box, customer_box]
    )

    reset.click(on_reset, outputs=[chatbot, state, verified_box, customer_box])

demo.queue()
demo.launch()
