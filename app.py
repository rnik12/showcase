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


def _normalize_ui_history(history):
    """
    Gradio 6.x Chatbot expects:
      [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    Be tolerant if history is None/empty or contains weird items.
    """
    if not history:
        return []
    cleaned = []
    for m in history:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            cleaned.append({"role": m["role"], "content": str(m.get("content", ""))})
    return cleaned


async def respond(user_message, history, state):
    state = state or {}
    state.setdefault("verified", False)
    state.setdefault("customer_id", None)
    state.setdefault("customer_summary", None)

    # UI history shown in Chatbot (user/assistant only)
    ui_history = _normalize_ui_history(history) or _normalize_ui_history(state.get("ui_history", []))

    # LLM history (includes tool messages). THIS is what the model sees next turn.
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

    # Update UI history (append, don't rebuild)
    ui_history.append({"role": "user", "content": user_message})
    ui_history.append({"role": "assistant", "content": assistant_text})

    # Persist both histories in state
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

    chatbot = gr.Chatbot(height=420)  # messages format in Gradio 6.x

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
