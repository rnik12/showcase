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

async def respond(user_message, history, state):
    # history in gradio is list[list[str,str]]; we’ll convert to OpenAI messages.
    openai_history = []
    for u, a in (history or []):
        openai_history.append({"role": "user", "content": u})
        openai_history.append({"role": "assistant", "content": a})

    state = state or {"verified": False, "customer_id": None, "customer_summary": None}

    async def mcp_call(name, args):
        return await mcp.call_tool(name, args)

    answer, new_openai_history, new_state = await run_agent_turn(
        user_text=user_message,
        chat_history=openai_history,
        state=state,
        mcp_call_fn=mcp_call,
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
    )

    # Convert back to gradio history
    new_pairs = []
    # Walk messages two at a time (user/assistant); ignore tool messages for display
    pending_user = None
    for m in new_openai_history:
        if m["role"] == "user":
            pending_user = m["content"]
        elif m["role"] == "assistant" and pending_user is not None:
            new_pairs.append([pending_user, m.get("content", "")])
            pending_user = None

    return "", new_pairs, new_state

with gr.Blocks(title="Computer Products Support Bot") as demo:
    gr.Markdown("## 🛠️ Customer Support Chatbot (MCP Tools + Mini LLM)\nAsk about monitors/printers, product details, or order status (verification required).")

    state = gr.State({"verified": False, "customer_id": None, "customer_summary": None})

    chatbot = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="E.g., 'Show me 27-inch monitors under $300' or 'Track my order'")

    with gr.Row():
        send = gr.Button("Send", variant="primary")
        reset = gr.Button("Reset")

    with gr.Accordion("🔐 Verification status", open=False):
        verified_box = gr.Markdown("Not verified yet.")
        customer_box = gr.Markdown("")

    async def update_status(st):
        if st.get("verified"):
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
