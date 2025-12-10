import json
from typing import Any, Dict, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def _to_text(resp: Any) -> str:
    """
    MCP SDK responses vary. We try the common patterns and fall back to JSON.
    Your tool output schema says it returns {"result": "..."}.
    """
    if hasattr(resp, "model_dump"):
        data = resp.model_dump()
        if isinstance(data, dict) and "result" in data:
            return str(data["result"])
        return json.dumps(data, indent=2, default=str)

    if isinstance(resp, dict):
        if "result" in resp:
            return str(resp["result"])
        return json.dumps(resp, indent=2, default=str)

    return str(resp)


class MCPToolClient:
    def __init__(self, mcp_url: str, headers: Optional[Dict[str, str]] = None):
        self.mcp_url = mcp_url
        self.headers = headers or {}

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        async with streamablehttp_client(self.mcp_url, headers=self.headers) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                resp = await session.call_tool(name, arguments=arguments)
                return _to_text(resp)
