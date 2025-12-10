# file: inspect_mcp_server.py
import asyncio
import json
from typing import Any

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = "https://vipfapwm3x.us-east-1.awsapprunner.com/mcp"
# If your server needs extra headers (Auth), add them here:
HEADERS = {
    # "Authorization": "Bearer YOUR_TOKEN",
    # ... add other headers if necessary
}

def safe_serialize(obj: Any) -> Any:
    """Try to convert mcp SDK objects to JSON-serializable forms."""
    # pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # pydantic v1
    if hasattr(obj, "dict"):
        return obj.dict()
    # dataclass / simple object
    if hasattr(obj, "__dict__"):
        return {k: safe_serialize(v) for k, v in obj.__dict__.items()}
    # fallback primitives / lists / dicts
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    return obj

async def inspect_server():
    # streamablehttp_client yields (read_stream, write_stream, extra) as per examples
    async with streamablehttp_client(MCP_URL, headers=HEADERS) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            # perform initialize handshake
            await session.initialize()

            out = {"tools": [], "prompts": [], "resources": []}

            # list tools
            tools_resp = await session.list_tools()
            for t in tools_resp.tools:
                out["tools"].append(safe_serialize(t))

            # list prompts (and optionally fetch the generated messages for a sample args)
            prompts_resp = await session.list_prompts()
            for p in prompts_resp.prompts:
                # try to call/get a sample of the prompt with no args (or sample args)
                try:
                    # NOTE: some prompts require args and may error; handle gracefully
                    sample_args = {}  # change if you know required args
                    prompt_result = await session.get_prompt(p.name, arguments=sample_args)
                    out_prompt = {
                        **safe_serialize(p),
                        "sample_messages": safe_serialize(prompt_result.messages),
                    }
                except Exception as e:
                    out_prompt = {**safe_serialize(p), "sample_error": repr(e)}
                out["prompts"].append(out_prompt)

            # list resources (and try a read of the first few if supported)
            resources_resp = await session.list_resources()
            for r in resources_resp.resources:
                res_entry = safe_serialize(r)
                # optionally try to read a small resource (if it looks like a resolvable URI)
                try:
                    # pick a sample URI: the resource's "uri" or pattern may require parameters
                    uri = getattr(r, "uri", None) or (getattr(r, "uriPattern", None) if hasattr(r, "uriPattern") else None)
                    if uri and "{" not in str(uri):  # only automatically read if URI is concrete
                        content_resp = await session.read_resource(types.AnyUrl(str(uri)))
                        res_entry["contents"] = safe_serialize(content_resp.contents)
                except Exception as e:
                    res_entry["read_error"] = repr(e)
                out["resources"].append(res_entry)

            # print a pretty JSON dump
            print(json.dumps(out, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(inspect_server())
