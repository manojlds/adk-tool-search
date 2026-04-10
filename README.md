# adk-tool-search

Dynamic tool search for Google ADK — load tools on demand instead of all at once.

Implements the [Anthropic Tool Search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool) pattern for Google's Agent Development Kit (ADK). Instead of loading all tool definitions into context upfront, the agent discovers and loads tools on demand using BM25 search.

## Why?

| Problem | Impact |
|---|---|
| **Context bloat** | A typical multi-MCP setup can consume 50k+ tokens in tool definitions before the agent does any work |
| **Tool selection accuracy** | LLM ability to pick the right tool degrades past 30-50 tools |
| **Gemini's 100-tool limit** | Hard cap on function declarations in the Gemini API |

This library reduces context usage by ~95% and keeps tool selection accurate across hundreds of tools.

## How it works

```
┌─────────────────────────────────────────────────────┐
│  Startup                                            │
│  1. Fetch tools from MCP servers / register funcs   │
│  2. Index all tools in BM25 registry                │
│  3. Agent starts with only: search_tools, load_tool │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Runtime (per user request)                         │
│  1. Agent calls search_tools("weather forecast")    │
│  2. Registry returns top-5 matches (name + snippet) │
│  3. Agent calls load_tool("get_forecast")           │
│  4. after_tool_callback injects tool into agent     │
│  5. Agent calls get_forecast(location="Tokyo")      │
└─────────────────────────────────────────────────────┘
```

## Install

```bash
pip install adk-tool-search
```

### Development setup

```bash
git clone https://github.com/manojlds/adk-tool-search.git
cd adk-tool-search
uv sync --all-extras
```

## Quick start

### With plain Python functions

```python
from adk_tool_search import ToolRegistry, create_tool_search_agent

def get_weather(location: str) -> dict:
    """Get current weather for a location."""
    return {"location": location, "temp": 22, "condition": "sunny"}

def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email."""
    return {"status": "sent"}

# Register tools
registry = ToolRegistry()
registry.register_many([get_weather, send_email])

# Create agent — starts with only search_tools + load_tool
agent = create_tool_search_agent(
    name="Assistant",
    model="gemini-2.5-flash",
    registry=registry,
)
```

### With MCP servers

```python
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from adk_tool_search import ToolRegistry, create_tool_search_agent

# Fetch tools from MCP server (but don't give to agent)
mcp = MCPToolset(connection_params=...)
mcp_tools = await mcp.get_tools()

# Index and create agent
registry = ToolRegistry()
registry.register_many(mcp_tools)

agent = create_tool_search_agent(
    name="Assistant",
    model="gemini-2.5-flash",
    registry=registry,
)
```

## Examples

```bash
# Plain function tools demo
python examples/function_tools_demo.py

# MCP server demo (requires GITHUB_TOKEN)
GITHUB_TOKEN=ghp_... python examples/mcp_demo.py
```

## API

### `ToolRegistry`
- `register(tool)` — Register a single tool (function, ADK tool, or MCP tool)
- `register_many(tools)` — Register multiple tools (rebuilds index once)
- `search(query, n=5)` — BM25 search, returns `["name: snippet", ...]`
- `get_tool(name)` — Get tool object by exact name
- `tool_count` / `tool_names` — Introspection properties

### `create_tool_search_agent(...)`
- `name`, `model` — Standard Agent params
- `registry` — A populated `ToolRegistry`
- `instruction` — Optional custom instruction
- `always_available_tools` — Tools that skip deferred loading
- `**agent_kwargs` — Forwarded to `Agent()`
