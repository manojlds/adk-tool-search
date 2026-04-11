# adk-tool-search

Dynamic tool search for Google ADK — load tools on demand instead of all at once.

Implements the [Anthropic Tool Search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool) pattern for Google's Agent Development Kit (ADK). Instead of loading all tool definitions into context upfront, the agent discovers and loads tools on demand using BM25 search.

Primary integration target: standard ADK `LlmAgent` wiring with `ToolRegistry` + callbacks.

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
│  4. Tool is marked loaded for this session           │
│  5. before_model_callback injects it next turn       │
│  6. Agent calls get_forecast(location="Tokyo")      │
└─────────────────────────────────────────────────────┘
```

Loaded tools are session-scoped. A tool loaded in one session is not exposed to other sessions.

Persistence model:
- Loaded tool names are written to ADK session state (`adk_tool_search.loaded_tools`) on `load_tool`.
- `before_model_callback` reads that state and injects only those tools for the current session.
- With persistent session services (SQLite/DB/Vertex), loaded tools survive process restarts.
- With in-memory session services, restart continuity is not available.

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

### Recommended: use standard ADK `LlmAgent`

#### With plain Python functions

```python
from google.adk.agents import LlmAgent
from adk_tool_search import (
    ToolRegistry,
    create_search_and_load_tools,
    create_session_scoped_loader_callbacks,
)

def get_weather(location: str) -> dict:
    """Get current weather for a location."""
    return {"location": location, "temp": 22, "condition": "sunny"}

def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email."""
    return {"status": "sent"}

# 1. Register tools in the search index
registry = ToolRegistry()
registry.register_many([get_weather, send_email])

# 2. Create the lightweight discovery tools
search_tools, load_tool = create_search_and_load_tools(registry)

# 3. Create session-scoped loader callbacks
before_model_callback, after_tool_callback = create_session_scoped_loader_callbacks(registry)

# 4. Wire into a normal ADK LlmAgent
agent = LlmAgent(
    name="Assistant",
    model="gemini-2.5-flash",
    instruction="Use search_tools to find tools, load_tool to activate them, then call them.",
    tools=[search_tools, load_tool],
    before_model_callback=before_model_callback,
    after_tool_callback=after_tool_callback,
)
```

#### With MCP servers

```python
from google.adk.agents import LlmAgent
from google.adk.tools.mcp import MCPToolset, StdioConnectionParams
from adk_tool_search import (
    ToolRegistry,
    create_search_and_load_tools,
    create_session_scoped_loader_callbacks,
)

# Fetch tools from MCP server (but don't give to agent)
mcp = MCPToolset(connection_params=StdioConnectionParams(command="npx", args=["-y", "@modelcontextprotocol/server-github"]))
mcp_tools = await mcp.get_tools()

# Index all MCP tools
registry = ToolRegistry()
registry.register_many(mcp_tools)

# Create search/load tools + callbacks
search_tools, load_tool = create_search_and_load_tools(registry)
before_model_callback, after_tool_callback = create_session_scoped_loader_callbacks(registry)

# Wire up a normal ADK LlmAgent
agent = LlmAgent(
    name="GitHubAssistant",
    model="gemini-2.5-flash",
    instruction="Use search_tools to find tools, load_tool to activate them, then call them.",
    tools=[search_tools, load_tool],
    before_model_callback=before_model_callback,
    after_tool_callback=after_tool_callback,
)
```

### Optional helper factory

If you prefer less boilerplate, `create_tool_search_agent` wraps the above wiring:

```python
from adk_tool_search import ToolRegistry, create_tool_search_agent

registry = ToolRegistry()
registry.register_many([get_weather, send_email])

agent = create_tool_search_agent(
    name="Assistant",
    model="gemini-2.5-flash",
    registry=registry,
)

# Tools loaded via load_tool are session-scoped.
# A tool loaded in one session is not visible to other sessions unless they load it too.
```

## Examples

```bash
# Plain function tools demo
uv run python examples/function_tools_demo.py

# MCP server demo (requires GITHUB_TOKEN)
GITHUB_TOKEN=ghp_... uv run python examples/mcp_demo.py
```

## API

### `ToolRegistry`
- `register(tool)` — Register a single tool (function, ADK tool, or MCP tool)
- `register_many(tools)` — Register multiple tools (rebuilds index once)
- `search(query, n=5)` — BM25-first search with lexical fallback for tiny registries, returns `["name: snippet", ...]`
- `get_tool(name)` — Get tool object by exact name
- `tool_count` / `tool_names` — Introspection properties

### `create_search_and_load_tools(registry)`
Returns `(search_tools, load_tool)` — the two lightweight functions to give your agent.

### `create_session_scoped_loader_callbacks(registry)`
Returns `(before_model_callback, after_tool_callback)` that keep loaded tools scoped to each session.

### `create_tool_search_agent(...)` (optional helper)
Convenience wrapper around manual `LlmAgent` wiring.
- `name`, `model` — Standard Agent params
- `registry` — A populated `ToolRegistry`
- `instruction` — Optional custom instruction
- `always_available_tools` — Tools that skip deferred loading
- `**agent_kwargs` — Forwarded to `Agent()`
