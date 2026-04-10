# adk-tool-search

Dynamic tool search for Google ADK. Implements Anthropic-style deferred tool loading
for Google's Agent Development Kit, using BM25 search to discover and load tools on demand.

## Architecture
- `adk_tool_search/registry.py` - BM25-based tool registry for indexing and searching tools
- `adk_tool_search/loader.py` - Dynamic loader callback and agent factory
- Supports plain Python functions, ADK tools, and MCP tools
