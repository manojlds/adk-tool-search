# DRS Project Context

## What adk-tool-search Is
Python library that adds Anthropic-style deferred tool loading to Google ADK.
Agents start with only `search_tools` and `load_tool`, then discover and load
relevant tools on demand using BM25 ranking.

## Core Flow
- Build a `ToolRegistry` from Python functions, ADK tools, or MCP tools.
- Run `search_tools(query)` to get top matching tools.
- Run `load_tool(tool_name)` to enable a selected tool.
- Invoke the loaded tool directly in subsequent turns.

## Architecture
- `adk_tool_search/registry.py`: tool metadata extraction, BM25 indexing/search,
  zero-score filtering for no-match behavior.
- `adk_tool_search/loader.py`: search/load tool factories, dynamic loading callbacks,
  and `create_tool_search_agent()` convenience factory.
- `tests/test_registry.py`: unit coverage for registration and retrieval behavior.
- `tests/llm/test_tool_search_agent.py`: integration tests with live LLM backend,
  including positive flows and no-match behavior.

## Technology Stack
- Language: Python 3.11+
- Agent framework: Google ADK
- Retrieval: `rank-bm25`
- Env/config: `python-dotenv`
- Package/build: `uv`, Hatchling
- Testing: `pytest`, `pytest-asyncio`
- Lint/format: `ruff`

## Security and Reliability Focus
- Avoid leaking tools across sessions; loaded tool availability should be scoped.
- Ensure no false-positive tool search results for unrelated queries.
- Validate tool lookup and missing-tool handling paths.
- Keep callback behavior deterministic and side-effect-aware.

## Review Focus
- Correctness of session-scoped tool loading and callback wiring.
- BM25 search quality and no-match handling.
- Test quality for both unit and LLM integration paths.
- Clear docs and examples that match current behavior.
