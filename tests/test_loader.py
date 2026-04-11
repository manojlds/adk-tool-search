"""Unit tests for loader helpers and session-scoped callbacks."""

from __future__ import annotations

from types import SimpleNamespace

from google.adk.tools.base_tool import BaseTool

from adk_tool_search import ToolRegistry
from adk_tool_search.loader import (
    create_search_and_load_tools,
    create_session_scoped_loader_callbacks,
)


def get_weather(location: str) -> dict:
    """Get weather for a location."""
    return {"location": location, "condition": "sunny"}


def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email."""
    return {"to": to, "subject": subject, "body": body}


class FakeLlmRequest:
    def __init__(self, tool_names: list[str] | None = None):
        self.tools_dict = {name: object() for name in (tool_names or [])}
        self.appended_tools: list[BaseTool] = []

    def append_tools(self, tools: list[BaseTool]) -> None:
        for tool in tools:
            self.appended_tools.append(tool)
            name = getattr(tool, "name", None)
            if isinstance(name, str) and name:
                self.tools_dict[name] = tool


def _context(user_id: str | None = "u1", session_id: str | None = "s1") -> SimpleNamespace:
    session = SimpleNamespace(id=session_id) if session_id is not None else None
    return SimpleNamespace(user_id=user_id, session=session)


def _named_tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def test_create_search_and_load_tools_smoke():
    registry = ToolRegistry()
    registry.register_many([get_weather, send_email])

    search_tools, load_tool = create_search_and_load_tools(registry)

    search_results = search_tools("weather")
    assert any(item.startswith("get_weather:") for item in search_results)

    assert "load requested" in load_tool("get_weather")
    assert "not found" in load_tool("unknown_tool")


async def test_session_scoped_callbacks_do_not_leak_across_sessions():
    registry = ToolRegistry()
    registry.register_many([get_weather, send_email])

    before_model_callback, after_tool_callback = create_session_scoped_loader_callbacks(registry)

    response = await after_tool_callback(
        _named_tool("load_tool"),
        {"tool_name": "get_weather"},
        _context(user_id="user-a", session_id="session-a"),
        None,
    )
    assert "now loaded" in response

    request_a = FakeLlmRequest(tool_names=["search_tools", "load_tool"])
    await before_model_callback(_context(user_id="user-a", session_id="session-a"), request_a)
    assert any(tool.name == "get_weather" for tool in request_a.appended_tools)
    assert all(isinstance(tool, BaseTool) for tool in request_a.appended_tools)

    request_b = FakeLlmRequest(tool_names=["search_tools", "load_tool"])
    await before_model_callback(_context(user_id="user-a", session_id="session-b"), request_b)
    assert request_b.appended_tools == []


async def test_session_scoped_callbacks_are_idempotent_and_skip_existing_tools():
    registry = ToolRegistry()
    registry.register(get_weather)

    before_model_callback, after_tool_callback = create_session_scoped_loader_callbacks(registry)

    first = await after_tool_callback(
        _named_tool("load_tool"),
        {"tool_name": "get_weather"},
        _context(user_id="u1", session_id="s1"),
        None,
    )
    second = await after_tool_callback(
        _named_tool("load_tool"),
        {"tool_name": "get_weather"},
        _context(user_id="u1", session_id="s1"),
        None,
    )

    assert "now loaded" in first
    assert "already loaded" in second

    request = FakeLlmRequest(tool_names=["search_tools", "load_tool", "get_weather"])
    await before_model_callback(_context(user_id="u1", session_id="s1"), request)
    assert request.appended_tools == []


async def test_after_tool_callback_handles_validation_errors():
    registry = ToolRegistry()
    registry.register(get_weather)

    _, after_tool_callback = create_session_scoped_loader_callbacks(registry)

    non_load_result = await after_tool_callback(
        _named_tool("search_tools"),
        {"query": "weather"},
        _context(user_id="u1", session_id="s1"),
        None,
    )
    assert non_load_result is None

    missing_tool_result = await after_tool_callback(
        _named_tool("load_tool"),
        {"tool_name": "not_real"},
        _context(user_id="u1", session_id="s1"),
        None,
    )
    assert "not found" in missing_tool_result

    missing_context_result = await after_tool_callback(
        _named_tool("load_tool"),
        {"tool_name": "get_weather"},
        _context(user_id="u1", session_id=None),
        None,
    )
    assert "Could not determine session context" in missing_context_result
