"""Unit tests for loader helpers and session-scoped callbacks."""

from __future__ import annotations

from types import SimpleNamespace

from google.adk.tools.base_tool import BaseTool

from adk_tool_search import ToolRegistry
from adk_tool_search.loader import (
    _extract_allowed_tool_tokens,
    _resolve_allowed_tool_names,
    create_search_and_load_tools,
    create_session_scoped_loader_callbacks,
    create_session_scoped_loader_callbacks_with_config,
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


def _context(
    user_id: str | None = "u1",
    session_id: str | None = "s1",
    state: dict | None = None,
) -> SimpleNamespace:
    session = SimpleNamespace(id=session_id) if session_id is not None else None
    return SimpleNamespace(
        user_id=user_id, session=session, state=state if state is not None else {}
    )


def _context_without_state(
    user_id: str | None = "u1",
    session_id: str | None = "s1",
) -> SimpleNamespace:
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

    session_a_state: dict = {}
    session_b_state: dict = {}
    context_a = _context(user_id="user-a", session_id="session-a", state=session_a_state)
    context_b = _context(user_id="user-a", session_id="session-b", state=session_b_state)

    response = await after_tool_callback(
        _named_tool("load_tool"),
        {"tool_name": "get_weather"},
        context_a,
        None,
    )
    assert "now loaded" in response

    request_a = FakeLlmRequest(tool_names=["search_tools", "load_tool"])
    await before_model_callback(context_a, request_a)
    assert any(tool.name == "get_weather" for tool in request_a.appended_tools)
    assert all(isinstance(tool, BaseTool) for tool in request_a.appended_tools)

    request_b = FakeLlmRequest(tool_names=["search_tools", "load_tool"])
    await before_model_callback(context_b, request_b)
    assert request_b.appended_tools == []


async def test_session_scoped_callbacks_are_idempotent_and_skip_existing_tools():
    registry = ToolRegistry()
    registry.register(get_weather)

    before_model_callback, after_tool_callback = create_session_scoped_loader_callbacks(registry)
    session_state: dict = {}
    context = _context(user_id="u1", session_id="s1", state=session_state)

    first = await after_tool_callback(
        _named_tool("load_tool"),
        {"tool_name": "get_weather"},
        context,
        None,
    )
    second = await after_tool_callback(
        _named_tool("load_tool"),
        {"tool_name": "get_weather"},
        context,
        None,
    )

    assert "now loaded" in first
    assert "already loaded" in second

    request = FakeLlmRequest(tool_names=["search_tools", "load_tool", "get_weather"])
    await before_model_callback(context, request)
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


async def test_loaded_tools_resume_after_callback_recreation_with_same_session_state():
    """Loaded tools should survive callback recreation when session state persists."""
    registry = ToolRegistry()
    registry.register(get_weather)

    persisted_state: dict = {}
    callback_context = _context(user_id="u1", session_id="s1", state=persisted_state)

    # First "process": load a tool for the session.
    _, after_tool_callback_1 = create_session_scoped_loader_callbacks(registry)
    result = await after_tool_callback_1(
        _named_tool("load_tool"),
        {"tool_name": "get_weather"},
        callback_context,
        None,
    )
    assert "now loaded" in result

    # Simulate restart by creating fresh callbacks with empty in-memory state.
    before_model_callback_2, _ = create_session_scoped_loader_callbacks(registry)
    request = FakeLlmRequest(tool_names=["search_tools", "load_tool"])
    await before_model_callback_2(callback_context, request)

    assert any(tool.name == "get_weather" for tool in request.appended_tools), (
        "Expected get_weather to be rehydrated from persisted session state after restart"
    )


async def test_callbacks_require_session_state_for_persistence_and_injection():
    """Without callback context state, loaded tools cannot persist across turns."""
    registry = ToolRegistry()
    registry.register(get_weather)

    before_model_callback, after_tool_callback = create_session_scoped_loader_callbacks(registry)

    load_result = await after_tool_callback(
        _named_tool("load_tool"),
        {"tool_name": "get_weather"},
        _context_without_state(user_id="u1", session_id="s1"),
        None,
    )
    assert "now loaded" in load_result

    request = FakeLlmRequest(tool_names=["search_tools", "load_tool"])
    await before_model_callback(_context_without_state(user_id="u1", session_id="s1"), request)
    assert request.appended_tools == []


def test_extract_allowed_tool_tokens_variants():
    assert _extract_allowed_tool_tokens({"allowed_tools": ["get_weather", "send_email"]}) == [
        "get_weather",
        "send_email",
    ]
    assert _extract_allowed_tool_tokens({"allowed_tools": "get_weather send_email"}) == [
        "get_weather",
        "send_email",
    ]
    assert _extract_allowed_tool_tokens({"allowed_tools_raw": "get_weather send_email"}) == [
        "get_weather",
        "send_email",
    ]
    assert _extract_allowed_tool_tokens({"allowed-tools": "get_weather send_email"}) == [
        "get_weather",
        "send_email",
    ]


def test_extract_allowed_tool_tokens_field_precedence_is_configurable():
    payload = {
        "allowed_tools": "get_weather",
        "allowed_tools_raw": "send_email",
        "allowed-tools": "ignored",
    }

    assert _extract_allowed_tool_tokens(payload) == ["get_weather"]
    assert _extract_allowed_tool_tokens(
        payload,
        field_keys=("allowed_tools_raw", "allowed_tools", "allowed-tools"),
    ) == ["send_email"]


def test_resolve_allowed_tool_names_with_token_variants():
    registry = ToolRegistry()
    registry.register_many([get_weather, send_email])

    resolved, unresolved = _resolve_allowed_tool_names(
        ["get_weather", "send-email", "Bash(git:*)"],
        registry,
    )

    assert "get_weather" in resolved
    assert "send_email" in resolved
    assert "Bash(git:*)" in unresolved


async def test_use_skill_allowed_tools_are_auto_loaded_in_session_state():
    registry = ToolRegistry()
    registry.register_many([get_weather, send_email])

    before_model_callback, after_tool_callback = create_session_scoped_loader_callbacks(registry)
    context = _context(user_id="u1", session_id="s1", state={})

    skill_payload = {
        "skill_name": "weather-skill",
        "allowed_tools": "get_weather send-email Bash(git:*)",
    }

    override_response = await after_tool_callback(
        _named_tool("use_skill"),
        {"name": "weather-skill"},
        context,
        skill_payload,
    )

    assert isinstance(override_response, dict)
    assert override_response["auto_loaded_tools"] == ["get_weather", "send_email"]
    assert "Bash(git:*)" in override_response["unresolved_allowed_tools"]

    request = FakeLlmRequest(tool_names=["search_tools", "load_tool"])
    await before_model_callback(context, request)
    injected_names = {tool.name for tool in request.appended_tools}
    assert {"get_weather", "send_email"}.issubset(injected_names)


async def test_injected_tools_are_deterministically_sorted_for_cache_stability():
    registry = ToolRegistry()
    registry.register_many([get_weather, send_email])

    before_model_callback, _ = create_session_scoped_loader_callbacks(registry)
    context = _context(
        user_id="u1",
        session_id="s1",
        state={"adk_tool_search.loaded_tools": ["send_email", "get_weather"]},
    )

    request = FakeLlmRequest(tool_names=["search_tools", "load_tool"])
    await before_model_callback(context, request)

    injected_order = [tool.name for tool in request.appended_tools]
    assert injected_order == ["get_weather", "send_email"]


async def test_configurable_auto_load_field_only_mode_allows_non_use_skill_tool():
    registry = ToolRegistry()
    registry.register_many([get_weather])

    before_model_callback, after_tool_callback = create_session_scoped_loader_callbacks_with_config(
        registry,
        auto_load_from_tool_names=None,
    )
    context = _context(user_id="u1", session_id="s1", state={})

    response = await after_tool_callback(
        _named_tool("emit_policy"),
        {},
        context,
        {"allowed_tools": "get_weather"},
    )
    assert isinstance(response, dict)
    assert response["auto_loaded_tools"] == ["get_weather"]

    request = FakeLlmRequest(tool_names=["search_tools", "load_tool"])
    await before_model_callback(context, request)
    injected_names = {tool.name for tool in request.appended_tools}
    assert "get_weather" in injected_names


async def test_configurable_auto_load_custom_predicate_takes_precedence():
    registry = ToolRegistry()
    registry.register_many([get_weather])

    def predicate(tool_name: str, args: dict, tool_response: dict | None) -> bool:
        return tool_name == "policy_router" and isinstance(tool_response, dict)

    before_model_callback, after_tool_callback = create_session_scoped_loader_callbacks_with_config(
        registry,
        auto_load_from_tool_names={"use_skill"},
        auto_load_when=predicate,
    )
    context = _context(user_id="u1", session_id="s1", state={})

    skipped = await after_tool_callback(
        _named_tool("use_skill"),
        {},
        context,
        {"allowed_tools": "get_weather"},
    )
    assert skipped is None

    loaded = await after_tool_callback(
        _named_tool("policy_router"),
        {},
        context,
        {"allowed_tools": "get_weather"},
    )
    assert isinstance(loaded, dict)
    assert loaded["auto_loaded_tools"] == ["get_weather"]

    request = FakeLlmRequest(tool_names=["search_tools", "load_tool"])
    await before_model_callback(context, request)
    injected_names = {tool.name for tool in request.appended_tools}
    assert "get_weather" in injected_names


async def test_configurable_auto_load_custom_resolver_is_used():
    registry = ToolRegistry()
    registry.register_many([get_weather])

    def resolver(tokens: list[str], _registry: ToolRegistry) -> tuple[set[str], list[str]]:
        if "weather-policy" in tokens:
            return {"get_weather"}, []
        return set(), tokens

    before_model_callback, after_tool_callback = create_session_scoped_loader_callbacks_with_config(
        registry,
        allowed_tool_token_resolver=resolver,
    )
    context = _context(user_id="u1", session_id="s1", state={})

    response = await after_tool_callback(
        _named_tool("use_skill"),
        {},
        context,
        {"allowed_tools": "weather-policy"},
    )
    assert isinstance(response, dict)
    assert response["auto_loaded_tools"] == ["get_weather"]

    request = FakeLlmRequest(tool_names=["search_tools", "load_tool"])
    await before_model_callback(context, request)
    injected_names = {tool.name for tool in request.appended_tools}
    assert "get_weather" in injected_names
