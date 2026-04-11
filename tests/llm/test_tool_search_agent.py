"""LLM integration tests — tool search agent with a real LLM backend.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import pytest
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner

from adk_tool_search import (
    ToolRegistry,
    create_tool_search_agent,
)
from tests.conftest import (
    make_litellm_model,
    run_agent,
    run_agent_with_call_args,
    run_agent_with_payloads,
    run_runner_session_turn,
)

pytestmark = [pytest.mark.llm]


# ── Dummy tools for the agent to discover ───────────────────────────────────


def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get the current weather for a location.

    Args:
        location: City name or coordinates.
        unit: Temperature unit - 'celsius' or 'fahrenheit'.
    """
    return {"location": location, "temperature": 22, "unit": unit, "condition": "sunny"}


def get_forecast(location: str, days: int = 5) -> dict:
    """Get a multi-day weather forecast for a location.

    Args:
        location: City name or coordinates.
        days: Number of days to forecast (1-14).
    """
    return {
        "location": location,
        "days": days,
        "forecast": [{"day": i, "temp": 20 + i} for i in range(days)],
    }


def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email to a recipient.

    Args:
        to: Email address of the recipient.
        subject: Email subject line.
        body: Email body text.
    """
    return {"status": "sent", "to": to, "subject": subject}


def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression.

    Args:
        expression: Math expression to evaluate (e.g., '2 + 2', 'sqrt(16)').
    """
    return {"expression": expression, "result": eval(expression)}  # noqa: S307


def translate_text(text: str, target_language: str) -> dict:
    """Translate text to a target language.

    Args:
        text: Text to translate.
        target_language: Target language code (e.g., 'es', 'fr', 'de').
    """
    return {"translated": f"[{target_language}] {text}", "source_language": "en"}


ALL_TOOLS = [get_weather, get_forecast, send_email, calculate, translate_text]


def _make_agent() -> LlmAgent:
    """Create a tool-search agent with all dummy tools registered."""
    registry = ToolRegistry()
    registry.register_many(ALL_TOOLS)

    return create_tool_search_agent(
        name="test_agent",
        model=make_litellm_model(),
        registry=registry,
        instruction=(
            "You are a helpful assistant with a library of tools.\n"
            "You start with only search_tools and load_tool.\n"
            "When you need to perform an action:\n"
            "1. Call search_tools with keywords describing what you need\n"
            "2. Call load_tool with the exact tool name from the results\n"
            "3. Then call the loaded tool directly\n"
        ),
    )


@pytest.mark.timeout(120)
async def test_search_and_use_weather_tool():
    """Agent should search for, load, and call the weather tool."""
    agent = _make_agent()
    texts, calls_with_args, responses = await run_agent_with_call_args(
        agent, "What's the weather in Tokyo?"
    )
    calls = [call["name"] for call in calls_with_args]

    # Agent should have called search_tools, load_tool, and get_weather
    assert "search_tools" in calls, f"Expected search_tools in calls, got: {calls}"
    assert "load_tool" in calls, f"Expected load_tool in calls, got: {calls}"
    assert "get_weather" in calls, f"Expected get_weather in calls, got: {calls}"

    weather_calls = [call for call in calls_with_args if call["name"] == "get_weather"]
    assert weather_calls, f"Expected get_weather call details, got: {calls_with_args}"

    weather_args = weather_calls[-1]["args"]
    assert isinstance(weather_args, dict), f"Expected get_weather args dict, got: {weather_args}"

    location = weather_args.get("location", "")
    assert isinstance(location, str) and "tokyo" in location.lower(), (
        f"Expected get_weather location to include 'Tokyo', got: {weather_args}"
    )

    # Final response should mention Tokyo
    full_text = " ".join(texts).lower()
    assert "tokyo" in full_text, f"Expected 'tokyo' in response, got: {full_text}"


@pytest.mark.timeout(120)
async def test_search_and_use_calculate_tool():
    """Agent should find and use the calculate tool for math."""
    agent = _make_agent()
    texts, calls, responses = await run_agent(agent, "What is 42 * 17?")

    full_text = " ".join(texts).lower()
    assert "714" in full_text, f"Expected '714' in response, got: {full_text}"

    if calls:
        assert "search_tools" in calls, f"Expected search_tools in calls, got: {calls}"
        assert "load_tool" in calls, f"Expected load_tool in calls, got: {calls}"
        assert "calculate" in calls, f"Expected calculate in calls, got: {calls}"


@pytest.mark.timeout(120)
async def test_factory_helper():
    """The create_tool_search_agent factory should produce a working agent."""
    registry = ToolRegistry()
    registry.register_many(ALL_TOOLS)

    agent = create_tool_search_agent(
        name="factory_test",
        model=make_litellm_model(),
        registry=registry,
    )

    texts, calls, responses = await run_agent(agent, "Translate 'hello world' to Spanish")

    assert "search_tools" in calls, f"Expected search_tools in calls, got: {calls}"
    assert "load_tool" in calls, f"Expected load_tool in calls, got: {calls}"
    assert "translate_text" in calls, f"Expected translate_text in calls, got: {calls}"


@pytest.mark.timeout(120)
async def test_search_with_no_matching_tool_does_not_load_or_call_domain_tool():
    """Agent handles empty search results without loading/calling unrelated tools."""
    registry = ToolRegistry()
    registry.register_many(ALL_TOOLS)

    agent = create_tool_search_agent(
        name="no_match_test",
        model=make_litellm_model(),
        registry=registry,
        instruction=(
            "You must use tools for actions. If search_tools returns no results, "
            "do not call load_tool and clearly say the capability is unavailable."
        ),
    )

    texts, calls_with_args, responses_with_payloads = await run_agent_with_payloads(
        agent,
        "Find and use a stock portfolio rebalance optimizer tool for a retirement portfolio.",
    )
    calls = [call["name"] for call in calls_with_args]

    assert "search_tools" in calls, f"Expected search_tools call, got: {calls_with_args}"

    search_responses = [item for item in responses_with_payloads if item["name"] == "search_tools"]
    assert search_responses, (
        f"Expected search_tools response payload, got: {responses_with_payloads}"
    )

    latest_payload = search_responses[-1]["response"]
    assert isinstance(latest_payload, dict), (
        f"Expected dict payload for search_tools response, got: {latest_payload}"
    )

    result_list = latest_payload.get("result")
    assert result_list == [], f"Expected empty search results, got: {latest_payload}"

    domain_tool_names = {"get_weather", "get_forecast", "send_email", "calculate", "translate_text"}
    called_domain_tools = [name for name in calls if name in domain_tool_names]
    assert not called_domain_tools, (
        f"Expected no domain tool calls when search is empty, got: {called_domain_tools}"
    )

    full_text = " ".join(texts).lower()
    assert any(
        word in full_text for word in ("unavailable", "not found", "no tools", "no matching")
    ), f"Expected unavailable/no-match message, got: {full_text}"


@pytest.mark.timeout(120)
async def test_loaded_tools_are_isolated_per_session():
    """A tool loaded in one session should not leak into another session."""
    registry = ToolRegistry()
    registry.register_many(ALL_TOOLS)

    tools_seen_by_session: dict[str, list[set[str]]] = {}

    async def capture_tools_before_model(callback_context, llm_request):
        session_id = callback_context.session.id
        tool_names = set(getattr(llm_request, "tools_dict", {}).keys())
        tools_seen_by_session.setdefault(session_id, []).append(tool_names)
        return None

    agent = create_tool_search_agent(
        name="session_isolation_test",
        model=make_litellm_model(),
        registry=registry,
        before_model_callback=capture_tools_before_model,
    )

    runner = InMemoryRunner(agent=agent, app_name="test")
    session_a = await runner.session_service.create_session(app_name="test", user_id="test_user")
    session_b = await runner.session_service.create_session(app_name="test", user_id="test_user")

    # Session A explicitly loads get_weather.
    _, calls_a1, _ = await run_runner_session_turn(
        runner,
        session_id=session_a.id,
        user_id="test_user",
        prompt="Call load_tool with tool_name 'get_weather'.",
    )
    assert "load_tool" in calls_a1, f"Expected load_tool call in session A, got: {calls_a1}"

    # Next turn in session A should have get_weather available.
    _, calls_a2, _ = await run_runner_session_turn(
        runner,
        session_id=session_a.id,
        user_id="test_user",
        prompt="Now call get_weather for Tokyo.",
    )
    assert "get_weather" in calls_a2, (
        f"Expected get_weather call in session A after loading, got: {calls_a2}"
    )

    # Session B should not see get_weather unless it loads it itself.
    _, calls_b1, _ = await run_runner_session_turn(
        runner,
        session_id=session_b.id,
        user_id="test_user",
        prompt="Without loading any tool, call get_weather for Paris if available.",
    )
    assert "get_weather" not in calls_b1, (
        "Expected get_weather to remain unavailable in session B until loaded there, "
        f"got calls: {calls_b1}"
    )

    # Verify tool visibility captured before each model call.
    assert session_a.id in tools_seen_by_session, "Expected tool snapshots for session A"
    assert session_b.id in tools_seen_by_session, "Expected tool snapshots for session B"

    first_tools_a = tools_seen_by_session[session_a.id][0]
    assert first_tools_a == {"search_tools", "load_tool"}, (
        f"Expected only meta-tools at start of session A, got: {first_tools_a}"
    )

    assert any("get_weather" in names for names in tools_seen_by_session[session_a.id]), (
        "Expected get_weather to be injected for session A after load_tool"
    )

    first_tools_b = tools_seen_by_session[session_b.id][0]
    assert first_tools_b == {"search_tools", "load_tool"}, (
        f"Expected only meta-tools at start of session B, got: {first_tools_b}"
    )
