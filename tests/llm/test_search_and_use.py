"""LLM test: basic search → load → use flow."""

from __future__ import annotations

import pytest

from adk_tool_search import ToolRegistry, create_tool_search_agent
from tests.conftest import make_litellm_model, run_agent, run_agent_with_call_args
from tests.llm.conftest import ALL_TOOLS, _make_agent, get_weather

pytestmark = [pytest.mark.llm]


@pytest.mark.timeout(120)
async def test_search_and_use_weather_tool():
    """Agent should search for, load, and call the weather tool."""
    agent = _make_agent()
    texts, calls_with_args, responses = await run_agent_with_call_args(
        agent, "What's the weather in Tokyo?"
    )
    calls = [call["name"] for call in calls_with_args]

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
async def test_single_tool_search_returns_match_for_relevant_query():
    """Regression: search should still find relevant tools in a tiny registry."""
    registry = ToolRegistry()
    registry.register(get_weather)

    agent = create_tool_search_agent(
        name="single_tool_search_regression",
        model=make_litellm_model(),
        registry=registry,
        instruction=(
            "Debug mode: call search_tools exactly once with query 'weather'. "
            "Do not call load_tool or any other tools. Then report the search result."
        ),
    )

    from tests.conftest import run_agent_with_payloads

    _, calls_with_args, responses_with_payloads = await run_agent_with_payloads(
        agent,
        "Run the debug search now.",
    )

    search_calls = [call for call in calls_with_args if call["name"] == "search_tools"]
    assert search_calls, f"Expected search_tools call, got: {calls_with_args}"

    search_responses = [item for item in responses_with_payloads if item["name"] == "search_tools"]
    assert search_responses, (
        f"Expected search_tools response payload, got: {responses_with_payloads}"
    )

    payload = search_responses[-1]["response"]
    assert isinstance(payload, dict), f"Expected dict payload, got: {payload}"

    result_list = payload.get("result")
    assert isinstance(result_list, list), f"Expected list search results, got: {payload}"
    assert any(str(item).startswith("get_weather:") for item in result_list), (
        f"Expected 'get_weather' in search_tools results for query 'weather', got: {result_list}"
    )
