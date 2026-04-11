"""LLM test: inline execution — load a tool and execute it in one turn."""

from __future__ import annotations

import pytest
from google.adk.runners import InMemoryRunner

from adk_tool_search import ToolRegistry, create_tool_search_agent
from tests.conftest import make_litellm_model, run_agent_with_call_args, run_runner_session_turn
from tests.llm.conftest import ALL_TOOLS, calculate, get_weather

pytestmark = [pytest.mark.llm]


@pytest.mark.timeout(120)
async def test_inline_execution_weather_tool():
    """Agent should be able to load and execute get_weather in one turn via load_tool args."""
    registry = ToolRegistry()
    registry.register_many([get_weather])

    agent = create_tool_search_agent(
        name="inline_weather_test",
        model=make_litellm_model(),
        registry=registry,
        instruction=(
            "You are a weather assistant. You have search_tools and load_tool.\n"
            "When you need a tool:\n"
            "1. Call search_tools to find it\n"
            "2. Call load_tool with tool_name AND args to load and execute it in one step\n"
            "   For example: load_tool(tool_name='get_weather', args={'location': 'Paris'})\n"
        ),
    )

    texts, calls_with_args, _ = await run_agent_with_call_args(
        agent, "What's the weather in Paris?"
    )
    calls = [call["name"] for call in calls_with_args]

    assert "search_tools" in calls, f"Expected search_tools in calls, got: {calls}"
    assert "load_tool" in calls, f"Expected load_tool in calls, got: {calls}"

    load_calls = [call for call in calls_with_args if call["name"] == "load_tool"]
    assert load_calls, "Expected at least one load_tool call"

    last_load_call = load_calls[-1]
    load_args = last_load_call.get("args", {})
    assert "tool_name" in load_args, f"Expected tool_name in load_tool args, got: {load_args}"
    assert "get_weather" in str(load_args.get("tool_name", "")).lower() or "weather" in str(
        load_args
    ), f"Expected get_weather as tool_name, got: {load_args}"

    full_text = " ".join(texts).lower()
    assert "paris" in full_text, f"Expected 'paris' in response, got: {full_text}"


@pytest.mark.timeout(120)
async def test_inline_execution_calculate_tool():
    """Agent should be able to load and execute calculate inline for a math question."""
    registry = ToolRegistry()
    registry.register_many([calculate])

    agent = create_tool_search_agent(
        name="inline_calculate_test",
        model=make_litellm_model(),
        registry=registry,
        instruction=(
            "You are a calculator assistant.\n"
            "When you need a tool:\n"
            "1. Call search_tools to find it\n"
            "2. Call load_tool with tool_name AND args to load and execute it in one step\n"
            "   For example: load_tool(tool_name='calculate', args={'expression': '2+2'})\n"
        ),
    )

    texts, calls_with_args, _ = await run_agent_with_call_args(agent, "What is 15 multiplied by 8?")
    calls = [call["name"] for call in calls_with_args]

    assert "load_tool" in calls, f"Expected load_tool in calls, got: {calls}"

    load_calls = [call for call in calls_with_args if call["name"] == "load_tool"]
    last_load = load_calls[-1]
    load_args = last_load.get("args", {})

    assert "args" in load_args or "tool_name" in load_args, (
        f"Expected load_tool to have tool_name and possibly args, got: {load_args}"
    )

    full_text = " ".join(texts).lower()
    has_result = "120" in full_text
    assert has_result, f"Expected '120' in response text, got: {full_text}"


@pytest.mark.timeout(120)
async def test_inline_execution_then_subsequent_call():
    """After inline execution, the loaded tool should be callable in subsequent turns."""
    registry = ToolRegistry()
    registry.register_many(ALL_TOOLS)

    agent = create_tool_search_agent(
        name="inline_then_call_test",
        model=make_litellm_model(),
        registry=registry,
        instruction=(
            "You have search_tools and load_tool.\n"
            "When you need a tool, you can:\n"
            "- load_tool(tool_name='...', args={...}) to load and execute it in one step\n"
            "- Or load_tool(tool_name='...') to load it for later use\n"
        ),
    )

    runner = InMemoryRunner(agent=agent, app_name="test")
    session = await runner.session_service.create_session(app_name="test", user_id="test_user")

    _, calls_1, _ = await run_runner_session_turn(
        runner,
        session_id=session.id,
        user_id="test_user",
        prompt="Load the get_weather tool with args for London and execute it.",
    )

    assert "load_tool" in calls_1, f"Expected load_tool in first turn, got: {calls_1}"

    _, calls_2, _ = await run_runner_session_turn(
        runner,
        session_id=session.id,
        user_id="test_user",
        prompt="Now call get_weather for Tokyo directly.",
    )

    assert "get_weather" in calls_2, (
        f"Expected get_weather in second turn after inline load, got: {calls_2}"
    )
