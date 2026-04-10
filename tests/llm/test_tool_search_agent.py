"""LLM integration tests — tool search agent with a real LLM backend.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import pytest
from google.adk.agents import LlmAgent

from adk_tool_search import (
    ToolRegistry,
    create_dynamic_loader_callback,
    create_search_and_load_tools,
)
from tests.conftest import make_litellm_model, run_agent

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

    search_tools, load_tool = create_search_and_load_tools(registry)
    agent_ref: list[LlmAgent] = []
    callback = create_dynamic_loader_callback(registry, agent_ref)

    agent = LlmAgent(
        name="test_agent",
        model=make_litellm_model(),
        instruction=(
            "You are a helpful assistant with a library of tools.\n"
            "You start with only search_tools and load_tool.\n"
            "When you need to perform an action:\n"
            "1. Call search_tools with keywords describing what you need\n"
            "2. Call load_tool with the exact tool name from the results\n"
            "3. Then call the loaded tool directly\n"
        ),
        tools=[search_tools, load_tool],
        after_tool_callback=callback,
    )
    agent_ref.append(agent)
    return agent


@pytest.mark.timeout(120)
async def test_search_and_use_weather_tool():
    """Agent should search for, load, and call the weather tool."""
    agent = _make_agent()
    texts, calls, responses = await run_agent(agent, "What's the weather in Tokyo?")

    # Agent should have called search_tools, load_tool, and get_weather
    assert "search_tools" in calls, f"Expected search_tools in calls, got: {calls}"
    assert "load_tool" in calls, f"Expected load_tool in calls, got: {calls}"
    assert "get_weather" in calls, f"Expected get_weather in calls, got: {calls}"

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
    from adk_tool_search import create_tool_search_agent

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
