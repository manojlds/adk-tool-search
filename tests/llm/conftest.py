"""Shared LLM test fixtures and dummy tools."""

from __future__ import annotations

import pytest
from google.adk.agents import LlmAgent

from adk_tool_search import ToolRegistry, create_tool_search_agent
from tests.conftest import make_litellm_model

pytestmark = [pytest.mark.llm]


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
    import math

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {"abs": abs, "round": round, "min": min, "max": max, "pow": pow, "sqrt": math.sqrt},
        )  # noqa: S307
    except Exception:
        result = "error"
    return {"expression": expression, "result": result}


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
