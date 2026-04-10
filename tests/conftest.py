"""Shared test fixtures and helpers."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def make_litellm_model():
    """Create a LiteLlm model from environment variables.

    Reads model from ADK_TOOL_SEARCH_MODEL first, then ADK_DEEPAGENTS_MODEL
    (for compatibility with copied .env files), then LITELLM_MODEL.
    """
    from google.adk.models.lite_llm import LiteLlm

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENCODE_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "https://opencode.ai/zen/v1")
    model = (
        os.environ.get("ADK_TOOL_SEARCH_MODEL")
        or os.environ.get("ADK_DEEPAGENTS_MODEL")
        or os.environ.get("LITELLM_MODEL", "openai/gpt-4o-mini")
    )

    return LiteLlm(model=model, api_key=api_key, api_base=api_base)


async def run_agent(agent, prompt: str) -> tuple[list[str], list[str], list[str]]:
    """Run an agent with a single prompt and return (texts, function_calls, function_responses)."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=agent, app_name="test")
    session = await runner.session_service.create_session(app_name="test", user_id="test_user")

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    texts: list[str] = []
    function_calls: list[str] = []
    function_responses: list[str] = []

    async for event in runner.run_async(
        session_id=session.id, user_id="test_user", new_message=content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    name = part.function_call.name
                    if isinstance(name, str) and name:
                        function_calls.append(name)
                if hasattr(part, "function_response") and part.function_response:
                    name = part.function_response.name
                    if isinstance(name, str) and name:
                        function_responses.append(name)

    return texts, function_calls, function_responses


async def run_agent_with_call_args(
    agent, prompt: str
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Run an agent and return function call names with args.

    Returns ``(texts, function_calls, function_responses)`` where
    ``function_calls`` is a list of dicts like ``{"name": str, "args": dict}``.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=agent, app_name="test")
    session = await runner.session_service.create_session(app_name="test", user_id="test_user")

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    texts: list[str] = []
    function_calls: list[dict[str, Any]] = []
    function_responses: list[str] = []

    async for event in runner.run_async(
        session_id=session.id, user_id="test_user", new_message=content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)

                if hasattr(part, "function_call") and part.function_call:
                    name = part.function_call.name
                    if isinstance(name, str) and name:
                        args = (
                            part.function_call.args
                            if isinstance(part.function_call.args, dict)
                            else {}
                        )
                        function_calls.append({"name": name, "args": args})

                if hasattr(part, "function_response") and part.function_response:
                    name = part.function_response.name
                    if isinstance(name, str) and name:
                        function_responses.append(name)

    return texts, function_calls, function_responses


async def run_agent_with_payloads(
    agent, prompt: str
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run an agent and return function call args plus response payloads."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=agent, app_name="test")
    session = await runner.session_service.create_session(app_name="test", user_id="test_user")

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    texts: list[str] = []
    function_calls: list[dict[str, Any]] = []
    function_responses: list[dict[str, Any]] = []

    async for event in runner.run_async(
        session_id=session.id, user_id="test_user", new_message=content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)

                function_call = getattr(part, "function_call", None)
                if function_call and isinstance(function_call.name, str) and function_call.name:
                    args = function_call.args if isinstance(function_call.args, dict) else {}
                    function_calls.append({"name": function_call.name, "args": args})

                function_response = getattr(part, "function_response", None)
                if (
                    function_response
                    and isinstance(function_response.name, str)
                    and function_response.name
                ):
                    function_responses.append(
                        {"name": function_response.name, "response": function_response.response}
                    )

    return texts, function_calls, function_responses
