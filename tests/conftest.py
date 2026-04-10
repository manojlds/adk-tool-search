"""Shared test fixtures and helpers."""

from __future__ import annotations

import os

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
