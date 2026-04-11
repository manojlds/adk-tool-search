"""LLM test: no-match handling, session isolation, persistence, and skill auto-load."""

from __future__ import annotations

from pathlib import Path

import pytest
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService

from adk_tool_search import ToolRegistry, create_tool_search_agent
from tests.conftest import (
    make_litellm_model,
    run_agent_with_payloads,
    run_runner_session_turn,
    run_runner_session_turn_allow_error,
)
from tests.llm.conftest import ALL_TOOLS

pytestmark = [pytest.mark.llm]


@pytest.mark.timeout(120)
async def test_search_with_no_matching_tool():
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
        word in full_text
        for word in (
            "unavailable",
            "not found",
            "no tools",
            "no matching",
            "not available",
            "couldn't find",
            "doesn't currently have",
            "no results",
            "not currently available",
        )
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

    _, calls_a1, _ = await run_runner_session_turn(
        runner,
        session_id=session_a.id,
        user_id="test_user",
        prompt="Call load_tool with tool_name 'get_weather'.",
    )
    assert "load_tool" in calls_a1, f"Expected load_tool call in session A, got: {calls_a1}"

    _, calls_a2, _ = await run_runner_session_turn(
        runner,
        session_id=session_a.id,
        user_id="test_user",
        prompt="Now call get_weather for Tokyo.",
    )
    assert "get_weather" in calls_a2, (
        f"Expected get_weather call in session A after loading, got: {calls_a2}"
    )

    _, calls_b1, _ = await run_runner_session_turn(
        runner,
        session_id=session_b.id,
        user_id="test_user",
        prompt="Call search_tools with query 'weather' and then stop.",
    )
    assert "search_tools" in calls_b1, f"Expected search_tools call in session B, got: {calls_b1}"

    _, calls_b2, _, session_b_error = await run_runner_session_turn_allow_error(
        runner,
        session_id=session_b.id,
        user_id="test_user",
        prompt="Without loading any tool first, call get_weather for Paris now.",
    )
    if "get_weather" in calls_b2:
        assert session_b_error is not None, (
            "Expected unavailable get_weather to fail if model attempts direct call"
        )
        assert "Tool 'get_weather' not found" in str(session_b_error), (
            f"Expected missing-tool error in session B, got: {session_b_error}"
        )
    else:
        if session_b_error is not None:
            assert "Tool 'get_weather' not found" in str(session_b_error), (
                f"Expected missing-tool error in session B, got: {session_b_error}"
            )

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


@pytest.mark.timeout(120)
async def test_loaded_tool_persists_across_runner_restart_with_sqlite_session(tmp_path: Path):
    """Loaded tools should still be available after process restart with persisted session."""
    db_path = tmp_path / "sessions.db"

    registry = ToolRegistry()
    registry.register_many(ALL_TOOLS)

    agent_1 = create_tool_search_agent(
        name="restart_persistence_test",
        model=make_litellm_model(),
        registry=registry,
    )
    session_service_1 = SqliteSessionService(str(db_path))
    runner_1 = Runner(agent=agent_1, app_name="test", session_service=session_service_1)

    session = await runner_1.session_service.create_session(app_name="test", user_id="test_user")

    _, calls_first, _ = await run_runner_session_turn(
        runner_1,
        session_id=session.id,
        user_id="test_user",
        prompt="Call load_tool with tool_name 'get_weather'.",
    )
    assert "load_tool" in calls_first, f"Expected load_tool call, got: {calls_first}"

    agent_2 = create_tool_search_agent(
        name="restart_persistence_test",
        model=make_litellm_model(),
        registry=registry,
    )
    session_service_2 = SqliteSessionService(str(db_path))
    runner_2 = Runner(agent=agent_2, app_name="test", session_service=session_service_2)

    _, calls_second, _ = await run_runner_session_turn(
        runner_2,
        session_id=session.id,
        user_id="test_user",
        prompt="Now call get_weather for Tokyo without calling load_tool again.",
    )

    assert "get_weather" in calls_second, (
        "Expected get_weather to remain available after restart when session is persisted, "
        f"got calls: {calls_second}"
    )


@pytest.mark.timeout(120)
async def test_use_skill_auto_loads_allowed_tools_for_session():
    """When a skill is loaded, allowed tools should become callable next turn."""

    def use_skill(name: str) -> dict:
        """Load a skill and return instructions plus allowed tools."""
        if name == "weather-skill":
            return {
                "skill_name": name,
                "instructions": "Use get_weather to answer weather questions.",
                "allowed_tools": "get_weather",
            }
        return {
            "skill_name": name,
            "instructions": "No special tools.",
            "allowed_tools": "",
        }

    registry = ToolRegistry()
    registry.register_many(ALL_TOOLS)

    agent = create_tool_search_agent(
        name="skill_allowed_tools_test",
        model=make_litellm_model(),
        registry=registry,
        always_available_tools=[use_skill],
    )

    runner = InMemoryRunner(agent=agent, app_name="test")
    session = await runner.session_service.create_session(app_name="test", user_id="test_user")

    _, calls_1, _ = await run_runner_session_turn(
        runner,
        session_id=session.id,
        user_id="test_user",
        prompt="Call use_skill with name 'weather-skill'.",
    )
    assert "use_skill" in calls_1, f"Expected use_skill call, got: {calls_1}"

    _, calls_2, _ = await run_runner_session_turn(
        runner,
        session_id=session.id,
        user_id="test_user",
        prompt="Now call get_weather for Tokyo without calling load_tool.",
    )
    assert "get_weather" in calls_2, (
        f"Expected get_weather to be auto-loaded from skill allowed_tools, got: {calls_2}"
    )
