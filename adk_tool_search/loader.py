from __future__ import annotations

from collections.abc import Callable
from typing import Any

from google.adk.agents import Agent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool

from adk_tool_search.registry import ToolRegistry

_SESSION_LOADED_TOOLS_STATE_KEY = "adk_tool_search.loaded_tools"

_DEFAULT_ALLOWED_TOOL_FIELD_KEYS = ("allowed_tools", "allowed_tools_raw", "allowed-tools")
_DEFAULT_AUTO_LOAD_TOOL_NAMES = {"use_skill"}


def _extract_allowed_tool_tokens(
    tool_response: Any,
    field_keys: tuple[str, ...] = _DEFAULT_ALLOWED_TOOL_FIELD_KEYS,
) -> list[str]:
    """Extract allowed tool tokens from a skill tool response payload."""
    if not isinstance(tool_response, dict):
        return []

    allowed_tools_value: Any | None = None
    for key in field_keys:
        if key in tool_response:
            allowed_tools_value = tool_response[key]
            break

    if isinstance(allowed_tools_value, list):
        return [
            token.strip()
            for token in allowed_tools_value
            if isinstance(token, str) and token.strip()
        ]
    if isinstance(allowed_tools_value, str):
        return [token.strip() for token in allowed_tools_value.split() if token.strip()]

    return []


def _resolve_allowed_tool_names(
    tokens: list[str],
    registry: ToolRegistry,
) -> tuple[set[str], list[str]]:
    """Resolve allowed-tools tokens into concrete registry tool names."""
    resolved: set[str] = set()
    unresolved: list[str] = []

    for token in tokens:
        base = token.split("(", 1)[0].strip()
        candidates: list[str] = []
        for value in (token, base):
            if not value:
                continue
            candidates.extend(
                [value, value.lower(), value.replace("-", "_"), value.lower().replace("-", "_")]
            )
        candidates = list(dict.fromkeys(candidates))

        matched_name: str | None = None
        for candidate in candidates:
            if registry.get_tool(candidate) is not None:
                matched_name = candidate
                break

        if matched_name is None:
            unresolved.append(token)
        else:
            resolved.add(matched_name)

    return resolved, unresolved


def _session_key_from_context(context: Any) -> tuple[str, str] | None:
    """Extract a stable (user_id, session_id) key from ADK context objects."""
    user_id = getattr(context, "user_id", None)
    session = getattr(context, "session", None)
    session_id = getattr(session, "id", None)

    if user_id is None or session_id is None:
        return None

    return str(user_id), str(session_id)


def create_session_scoped_loader_callbacks(registry: ToolRegistry):
    """Create callbacks that load tools per session instead of globally.

    Returns:
        before_model_callback: Injects previously loaded tools for current session.
        after_tool_callback: Handles load_tool and records loaded tools per session.
    """

    return create_session_scoped_loader_callbacks_with_config(registry)


def create_session_scoped_loader_callbacks_with_config(
    registry: ToolRegistry,
    *,
    auto_load_from_tool_names: set[str] | None = None,
    auto_load_field_keys: tuple[str, ...] = _DEFAULT_ALLOWED_TOOL_FIELD_KEYS,
    auto_load_when: Callable[[str, dict[str, Any], Any], bool] | None = None,
    allowed_tool_token_resolver: Callable[[list[str], ToolRegistry], tuple[set[str], list[str]]]
    | None = None,
):
    """Create session-scoped callbacks with configurable auto-load behavior.

    Args:
        registry: Registry used to resolve and inject tool definitions.
        auto_load_from_tool_names: Tool names eligible for response-based auto-loading.
            Defaults to {"use_skill"}. Set to None for field-only behavior.
        auto_load_field_keys: Ordered response field keys to inspect for allowed-tool tokens.
        auto_load_when: Optional custom predicate
            (tool_name, args, tool_response) -> bool.
            If provided, takes precedence over ``auto_load_from_tool_names``.
        allowed_tool_token_resolver: Optional custom resolver
            (tokens, registry) -> (resolved_names, unresolved_tokens).

    Returns:
        before_model_callback, after_tool_callback
    """
    if auto_load_from_tool_names is None:
        effective_tool_names: set[str] | None = None
    else:
        effective_tool_names = set(auto_load_from_tool_names)
    token_resolver = allowed_tool_token_resolver or _resolve_allowed_tool_names

    def _as_base_tool(tool: Any) -> BaseTool | None:
        if isinstance(tool, BaseTool):
            return tool
        if callable(tool):
            return FunctionTool(tool)
        return None

    def _get_loaded_names_from_state(context: Any) -> set[str]:
        state = getattr(context, "state", None)
        if state is None:
            return set()

        persisted = state.get(_SESSION_LOADED_TOOLS_STATE_KEY, [])
        if not isinstance(persisted, list):
            return set()

        return {name for name in persisted if isinstance(name, str) and name}

    def _set_loaded_names_in_state(context: Any, loaded_names: set[str]) -> None:
        state = getattr(context, "state", None)
        if state is not None:
            state[_SESSION_LOADED_TOOLS_STATE_KEY] = sorted(loaded_names)

    async def before_model_callback(callback_context: Any, llm_request: Any) -> None:
        session_key = _session_key_from_context(callback_context)
        if session_key is None:
            return None

        loaded_names = _get_loaded_names_from_state(callback_context)

        if not loaded_names:
            return None

        existing_tool_names = set(getattr(llm_request, "tools_dict", {}).keys())
        tools_to_append: list[BaseTool] = []
        for tool_name in loaded_names:
            if tool_name in existing_tool_names:
                continue

            tool = registry.get_tool(tool_name)
            if tool is None:
                continue

            base_tool = _as_base_tool(tool)
            if base_tool is not None:
                tools_to_append.append(base_tool)

        if tools_to_append:
            llm_request.append_tools(tools_to_append)

        return None

    def _should_auto_load_from_response(tool_name: str, args: dict, tool_response: Any) -> bool:
        if auto_load_when is not None:
            return bool(auto_load_when(tool_name, args, tool_response))

        if effective_tool_names is None:
            return isinstance(tool_response, dict) and any(
                key in tool_response for key in auto_load_field_keys
            )

        return (
            tool_name in effective_tool_names
            and isinstance(tool_response, dict)
            and any(key in tool_response for key in auto_load_field_keys)
        )

    async def after_tool_callback(
        tool: Any, args: dict, tool_context: Any, tool_response: Any
    ) -> Any:
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))

        session_key = _session_key_from_context(tool_context)
        if session_key is None:
            return "Error: Could not determine session context for tool loading."

        if tool_name == "load_tool":
            requested_name = args.get("tool_name", "")
            new_tool = registry.get_tool(requested_name)
            if not new_tool:
                return f"Error: Tool '{requested_name}' not found in registry."

            loaded_names = _get_loaded_names_from_state(tool_context)
            if requested_name in loaded_names:
                return (
                    f"Tool '{requested_name}' is already loaded and ready to use in this session."
                )

            loaded_names.add(requested_name)
            _set_loaded_names_in_state(tool_context, loaded_names)

            return (
                f"Tool '{requested_name}' is now loaded for this session and ready to use. "
                "You can call it directly."
            )

        if _should_auto_load_from_response(tool_name, args, tool_response):
            tokens = _extract_allowed_tool_tokens(tool_response, field_keys=auto_load_field_keys)
            if not tokens:
                return None

            resolved_names, unresolved_tokens = token_resolver(tokens, registry)
            if not resolved_names and not unresolved_tokens:
                return None

            loaded_names = _get_loaded_names_from_state(tool_context)
            loaded_names.update(resolved_names)
            _set_loaded_names_in_state(tool_context, loaded_names)

            if not isinstance(tool_response, dict):
                return None

            enhanced_response = dict(tool_response)
            enhanced_response["auto_loaded_tools"] = sorted(resolved_names)
            if unresolved_tokens:
                enhanced_response["unresolved_allowed_tools"] = unresolved_tokens
            return enhanced_response

        return None

    return before_model_callback, after_tool_callback


def create_search_and_load_tools(registry: ToolRegistry):
    """Create the two lightweight tools the agent starts with."""

    def search_tools(query: str) -> list[str]:
        """Search the tool library for tools matching a query.

        Use this when you need a tool you don't currently have.
        Returns a list of 'tool_name: description' entries.

        Args:
            query: Keywords describing what you need (e.g., 'github issues', 'weather forecast', 'send email').
        """
        return registry.search(query, n=5)

    def load_tool(tool_name: str) -> str:
        """Load a tool into your active toolkit so you can use it.

        Call this after finding a tool with search_tools.

        Args:
            tool_name: The exact tool name from search_tools results.
        """
        tool = registry.get_tool(tool_name)
        if tool:
            return f"Tool '{tool_name}' load requested."
        return f"Error: Tool '{tool_name}' not found in registry."

    return search_tools, load_tool


def create_tool_search_agent(
    *,
    name: str,
    model: str,
    registry: ToolRegistry,
    instruction: str | None = None,
    always_available_tools: list[Any] | None = None,
    **agent_kwargs: Any,
) -> Agent:
    """Create an ADK Agent with dynamic tool search wired in.

    Args:
        name: Agent name.
        model: Model identifier (e.g., 'gemini-2.5-flash').
        registry: A ToolRegistry populated with tools to search.
        instruction: Optional system instruction. A default is provided if omitted.
        always_available_tools: Tools that should always be in context (not deferred).
        **agent_kwargs: Extra kwargs forwarded to Agent().
    """
    search_tools, load_tool = create_search_and_load_tools(registry)

    before_model_callback, after_tool_callback = create_session_scoped_loader_callbacks(registry)

    existing_before_model_callback = agent_kwargs.pop("before_model_callback", None)
    existing_after_tool_callback = agent_kwargs.pop("after_tool_callback", None)

    if existing_before_model_callback:
        if isinstance(existing_before_model_callback, list):
            before_model_callback = [before_model_callback, *existing_before_model_callback]
        else:
            before_model_callback = [before_model_callback, existing_before_model_callback]

    if existing_after_tool_callback:
        if isinstance(existing_after_tool_callback, list):
            after_tool_callback = [after_tool_callback, *existing_after_tool_callback]
        else:
            after_tool_callback = [after_tool_callback, existing_after_tool_callback]

    default_instruction = (
        f"You are {name}. You have a library of {registry.tool_count} tools available.\n"
        "You start with only search_tools and load_tool.\n"
        "When you need to perform an action:\n"
        "1. Call search_tools with keywords describing what you need\n"
        "2. Call load_tool with the exact tool name from the results\n"
        "3. Then call the loaded tool directly\n\n"
        f"Available tool categories: {', '.join(_guess_categories(registry))}"
    )

    tools: list[Any] = [search_tools, load_tool]
    if always_available_tools:
        tools.extend(always_available_tools)

    agent = Agent(
        name=name,
        model=model,
        instruction=instruction or default_instruction,
        tools=tools,
        before_model_callback=before_model_callback,
        after_tool_callback=after_tool_callback,
        **agent_kwargs,
    )
    return agent


def _guess_categories(registry: ToolRegistry) -> list[str]:
    """Derive rough category names from tool name prefixes."""
    prefixes = set()
    for name in registry.tool_names:
        parts = name.replace("-", "_").split("_")
        if len(parts) > 1:
            prefixes.add(parts[0])
        else:
            prefixes.add(name)
    return sorted(prefixes)[:10]
