from __future__ import annotations

from typing import Any

from google.adk.agents import Agent

from adk_tool_search.registry import ToolRegistry


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


def create_dynamic_loader_callback(
    registry: ToolRegistry,
    agent_ref: list[Agent],
):
    """Create an after_tool_callback that injects tools on load_tool calls.

    agent_ref is a single-element list holding the agent, used as a mutable
    reference so the callback can access the agent after it's created.
    """

    async def dynamic_loader_callback(
        tool: Any, args: dict, tool_context: Any, tool_response: Any
    ) -> Any:
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))

        if tool_name != "load_tool":
            return tool_response

        requested_name = args.get("tool_name", "")
        new_tool = registry.get_tool(requested_name)
        if not new_tool:
            return f"Error: Tool '{requested_name}' not found in registry."

        agent = agent_ref[0]
        current_names = set()
        for t in agent.tools:
            if hasattr(t, "name"):
                current_names.add(t.name)
            elif hasattr(t, "__name__"):
                current_names.add(t.__name__)

        if requested_name in current_names:
            return f"Tool '{requested_name}' is already loaded and ready to use."

        agent.tools.append(new_tool)
        return f"Tool '{requested_name}' is now loaded and ready to use. You can call it directly."

    return dynamic_loader_callback


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

    # Mutable ref so callback can access agent
    agent_ref: list[Agent] = []
    callback = create_dynamic_loader_callback(registry, agent_ref)

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
        after_tool_callback=callback,
        **agent_kwargs,
    )
    agent_ref.append(agent)
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
