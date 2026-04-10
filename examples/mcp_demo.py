"""Demo: Tool search with MCP server tools (e.g., GitHub MCP server)."""

import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp import MCPToolset, StdioConnectionParams
from google.genai import types

from adk_tool_search import ToolRegistry, create_tool_search_agent


async def main():
    # 1. Connect to MCP server and fetch tools
    mcp_toolset = MCPToolset(
        connection_params=StdioConnectionParams(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
        )
    )
    mcp_tools = await mcp_toolset.get_tools()
    print(f"Fetched {len(mcp_tools)} tools from GitHub MCP server")

    # 2. Register all MCP tools (but don't give them to the agent)
    registry = ToolRegistry()
    registry.register_many(mcp_tools)
    print(f"Indexed {registry.tool_count} tools for search")

    # 3. Create agent with tool search
    agent = create_tool_search_agent(
        name="GitHubAssistant",
        model="gemini-2.5-flash",
        registry=registry,
        instruction=(
            "You are a GitHub assistant. You have access to a library of GitHub tools.\n"
            "When you need to interact with GitHub, first search for the right tool, "
            "load it, then use it.\n"
            "Use search_tools to find tools and load_tool to activate them."
        ),
    )
    print(f"Agent created with {len(agent.tools)} initial tools")

    # 4. Run
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="github_demo", session_service=session_service)
    session = await session_service.create_session(app_name="github_demo", user_id="user1")

    user_message = "List the open issues in the google/adk-python repository"
    print(f"\nUser: {user_message}")

    content = types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
    async for event in runner.run_async(
        session_id=session.id, user_id="user1", new_message=content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent: {part.text}")
                if part.function_call:
                    print(
                        f"  [tool call] {part.function_call.name}({dict(part.function_call.args)})"
                    )


if __name__ == "__main__":
    asyncio.run(main())
