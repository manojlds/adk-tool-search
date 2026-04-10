"""Demo: Tool search with plain Python function tools (no MCP server needed)."""

import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_tool_search import ToolRegistry, create_tool_search_agent

# ── Define a bunch of plain function tools ──────────────────────────────────


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


def search_contacts(query: str) -> list[dict]:
    """Search the contact directory for people.

    Args:
        query: Name or keyword to search for.
    """
    return [
        {"name": "Alice Smith", "email": "alice@example.com"},
        {"name": "Bob Jones", "email": "bob@example.com"},
    ]


def create_calendar_event(title: str, date: str, duration_minutes: int = 60) -> dict:
    """Create a new calendar event.

    Args:
        title: Event title.
        date: Event date in YYYY-MM-DD format.
        duration_minutes: Duration in minutes.
    """
    return {"status": "created", "title": title, "date": date, "duration": duration_minutes}


def list_calendar_events(date: str) -> list[dict]:
    """List all calendar events for a specific date.

    Args:
        date: Date in YYYY-MM-DD format.
    """
    return [{"title": "Team standup", "time": "09:00"}, {"title": "Lunch", "time": "12:00"}]


def translate_text(text: str, target_language: str, source_language: str = "auto") -> dict:
    """Translate text to a target language.

    Args:
        text: Text to translate.
        target_language: Target language code (e.g., 'es', 'fr', 'de').
        source_language: Source language code, or 'auto' for auto-detection.
    """
    return {"translated": f"[{target_language}] {text}", "source_language": "en"}


def summarize_text(text: str, max_sentences: int = 3) -> dict:
    """Summarize a long piece of text into key points.

    Args:
        text: The text to summarize.
        max_sentences: Maximum number of sentences in summary.
    """
    return {"summary": f"Summary of text ({max_sentences} sentences)", "original_length": len(text)}


def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression.

    Args:
        expression: Math expression to evaluate (e.g., '2 + 2', 'sqrt(16)').
    """
    try:
        result = eval(expression)  # noqa: S307 — demo only
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between currencies.

    Args:
        amount: Amount to convert.
        from_currency: Source currency code (e.g., 'USD').
        to_currency: Target currency code (e.g., 'EUR').
    """
    rates = {"USD_EUR": 0.92, "EUR_USD": 1.09, "USD_GBP": 0.79, "GBP_USD": 1.27}
    key = f"{from_currency}_{to_currency}"
    rate = rates.get(key, 1.0)
    return {
        "amount": amount,
        "from": from_currency,
        "to": to_currency,
        "converted": amount * rate,
        "rate": rate,
    }


# ── Main ────────────────────────────────────────────────────────────────────

ALL_TOOLS = [
    get_weather,
    get_forecast,
    send_email,
    search_contacts,
    create_calendar_event,
    list_calendar_events,
    translate_text,
    summarize_text,
    calculate,
    convert_currency,
]


async def main():
    # 1. Create registry and register all tools
    registry = ToolRegistry()
    registry.register_many(ALL_TOOLS)
    print(f"Registered {registry.tool_count} tools: {registry.tool_names}")

    # 2. Test search
    print("\n--- Search test: 'weather temperature' ---")
    for result in registry.search("weather temperature"):
        print(f"  {result}")

    print("\n--- Search test: 'send message email' ---")
    for result in registry.search("send message email"):
        print(f"  {result}")

    # 3. Create agent with tool search
    agent = create_tool_search_agent(
        name="Assistant",
        model="gemini-2.5-flash",
        registry=registry,
    )
    print(
        f"\nAgent created with {len(agent.tools)} initial tools (out of {registry.tool_count} available)"
    )

    # 4. Run a conversation
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="demo", session_service=session_service)
    session = await session_service.create_session(app_name="demo", user_id="user1")

    user_message = "What's the weather like in Tokyo?"
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
                if part.function_response:
                    print(
                        f"  [tool result] {part.function_response.name}: {part.function_response.response}"
                    )


if __name__ == "__main__":
    asyncio.run(main())
