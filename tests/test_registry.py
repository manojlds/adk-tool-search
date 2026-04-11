"""Tests for ToolRegistry."""

import pytest

from adk_tool_search.registry import ToolRegistry


def sample_func_a():
    """Get the weather for a location."""
    pass


def sample_func_b():
    """Send an email to a recipient."""
    pass


def sample_func_c():
    """Create a calendar event."""
    pass


class TestToolRegistry:
    def test_register_function(self):
        registry = ToolRegistry()
        registry.register(sample_func_a)
        assert registry.tool_count == 1
        assert "sample_func_a" in registry.tool_names

    def test_register_duplicate_ignored(self):
        registry = ToolRegistry()
        registry.register(sample_func_a)
        registry.register(sample_func_a)
        assert registry.tool_count == 1

    def test_register_many(self):
        registry = ToolRegistry()
        registry.register_many([sample_func_a, sample_func_b, sample_func_c])
        assert registry.tool_count == 3

    def test_search_returns_relevant_results(self):
        registry = ToolRegistry()
        registry.register_many([sample_func_a, sample_func_b, sample_func_c])
        results = registry.search("weather")
        assert len(results) > 0
        assert any("sample_func_a" in r for r in results)

    def test_search_email(self):
        registry = ToolRegistry()
        registry.register_many([sample_func_a, sample_func_b, sample_func_c])
        results = registry.search("email send")
        assert len(results) > 0
        assert any("sample_func_b" in r for r in results)

    def test_search_empty_registry(self):
        registry = ToolRegistry()
        results = registry.search("anything")
        assert results == []

    def test_search_no_relevant_match_returns_empty(self):
        registry = ToolRegistry()
        registry.register_many([sample_func_a, sample_func_b, sample_func_c])
        results = registry.search("portfolio rebalance optimizer")
        assert results == []

    def test_search_single_tool_registry_still_returns_relevant_match(self):
        registry = ToolRegistry()
        registry.register(sample_func_a)
        results = registry.search("weather")
        assert any("sample_func_a" in item for item in results)

    def test_search_two_tool_registry_returns_relevant_match(self):
        registry = ToolRegistry()
        registry.register_many([sample_func_a, sample_func_b])
        results = registry.search("weather")
        assert any("sample_func_a" in item for item in results)

    def test_search_matches_underscore_tool_names_with_space_query(self):
        registry = ToolRegistry()
        registry.register(sample_func_a)
        results = registry.search("sample func")
        assert any("sample_func_a" in item for item in results)

    def test_get_tool(self):
        registry = ToolRegistry()
        registry.register(sample_func_a)
        tool = registry.get_tool("sample_func_a")
        assert tool is sample_func_a

    def test_get_tool_missing(self):
        registry = ToolRegistry()
        assert registry.get_tool("nonexistent") is None

    def test_register_object_with_name(self):
        """Test registering an object with a name attribute (like ADK tools)."""

        class FakeTool:
            name = "fake_tool"
            description = "A fake tool for testing"

        registry = ToolRegistry()
        registry.register(FakeTool())
        assert registry.tool_count == 1
        assert "fake_tool" in registry.tool_names

    def test_register_non_callable_raises(self):
        registry = ToolRegistry()
        with pytest.raises(TypeError):
            registry.register(42)
