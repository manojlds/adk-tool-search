import inspect
import re
from typing import Any

from rank_bm25 import BM25Okapi


class ToolRegistry:
    """Indexes tools for lightweight BM25 search without loading full schemas into context."""

    def __init__(self):
        self._tools: dict[str, Any] = {}
        self._descriptions: list[str] = []
        self._tokenized_descriptions: list[list[str]] = []
        self._tool_names: list[str] = []
        self._bm25: BM25Okapi | None = None

    def register(self, tool: Any) -> None:
        """Register a tool (function, ADK BaseTool, or MCP tool) for search."""
        name, doc = self._extract_metadata(tool)
        if name in self._tools:
            return  # already registered

        self._tools[name] = tool
        self._tool_names.append(name)
        self._descriptions.append(f"{name} {doc}")
        self._rebuild_index()

    def register_many(self, tools: list[Any]) -> None:
        """Register multiple tools at once, rebuilding the index only once."""
        added = False
        for tool in tools:
            name, doc = self._extract_metadata(tool)
            if name in self._tools:
                continue
            self._tools[name] = tool
            self._tool_names.append(name)
            self._descriptions.append(f"{name} {doc}")
            added = True
        if added:
            self._rebuild_index()

    def search(self, query: str, n: int = 5) -> list[str]:
        """Return top-N relevant tool summaries as 'name: description snippet'."""
        if not self._bm25:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        positive_ranked_indices = [
            idx
            for idx in sorted(
                range(len(self._descriptions)),
                key=lambda item_idx: float(scores[item_idx]),
                reverse=True,
            )
            if float(scores[idx]) > 0
        ]

        if positive_ranked_indices:
            ranked_indices = positive_ranked_indices
        else:
            query_terms = set(tokenized_query)
            lexical_ranked_indices = sorted(
                range(len(self._descriptions)),
                key=lambda item_idx: len(
                    query_terms.intersection(self._tokenized_descriptions[item_idx])
                ),
                reverse=True,
            )
            ranked_indices = [
                idx
                for idx in lexical_ranked_indices
                if len(query_terms.intersection(self._tokenized_descriptions[idx])) > 0
            ]

        results = []
        for idx in ranked_indices:
            if len(results) >= n:
                break

            doc = self._descriptions[idx]
            name = self._tool_names[idx]
            snippet = doc.split("\n")[0][:120]
            results.append(f"{name}: {snippet}")

        return results

    def get_tool(self, name: str) -> Any | None:
        """Get the full tool object by exact name."""
        return self._tools.get(name)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def _extract_metadata(self, tool: Any) -> tuple[str, str]:
        """Extract (name, description) from any tool type."""
        if hasattr(tool, "name"):
            # ADK BaseTool / MCP tool
            name = tool.name
            doc = getattr(tool, "description", "") or ""
        elif callable(tool):
            # Plain Python function
            name = tool.__name__
            doc = inspect.getdoc(tool) or ""
        else:
            raise TypeError(
                f"Cannot register {type(tool)}: expected a callable or an object with a 'name' attribute"
            )
        return name, doc

    def _rebuild_index(self) -> None:
        self._tokenized_descriptions = [self._tokenize(desc) for desc in self._descriptions]
        self._bm25 = BM25Okapi(self._tokenized_descriptions)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
