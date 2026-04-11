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
            overlaps = [
                (idx, len(query_terms.intersection(tokens)))
                for idx, tokens in enumerate(self._tokenized_descriptions)
            ]
            ranked_indices = [
                idx
                for idx, overlap in sorted(overlaps, key=lambda item: item[1], reverse=True)
                if overlap > 0
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
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        return re.findall(r"[a-z0-9]+", text.lower())

    def guess_categories(self, max_categories: int = 10) -> list[str]:
        """Derive category labels from tool names using meaningful name segments.

        Extracts the most significant noun-like segments from tool names.
        For example, from ``browser_navigate``, ``browser_click``,
        ``get_file_contents``, ``list_issues`` extracts ``browser``,
        ``file``, ``issue`` rather than just ``browser``, ``get``, ``list``.

        Uses description text to prefer segments that appear in descriptions,
        since those are more likely to represent meaningful domain concepts.
        """
        if not self._tool_names:
            return []

        all_segments: dict[str, int] = {}
        segment_in_description: dict[str, int] = {}

        for idx, name in enumerate(self._tool_names):
            split = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
            parts = re.split(r"[_\-]", split)
            description = self._descriptions[idx].lower() if idx < len(self._descriptions) else ""

            for part in parts:
                token = part.lower().strip()
                if len(token) <= 1:
                    continue
                all_segments[token] = all_segments.get(token, 0) + 1
                if token in description:
                    segment_in_description[token] = segment_in_description.get(token, 0) + 1

        action_verbs = {
            "get",
            "list",
            "create",
            "update",
            "delete",
            "add",
            "remove",
            "search",
            "find",
            "set",
            "manage",
            "handle",
            "run",
            "send",
            "read",
            "write",
            "push",
            "pull",
            "merge",
            "fork",
            "star",
            "unstar",
            "dismiss",
            "mark",
            "assign",
            "request",
            "enable",
            "take",
            "save",
            "start",
            "stop",
            "resume",
            "close",
            "open",
            "click",
            "type",
            "hover",
            "drag",
            "fill",
            "press",
            "select",
            "navigate",
            "snapshot",
            "evaluate",
            "verify",
            "generate",
            "trigger",
            "clear",
            "upload",
        }

        scored: list[tuple[str, float]] = []
        for segment, count in all_segments.items():
            if segment in action_verbs:
                continue
            desc_boost = segment_in_description.get(segment, 0) * 2
            score = count + desc_boost
            scored.append((segment, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [segment for segment, _ in scored[:max_categories]]
