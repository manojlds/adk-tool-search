# adk-tool-search

Dynamic tool search for Google ADK. Implements Anthropic-style deferred tool loading using BM25 search to discover and load tools on demand.

## Commands

Install/update dependencies:

```bash
uv sync --all-extras
```

After ANY code change, run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Run LLM integration tests (requires API key in `.env`):

```bash
uv run pytest -m llm
```

## Quality Checks

- Do not consider a change complete until lint and tests pass.
- Default `uv run pytest` excludes `llm`-marked tests (no API key needed).
- LLM tests run with `uv run pytest -m llm` and require a `.env` file (see `.env.example`).

## Conventions

- Use `uv` for dependency management and command execution.
- Python 3.11+, line length 100 characters.
- Run `ruff format` and `ruff check` before committing.
- Unit tests in `tests/`, LLM integration tests in `tests/llm/`.
- Mark all LLM integration tests with `@pytest.mark.llm`.

## Directory Structure

```text
adk_tool_search/     # Library source (registry, loader)
tests/               # Unit tests
tests/llm/           # LLM integration tests (marked with @pytest.mark.llm)
examples/            # Usage examples
.github/workflows/   # CI workflows
```

## Testing Patterns

- Test runner: `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`).
- Unit tests: pure logic, no API calls, run in CI.
- LLM tests: require a live LLM backend, excluded from CI via `-m "not llm"`.
- LLM tests use `litellm` via `make_litellm_model()` from `tests/conftest.py`.
- Copy `.env.example` to `.env` and set your API key to run LLM tests.
