# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python email-routing agent built around LangGraph. `main.py` loads environment variables and exposes the compiled app. `graph.py` defines the workflow and connects node functions from `nodes/` such as `classify_intent.py`, `draft_response.py`, and `human_review.py`. Shared state types live in `states.py`, Redis helpers live in `redisClient.py`, and exploratory or legacy material is kept under `deprecated/` and `doc/`. Tests currently live in `test/` plus the root-level workflow script `test_v2.py`.

## Build, Test, and Development Commands
Create a virtual environment and install dependencies before running anything:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the agent entrypoint with `python main.py`. Run the workflow checks with `python test_v2.py`. Run the Redis smoke test with `python -m test.testRedis`. If you add `pytest`, keep existing scripts working until the repo is migrated fully.

## Coding Style & Naming Conventions
Use 4-space indentation and follow PEP 8 for imports, spacing, and line length. Keep modules and functions in `snake_case`; use `TypedDict` or class names in `PascalCase` such as `EmailAgentState`. New workflow steps should be added as one file per node under `nodes/`, with names matching the graph node ID where practical. Prefer small, single-purpose functions and keep prompt or API configuration near the relevant node.

## Testing Guidelines
Add focused test coverage for every new branch in the graph. Place reusable tests under `test/` and name files `test_*.py` for future `pytest` compatibility, even if the current suite is script-driven. For workflow tests, use stable `thread_id` values and assert final state fields, not just printed output. Document any required services such as Redis or OpenAI-compatible endpoints in the test setup.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects like `Implement Redis client with basic CRUD operations`. Follow that pattern and keep each commit scoped to one change. Pull requests should include a brief summary, impacted files or nodes, the commands you ran, and any environment or API assumptions. Include sample input/output when behavior changes in the email flow.

## Security & Configuration Tips
Secrets belong in `.env`; do not commit API keys, Redis passwords, or endpoint URLs. `nodes/config.py` reads `OPENAI_API_KEY` and `OPENAI_API_BASE`, so verify those values locally before running graph tests.
