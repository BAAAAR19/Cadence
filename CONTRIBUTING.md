# Contributing

Contributions that improve scheduler correctness, measurement quality, cache
management, admission policy, or reproducibility are welcome.

## Development setup

```bash
git clone https://github.com/BAAAAR19/Cadence.git
cd Cadence
uv sync --all-groups

uv run ruff check src bench tests
uv run mypy src/cadence
uv run pytest -q -m "not slow"
```

Tests marked `slow` need the Qwen GGUF described in the README and a working
llama.cpp backend:

```bash
uv run pytest -q -m slow
```

## Pull requests

- Add tests for behavioural changes.
- Keep the scheduler and API paths free of blocking I/O on the event loop.
- Do not commit GGUF files, credentials, private prompts, or production traces.
- If a change can affect a reported number, run the relevant benchmark with a
  recorded seed and update its source hash, metadata, generated tables, and
  figures together.
- Report negative or flat results rather than selecting only favourable runs.
- Explain workload, offered-load grid, duration, warm-up, SLO, seed, and
  hardware for new performance claims.

Security-sensitive findings belong in the private process described in
[SECURITY.md](SECURITY.md), not in a public issue.
