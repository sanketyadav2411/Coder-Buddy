# Changes in this version (secure-coding scanner + local LLM)

This archive contains **only the files that were created or changed**. Copy it over your project folder
(`coder-buddy-main/`), keeping your own `Dockerfile`, `.sample_env`, `.env`, `.gitignore`, `.dockerignore`,
`.python-version`, `uv.lock`, `resources/` and the `pre_generated_project_*` folders.

## New files
| File | Purpose |
|---|---|
| `agent/llm.py` | LLM factory (Groq or local Ollama) and the "ask for JSON, validate it, retry with the error" helper |
| `agent/rules.py` | Rule catalogue (regex) mapped to CWE / OWASP Top 10 2021 / CERT C-C++; secret masking |
| `agent/secure.py` | Scan and fix engine, safe upload handling, patching with rollback, reports, scanner/fixer graph nodes |
| `tests/` | 200 offline tests (scripted fake model) + one optional live-Ollama smoke test |
| `.sample_env.secure-scan` | New environment settings to append to your `.sample_env` / `.env` |
| `CHANGES.md` | This file |

## Changed files
| File | Change |
|---|---|
| `agent/states.py` | Added `Violation`, `Patch`, `Report` (+ lenient LLM response models) and a typed `GraphState`. Original models untouched. |
| `agent/prompts.py` | Added `scanner_prompt` and `fixer_prompt` (injection-resistant). Original three prompts untouched. |
| `agent/graph.py` | `get_llm` now comes from `agent/llm.py` (default is still Groq + `openai/gpt-oss-120b`). Error detection widened as the README already claimed. Coder routing: ends exactly as before unless `secure_scan` is set, then goes to the scanner/fixer loop. Added the standalone `scan_agent`. |
| `app.py` | Provider selector (Groq/Ollama), scan toggle, **Scan existing code** mode with safe upload, Security report view (table, diffs, JSON/Markdown/ZIP downloads). Generate flow and live preview unchanged. |
| `main.py` | Added `--prompt`, `--secure`, `--scan`, `--no-fix`, `--no-llm`, `--max-rounds`, `--out`, `--provider`, `--model`, `--base-url`, `--json`. Interactive prompt still works. |
| `pyproject.toml` | Added `langchain-ollama`, `pytest` (dev group), pytest settings; version 0.2.0. |
| `docker-compose.yml` | Reaches host Ollama (`host.docker.internal`), mounts `scan_workspace/` and `scan_reports/`. |
| `README.md` | Documents the feature; sections 5, 8, 9 now match the code. |

## Manual steps (files I did not receive, so did not edit)
1. Append `.sample_env.secure-scan` to `.sample_env`.
2. Add these lines to `.gitignore` **and** `.dockerignore`:
   ```
   scan_workspace/
   scan_reports/
   ```
3. Refresh the lock file and install: `uv lock` then `uv pip install -r pyproject.toml`.
4. Docker users: `docker compose up --build` once (new dependency). Your `Dockerfile` should need no change.
5. Local model: install Ollama, `ollama pull qwen2.5-coder:7b`, choose "Ollama (local)" in the sidebar.

## Not done (needs your decision)
- The optional hardening of the coder step (verifying that `write_file` was actually called) was left out: it changes
  existing generation behaviour, which you asked to keep.
- The stray `pip>=25.2` dependency was left in `pyproject.toml`.

# Fix release: reliable generation, faster scans, new UI

- **Generation no longer silently returns an empty project.** `agent/graph.py` checks that each file was really written; if the model "answered" without calling `write_file`, the file is requested as plain text and written by the app. With the Ollama provider tools are never used (planner/architect use plain JSON via `structured_call`, coder uses plain text with project context).
- Architect falls back to one task per planned file if the model gives no tasks. Clear errors instead of a silent "Done".
- **Speed:** parallel scan windows for cloud models (`SCAN_WORKERS`), 1 retry instead of 2, 1 fix round by default, output length caps, Ollama `keep_alive`, Groq `reasoning_effort=low`.
- **New UI** (`app.py`, `ui/style.css`, `.streamlit/config.toml`): pipeline stepper, live activity log, Stop button, runs in a background thread (`agent/jobs.py`), installed-model picker for Ollama, device-size preview, filterable security report with severity bar, diffs and downloads.
- `.sample_env` now contains a placeholder instead of a real key; duplicate `OLLAMA_MODEL` removed.
- 6 new tests (206 total).
