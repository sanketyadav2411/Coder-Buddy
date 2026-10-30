# Coder Buddy

An AI project generator built with **LangGraph**. You describe a project in plain English; three agents (Planner, Architect, Coder) plan it, break it into tasks and write the files for you. It has a **Streamlit web UI** and a plain terminal mode. The LLM is served by **Groq** (`openai/gpt-oss-120b`) or by a **local model through Ollama**.

New in this version: **Local LLM-based automated detection and correction of secure-coding violations** (section 9). It scans generated or uploaded code against CWE / OWASP Top 10 / CERT C-C++ categories, fixes what it can, and re-scans to verify every fix.

Original project: Codebasics Inc. This copy includes reliability fixes (section 8) and the secure-coding scanner (section 9).

---

## 1. Quick start (Windows)

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/), and either a free Groq API key from https://console.groq.com/keys **or** [Ollama](https://ollama.com) for a fully local run.

```cmd
cd C:\path\to\coder-buddy-main
uv venv
.venv\Scripts\activate
uv pip install -r pyproject.toml
copy .sample_env .env
notepad .env
streamlit run app.py
```

In Notepad, replace `<YOUR_API_KEY_HERE>` with your real key, save, close. The app opens at http://localhost:8501.

macOS / Linux: same steps, but activate with `source .venv/bin/activate` and copy with `cp .sample_env .env`.

**Local model (no cloud, no API key):**

```cmd
ollama pull qwen2.5-coder:7b
```

Then pick **Ollama (local)** in the sidebar, or set `LLM_PROVIDER=ollama` in `.env`.

---

## 2. Start, stop, sleep, wake

| Action | What to do |
|---|---|
| **Start** | `cd` into the folder, `.venv\Scripts\activate`, `streamlit run app.py` |
| **Start without activating** | `uv run streamlit run app.py` |
| **Stop** | Click the terminal running Streamlit, press **Ctrl+C**. Closing the browser tab does NOT stop it. |
| **Put to sleep (safe shutdown)** | 1) Wait for any running generation or scan to finish. 2) Ctrl+C. 3) `deactivate`. 4) If you used Docker: `docker compose down`. 5) Copy anything you want to keep out of `generated_project\` (it is cleared on the next generation by default) and `scan_reports\`. 6) Never share or upload `.env` (it holds your API key). |
| **Wake up** | Run the Start steps again. Nothing needs reinstalling. |
| **Terminal mode** | `python main.py`, then type your prompt. Add `-r 150` to raise the recursion limit. More options in section 3b. |

Remember: every new terminal window needs `.venv\Scripts\activate` again (or use `uv run ...`).

---

## 3. Using the app

**Generate project mode (default)**

1. Choose a provider in the sidebar (Groq needs a key; Ollama needs a running server and a pulled model).
2. Click an example prompt or type your own.
3. (Optional) tick **Run secure-coding scan after generation**.
4. Click **Generate**. Progress shows Planner, Architect, Coder (and Scanner / Fixer when scanning) and a per-file task list.
5. When done: browse files in the **Files** tab, see the site in **Live preview** (static HTML/CSS/JS only), read the **Security report**, or **Download project as ZIP**.

**Scan existing code mode**

1. Switch **Mode** to *Scan existing code*.
2. Upload source files or a `.zip` of a project.
3. Click **Scan**. A **copy** is scanned and fixed; your originals are never modified.
4. Review the violations table, the before/after diffs, and download the report (JSON / Markdown) or the fixed copy (ZIP). Use *Delete the uploaded copy* when you are done.

Sidebar settings: **LLM provider**, **Recursion limit** (max agent steps, default 100), **Clear previous generated files first** (default on), and the scan options: **Fix violations automatically**, **Use the LLM** (off = rule-based checks only), **Max fix rounds** (default 2).

Tips for better results:
- Name the exact files you want ("create index.html, style.css, script.js").
- Ask for ASCII-only code and short comments (long comments with odd symbols are what used to break tool calls).
- Keep it to about 3 to 5 files per run.

### 3b. Terminal mode

```cmd
python main.py                                         :: interactive prompt (original behaviour)
python main.py -p "Create a todo app" --secure         :: generate, then scan + fix the result
python main.py --scan .\some_project                   :: scan + fix a COPY of existing code
python main.py --scan app.py --no-fix --no-llm         :: rules only, nothing modified
python main.py --scan .\src --provider ollama --model qwen2.5-coder:7b --out .\fixed
```

| Flag | Meaning |
|---|---|
| `--prompt`, `-p` | Project prompt (skips the interactive question) |
| `--secure` | After generating, scan and fix the generated project |
| `--scan PATH` | Scan an existing file or folder (a copy is scanned) |
| `--no-fix` / `--no-llm` | Report only / rule-based checks only |
| `--max-rounds N` | Max fix rounds, 0 to 5 |
| `--out DIR` | Where to put the fixed copy (must not exist or be empty) |
| `--provider`, `--model`, `--base-url` | Override the LLM settings for this run |
| `--json` | Print the report as JSON |

Exit codes: `0` finished and nothing left to fix, `1` error, `2` scan finished but violations remain (useful in CI).

---

## 4. How it works

```
user prompt -> PLANNER -> ARCHITECT -> CODER (loops once per task) -> done            (default)
                                                     |
                                           secure_scan on?
                                                     v
                                        SCANNER -> FIXER -> SCANNER ... -> report
existing code ----------------------------> SCANNER -> FIXER -> SCANNER ... -> report
```

- **Planner** turns the prompt into a `Plan` (name, description, tech stack, features, files).
- **Architect** turns the plan into a `TaskPlan`: an ordered list of `ImplementationTask` (file path + detailed description).
- **Coder** takes one task at a time, reads the current file, and uses tools to write the full file. It loops until all tasks are done.
- **Scanner / Fixer** are described in section 9. They only run when you ask for them.

The Planner and Architect use structured output (Pydantic models). The Coder is a ReAct agent with four tools: `read_file`, `write_file`, `list_files`, `get_current_directory`. All file access is sandboxed to `generated_project/`.

Diagram: `resources/coder_buddy_diagram.png`

---

## 5. What is where

```
coder-buddy-main/
|-- app.py                  Streamlit web UI (run this): generate mode + scan mode
|-- main.py                 Terminal version (python main.py), incl. --secure / --scan
|-- agent/
|   |-- graph.py            LangGraph pipelines: planner, architect, coder (+ scanner, fixer);
|   |                       retry + fallback logic; `agent` (generate) and `scan_agent` (scan only)
|   |-- states.py           Pydantic models: Plan, File, TaskPlan, ImplementationTask, CoderState,
|   |                       Violation, Patch, Report, GraphState
|   |-- prompts.py          Prompt text for each agent + scanner_prompt / fixer_prompt
|   |-- llm.py              LLM factory (Groq or Ollama) + "ask for JSON, validate it" helper
|   |-- rules.py            Deterministic rule catalogue mapped to CWE / OWASP / CERT
|   |-- secure.py           Scan + fix engine: loading, chunking, validation, patching, reports,
|   |                       safe upload handling, the scanner and fixer graph nodes
|   |-- tools.py            File tools + path sandbox (generated_project/)
|-- tests/                  pytest suite (runs offline with a scripted fake model)
|-- generated_project/      OUTPUT folder, generated files land here
|-- scan_workspace/         COPIES of uploaded / scanned code (originals are never modified)
|-- scan_reports/           report.json, report.md and backups of every changed file
|-- pre_generated_project_calculator/   Sample output (calculator)
|-- pre_generated_project_todo_app/     Sample output (to-do app)
|-- resources/              Architecture diagram (.png and .mmd source)
|-- pyproject.toml          Dependencies (Python >= 3.11) and pytest settings
|-- uv.lock                 Locked dependency versions (re-run `uv lock` after pulling this version)
|-- .python-version         Python version for uv (3.11)
|-- .sample_env             Template for .env (see .sample_env.secure-scan for the new settings)
|-- .env                    YOUR secrets (you create it; never share it)
|-- Dockerfile, docker-compose.yml, .dockerignore   Docker setup
|-- .gitignore              Keeps .env, .venv, generated files out of git
```

---

## 6. Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `groq` (default, the original behaviour) or `ollama` (local). |
| `GROQ_API_KEY` | Required for the Groq provider. Your Groq key. |
| `GROQ_MODEL` | Optional. Override the model (default `openai/gpt-oss-120b`). |
| `OLLAMA_MODEL` | Local model (default `qwen2.5-coder:7b`). Must be pulled first. |
| `OLLAMA_BASE_URL` | Ollama server (default `http://localhost:11434`). |
| `OLLAMA_NUM_CTX` | Context window in tokens (default 8192). Ollama silently truncates longer prompts. |
| `OLLAMA_TIMEOUT`, `OLLAMA_TEMPERATURE` | Optional request timeout (s, default 300) and temperature. |
| `MAX_FIX_ROUNDS` | Scan/fix rounds (default 2, max 5). |
| `SCAN_MAX_FILES`, `SCAN_MAX_FILE_KB`, `SCAN_MAX_TOTAL_MB` | Input limits (200 files, 200 KB per file, 5 MB total). |
| `SCAN_CHUNK_LINES`, `SCAN_CHUNK_CHARS` | Window size for large files (200 lines / 12000 characters). |
| `CODER_BUDDY_DEBUG=1` | Optional. Print full LangChain debug logs. |

The new settings are collected in `.sample_env.secure-scan`; append them to your `.env`.

---

## 7. Docker (optional)

```cmd
copy .sample_env .env
docker compose up --build
```
Open http://localhost:8501. Output appears in `./generated_project`; scan copies in `./scan_workspace`; reports in `./scan_reports`. Stop with Ctrl+C, then `docker compose down`.

Using a local model from inside Docker: Ollama runs on the **host**, so the container must reach it. `docker-compose.yml` sets `OLLAMA_BASE_URL=http://host.docker.internal:11434` and adds `extra_hosts: host.docker.internal:host-gateway` (needed on Linux; Docker Desktop provides the name already). On Linux, Ollama must listen on an address the container can reach: start it with `OLLAMA_HOST=0.0.0.0` (only on a trusted network). Rebuild once (`docker compose up --build`) so the new `langchain-ollama` dependency is installed. Docker Compose 2.24 or newer is required for the `env_file` syntax used here.

---

## 8. Fixes made in this version

Problem: Groq sometimes returned `400 tool_use_failed: Failed to parse tool call arguments as JSON` (or `output_parse_failed`). The model wrote broken JSON when putting a long file inside a tool call (for example a stray unicode symbol or mangled comment markers). This is random model behaviour, not a bug in the app logic.

Changes in `agent/graph.py`:
- `_is_tool_call_failure()` recognises these Groq errors (`tool_use_failed`, `Failed to parse tool call`, `output_parse_failed`, `Parsing failed`) and also local models that do not support tool calls (`does not support tools`).
- `_invoke_with_retry()` retries the call up to 2 times; used by planner, architect and coder.
- Coder fallback: if tool calling still fails, the model is asked for the file as plain text in a code fence; `_extract_code()` pulls it out and the app writes the file itself (no JSON involved).
- `GROQ_MODEL` environment override.

---

## 9. Secure-coding scanner and fixer (local LLM)

### What it does

1. **Detects** violations with a hybrid approach: a deterministic rule catalogue (`agent/rules.py`, regex rules mapped to CWE / OWASP Top 10 2021 / CERT C-C++) **plus** semantic analysis by the LLM, which also receives the rule hits as hints.
2. **Validates** every LLM finding before it is reported: the model must quote the offending code, we locate that quote in the real file, correct the line numbers, and drop anything we cannot find (hallucination guard).
3. **Fixes** by asking the model for line-range patches. Our code checks, applies and syntax-checks them (Python `compile`, JavaScript `node --check` when Node is installed, JSON); a patch set that breaks the file is rolled back.
4. **Verifies** by re-scanning every changed file. A violation only counts as `fixed` when the re-scan no longer finds it. Fixes that fail stay `unresolved`; problems a fix introduces appear as `new`. The loop stops after `MAX_FIX_ROUNDS`, or earlier if a round resolves nothing.
5. **Reports** to the UI and to `scan_reports/scan-<time>/report.json` and `report.md`, with before/after diffs. Changed files are backed up in `scan_reports/scan-<time>/originals/`.

Covered: Python, JavaScript/TypeScript, HTML, C/C++ (rules + LLM); Java, Go, PHP, Ruby, C#, shell, SQL and similar (LLM + generic rules); JSON/YAML/TOML/.env (rules only, for example hardcoded secrets).

### Why a local model

Source code is sensitive. With `LLM_PROVIDER=ollama` nothing leaves your machine. The app warns when the provider is the Groq cloud or when `OLLAMA_BASE_URL` is not a local or private address.

### Recommended local models

Any Ollama model that follows JSON instructions works. `qwen2.5-coder:7b` (default) needs roughly 6 GB of VRAM or 8 GB of RAM; larger coder models detect more but are slower. If a small model returns unusable output the run does **not** fail: it falls back to rule-based results and says so in the report warnings.

### Security design

| Risk | Mitigation |
|---|---|
| Prompt injection from scanned code | Code is sent as delimited data under a random per-call tag; scan and fix calls get **no tools**; replies are only ever parsed as JSON and never executed |
| Path traversal / zip-slip | Uploaded archives are rejected on `..`, absolute or drive paths, symlinks; every written path is resolved and must stay inside the workspace |
| Zip bombs / huge inputs | Limits on entries, per-file size, total extracted size and file count; binaries and non-UTF-8 files are skipped |
| A bad patch corrupting code | Patches are range-checked, overlap-checked, limited to the window the model saw, syntax-checked and written atomically; originals are backed up |
| Secret leakage | Credentials are masked in findings, diffs and reports; keys live only in the environment |
| Data egress | Local provider by default for scanning guidance; warnings for cloud or remote endpoints |
| UI injection | Findings are shown with `st.code` / `st.text`, never as raw HTML; titles are markdown-escaped |

### Limits you should know about

- LLM findings are anchored to real code but are **not proven exploitable**; the rule catalogue is intentionally small and conservative. This is an assistant, not a replacement for a SAST tool or a security review.
- Automatic fixes change behaviour only where needed, but **review every diff** before using the result.
- Small models miss things and occasionally propose bad fixes; the verification step catches many of these but not all.

---

## 10. Troubleshooting

| Message | Cause and fix |
|---|---|
| `'source' is not recognized` | You are on Windows. Use `.venv\Scripts\activate`. |
| `'streamlit' is not recognized` | Venv not active in this terminal. Activate it, or `uv run streamlit run app.py`. |
| `The system cannot find the path specified` on activate | Broken/missing venv. `rmdir /s /q .venv`, then `uv venv` and reinstall. |
| `ModuleNotFoundError` | Dependencies not installed. `uv pip install -r pyproject.toml`. |
| `Please enter your Groq API key` | Fill `.env` or paste the key in the sidebar (Groq provider only). |
| `needs the 'langchain-ollama' package` | `uv pip install -r pyproject.toml` again after updating. |
| Connection refused / `LLM unavailable` | Ollama is not running (`ollama serve`) or `OLLAMA_BASE_URL` is wrong. The scan continues with rules only. |
| `model ... not found` | Pull it first: `ollama pull qwen2.5-coder:7b`. |
| "The model never returned usable output" | The model is too small or not following JSON. Try a larger coder model; raise `OLLAMA_NUM_CTX`. |
| Scan very slow | Lower `SCAN_CHUNK_LINES`, scan fewer files, or use a smaller/faster model. |
| `400 tool_use_failed` / `output_parse_failed` | Handled by retry and fallback. If it persists, click Generate again or set another `GROQ_MODEL`. |
| Rate limit / 429 | Groq free-tier limit. Wait a minute and retry. |
| `Another generation is already running` | One run at a time (shared output folders). Wait for it to finish. |
| Live preview blank | Preview only works for static sites with an `index.html`. Download the ZIP instead. |
| Generated C++ will not compile | Check for non-ASCII symbols (unicode minus signs, `≤`) and mangled `//` comments; fix by hand. |
| Docker cannot reach Ollama | See section 7 (`host.docker.internal`, `OLLAMA_HOST=0.0.0.0`). |

---

## 11. Security notes

- `.env` holds your API key. It is in `.gitignore`; do not zip, email or upload it.
- Generated code is written by an AI. Read and test it before using it.
- Scan reports contain code excerpts: treat `scan_reports/` and `scan_workspace/` as sensitive (both should be in `.gitignore` and `.dockerignore`).
- The live preview runs generated JavaScript in an iframe; only preview code you are comfortable running.

---

## 12. Tests

```cmd
uv pip install pytest
pytest                  :: ~200 tests, offline, a few seconds
pytest -m ollama -s     :: optional smoke test against a real local Ollama model
```

The suite uses a scripted fake model, so it needs no network or GPU. It covers the original pipeline (regression with scanning off), rule detection on vulnerable and clean fixtures, JSON parsing and retries, hallucination rejection, patching and rollback, upload safety (zip-slip, symlinks, zip bombs), the fix loop and its limits, the CLI and the UI.
