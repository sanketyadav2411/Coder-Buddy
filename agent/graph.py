import os
import re

from dotenv import load_dotenv
from langchain_core.globals import set_verbose, set_debug
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from agent.events import emit
from agent.llm import (MODEL_NAME, LLMOutputError, current_provider, get_llm,  # noqa: F401  (MODEL_NAME kept for backwards compatibility)
                       structured_call)
from agent.prompts import *
from agent.secure import fixer_agent, route_after_scan, scanner_agent
from agent.states import *
from agent.tools import (get_current_directory, list_files, read_file, safe_path_for_project,
                         write_file)

_ = load_dotenv()

# Verbose LangChain logging dumps every prompt/response to the console.
# Enable it only when needed:  CODER_BUDDY_DEBUG=1
_DEBUG = os.getenv("CODER_BUDDY_DEBUG", "").strip().lower() in {"1", "true", "yes"}
set_debug(_DEBUG)
set_verbose(_DEBUG)

# How many times to retry when the model emits a malformed tool call.
_TOOL_RETRIES = 2


def _is_tool_call_failure(exc: Exception) -> bool:
    """True for Groq's 400 'tool_use_failed' / 'Failed to parse tool call arguments' errors
    (also 'output_parse_failed' / 'Parsing failed'), and for local models that cannot do tool calls.

    These happen when the model writes broken JSON for a tool call (typically when the
    file content is long or contains odd characters). They are random, so retrying or
    falling back to plain-text generation fixes them.
    """
    msg = str(exc)
    return any(s in msg for s in (
        "tool_use_failed", "Failed to parse tool call", "output_parse_failed",
        "Parsing failed", "does not support tools",
    ))


def _extract_code(text: str) -> str:
    """Pull file content out of a fenced code block (or return the text as-is)."""
    text = text.strip()
    m = re.match(r"^```[^\n]*\n(.*)\n```\s*$", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```[^\n]*\n(.*?)\n```", text, re.DOTALL)
    return m.group(1) if m else text


def _invoke_with_retry(fn):
    """Run fn(); retry on malformed-tool-call errors, re-raise anything else."""
    last = None
    for _ in range(_TOOL_RETRIES + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not _is_tool_call_failure(e):
                raise
            last = e
    raise last


# Reply shapes shown to the model when it has to answer with plain JSON (no tool calling).
_PLAN_HINT = ('{"name": "...", "description": "...", "techstack": "...", "features": ["..."], '
              '"files": [{"path": "index.html", "purpose": "..."}]}')
_TASKS_HINT = '{"implementation_steps": [{"filepath": "index.html", "task_description": "..."}]}'


def _json_prompt(prompt: str, hint: str) -> str:
    return (prompt.rstrip() + "\n\nReply with ONLY one JSON object, no prose and no code fences, "
            f"shaped like this example:\n{hint}\n")


def _structured(model_cls, prompt: str, hint: str, max_tokens: int = 1800):
    """Get a validated `model_cls` from the model.

    Groq: native structured output (tool calling) with retries, and a plain-JSON fallback if the
    model keeps breaking its tool call. Ollama: plain JSON straight away, because small local models
    are unreliable at tool calling (that was the main reason projects came back empty)."""
    if current_provider() != "ollama":
        try:
            resp = _invoke_with_retry(lambda: get_llm().with_structured_output(model_cls).invoke(prompt))
            if resp is not None:
                return resp
        except Exception as e:  # noqa: BLE001
            if not _is_tool_call_failure(e):
                raise
            emit("warn", "Structured output failed; retrying with plain JSON")
    return structured_call(get_llm(temperature=0, json_mode=True, max_tokens=max_tokens),
                           _json_prompt(prompt, hint), model_cls, retries=2)


def planner_agent(state: dict) -> dict:
    """Converts user prompt into a structured Plan."""
    emit("log", "Planner: turning your prompt into a project plan")
    user_prompt = state["user_prompt"]
    try:
        resp = _structured(Plan, planner_prompt(user_prompt), _PLAN_HINT)
    except LLMOutputError as e:
        raise ValueError(f"The model could not produce a project plan ({e}). "
                         "Try a larger model or a shorter, clearer prompt.") from e
    if resp is None:
        raise ValueError("Planner did not return a valid plan. Try rephrasing the prompt.")
    emit("log", f"Planner: {resp.name} - {len(resp.files)} file(s)")
    return {"plan": resp}


def _tasks_from_plan(plan: Plan) -> TaskPlan:
    """Last-resort task list: one task per planned file."""
    return TaskPlan(implementation_steps=[
        ImplementationTask(filepath=f.path, task_description=f"Implement this file completely. Purpose: {f.purpose}. "
                                                              f"Project: {plan.name} - {plan.description}")
        for f in plan.files])


def architect_agent(state: dict) -> dict:
    """Creates TaskPlan from Plan."""
    emit("log", "Architect: breaking the plan into implementation tasks")
    plan: Plan = state["plan"]
    try:
        resp = _structured(TaskPlan, architect_prompt(plan=plan.model_dump_json()), _TASKS_HINT, 2000)
    except LLMOutputError:
        resp = None
    steps = [t for t in (resp.implementation_steps if resp else []) if t.filepath.strip()]
    if not steps:
        if not plan.files:
            raise ValueError("The model produced neither tasks nor a file list. Try rephrasing the prompt.")
        emit("warn", "Architect gave no usable tasks; using one task per planned file")
        resp = _tasks_from_plan(plan)
    resp.plan = plan
    emit("log", f"Architect: {len(resp.implementation_steps)} task(s)")
    return {"task_plan": resp}


def _project_context(task_plan: TaskPlan, upto: int, budget: int = 7000) -> str:
    """Overview + already written files, so a model without tool use still keeps files consistent."""
    plan = getattr(task_plan, "plan", None)
    parts = []
    if plan is not None:
        parts.append(f"Project: {plan.name} - {plan.description}\nTech stack: {plan.techstack}")
        parts.append("Planned files:\n" + "\n".join(f"- {f.path}: {f.purpose}" for f in plan.files))
    used = 0
    seen = set()
    for t in task_plan.implementation_steps[:upto]:
        if t.filepath in seen:
            continue
        seen.add(t.filepath)
        text = read_file.run(t.filepath)
        if not text or text.startswith("ERROR:"):
            continue
        if used + len(text) > budget:
            text = text[: max(0, budget - used)] + "\n...[truncated]"
        parts.append(f"--- already written: {t.filepath} ---\n{text}")
        used += len(text)
        if used >= budget:
            break
    return "\n\n".join(parts)


def _written(path: str) -> bool:
    try:
        p = safe_path_for_project(path)
    except ValueError:
        return False
    return p.is_file() and p.stat().st_size > 0


def _plain_text_file(system_prompt: str, user_prompt: str, filepath: str, max_tokens: int = 3500) -> None:
    """Ask for the file as plain text in a code fence and write it ourselves (no tool call, no JSON)."""
    last = "the model returned an empty reply"
    for _ in range(2):
        reply = get_llm(temperature=0.2, max_tokens=max_tokens).invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + (
                "\n\nDo NOT call any tools. Reply with ONLY the complete final content of "
                f"{filepath} inside a single fenced code block, with no explanation.")},
        ])
        text = reply.content if isinstance(reply.content, str) else "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in reply.content)
        code = _extract_code(text)
        if not code.strip():
            continue
        result = write_file.run({"path": filepath, "content": code if code.endswith("\n") else code + "\n"})
        if str(result).startswith("ERROR:"):
            last = str(result)
            continue
        return
    raise RuntimeError(f"Could not generate {filepath}: {last}")


def coder_agent(state: dict) -> dict:
    """Writes one file per call. Always verifies that the file really exists afterwards."""
    coder_state: CoderState = state.get("coder_state")
    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    idx = coder_state.current_step_idx
    current_task = steps[idx]
    emit("log", f"Coder: writing {current_task.filepath} ({idx + 1}/{len(steps)})",
         file=current_task.filepath, index=idx, total=len(steps))
    existing_content = read_file.run(current_task.filepath)
    if existing_content.startswith("ERROR:"):
        # Bad path from the model: treat as a new/empty file instead of crashing the run.
        existing_content = ""

    system_prompt = coder_system_prompt()
    context = _project_context(coder_state.task_plan, idx)
    user_prompt = (
        (f"{context}\n\n" if context else "")
        + f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )

    use_tools = current_provider() != "ollama" and os.getenv("CODER_USE_TOOLS", "1") != "0"
    done = False
    if use_tools:
        coder_tools = [read_file, write_file, list_files, get_current_directory]
        react_agent = create_react_agent(get_llm(), coder_tools)
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        try:
            _invoke_with_retry(lambda: react_agent.invoke({"messages": messages}))
        except Exception as e:  # noqa: BLE001
            if not _is_tool_call_failure(e):
                raise
            emit("warn", f"Tool call failed for {current_task.filepath}; using plain-text mode")
        # The model may have "answered" without ever calling write_file; that raises no error, so check.
        done = _written(current_task.filepath)
        if not done:
            emit("warn", f"{current_task.filepath} was not written by the tool call; using plain-text mode")
    if not done:
        _plain_text_file(system_prompt, user_prompt, current_task.filepath)

    emit("file", f"Wrote {current_task.filepath}", path=current_task.filepath)
    coder_state.current_step_idx += 1
    return {"coder_state": coder_state}


def _after_coder(s: dict) -> str:
    """Coder loops until DONE; then either finish (default) or run the secure-coding scan."""
    if s.get("status") != "DONE":
        return "coder"
    return "scanner" if s.get("secure_scan") else "END"


# ---- generation pipeline (the original behaviour; scanning only runs when secure_scan is set) ----
graph = StateGraph(GraphState)

graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)
graph.add_node("scanner", scanner_agent)
graph.add_node("fixer", fixer_agent)

graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")
graph.add_conditional_edges(
    "coder",
    _after_coder,
    {"END": END, "coder": "coder", "scanner": "scanner"}
)
graph.add_conditional_edges("scanner", route_after_scan, {"fixer": "fixer", "END": END})
graph.add_edge("fixer", "scanner")

graph.set_entry_point("planner")
agent = graph.compile()

# ---- standalone pipeline: scan (and fix) code that already exists --------------------------------
scan_graph = StateGraph(GraphState)
scan_graph.add_node("scanner", scanner_agent)
scan_graph.add_node("fixer", fixer_agent)
scan_graph.add_conditional_edges("scanner", route_after_scan, {"fixer": "fixer", "END": END})
scan_graph.add_edge("fixer", "scanner")
scan_graph.set_entry_point("scanner")
scan_agent = scan_graph.compile()

if __name__ == "__main__":
    result = agent.invoke({"user_prompt": "Build a colourful modern todo app in html css and js"},
                          {"recursion_limit": 100})
    print("Final State:", result)
