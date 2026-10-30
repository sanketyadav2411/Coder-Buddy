"""Regression tests for the 'no project files generated' bug and the background job runner."""
import json
import time
import types

import pytest

import agent.graph as G
import agent.tools as T
from agent import events
from agent.jobs import JobManager
from conftest import FakeLLM

PLAN_JSON = json.dumps({"name": "demo", "description": "d", "techstack": "html", "features": ["f"],
                        "files": [{"path": "index.html", "purpose": "page"}]})
TASKS_JSON = json.dumps({"implementation_steps": [{"filepath": "index.html", "task_description": "write it"}]})
CODE = "<h1>hello</h1>\n"


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "generated_project"
    root.mkdir()
    monkeypatch.setattr(T, "PROJECT_ROOT", root)
    return root


def script(prompt):
    if "PLANNER" in prompt:
        return PLAN_JSON
    if "ARCHITECT" in prompt:
        return TASKS_JSON
    return f"```html\n{CODE}```"


class NoopReactAgent:
    """A model that 'answers' in text and never calls write_file (what small local models do)."""

    def __init__(self, llm, tools):
        pass

    def invoke(self, payload):
        return {"messages": []}


def test_tool_call_that_writes_nothing_falls_back_to_plain_text(sandbox, monkeypatch):
    class Struct(FakeLLM):
        def with_structured_output(self, model):
            from agent.states import ImplementationTask, Plan, TaskPlan

            class R:
                def invoke(_, prompt):
                    if model is Plan:
                        return Plan.model_validate_json(PLAN_JSON)
                    return TaskPlan(implementation_steps=[ImplementationTask(filepath="index.html", task_description="x")])
            return R()

    llm = Struct(script)
    monkeypatch.setattr(G, "get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(G, "create_react_agent", NoopReactAgent)
    G.agent.invoke({"user_prompt": "x"}, {"recursion_limit": 30})
    assert (sandbox / "index.html").read_text() == CODE


def test_ollama_provider_never_uses_tool_calling(sandbox, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    llm = FakeLLM(script)

    def boom(*a, **k):
        raise AssertionError("tool calling must not be used with Ollama")

    monkeypatch.setattr(G, "get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(G, "create_react_agent", boom)
    result = G.agent.invoke({"user_prompt": "make a page"}, {"recursion_limit": 30})
    assert (sandbox / "index.html").read_text() == CODE
    assert result["plan"].name == "demo"


def test_empty_model_reply_raises_a_clear_error(sandbox, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    llm = FakeLLM(lambda p: PLAN_JSON if "PLANNER" in p else TASKS_JSON if "ARCHITECT" in p else "")
    monkeypatch.setattr(G, "get_llm", lambda *a, **k: llm)
    with pytest.raises(Exception, match="Could not generate index.html"):
        G.agent.invoke({"user_prompt": "x"}, {"recursion_limit": 30})


# ---- background jobs ----------------------------------------------------------------------------------
class FakeGraph:
    def __init__(self, chunks=(), error=None, delay=0.0):
        self.chunks, self.error, self.delay = chunks, error, delay

    def stream(self, inputs, config, stream_mode=None):
        for c in self.chunks:
            time.sleep(self.delay)
            events.emit("log", "step")
            yield c
        if self.error:
            raise self.error


def wait(job, timeout=5):
    end = time.time() + timeout
    while job.state == "running" and time.time() < end:
        time.sleep(0.02)
    return job


def test_job_finishes_and_tracks_stages():
    plan = types.SimpleNamespace(name="p")
    steps = [types.SimpleNamespace(filepath="a")]
    chunks = [{"planner": {"plan": plan}}, {"architect": {"task_plan": types.SimpleNamespace(implementation_steps=steps)}},
              {"coder": {"coder_state": types.SimpleNamespace(current_step_idx=1), "status": "DONE"}}]
    job = wait(JobManager().start("generate", FakeGraph(chunks), {}, ["planner", "architect", "coder"]))
    assert job.state == "done" and job.plan is plan and job.done_count == 1
    assert [job.stages[n] for n in job.stage_order] == ["done", "done", "done"]


def test_job_reports_errors_and_marks_the_running_stage():
    job = wait(JobManager().start("generate", FakeGraph(error=RuntimeError("boom")), {}, ["planner", "architect"]))
    assert job.state == "error" and "boom" in job.error and job.stages["planner"] == "error"


def test_job_can_be_cancelled_and_only_one_runs_at_a_time():
    mgr = JobManager()
    chunks = [{"planner": {"plan": None}}] * 50
    job = mgr.start("generate", FakeGraph(chunks, delay=0.05), {}, ["planner"])
    assert mgr.start("generate", FakeGraph(), {}, ["planner"]) is None      # busy
    mgr.cancel()
    wait(job)
    assert job.state == "cancelled"
    assert mgr.start("generate", FakeGraph(), {}, ["planner"]) is not None   # free again
