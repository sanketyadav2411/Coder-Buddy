"""Background job runner used by the web UI.

A generation or scan runs in a worker thread so the page stays responsive: the UI polls the ``Job``
object for stage changes, the plan, progress and a live activity log, and can ask it to stop.
No Streamlit code lives here, so it can be tested on its own.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from agent import events

LOG_LIMIT = 400


@dataclass
class Job:
    kind: str                                   # "generate" | "scan"
    stage_order: list[str]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    state: str = "running"                      # running | done | error | cancelled
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    stages: dict[str, str] = field(default_factory=dict)          # waiting|running|done|skipped|error
    stage_times: dict[str, list[Optional[float]]] = field(default_factory=dict)
    plan: Any = None
    steps: list = field(default_factory=list)
    done_count: int = 0
    current_file: str = ""
    files_written: list[str] = field(default_factory=list)
    report: Any = None
    progress_done: int = 0
    progress_total: int = 0
    progress_text: str = ""
    log: list[tuple[float, str, str]] = field(default_factory=list)   # (elapsed s, level, text)
    error: str = ""
    trace: str = ""
    cancelling: bool = False
    fix_ran: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        for name in self.stage_order:
            self.stages[name] = "waiting"
            self.stage_times[name] = [None, None]

    # ---- helpers -------------------------------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    def add_log(self, text: str, level: str = "info") -> None:
        with self.lock:
            if self.log and self.log[-1][2] == text:
                return
            self.log.append((time.time() - self.started, level, text))
            if len(self.log) > LOG_LIMIT:
                del self.log[: len(self.log) - LOG_LIMIT]

    def set_stage(self, name: str, state: str) -> None:
        if name not in self.stages:
            return
        now = time.time()
        times = self.stage_times[name]
        if state == "running" and times[0] is None:
            times[0] = now
        if state in ("done", "skipped", "error") and times[1] is None:
            times[1] = now
            if times[0] is None:
                times[0] = now
        self.stages[name] = state

    def stage_seconds(self, name: str) -> Optional[float]:
        start, end = self.stage_times.get(name, [None, None])
        if start is None:
            return None
        return (end or time.time()) - start

    # ---- events coming from inside the agent nodes ---------------------------------------------
    def on_event(self, ev: dict) -> None:
        kind, msg = ev.get("kind"), ev.get("message", "")
        if kind == "progress":
            self.progress_text = msg
            if "total" in ev:
                self.progress_done, self.progress_total = int(ev.get("done", 0)), int(ev["total"])
            if ev.get("file"):
                self.add_log(msg)
        elif kind == "file":
            if ev.get("path") and ev["path"] not in self.files_written:
                self.files_written.append(ev["path"])
            self.add_log(msg)
        elif kind == "warn":
            self.add_log(msg, "warn")
        else:
            if ev.get("file"):
                self.current_file = ev["file"]
            self.add_log(msg)

    # ---- graph "updates" stream ----------------------------------------------------------------
    def apply(self, chunk: dict) -> None:
        for node, update in chunk.items():
            if not update:
                continue
            if node == "planner":
                self.set_stage("planner", "done")
                self.set_stage("architect", "running")
                self.plan = update["plan"]
            elif node == "architect":
                self.set_stage("architect", "done")
                self.set_stage("coder", "running")
                self.steps = update["task_plan"].implementation_steps
            elif node == "coder":
                cs = update.get("coder_state")
                self.done_count = cs.current_step_idx if cs else len(self.steps)
                if update.get("status") == "DONE":
                    self.set_stage("coder", "done")
                    self.set_stage("scanner", "running")
            elif node == "scanner":
                self.set_stage("scanner", "done")
                if update.get("scan_next") == "fixer":
                    self.set_stage("fixer", "running")
                else:
                    self.set_stage("fixer", "done" if self.fix_ran else "skipped")
                    if update.get("report") is not None:
                        self.report = update["report"]
            elif node == "fixer":
                self.fix_ran = True
                self.set_stage("fixer", "done")
                self.set_stage("scanner", "running")     # re-scan the files that were changed


class JobManager:
    """Runs one job at a time (generation and scans share the same output folders)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current: Optional[Job] = None

    @property
    def busy(self) -> bool:
        j = self.current
        return j is not None and j.state == "running"

    def start(self, kind: str, graph_agent: Any, inputs: dict, stages: list[str],
              recursion_limit: int = 100) -> Optional[Job]:
        with self._lock:
            if self.busy:
                return None
            job = Job(kind=kind, stage_order=stages)
            first = stages[0]
            job.set_stage(first, "running")
            self.current = job
        t = threading.Thread(target=self._run, args=(job, graph_agent, inputs, recursion_limit),
                             name=f"coder-buddy-{job.id}", daemon=True)
        t.start()
        return job

    @staticmethod
    def _run(job: Job, graph_agent: Any, inputs: dict, limit: int) -> None:
        events.reset_cancel()
        events.set_listener(job.on_event)
        job.add_log("Started")
        try:
            for chunk in graph_agent.stream(inputs, {"recursion_limit": limit}, stream_mode="updates"):
                events.check_cancel()
                job.apply(chunk)
            job.add_log("Finished")
            outcome = "done"
        except events.Cancelled:
            job.add_log("Stopped by user", "warn")
            outcome = "cancelled"
        except BaseException as e:  # noqa: BLE001 - report everything to the UI
            job.error = f"{e.__class__.__name__}: {e}"
            job.trace = traceback.format_exc()
            for name, st in job.stages.items():
                if st == "running":
                    job.set_stage(name, "error")
            job.add_log(f"Error: {job.error}", "error")
            outcome = "error"
        events.set_listener(None)
        job.finished = time.time()
        job.state = outcome        # last: the UI treats a non-running state as "everything is final"

    def cancel(self) -> None:
        j = self.current
        if j is not None and j.state == "running":
            j.cancelling = True
            j.add_log("Stop requested - finishing the current model call...", "warn")
            events.request_cancel()


MANAGER = JobManager()
