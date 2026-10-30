"""Coder Buddy - terminal entry point.

    python main.py                                   # interactive prompt (original behaviour)
    python main.py -p "Create a todo app" --secure   # generate, then scan + fix the result
    python main.py --scan ./some_project             # scan + fix a COPY of existing code
    python main.py --scan app.py --no-fix --no-llm   # rule-based detection only, nothing modified

Exit codes: 0 = finished (and nothing left to fix), 1 = error, 2 = scan finished with violations
still open (handy for CI).
"""
import argparse
import os
import re
import sys
import traceback
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run engineering project planner")
    parser.add_argument("--recursion-limit", "-r", type=int, default=100,
                        help="Recursion limit for processing (default: 100)")
    parser.add_argument("--prompt", "-p", help="Project prompt (skips the interactive question)")

    sec = parser.add_argument_group("secure-coding scan")
    sec.add_argument("--secure", action="store_true",
                     help="after generating the project, scan it for secure-coding violations and fix them")
    sec.add_argument("--scan", metavar="PATH",
                     help="scan an existing file/folder instead of generating (works on a copy; "
                          "the original is never modified)")
    sec.add_argument("--no-fix", action="store_true", help="detect and report only; do not modify code")
    sec.add_argument("--no-llm", action="store_true", help="rule-based detection only (no model calls)")
    sec.add_argument("--max-rounds", type=int, help="maximum fix rounds (default 2, env MAX_FIX_ROUNDS)")
    sec.add_argument("--out", metavar="DIR",
                     help="where to put the (fixed) copy for --scan; must not exist or be empty "
                          "(default: scan_workspace/cli-<name>)")
    sec.add_argument("--json", action="store_true", help="print the scan report as JSON")

    llm = parser.add_argument_group("model")
    llm.add_argument("--provider", choices=["groq", "ollama"],
                     help="LLM provider (default: env LLM_PROVIDER or groq)")
    llm.add_argument("--model", help="model name for the chosen provider")
    llm.add_argument("--base-url", help="Ollama server URL (default http://localhost:11434)")
    return parser


def apply_llm_args(args) -> None:
    """CLI flags override the environment for this process (agent/llm.py reads the environment)."""
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
    if args.model:
        os.environ["OLLAMA_MODEL" if provider == "ollama" else "GROQ_MODEL"] = args.model
    if args.base_url:
        os.environ["OLLAMA_BASE_URL"] = args.base_url


def scan_options(args) -> dict:
    opts: dict = {"auto_fix": not args.no_fix, "use_llm": not args.no_llm}
    if args.max_rounds is not None:
        opts["max_fix_rounds"] = args.max_rounds
    return opts


def format_summary(report) -> str:
    s = report.summary
    lines = [
        "",
        "=== Secure-coding scan ===",
        f"Files scanned : {s.get('files_scanned', 0)}"
        f"   (LLM: {report.provider + '/' + report.model if report.llm_used else 'not used - rules only'})",
        f"Found         : {s.get('found', 0)}",
        f"Fixed         : {s.get('fixed', 0)}",
        f"Unresolved    : {s.get('unresolved', 0)}",
        f"New after fix : {s.get('new', 0)}",
        f"Remaining     : {s.get('remaining', 0)}",
    ]
    for v in report.violations:
        if v.status in ("open", "unresolved", "new"):
            lines.append(f"  [{v.severity.upper():8}] {v.file}:{v.line_start}  {v.cwe}  {v.title}  ({v.status})")
    for w in report.warnings:
        lines.append(f"  warning: {w}")
    if report.artifacts:
        lines.append(f"Report        : {report.artifacts.get('markdown', '')}")
    return "\n".join(lines)


def _safe_name(p: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(p).resolve().name or "scan")[:60] or "scan"


def run_scan(args) -> int:
    from agent.graph import scan_agent
    from agent.secure import ScanConfig, collect_sources, reset_workspace, write_text_files

    opts = scan_options(args)
    cfg = ScanConfig.from_env(opts)
    src = Path(args.scan)
    if not src.exists():
        print(f"Error: {src} does not exist.", file=sys.stderr)
        return 1
    files, skipped = collect_sources(src, cfg)
    if not files:
        print("Error: no scannable source files found "
              f"({len(skipped)} skipped).", file=sys.stderr)
        return 1

    if args.out:
        out = Path(args.out)
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            print(f"Error: output directory {out} is not empty; refusing to overwrite.", file=sys.stderr)
            return 1
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = reset_workspace(Path("scan_workspace"), f"cli-{_safe_name(args.scan)}")
    write_text_files(out, files)
    print(f"Scanning a copy of {src} in {out} ...")

    result = scan_agent.invoke({"scan_root": str(out), "scan_options": opts},
                               {"recursion_limit": args.recursion_limit})
    report = result["report"]
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(format_summary(report))
        if report.diffs:
            print(f"Fixed copy    : {out}")
    return 2 if report.summary.get("remaining", 0) else 0


def run_generate(args) -> int:
    from agent.graph import agent

    user_prompt = args.prompt if args.prompt else input("Enter your project prompt: ")
    state = {"user_prompt": user_prompt}
    if args.secure:
        state["secure_scan"] = True
        state["scan_options"] = scan_options(args)
    result = agent.invoke(state, {"recursion_limit": args.recursion_limit})
    if args.secure and result.get("report") is not None:
        report = result["report"]
        print(format_summary(report) if not args.json else report.model_dump_json(indent=2))
        return 2 if report.summary.get("remaining", 0) else 0
    print("Final State:", result)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.scan and (args.prompt or args.secure):
        parser.error("--scan cannot be combined with --prompt/--secure (scan existing code OR generate)")
    if args.max_rounds is not None and not 0 <= args.max_rounds <= 5:
        parser.error("--max-rounds must be between 0 and 5")

    try:
        apply_llm_args(args)
        code = run_scan(args) if args.scan else run_generate(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(code)


if __name__ == "__main__":
    main()
