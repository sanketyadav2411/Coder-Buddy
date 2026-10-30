import pathlib

from langchain_core.tools import tool

PROJECT_ROOT = pathlib.Path.cwd() / "generated_project"


def _normalize(path: str) -> str:
    """Make model-supplied paths relative to the project root.

    Models often return '/index.html', './index.html' or 'generated_project/index.html'.
    All of these are treated as a path inside the project root.
    """
    path = (path or ".").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    path = path.lstrip("/")
    if path.startswith("generated_project/"):
        path = path[len("generated_project/"):]
    return path or "."


def safe_path_for_project(path: str) -> pathlib.Path:
    root = PROJECT_ROOT.resolve()
    p = (root / _normalize(path)).resolve()
    if p != root and root not in p.parents:
        raise ValueError("Attempt to access a path outside the project root")
    return p


@tool
def write_file(path: str, content: str) -> str:
    """Writes content to a file at the specified path within the project root."""
    try:
        p = safe_path_for_project(path)
    except ValueError as e:
        return f"ERROR: {e}"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return f"WROTE:{p}"


@tool
def read_file(path: str) -> str:
    """Reads content from a file at the specified path within the project root."""
    try:
        p = safe_path_for_project(path)
    except ValueError as e:
        return f"ERROR: {e}"
    if not p.is_file():
        return ""
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


@tool
def get_current_directory() -> str:
    """Returns the project root directory."""
    return str(PROJECT_ROOT)


@tool
def list_files(directory: str = ".") -> str:
    """Lists all files in the specified directory within the project root."""
    try:
        p = safe_path_for_project(directory)
    except ValueError as e:
        return f"ERROR: {e}"
    if not p.is_dir():
        return f"ERROR: {directory} is not a directory"
    root = PROJECT_ROOT.resolve()
    files = [str(f.relative_to(root)) for f in p.glob("**/*") if f.is_file()]
    return "\n".join(files) if files else "No files found."


def init_project_root():
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    return str(PROJECT_ROOT)
