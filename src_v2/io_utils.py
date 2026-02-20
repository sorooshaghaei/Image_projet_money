from pathlib import Path
from typing import Iterable, List, Sequence


def list_image_paths(path_text: str, valid_extensions: Sequence[str]) -> List[Path]:
    """Return sorted image files from a file path or recursively from a directory."""
    root = Path(path_text)
    if root.is_file():
        return [root]

    if not root.exists():
        return []

    exts = {ext.lower() for ext in valid_extensions}
    out = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    out.sort(key=lambda p: p.as_posix().lower())
    return out


def short_path(path: Path, root: Path) -> str:
    """Stable short path for terminal and CSV tracing."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def ensure_parent_dir(path_text: str) -> None:
    """Create parent directory for report outputs when needed."""
    Path(path_text).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
