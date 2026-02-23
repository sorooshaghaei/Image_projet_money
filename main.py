from pathlib import Path
import sys


if __name__ == "__main__":
    # Ensure local `src` imports work when launching from repository root.
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.runner import AppRunner

    AppRunner().main()
