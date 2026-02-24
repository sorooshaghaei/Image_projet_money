"""Application entrypoint for CLI execution."""

from __future__ import annotations

from src.app.cli import AppRunner


def main() -> None:
    AppRunner().run()


if __name__ == "__main__":
    main()
