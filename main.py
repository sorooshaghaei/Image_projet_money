"""CLI entry point for the euro coin detection pipeline."""

from src.runner import AppRunner


if __name__ == "__main__":
    # Keep main.py tiny so all runtime logic stays centralized in src/runner.py.
    AppRunner().main()
