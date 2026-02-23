"""CLI entrypoint and application runner."""

from __future__ import annotations

import argparse

import onefiler as _legacy


class AppRunner:
    """Application runner keeping CLI behavior faithful to onefiler.py."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Return the CLI parser used by the legacy runtime."""

        return _legacy.OneFileRunner._build_parser()

    def run(self) -> None:
        """Run the full evaluation/report/viewer flow."""

        _legacy.OneFileRunner().run()

    def main(self) -> None:
        """Main method used by ``main.py``."""

        self.run()


def main() -> None:
    """Standalone module entrypoint."""

    AppRunner().run()


__all__ = ["AppRunner", "main"]
