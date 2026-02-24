"""Formatting and CLI token parsing helpers."""

from __future__ import annotations

from pathlib import Path

from src.data.dataset import normalize_group_name


def group_from_relative_path(relative_path: Path) -> str:
    """Infer normalized group name from dataset-relative path."""
    parts = relative_path.parts
    if len(parts) <= 1:
        return ""
    return normalize_group_name(parts[0])


def parse_eval_groups(raw_groups: list[str] | None) -> set[str] | None:
    """Parse `--eval-groups` values supporting comma or space separators."""
    if raw_groups is None:
        return None
    groups: set[str] = set()
    for token in raw_groups:
        for chunk in token.split(","):
            group = normalize_group_name(chunk.strip())
            if group:
                groups.add(group)
    return groups if groups else None


def format_total_cents(total_cents: int) -> str:
    """Format cents as `X EUR YY c`."""
    euros = total_cents // 100
    cents = total_cents % 100
    return f"{euros} EUR {cents:02d} c"


def format_signed_cents(diff_cents: int) -> str:
    """Format signed cents as `+X EUR YY c` or `-X EUR YY c`."""
    sign = "+" if diff_cents >= 0 else "-"
    euros_abs = abs(int(diff_cents)) // 100
    cents_abs = abs(int(diff_cents)) % 100
    return f"{sign}{euros_abs} EUR {cents_abs:02d} c"


def fmt_optional_score(value: float | int | None) -> str:
    """Format optional numeric metric or `n/a`."""
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def format_cents_compact(total_cents: int) -> str:
    """Format cents compactly as decimal euros with two digits."""
    cents = int(total_cents)
    sign = "-" if cents < 0 else ""
    cents_abs = abs(cents)
    return f"{sign}{cents_abs // 100}.{cents_abs % 100:02d}"


def format_diff_cents_compact(diff_cents: int) -> str:
    """Format signed compact euro difference from cents input."""
    cents = int(diff_cents)
    sign = "+" if cents >= 0 else "-"
    return f"{sign}{format_cents_compact(abs(cents))}"
