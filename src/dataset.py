"""Dataset listing and ground-truth repository."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence


def normalize_group_name(group: str) -> str:
    cleaned = (group or "").strip().lower()
    if cleaned.startswith("grp"):
        return "gp" + cleaned[3:]
    return cleaned


@dataclass(frozen=True)
class DatasetImage:
    path: Path
    relative_path: Path


class ImageDataset:
    def __init__(self, root_dir: Path, valid_extensions: Sequence[str]):
        self._root_dir = Path(root_dir)
        self._valid_extensions = {ext.lower() for ext in valid_extensions}

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def list_images(self, limit: int | None = None) -> list[DatasetImage]:
        if not self._root_dir.exists():
            return []

        images: list[DatasetImage] = []
        for path in self._root_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self._valid_extensions:
                continue
            relative = path.relative_to(self._root_dir)
            images.append(DatasetImage(path=path, relative_path=relative))

        images.sort(key=lambda item: str(item.relative_path).lower())
        if limit is not None:
            return images[: max(0, limit)]
        return images


@dataclass(frozen=True)
class GroundTruthEntry:
    filename: str
    group: str
    coin_count: int
    value_cents: int | None = None


class GroundTruthRepository:
    def __init__(self, rows: Iterable[GroundTruthEntry] | None = None):
        entries = list(rows) if rows is not None else self._parse_default_rows()
        self._index: Dict[tuple[str, str], GroundTruthEntry] = {}
        for entry in entries:
            key = (normalize_group_name(entry.group), entry.filename.lower())
            self._index[key] = GroundTruthEntry(
                filename=entry.filename,
                group=normalize_group_name(entry.group),
                coin_count=int(entry.coin_count),
                value_cents=None if entry.value_cents is None else int(entry.value_cents),
            )

    def find(self, filename: str, group: str) -> GroundTruthEntry | None:
        key = (normalize_group_name(group), filename.lower())
        return self._index.get(key)

    def _parse_default_rows(self) -> list[GroundTruthEntry]:
        entries: list[GroundTruthEntry] = []
        for raw_line in RAW_ANNOTATIONS.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            filename = parts[0]
            pieces_text = parts[1]
            group = parts[-1]
            value_token = parts[2] if len(parts) >= 4 else None
            if not pieces_text.isdigit():
                continue
            entries.append(
                GroundTruthEntry(
                    filename=filename,
                    group=group,
                    coin_count=int(pieces_text),
                    value_cents=_parse_value_cents(value_token),
                )
            )
        return entries


def _parse_value_cents(raw: str | None) -> int | None:
    if raw is None:
        return None
    cleaned = raw.strip().lower().replace(",", ".")
    if not cleaned or cleaned in {"nan", "na", "n/a", "-"}:
        return None
    try:
        value_eur = float(cleaned)
    except ValueError:
        return None
    if not (value_eur >= 0.0):
        return None
    return int(round(value_eur * 100.0))


RAW_ANNOTATIONS = """
exemple1.png 4 7.25 gp1
10.jpg 9 3.13 gp5
11.jpg 12 6,18 gp5
12.jpg 16 8,83 gp5
13.jpg 19 12,33 gp5
14.jpg 28 15.69 gp5
15.jpg 35 17.32 gp5
16.jpg 48 18.69 gp5
17.jpg 48 18.20 gp5
0.jpeg 2 2.2 gp5
1.jpeg 4 4.22 gp5
2.jpeg 3 3.2 gp5
3.jpeg 4 0.8 gp5
4.jpeg 3 3 gp5
5.jpeg 2 1.20 gp5
6.jpeg 11 10.26 gp5
7.jpeg 3 1.7 gp5
8.jpg 6 NAN gp5
9.jpg 8 3.88 gp5
18.png 7 4.31 gp1
19.png 4 1.60 gp1
20.png 8 4.81 gp1
21.png 6 3.76 gp1
22.png 5 2.25 gp1
23.png 8 4.34 gp1
24.png 3 2.55 gp1
25.png 10 4.40 gp1
26.jpg 8 3.51 gp1
27.jpg 9 0.88 gp1
28.jpg 3 0.21 gp1
29.jpg 5 0.36 gp1
30.jpg 7 3.72 gp1
31.jpg 4 1.7 gp1
3_1.jpg 8 5 grp3
3_2.jpg 16 4.8 grp3
3_3.jpg 8 5 grp3
3_4.jpg 10 04.03 grp3
3_5.jpg 25 12.5 grp3
3_6.jpg 8 16 grp3
3_7.jpg 8 16 grp3
3_8.jpg 50 5 grp3
3_9.jpg 24 24 grp3
3_10.jpg 35 3.5 grp3
18.jpg 8 02.01 grp5
19.jpg 10 3.19 grp5
20.jpg 12 4.17 grp5
21.jpg 8 4.22 grp5
22.jpg 12 6.19 grp5
23.jpg 20 8.88 grp5
24.jpg 26 10.05 grp5
1.jpg 2 1.50 grp4
2.jpg 4 2.27 grp4
3.jpg 5 3.27 grp4
4.jpg 7 1.88 grp4
5.jpg 8 4.38 grp4
6.jpg 7 2.37 grp4
7.jpg 8 3.88 grp4
8.jpg 8 3.88 grp4
9.jpg 4 2.65 grp4
10.jpg 7 5.12 grp4
60.jpg 13 6,33 gp6
61.jpg 11 5,53 gp6
62.jpg 9 6,86 gp6
63.jpg 9 5,34 gp6
64.jpg 12 7,07 gp6
65.jpg 13 2,63 gp6
66.jpg 7 0,77 gp6
67.jpg 10 3,31 gp6
68.jpg 11 5,41 gp6
69.jpg 9 7,4 gp6
gp7_01.webp 7 3,79 gp7
gp7_02.webp 12 1,85 gp7
gp7_03.webp 12 4,6 gp7
gp7_04.webp 13 4,65 gp7
gp7_05.webp 12 4,15 gp7
gp7_06.webp 12 4,74 gp7
gp7_07.webp 11 3,74 gp7
gp7_08.webp 10 4,19 gp7
gp7_09.webp 11 2,55 gp7
gp7_10.webp 9 4,46 gp7
gp7_11.webp 10 4,03 gp7
gp7_12.webp 14 4,95 gp7
IMG_1136.png 5 0,83 gp8
IMG_1137.png 10 2,16 gp8
IMG_1138.png 9 2,17 gp8
IMG_1139.png 4 1,21 gp8
IMG_1140.png 11 2,47 gp8
IMG_1141.png 7 1,36 gp8
IMG_1142.png 4 1,52 gp8
IMG_1143.png 17 1,4 gp8
IMG_1144.png 16 0,43 gp8
IMG_1145.png 5 3,86 gp8
1.jpeg 8 gp2
2.jpeg 2 3 gp2
3.jpeg 3 2,7 gp2
4.jpeg 8 3,86 gp2
5.jpeg 3 0,24 gp2
6.jpeg 9 3,98 gp2
7.jpeg 9 3.98 gp2
8.jpeg 3 3,5 gp2
9.jpeg 6 0,96 gp2
10.jpeg 6 0,96 gp2
11.jpeg 9 3,37 gp2
12.jpeg 2 3 gp2
13.jpeg 9 3,87 gp2
14.jpeg 4 2,45 gp2
15.jpeg 5 3,9 gp2
"""
