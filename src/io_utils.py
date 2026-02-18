from pathlib import Path
from typing import Optional


class ImagePathResolver:
    def __init__(self, base_dir: str):
        self._base_dir = Path(base_dir)

    def resolve(self, filename: str, group: str) -> Optional[str]:
        path_grouped = self._base_dir / group / filename
        if path_grouped.exists():
            return str(path_grouped)

        if group.startswith("grp"):
            alt_group = group.replace("grp", "gp")
            path_alt = self._base_dir / alt_group / filename
            if path_alt.exists():
                return str(path_alt)

        path_flat = self._base_dir / filename
        if path_flat.exists():
            return str(path_flat)

        return None
