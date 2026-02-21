from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


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

