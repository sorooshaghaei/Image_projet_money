import os
from typing import Optional


def get_image_path(base_dir: str, filename: str, group: str) -> Optional[str]:
    path_grouped = os.path.join(base_dir, group, filename)
    if os.path.exists(path_grouped):
        return path_grouped

    if group.startswith("grp"):
        alt_group = group.replace("grp", "gp")
        path_alt = os.path.join(base_dir, alt_group, filename)
        if os.path.exists(path_alt):
            return path_alt

    path_flat = os.path.join(base_dir, filename)
    if os.path.exists(path_flat):
        return path_flat

    return None
