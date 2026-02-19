from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .coin_metadata import (
    COIN_DIAMETER_MM,
    COLOR_BIMETAL_1E,
    COLOR_BIMETAL_2E,
    COLOR_BRONZE,
    COLOR_GOLD,
    COLOR_TO_DENOMS,
    COLOR_UNKNOWN,
)


class CoinColorClassifier:
    """OpenCV-only material grouping with inner-vs-ring evidence."""

    def __init__(
        self,
        *,
        bimetal_min_contrast: float = 0.20,
        unknown_label_threshold: float = 0.16,
    ) -> None:
        self._bimetal_min_contrast = float(bimetal_min_contrast)
        self._unknown_label_threshold = float(unknown_label_threshold)

    def extract_coin_features(self, img_bgr: np.ndarray, circle: np.ndarray) -> Dict[str, float]:
        x, y, r = [int(v) for v in circle]
        h, w = img_bgr.shape[:2]
        x = int(np.clip(x, 0, max(0, w - 1)))
        y = int(np.clip(y, 0, max(0, h - 1)))
        r = int(np.clip(r, 1, max(1, min(h, w) - 1)))

        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - x) ** 2 + (yy - y) ** 2
        mask_coin = dist2 <= (r**2)
        mask_inner = dist2 <= int((0.56 * r) ** 2)
        mask_ring = np.logical_and(dist2 >= int((0.66 * r) ** 2), dist2 <= int((0.95 * r) ** 2))

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        coin = self._region_stats(lab=lab, hsv=hsv, mask=mask_coin)
        inner = self._region_stats(lab=lab, hsv=hsv, mask=mask_inner)
        ring = self._region_stats(lab=lab, hsv=hsv, mask=mask_ring)

        yellow_diff = abs(inner["yellow"] - ring["yellow"])
        sat_diff = abs(inner["sat"] - ring["sat"])
        chroma_diff = abs(inner["chroma"] - ring["chroma"])
        bimetal_contrast = float(np.clip(0.48 * yellow_diff + 0.30 * chroma_diff + 0.22 * sat_diff, 0.0, 1.0))

        return {
            "coin_yellow": coin["yellow"],
            "coin_copper": coin["copper"],
            "coin_silver": coin["silver"],
            "coin_sat": coin["sat"],
            "coin_chroma": coin["chroma"],
            "inner_yellow": inner["yellow"],
            "inner_silver": inner["silver"],
            "ring_yellow": ring["yellow"],
            "ring_silver": ring["silver"],
            "bimetal_contrast": bimetal_contrast,
        }

    def classify_color_groups(self, features: Sequence[Dict[str, float]]) -> Tuple[List[str], List[Dict[str, float]]]:
        labels: List[str] = []
        score_maps: List[Dict[str, float]] = []
        for feat in features:
            scores = self._score_groups(feat)
            score_maps.append(scores)

            best_label, best_score = max(scores.items(), key=lambda kv: kv[1])
            if best_score < self._unknown_label_threshold:
                labels.append(COLOR_UNKNOWN)
            else:
                labels.append(best_label)
        return labels, score_maps

    def candidate_denoms_from_group_scores(self, group_scores: Dict[str, float]) -> List[int]:
        all_denoms = sorted(COIN_DIAMETER_MM.keys())
        if not group_scores:
            return all_denoms

        ordered = sorted(group_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_group, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0

        if best_score < 0.16:
            return all_denoms

        if best_group in (COLOR_BIMETAL_1E, COLOR_BIMETAL_2E) and (best_score - second_score) >= 0.08:
            return list(COLOR_TO_DENOMS[best_group])

        selected = [best_group]
        if len(ordered) > 1:
            selected.append(ordered[1][0])

        denoms: List[int] = []
        for group in selected:
            denoms.extend(COLOR_TO_DENOMS.get(group, ()))
        denoms = sorted(set(denoms))
        return denoms if denoms else all_denoms

    def _score_groups(self, feat: Dict[str, float]) -> Dict[str, float]:
        copper = float(np.clip(0.62 * feat["coin_copper"] + 0.20 * feat["coin_yellow"] + 0.18 * feat["coin_sat"], 0.0, 1.0))
        gold = float(np.clip(0.58 * feat["coin_yellow"] + 0.24 * feat["coin_chroma"] + 0.18 * (1.0 - feat["coin_silver"]), 0.0, 1.0))

        bimetal_contrast = float(feat["bimetal_contrast"])
        if bimetal_contrast < self._bimetal_min_contrast:
            bimetal_conf = 0.0
        else:
            bimetal_conf = float(np.clip((bimetal_contrast - self._bimetal_min_contrast) / (1.0 - self._bimetal_min_contrast), 0.0, 1.0))

        one_euro_dir = min(feat["ring_yellow"], feat["inner_silver"])
        two_euro_dir = min(feat["ring_silver"], feat["inner_yellow"])

        return {
            COLOR_BRONZE: copper,
            COLOR_GOLD: gold,
            COLOR_BIMETAL_1E: float(np.clip(bimetal_conf * one_euro_dir, 0.0, 1.0)),
            COLOR_BIMETAL_2E: float(np.clip(bimetal_conf * two_euro_dir, 0.0, 1.0)),
        }

    def _region_stats(self, *, lab: np.ndarray, hsv: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        if not np.any(mask):
            return {"sat": 0.0, "chroma": 0.0, "yellow": 0.0, "silver": 0.0, "copper": 0.0}

        lab_pix = lab[mask].astype(np.float32)
        hsv_pix = hsv[mask].astype(np.float32)

        l = lab_pix[:, 0]
        a = lab_pix[:, 1] - 128.0
        b = lab_pix[:, 2] - 128.0

        sat_mean = float(np.mean(hsv_pix[:, 1] / 255.0))

        chroma = np.sqrt(a * a + b * b)
        chroma_mean = float(np.mean(chroma))
        chroma_norm = float(np.clip(chroma_mean / 35.0, 0.0, 1.0))

        a_mean = float(np.mean(a))
        b_mean = float(np.mean(b))
        l_mean = float(np.mean(l))

        yellow = float(np.clip((b_mean - 0.35 * abs(a_mean) - 1.5) / 18.0, 0.0, 1.0))
        neutral = float(np.clip(1.0 - np.hypot(a_mean, b_mean) / 18.0, 0.0, 1.0))
        silver = float(np.clip(0.55 * neutral + 0.30 * (1.0 - sat_mean) + 0.15 * (1.0 - chroma_norm), 0.0, 1.0))

        copper_warm = float(np.clip((a_mean + b_mean - 6.0) / 24.0, 0.0, 1.0))
        copper_dark = float(np.clip((170.0 - l_mean) / 120.0, 0.0, 1.0))
        copper = float(np.clip(0.58 * copper_warm + 0.24 * sat_mean + 0.18 * copper_dark, 0.0, 1.0))

        return {
            "sat": sat_mean,
            "chroma": chroma_norm,
            "yellow": yellow,
            "silver": silver,
            "copper": copper,
        }
