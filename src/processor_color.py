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
    """Color feature extraction and color-group-to-denomination priors."""

    def __init__(
        self,
        *,
        bimetal_high_confidence: float = 0.58,
        bimetal_strong_2e_margin: float = 0.10,
        unknown_label_threshold: float = 0.18,
    ) -> None:
        self._bimetal_high_confidence = float(bimetal_high_confidence)
        self._bimetal_strong_2e_margin = float(bimetal_strong_2e_margin)
        self._unknown_label_threshold = float(unknown_label_threshold)

    def extract_coin_features(self, img_bgr: np.ndarray, circle: np.ndarray) -> Dict[str, float]:
        """
        Extract coin-level color descriptors from full area + center/ring subregions.

        Why center/ring split:
        - 1EUR and 2EUR are bimetallic and differ mainly by center-vs-ring material order.
        """
        x, y, r = [int(v) for v in circle]
        h, w = img_bgr.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        r = int(np.clip(r, 1, min(w, h) - 1))

        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - x) ** 2 + (yy - y) ** 2
        mask_coin = dist2 <= (r**2)
        center_r2 = float(0.55 * r) ** 2
        ring_inner_r2 = float(0.65 * r) ** 2
        ring_outer_r2 = float(0.95 * r) ** 2
        mask_center = dist2 <= center_r2
        mask_ring = np.logical_and(dist2 >= ring_inner_r2, dist2 <= ring_outer_r2)

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        coin_lab = lab[mask_coin]
        ring_lab = lab[mask_ring]
        center_lab = lab[mask_center]
        center_stats = self._region_color_stats(hsv=hsv, lab=lab, mask=mask_center)
        ring_stats = self._region_color_stats(hsv=hsv, lab=lab, mask=mask_ring)

        if coin_lab.size == 0:
            return self._empty_features()

        bronze_coin, gold_coin, silver_coin = self._lab_material_scores(coin_lab)
        bronze_ring, gold_ring, silver_ring = self._lab_material_scores(ring_lab)
        bronze_center, gold_center, silver_center = self._lab_material_scores(center_lab)
        center_sat = center_stats["sat_norm"]
        ring_sat = ring_stats["sat_norm"]
        center_chroma = center_stats["chroma_norm"]
        ring_chroma = ring_stats["chroma_norm"]
        center_yellow = center_stats["yellow_score"]
        ring_yellow = ring_stats["yellow_score"]

        sat_diff_rel = abs(center_sat - ring_sat) / max(center_sat + ring_sat, 1e-6)
        chroma_diff_rel = abs(center_chroma - ring_chroma) / max(center_chroma + ring_chroma, 1e-6)
        yellow_diff_rel = abs(center_yellow - ring_yellow) / max(center_yellow + ring_yellow, 1e-6)
        hue_diff = 0.0
        if center_stats["hue_valid"] > 0.5 and ring_stats["hue_valid"] > 0.5:
            hue_diff = self._angular_distance_deg(center_stats["hue_deg"], ring_stats["hue_deg"]) / 180.0

        bimetal_contrast = float(
            np.clip(0.45 * yellow_diff_rel + 0.30 * chroma_diff_rel + 0.20 * sat_diff_rel + 0.05 * hue_diff, 0.0, 1.0)
        )
        split_silver_gold = max(
            center_stats["yellow_score"] * ring_stats["silver_score"],
            ring_stats["yellow_score"] * center_stats["silver_score"],
        )
        bimetal_confidence = float(np.clip(0.65 * bimetal_contrast + 0.35 * split_silver_gold, 0.0, 1.0))

        return {
            "bronze_score": bronze_coin,
            "gold_score": gold_coin,
            "silver_score": silver_coin,
            "ring_bronze_score": bronze_ring,
            "center_bronze_score": bronze_center,
            "ring_gold_score": gold_ring,
            "center_gold_score": gold_center,
            "ring_silver_score": silver_ring,
            "center_silver_score": silver_center,
            "bimetal_confidence": bimetal_confidence,
            "center_yellow_score": center_stats["yellow_score"],
            "ring_yellow_score": ring_stats["yellow_score"],
        }

    def classify_color_groups(self, features: Sequence[Dict[str, float]]) -> Tuple[List[str], List[Dict[str, float]]]:
        """Map raw color features to color-group labels + per-group confidence scores."""
        labels: List[str] = []
        group_scores_per_coin: List[Dict[str, float]] = []
        for feat in features:
            group_scores = self._compute_color_group_scores(feat)
            group_scores_per_coin.append(group_scores)

            best_label, best_score = max(group_scores.items(), key=lambda kv: kv[1])
            if best_score < self._unknown_label_threshold:
                labels.append(COLOR_UNKNOWN)
                continue
            labels.append(best_label)
        return labels, group_scores_per_coin

    def candidate_denoms_from_group_scores(self, group_scores: Dict[str, float]) -> List[int]:
        """
        Build denomination candidate set from color scores.

        Critical guardrail:
        - strong bimetal confidence returns [200] or [100, 200].
        - this blocks false mapping to 1c/2c/5c.
        """
        all_denoms = sorted(COIN_DIAMETER_MM.keys())
        if not group_scores:
            return all_denoms

        bimetal_1e = float(group_scores.get(COLOR_BIMETAL_1E, 0.0))
        bimetal_2e = float(group_scores.get(COLOR_BIMETAL_2E, 0.0))
        bimetal_conf = max(bimetal_1e, bimetal_2e)
        if bimetal_conf >= self._bimetal_high_confidence:
            if (bimetal_2e - bimetal_1e) >= self._bimetal_strong_2e_margin:
                return [200]
            return [100, 200]

        ordered = sorted(group_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_group, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0

        if best_score < 0.20:
            return all_denoms

        selected_groups = [best_group]
        if len(ordered) > 1:
            selected_groups.append(ordered[1][0])
        if len(ordered) > 2 and (best_score < 0.35 or (best_score - second_score) < 0.08):
            selected_groups.append(ordered[2][0])

        # Keep strong bimetal decisions tight to avoid 1EUR/2EUR confusion.
        if (
            best_group in (COLOR_BIMETAL_1E, COLOR_BIMETAL_2E)
            and best_score >= 0.28
            and (best_score - second_score) >= 0.10
        ):
            selected_groups = [best_group]

        denoms: List[int] = []
        for group in selected_groups:
            denoms.extend(COLOR_TO_DENOMS.get(group, ()))
        unique_denoms = sorted(set(denoms))
        return unique_denoms if unique_denoms else all_denoms

    def _compute_color_group_scores(self, feat: Dict[str, float]) -> Dict[str, float]:
        """
        Convert extracted features into confidence scores for each color family.

        Includes direction-aware bimetal scoring:
        - center more yellow -> 2EUR-like
        - ring more yellow   -> 1EUR-like
        """
        score_bronze = max(feat["bronze_score"], min(feat["ring_bronze_score"], feat["center_bronze_score"]))
        score_gold = feat["gold_score"]
        ring_gold = feat["ring_gold_score"]
        center_gold = feat["center_gold_score"]
        ring_silver = feat["ring_silver_score"]
        center_silver = feat["center_silver_score"]
        bimetal_conf = float(np.clip(feat.get("bimetal_confidence", 0.0), 0.0, 1.0))

        # Legacy directional evidence from region-wise material compatibility.
        legacy_1e = min(ring_gold, center_silver)  # 1EUR: gold ring + silver center
        legacy_2e = min(ring_silver, center_gold)  # 2EUR: silver ring + gold center

        # Relative yellow split gives direction robustness under warm/cool global lighting.
        center_yellow = float(np.clip(feat.get("center_yellow_score", center_gold), 0.0, 1.0))
        ring_yellow = float(np.clip(feat.get("ring_yellow_score", ring_gold), 0.0, 1.0))
        direction = float(np.clip(0.5 + 0.5 * (center_yellow - ring_yellow), 0.0, 1.0))

        bimetal_2e = bimetal_conf * float(np.clip(0.55 * legacy_2e + 0.45 * direction, 0.0, 1.0))
        bimetal_1e = bimetal_conf * float(np.clip(0.55 * legacy_1e + 0.45 * (1.0 - direction), 0.0, 1.0))

        return {
            COLOR_BRONZE: float(np.clip(score_bronze, 0.0, 1.0)),
            COLOR_GOLD: float(np.clip(score_gold, 0.0, 1.0)),
            COLOR_BIMETAL_1E: float(np.clip(bimetal_1e, 0.0, 1.0)),
            COLOR_BIMETAL_2E: float(np.clip(bimetal_2e, 0.0, 1.0)),
        }

    def _lab_material_scores(self, lab_pixels: np.ndarray) -> Tuple[float, float, float]:
        """
        Return soft bronze/gold/silver scores from LAB means using broad prototypes.

        Broad sigmas intentionally trade precision for lighting robustness.
        """
        if lab_pixels.size == 0:
            return 0.0, 0.0, 0.0

        l = lab_pixels[:, 0].astype(np.float32)
        a = lab_pixels[:, 1].astype(np.float32) - 128.0
        b = lab_pixels[:, 2].astype(np.float32) - 128.0

        l_mean = float(np.mean(l))
        a_mean = float(np.mean(a))
        b_mean = float(np.mean(b))

        def gaussian_score(value: float, mean: float, sigma: float) -> float:
            sigma = max(1e-6, sigma)
            z = (value - mean) / sigma
            return float(np.exp(-0.5 * z * z))

        # Prototypes are intentionally broad to tolerate phone-camera lighting.
        gold = (
            gaussian_score(a_mean, 0.0, 9.0)
            * gaussian_score(b_mean, 7.0, 9.0)
            * float(np.clip((l_mean - 65.0) / 95.0, 0.15, 1.0))
        )
        bronze = (
            gaussian_score(a_mean, 8.0, 10.0)
            * gaussian_score(b_mean, 10.0, 10.0)
            * float(np.clip((175.0 - l_mean) / 120.0, 0.15, 1.0))
        )
        silver = (
            gaussian_score(a_mean, 0.0, 6.5)
            * gaussian_score(b_mean, 0.0, 6.5)
            * float(np.clip((l_mean - 70.0) / 105.0, 0.15, 1.0))
        )
        return bronze, gold, silver

    def _region_color_stats(self, hsv: np.ndarray, lab: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Compute robust region-level statistics used by bimetal detection.

        Returned values are normalized or bounded so comparisons across images are stable.
        """
        if not np.any(mask):
            return {
                "sat_norm": 0.0,
                "chroma_norm": 0.0,
                "hue_deg": 0.0,
                "hue_valid": 0.0,
                "yellow_score": 0.0,
                "silver_score": 0.0,
            }

        hsv_region = hsv[mask].astype(np.float32)
        lab_region = lab[mask].astype(np.float32)
        sat = hsv_region[:, 1]
        val = hsv_region[:, 2]
        hue_deg = hsv_region[:, 0] * 2.0

        a = lab_region[:, 1] - 128.0
        b = lab_region[:, 2] - 128.0
        chroma = np.sqrt(a * a + b * b)

        sat_norm = float(np.mean(sat) / 255.0)
        chroma_mean = float(np.mean(chroma))
        chroma_norm = float(np.clip(chroma_mean / 36.0, 0.0, 1.0))

        hue_ok = np.logical_and(sat >= 35.0, val >= 35.0)
        hue_valid = float(np.mean(hue_ok)) if hue_ok.size > 0 else 0.0
        mean_hue = 0.0
        hue_gold_match = 0.0
        if np.any(hue_ok):
            hue_subset = hue_deg[hue_ok]
            rad = np.deg2rad(hue_subset)
            mean_hue = float((np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) + 360.0) % 360.0)
            hue_gold_match = float(np.exp(-0.5 * (self._angular_distance_deg(mean_hue, 50.0) / 24.0) ** 2))

        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        lab_yellow = float(np.clip((mean_b - 0.35 * abs(mean_a) - 2.0) / 18.0, 0.0, 1.0))
        yellow_score = float(np.clip(0.50 * lab_yellow + 0.30 * hue_gold_match + 0.20 * sat_norm, 0.0, 1.0))

        neutral_ab = float(np.clip(1.0 - np.hypot(mean_a, mean_b) / 18.0, 0.0, 1.0))
        silver_score = float(np.clip(0.55 * (1.0 - chroma_norm) + 0.30 * (1.0 - sat_norm) + 0.15 * neutral_ab, 0.0, 1.0))

        return {
            "sat_norm": sat_norm,
            "chroma_norm": chroma_norm,
            "hue_deg": mean_hue,
            "hue_valid": hue_valid,
            "yellow_score": yellow_score,
            "silver_score": silver_score,
        }

    def _angular_distance_deg(self, a_deg: float, b_deg: float) -> float:
        """Circular hue distance in degrees (e.g., 5 deg and 355 deg are 10 deg apart)."""
        diff = abs(float(a_deg) - float(b_deg)) % 360.0
        return float(min(diff, 360.0 - diff))

    def _empty_features(self) -> Dict[str, float]:
        return {
            "bronze_score": 0.0,
            "gold_score": 0.0,
            "silver_score": 0.0,
            "ring_bronze_score": 0.0,
            "center_bronze_score": 0.0,
            "ring_gold_score": 0.0,
            "center_gold_score": 0.0,
            "ring_silver_score": 0.0,
            "center_silver_score": 0.0,
            "bimetal_confidence": 0.0,
            "center_yellow_score": 0.0,
            "ring_yellow_score": 0.0,
        }
