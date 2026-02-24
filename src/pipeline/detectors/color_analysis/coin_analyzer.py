"""Coin color/material analysis stage.

This module contains the full color-analysis algorithm used between circle
detection and denomination estimation. The design is intentionally heuristic:
it combines physically meaningful color statistics with structural cues to
classify each coin as:
- one-color (bronze/gold-like),
- bi-metal (1-euro-like / 2-euro-like),
- uncertain.

Pipeline summary
----------------
1. For every detected circle, split pixels into inner and border regions.
2. Compute robust HSV summaries (circular hue average + median S/V).
3. Build color-difference evidence (`sat`, `hue`, `value` terms).
4. Add structural evidence:
   - radial KMeans separation in HSV feature space,
   - strongest radial step-change score,
   - contour roughness score near expected edge radius.
5. Fuse all votes with deterministic threshold rules.
6. Render an RGB debug visualization and emit per-coin stats.
"""

from __future__ import annotations

import cv2
import numpy as np

from .coin_analyzer_config import CoinAnalyzerConfig

# Lab-prototype anchor thresholds (coin-level).
LAB_PROTO_MIN_CHROMA = 5.0
LAB_PROTO_ANCHOR_SCORE_MIN = 0.52
LAB_PROTO_ANCHOR_MARGIN_MIN = 0.14
LAB_PROTO_BRONZE_ANCHOR_H_MAX = 16
LAB_PROTO_BRONZE_ANCHOR_S_MIN = 55
LAB_PROTO_GOLD_ANCHOR_H_MIN = 18
LAB_PROTO_GOLD_ANCHOR_S_MIN = 75
LAB_PROTO_WEIGHT_BASE = 0.2
LAB_PROTO_WEIGHT_CHROMA_SCALE = 0.03

# Lab-prototype decision thresholds.
LAB_PROTO_CONF_MARGIN_SCALE = 2.5
LAB_PROTO_MARGIN_FALLBACK_MAX = 0.06
LAB_PROTO_ONE_PROTO_MAX_DIST = 16.0
LAB_PROTO_NO_ANCHOR_BRONZE_S_MAX = 75
LAB_PROTO_NO_ANCHOR_BRONZE_H_MAX = 15
LAB_PROTO_NO_ANCHOR_BRONZE_CONF = 0.30
LAB_PROTO_GOLD_GUARD_S_MAX = 70
LAB_PROTO_GOLD_GUARD_H_MAX = 14
LAB_PROTO_GOLD_GUARD_CONF_MAX = 0.45
LAB_PROTO_GOLD_GUARD_BRONZE_CONF = 0.35

# Scene-level Lab clustering thresholds (image-level).
SCENE_CLUSTER_MIN_CHROMA = 2.0
SCENE_CLUSTER_CHROMA_NORM_SCALE = 36.0
SCENE_CLUSTER_MIN_COINS = 5
SCENE_CLUSTER_MIN_CLUSTER_SIZE = 2
SCENE_CLUSTER_MIN_CENTER_DIST = 0.16
SCENE_CLUSTER_B_WEIGHT = 0.08
SCENE_CLUSTER_CONF_BIAS = 0.5
SCENE_CLUSTER_CONF_SCALE = 1.4
LAB_SCENE_OVERRIDE_DELTA = 0.12


class CoinAnalyzer:
    """Facade object around the low-level coin color analysis function."""

    def __init__(self, config: CoinAnalyzerConfig | None = None):
        self._cfg = config or CoinAnalyzerConfig()

    def analyze(self, image_bgr: np.ndarray, circles: np.ndarray | None) -> tuple[np.ndarray, list[dict]]:
        """Return `(debug_rgb, per_coin_stats)` for the provided circle set."""
        return draw_and_analyze_circle_inner_border_colors(
            image_bgr,
            circles,
            border_ratio=self._cfg.border_ratio,
            sat_delta_threshold=self._cfg.sat_delta_threshold,
            bimetal_mode=self._cfg.bimetal_mode,
            material_mode=self._cfg.material_mode,
        )

    @staticmethod
    def draw_circles_filled_with_average_hue(
        image_bgr: np.ndarray, circles: np.ndarray | None
    ) -> tuple[np.ndarray, int]:
        """Render each circle with its average hue for fast visual QA.

        This utility is intentionally simple and independent from the final
        classifier. It is useful when tuning color preprocessing.
        """
        output_bgr = image_bgr.copy()
        if circles is None:
            return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), 0

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        circles_int = np.round(circles[0, :]).astype(int)
        h_img, w_img = image_bgr.shape[:2]

        for idx, (x, y, r) in enumerate(circles_int, start=1):
            if r <= 1:
                continue
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)

            pix = hsv[mask > 0]
            if pix.size == 0:
                continue

            h_vals = pix[:, 0].astype(np.float32)
            s_vals = pix[:, 1].astype(np.float32)
            v_vals = pix[:, 2].astype(np.float32)

            theta = h_vals * (2.0 * np.pi / 180.0)
            weights = np.clip(s_vals, 1.0, 255.0)
            mean_sin = float(np.sum(np.sin(theta) * weights) / np.sum(weights))
            mean_cos = float(np.sum(np.cos(theta) * weights) / np.sum(weights))
            hue = np.degrees(np.arctan2(mean_sin, mean_cos))
            if hue < 0.0:
                hue += 360.0

            hue_cv = int(np.clip(round(hue / 2.0), 0, 179))
            sat_cv = int(np.clip(round(np.median(s_vals)), 35, 255))
            val_cv = int(np.clip(round(np.median(v_vals)), 35, 255))
            fill_hsv = np.uint8([[[hue_cv, sat_cv, val_cv]]])
            fill_bgr = cv2.cvtColor(fill_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            cv2.circle(output_bgr, (int(x), int(y)), int(r), tuple(int(c) for c in fill_bgr), -1)

            text = str(idx)
            cv2.putText(
                output_bgr,
                text,
                (int(x) - 8, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                output_bgr,
                text,
                (int(x) - 8, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), int(len(circles_int))


def mean_hsv_from_pixels(hsv_pixels: np.ndarray) -> tuple[int, int, int]:
    """Compute robust HSV center from a pixel cloud.

    Hue is circular, so we use weighted mean on unit circle (weight by
    saturation). Saturation and value are summarized with medians for outlier
    robustness.
    """
    if hsv_pixels is None or len(hsv_pixels) == 0:
        return 0, 0, 0

    h_vals = hsv_pixels[:, 0].astype(np.float32)
    s_vals = hsv_pixels[:, 1].astype(np.float32)
    v_vals = hsv_pixels[:, 2].astype(np.float32)

    theta = h_vals * (2.0 * np.pi / 180.0)
    weights = np.clip(s_vals, 1.0, 255.0)
    mean_sin = float(np.sum(np.sin(theta) * weights) / np.sum(weights))
    mean_cos = float(np.sum(np.cos(theta) * weights) / np.sum(weights))

    hue_deg = np.degrees(np.arctan2(mean_sin, mean_cos))
    if hue_deg < 0.0:
        hue_deg += 360.0

    h_cv = int(np.clip(round(hue_deg / 2.0), 0, 179))
    s_cv = int(np.clip(round(np.median(s_vals)), 0, 255))
    v_cv = int(np.clip(round(np.median(v_vals)), 0, 255))
    return h_cv, s_cv, v_cv


def mean_lab_from_hsv_pixels(hsv_pixels: np.ndarray) -> tuple[float, float, float]:
    """Compute robust Lab center from HSV pixels."""
    if hsv_pixels is None or len(hsv_pixels) == 0:
        return 0.0, 0.0, 0.0

    hsv_u8 = np.asarray(hsv_pixels, dtype=np.uint8)
    hsv_u8 = hsv_u8.reshape(-1, 1, 3)
    bgr = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    l_med = float(np.median(lab[:, 0]))
    a_med = float(np.median(lab[:, 1]))
    b_med = float(np.median(lab[:, 2]))
    return l_med, a_med, b_med


def build_lab_material_prototypes(rows: list[dict]) -> dict[str, object]:
    """Build bronze/gold Lab(ab) prototypes from high-confidence coins in one image."""
    bronze_ab: list[tuple[float, float]] = []
    bronze_w: list[float] = []
    gold_ab: list[tuple[float, float]] = []
    gold_w: list[float] = []

    for row in rows:
        material_hsv = row.get("material_hsv")
        material_lab = row.get("material_lab")
        if not isinstance(material_hsv, (tuple, list)) or len(material_hsv) < 3:
            continue
        if not isinstance(material_lab, (tuple, list)) or len(material_lab) < 3:
            continue

        h = int(material_hsv[0])
        s = int(material_hsv[1])
        v = int(material_hsv[2])
        _, a_lab, b_lab = float(material_lab[0]), float(material_lab[1]), float(material_lab[2])
        chroma = float(np.hypot(a_lab - 128.0, b_lab - 128.0))
        if chroma < LAB_PROTO_MIN_CHROMA:
            continue

        bronze_score = bronze_score_from_inner_hsv(h, s, v)
        gold_score = gold_score_from_inner_hsv(h, s, v)
        margin = abs(bronze_score - gold_score)
        weight = float((LAB_PROTO_WEIGHT_BASE + margin) * (1.0 + LAB_PROTO_WEIGHT_CHROMA_SCALE * chroma))

        if (
            bronze_score >= LAB_PROTO_ANCHOR_SCORE_MIN
            and (bronze_score - gold_score) >= LAB_PROTO_ANCHOR_MARGIN_MIN
            and h <= LAB_PROTO_BRONZE_ANCHOR_H_MAX
            and s >= LAB_PROTO_BRONZE_ANCHOR_S_MIN
        ):
            bronze_ab.append((a_lab, b_lab))
            bronze_w.append(weight)
        elif (
            gold_score >= LAB_PROTO_ANCHOR_SCORE_MIN
            and (gold_score - bronze_score) >= LAB_PROTO_ANCHOR_MARGIN_MIN
            and h >= LAB_PROTO_GOLD_ANCHOR_H_MIN
            and s >= LAB_PROTO_GOLD_ANCHOR_S_MIN
        ):
            gold_ab.append((a_lab, b_lab))
            gold_w.append(weight)

    out: dict[str, object] = {
        "has_bronze": False,
        "has_gold": False,
        "bronze_ab": (0.0, 0.0),
        "gold_ab": (0.0, 0.0),
        "anchor_counts": {"bronze": len(bronze_ab), "gold": len(gold_ab)},
    }

    if len(bronze_ab) > 0:
        arr = np.asarray(bronze_ab, dtype=np.float32)
        ww = np.asarray(bronze_w, dtype=np.float32)
        wsum = float(np.sum(ww))
        if wsum > 1e-6:
            a = float(np.sum(arr[:, 0] * ww) / wsum)
            b = float(np.sum(arr[:, 1] * ww) / wsum)
            out["has_bronze"] = True
            out["bronze_ab"] = (a, b)

    if len(gold_ab) > 0:
        arr = np.asarray(gold_ab, dtype=np.float32)
        ww = np.asarray(gold_w, dtype=np.float32)
        wsum = float(np.sum(ww))
        if wsum > 1e-6:
            a = float(np.sum(arr[:, 0] * ww) / wsum)
            b = float(np.sum(arr[:, 1] * ww) / wsum)
            out["has_gold"] = True
            out["gold_ab"] = (a, b)

    return out


def infer_material_label_from_lab_prototypes(
    row: dict,
    prototypes: dict[str, object],
    fallback_label: str,
) -> tuple[str, float]:
    """Infer bronze/gold from Lab(ab) distance to image-level prototypes."""
    material_lab = row.get("material_lab")
    material_hsv = row.get("material_hsv")
    if not isinstance(material_lab, (tuple, list)) or len(material_lab) < 3:
        return fallback_label, 0.0
    if not isinstance(material_hsv, (tuple, list)) or len(material_hsv) < 3:
        return fallback_label, 0.0

    a = float(material_lab[1])
    b = float(material_lab[2])
    h = int(material_hsv[0])
    s = int(material_hsv[1])
    has_bronze = bool(prototypes.get("has_bronze", False))
    has_gold = bool(prototypes.get("has_gold", False))

    if not has_bronze and not has_gold:
        if s < LAB_PROTO_NO_ANCHOR_BRONZE_S_MAX and h <= LAB_PROTO_NO_ANCHOR_BRONZE_H_MAX:
            return "bronze", LAB_PROTO_NO_ANCHOR_BRONZE_CONF
        return fallback_label, 0.0

    if has_bronze:
        pb = prototypes.get("bronze_ab", (0.0, 0.0))
        db = float(np.hypot(a - float(pb[0]), b - float(pb[1])))
    else:
        db = 1e9
    if has_gold:
        pg = prototypes.get("gold_ab", (0.0, 0.0))
        dg = float(np.hypot(a - float(pg[0]), b - float(pg[1])))
    else:
        dg = 1e9

    if has_bronze and has_gold:
        label = "bronze" if db < dg else "gold"
        margin = abs(db - dg) / max(db + dg, 1e-6)
        confidence = float(np.clip(margin * LAB_PROTO_CONF_MARGIN_SCALE, 0.0, 1.0))
        if (
            label == "gold"
            and s < LAB_PROTO_GOLD_GUARD_S_MAX
            and h <= LAB_PROTO_GOLD_GUARD_H_MAX
            and confidence < LAB_PROTO_GOLD_GUARD_CONF_MAX
        ):
            return "bronze", confidence
        if margin < LAB_PROTO_MARGIN_FALLBACK_MAX and fallback_label in {"bronze", "gold"}:
            return fallback_label, confidence
        return label, confidence

    if has_bronze and db <= LAB_PROTO_ONE_PROTO_MAX_DIST:
        return "bronze", float(np.clip(1.0 - (db / LAB_PROTO_ONE_PROTO_MAX_DIST), 0.0, 1.0))
    if has_gold and dg <= LAB_PROTO_ONE_PROTO_MAX_DIST:
        if s < LAB_PROTO_GOLD_GUARD_S_MAX and h <= LAB_PROTO_GOLD_GUARD_H_MAX:
            return "bronze", LAB_PROTO_GOLD_GUARD_BRONZE_CONF
        return "gold", float(np.clip(1.0 - (dg / LAB_PROTO_ONE_PROTO_MAX_DIST), 0.0, 1.0))
    if s < LAB_PROTO_NO_ANCHOR_BRONZE_S_MAX and h <= LAB_PROTO_NO_ANCHOR_BRONZE_H_MAX:
        return "bronze", LAB_PROTO_NO_ANCHOR_BRONZE_CONF
    return fallback_label, 0.0


def build_scene_lab_cluster_labels(rows: list[dict]) -> dict[int, tuple[str, float]]:
    """Infer bronze/gold from scene-level clustering of coin Lab directions."""
    feats: list[list[float]] = []
    row_ids: list[int] = []
    hue_vals: list[int] = []
    b_vals: list[float] = []

    for row in rows:
        material_lab = row.get("material_lab")
        material_hsv = row.get("material_hsv")
        if not isinstance(material_lab, (tuple, list)) or len(material_lab) < 3:
            continue
        if not isinstance(material_hsv, (tuple, list)) or len(material_hsv) < 3:
            continue

        a = float(material_lab[1])
        b = float(material_lab[2])
        da = a - 128.0
        db = b - 128.0
        chroma = float(np.hypot(da, db))
        if chroma < SCENE_CLUSTER_MIN_CHROMA:
            continue

        dir_a = da / (chroma + 1e-6)
        dir_b = db / (chroma + 1e-6)
        chroma_norm = float(np.clip(chroma / SCENE_CLUSTER_CHROMA_NORM_SCALE, 0.0, 1.0))
        feats.append([dir_a, dir_b, chroma_norm])
        row_ids.append(int(row.get("id", -1)))
        hue_vals.append(int(material_hsv[0]))
        b_vals.append(float(b))

    if len(feats) < SCENE_CLUSTER_MIN_COINS:
        return {}

    data = np.asarray(feats, dtype=np.float32)
    attempts = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    compactness, labels, centers = cv2.kmeans(
        data,
        2,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )
    _ = compactness
    labels = labels.reshape(-1)
    centers = centers.reshape(2, -1)

    counts = [int(np.sum(labels == 0)), int(np.sum(labels == 1))]
    if min(counts) < SCENE_CLUSTER_MIN_CLUSTER_SIZE:
        return {}

    center_dist = float(np.linalg.norm(centers[0] - centers[1]))
    if center_dist < SCENE_CLUSTER_MIN_CENTER_DIST:
        return {}

    cluster_scores: dict[int, float] = {}
    for c in (0, 1):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            cluster_scores[c] = -1e9
            continue
        mean_h = float(np.mean([hue_vals[int(i)] for i in idx]))
        mean_b = float(np.mean([b_vals[int(i)] for i in idx]))
        cluster_scores[c] = mean_h + SCENE_CLUSTER_B_WEIGHT * (mean_b - 128.0)

    gold_cluster = 0 if cluster_scores[0] >= cluster_scores[1] else 1
    bronze_cluster = 1 - gold_cluster

    out: dict[int, tuple[str, float]] = {}
    for i, rid in enumerate(row_ids):
        if rid < 0:
            continue
        c = int(labels[i])
        own = centers[c]
        other = centers[1 - c]
        d_own = float(np.linalg.norm(data[i] - own))
        d_other = float(np.linalg.norm(data[i] - other))
        margin = (d_other - d_own) / max(d_other + d_own, 1e-6)
        conf = float(np.clip(SCENE_CLUSTER_CONF_BIAS + SCENE_CLUSTER_CONF_SCALE * margin, 0.0, 1.0))
        if c == gold_cluster:
            out[rid] = ("gold", conf)
        elif c == bronze_cluster:
            out[rid] = ("bronze", conf)
    return out


def fuse_lab_and_scene_material_labels(
    fallback_label: str,
    lab_label_raw: str,
    lab_conf_raw: float,
    scene_label: str,
    scene_conf: float,
) -> tuple[str, float]:
    """Combine per-coin Lab prototype decision with scene-level cluster label."""
    final_label = lab_label_raw
    final_conf = float(lab_conf_raw)

    if final_label not in {"bronze", "gold"} and scene_label in {"bronze", "gold"}:
        final_label = scene_label
        final_conf = float(scene_conf)
    elif scene_label in {"bronze", "gold"}:
        if final_label == scene_label:
            final_conf = max(final_conf, float(scene_conf))
        elif float(scene_conf) > (final_conf + LAB_SCENE_OVERRIDE_DELTA):
            final_label = scene_label
            final_conf = float(scene_conf)

    if final_label not in {"bronze", "gold"}:
        final_label = fallback_label
    return final_label, final_conf


def enrich_rows_with_lab_material_labels(rows: list[dict]) -> None:
    """Populate Lab-based material labels and confidences for every coin row."""
    lab_prototypes = build_lab_material_prototypes(rows)
    scene_lab_labels = build_scene_lab_cluster_labels(rows)

    for row in rows:
        fallback_for_lab = str(row.get("material_label_hsv_kmeans", "borderline"))
        lab_label_raw, lab_conf_raw = infer_material_label_from_lab_prototypes(
            row=row,
            prototypes=lab_prototypes,
            fallback_label=fallback_for_lab,
        )
        scene_label, scene_conf = scene_lab_labels.get(int(row.get("id", -1)), ("n/a", 0.0))
        final_label, final_conf = fuse_lab_and_scene_material_labels(
            fallback_label=fallback_for_lab,
            lab_label_raw=lab_label_raw,
            lab_conf_raw=float(lab_conf_raw),
            scene_label=scene_label,
            scene_conf=float(scene_conf),
        )

        row["material_label_lab_proto_raw"] = lab_label_raw
        row["material_lab_proto_raw_conf"] = float(lab_conf_raw)
        row["material_label_lab_scene"] = scene_label
        row["material_lab_scene_conf"] = float(scene_conf)
        row["material_label_lab_proto"] = final_label
        row["material_lab_proto_conf"] = float(final_conf)


def pick_material_label_by_mode(
    material_mode: str,
    material_label_hybrid: str,
    material_label_hsv_kmeans: str,
    material_label_lab_proto: str,
    bronze_score_hsv: float,
    bronze_score_hsv_kmeans: float,
) -> tuple[str, float]:
    """Return `(material_label, bronze_score)` for the configured material mode."""
    if material_mode == "hsv_kmeans":
        return material_label_hsv_kmeans, float(bronze_score_hsv_kmeans)
    if material_mode == "lab_proto":
        return material_label_lab_proto, float(bronze_score_hsv_kmeans)
    return material_label_hybrid, float(bronze_score_hsv)


def hsv_similarity_score(hsv_a: tuple[int, int, int], hsv_b: tuple[int, int, int]) -> float:
    """Simple similarity score in `[0, 1]` based on saturation distance."""
    _, sa, _ = hsv_a
    _, sb, _ = hsv_b

    ds = abs(float(sa) - float(sb)) / 255.0
    sim = 1.0 - ds
    return float(np.clip(sim, 0.0, 1.0))


def hue_circular_delta_cv(h1: int, h2: int) -> float:
    """Minimum circular hue difference in OpenCV hue space (`0..179`)."""
    d = abs(float(h1) - float(h2))
    return float(min(d, 180.0 - d))


def choose_dynamic_sat_delta_threshold(sat_deltas: list[float], default: float = 18.0) -> float:
    """Estimate an outlier threshold from sorted color deltas.

    The function searches for the largest stable gap in ordered values, with
    guards against weak or unbalanced splits. If no reliable split is found,
    it returns a value above max(delta), effectively disabling outlier votes.
    """
    if sat_deltas is None:
        return float(default)

    vals = np.asarray(sat_deltas, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(default)

    vals = np.sort(vals)
    n = int(vals.size)
    if n < 4:
        return float(np.max(vals) + 1.0)

    gaps = vals[1:] - vals[:-1]
    if gaps.size == 0:
        return float(np.max(vals) + 1.0)

    median_gap = float(np.median(gaps))
    min_low = max(3, int(np.ceil(0.55 * n)))

    valid_idx = [i for i in range(len(gaps)) if (i + 1) >= min_low and (n - i - 1) >= 1]
    if len(valid_idx) == 0:
        return float(np.max(vals) + 1.0)

    best_i = max(valid_idx, key=lambda i: float(gaps[i]))
    best_gap = float(gaps[best_i])

    if best_gap < max(6.0, 2.2 * max(median_gap, 1.0)):
        return float(np.max(vals) + 1.0)

    low = vals[: best_i + 1]
    high = vals[best_i + 1 :]

    n_high = int(high.size)
    high_frac = float(n_high / n)
    if high_frac > 0.45:
        return float(np.max(vals) + 1.0)

    sep = float(np.mean(high) - np.mean(low))
    if sep < max(8.0, 1.3 * float(np.std(low) + 1.0)):
        return float(np.max(vals) + 1.0)

    thr = float(0.5 * (vals[best_i] + vals[best_i + 1]))
    p60 = float(np.percentile(vals, 60))
    p99 = float(np.percentile(vals, 99))
    # Clamp to robust percentile range to avoid pathological thresholds.
    return float(np.clip(thr, p60, p99))

def clamp_value(x: float, lo: float, hi: float) -> float:
    """Clamp helper used in handcrafted color-confidence curves."""
    return lo if x < lo else hi if x > hi else x


def bronze_score_from_inner_hsv(h: int, s: int, v: int) -> float:
    """Estimate bronze-likeness from hue/chroma (minimal value influence)."""
    _ = v
    h_term = 1.0 - clamp_value(abs(float(h) - 9.5) / 6.5, 0.0, 1.0)
    sat_conf = clamp_value((float(s) - 35.0) / 95.0, 0.0, 1.0)
    conf = 0.45 + 0.55 * sat_conf
    return float(np.clip(h_term * conf, 0.0, 1.0))


def gold_score_from_inner_hsv(h: int, s: int, v: int) -> float:
    """Estimate gold-likeness from hue/chroma (minimal value influence)."""
    _ = v
    h_term = 1.0 - clamp_value(abs(float(h) - 20.5) / 8.5, 0.0, 1.0)
    sat_conf = clamp_value((float(s) - 35.0) / 95.0, 0.0, 1.0)
    conf = 0.50 + 0.50 * sat_conf
    return float(np.clip(h_term * conf, 0.0, 1.0))


def label_material_from_inner_hsv(h: int, s: int, v: int) -> str:
    """Primary material label from coarse hue/chroma rules."""
    _ = v
    if s < 40:
        return "borderline"
    if h <= 13:
        return "bronze"
    if h >= 16:
        return "gold"
    return "borderline"


def label_bimetal_euro_from_saturation(inner_s: int, border_s: int) -> str:
    """Distinguish 1e-like vs 2e-like using inner-vs-border saturation."""
    if inner_s < border_s:
        return "1-euro-like"
    if inner_s > border_s:
        return "2-euro-like"
    return "bi-metal-euro-uncertain"


def fallback_material_from_inner_hsv(h: int, s: int, v: int) -> str:
    """Resolve borderline material cases with soft scores + deterministic tie-break."""
    material = label_material_from_inner_hsv(h, s, v)
    if material != "borderline":
        return material

    bronze_score = bronze_score_from_inner_hsv(h, s, v)
    gold_score = gold_score_from_inner_hsv(h, s, v)
    if max(bronze_score, gold_score) >= 0.30 and abs(bronze_score - gold_score) >= 0.02:
        return "bronze" if bronze_score > gold_score else "gold"

    # Keep a deterministic fallback for very low-confidence borderline colors.
    return "bronze" if h <= 17 else "gold"


def _filter_material_pixels_from_v_band(hsv_pixels: np.ndarray) -> np.ndarray:
    """Remove shadow/highlight extremes and keep stable chromatic pixels."""
    if hsv_pixels is None or int(len(hsv_pixels)) == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    hsv_u8 = np.asarray(hsv_pixels, dtype=np.uint8)
    if hsv_u8.ndim != 2 or hsv_u8.shape[1] < 3:
        return hsv_u8

    work = hsv_u8.copy()
    n = int(work.shape[0])
    min_count = max(30, int(0.18 * n))

    sat = work[:, 1].astype(np.float32)
    sat_mask = sat >= 30.0
    if int(np.sum(sat_mask)) >= min_count:
        work = work[sat_mask]

    v = work[:, 2].astype(np.float32)
    if int(v.size) == 0:
        return hsv_u8

    lo = float(np.percentile(v, 20.0))
    hi = float(np.percentile(v, 85.0))
    stable = work[(v >= lo) & (v <= hi)]
    if int(stable.shape[0]) < min_count:
        lo = float(np.percentile(v, 10.0))
        hi = float(np.percentile(v, 90.0))
        stable = work[(v >= lo) & (v <= hi)]

    return stable if int(stable.shape[0]) > 0 else work


def material_pixels_from_stable_ring(
    hsv_image: np.ndarray,
    x: int,
    y: int,
    r_draw: int,
    ring_inner_ratio: float = 0.45,
    ring_outer_ratio: float = 0.80,
) -> np.ndarray:
    """Sample material colors from stable ring area (exclude center/highlights)."""
    h_img, w_img = hsv_image.shape[:2]
    r_outer = int(max(1, round(float(r_draw) * float(ring_outer_ratio))))
    r_inner = int(max(1, round(float(r_draw) * float(ring_inner_ratio))))
    r_inner = min(r_inner, max(1, r_outer - 1))

    outer_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    inner_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.circle(outer_mask, (x, y), r_outer, 255, -1)
    cv2.circle(inner_mask, (x, y), r_inner, 255, -1)
    ring_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))

    pix_ring = hsv_image[ring_mask > 0]
    if int(pix_ring.shape[0]) < 40:
        fallback_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.circle(fallback_mask, (x, y), max(1, int(r_draw)), 255, -1)
        pix_ring = hsv_image[fallback_mask > 0]

    return _filter_material_pixels_from_v_band(pix_ring)


def material_from_hsv_kmeans_auto(
    hsv_pixels: np.ndarray,
    fallback_hsv: tuple[int, int, int],
) -> dict[str, float | int | str | bool]:
    """Infer bronze/gold via HSV KMeans with automatic k in {1, 2}."""
    h_fb = int(fallback_hsv[0])
    s_fb = int(fallback_hsv[1])
    v_fb = int(fallback_hsv[2])
    fallback_label = fallback_material_from_inner_hsv(h_fb, s_fb, v_fb)
    fallback_bronze = bronze_score_from_inner_hsv(h_fb, s_fb, v_fb)
    fallback_gold = gold_score_from_inner_hsv(h_fb, s_fb, v_fb)

    if hsv_pixels is None or int(len(hsv_pixels)) < 40:
        return {
            "material_label": fallback_label,
            "bronze_score": float(fallback_bronze),
            "gold_score": float(fallback_gold),
            "chosen_k": 1,
            "k2_gain": 0.0,
            "bronze_share": 1.0 if fallback_label == "bronze" else 0.0,
            "gold_share": 1.0 if fallback_label == "gold" else 0.0,
            "used_fallback": True,
        }

    hsv = np.asarray(hsv_pixels, dtype=np.float32)
    n = int(hsv.shape[0])

    h = hsv[:, 0] * (2.0 * np.pi / 180.0)
    s = hsv[:, 1] / 255.0
    v = hsv[:, 2] / 255.0
    hue_weight = np.clip(s, 0.2, 1.0)
    feats = np.stack([np.cos(h) * hue_weight, np.sin(h) * hue_weight, 0.90 * s, 0.25 * v], axis=1).astype(np.float32)

    if n > 6000:
        rng = np.random.default_rng(2026)
        idx = rng.choice(n, size=6000, replace=False)
        feats_fit = feats[idx]
    else:
        feats_fit = feats

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.02)
    cv2.setRNGSeed(4242)
    _, _, centers_1 = cv2.kmeans(feats_fit, 1, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    cv2.setRNGSeed(4242)
    _, _, centers_2 = cv2.kmeans(feats_fit, 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

    dist_k1 = np.sum((feats - centers_1[0]) ** 2, axis=1)
    compact_k1 = float(np.sum(dist_k1))

    d20 = np.sum((feats - centers_2[0]) ** 2, axis=1)
    d21 = np.sum((feats - centers_2[1]) ** 2, axis=1)
    labels_k2 = (d21 < d20).astype(np.int32)
    compact_k2 = float(np.sum(np.minimum(d20, d21)))
    k2_gain = float((compact_k1 - compact_k2) / max(compact_k1, 1e-9))

    counts_k2 = np.bincount(labels_k2, minlength=2).astype(np.int32)
    min_count_k2 = int(np.min(counts_k2))
    min_share_k2 = float(min_count_k2 / max(n, 1))
    min_count_needed = max(30, int(0.08 * n))
    use_k2 = bool((k2_gain >= 0.12) and (min_count_k2 >= min_count_needed) and (min_share_k2 >= 0.08))

    if use_k2:
        chosen_k = 2
        labels = labels_k2
    else:
        chosen_k = 1
        labels = np.zeros((n,), dtype=np.int32)

    bronze_mass = 0.0
    gold_mass = 0.0
    bronze_share = 0.0
    gold_share = 0.0
    for cluster_id in range(chosen_k):
        mask = labels == int(cluster_id)
        count = int(np.sum(mask))
        if count <= 0:
            continue
        weight = float(count / max(n, 1))
        center_hsv = mean_hsv_from_pixels(hsv_pixels[mask])
        h_c, s_c, v_c = int(center_hsv[0]), int(center_hsv[1]), int(center_hsv[2])
        bronze_c = bronze_score_from_inner_hsv(h_c, s_c, v_c)
        gold_c = gold_score_from_inner_hsv(h_c, s_c, v_c)
        bronze_mass += weight * float(bronze_c)
        gold_mass += weight * float(gold_c)

        cluster_material = fallback_material_from_inner_hsv(h_c, s_c, v_c)
        if cluster_material == "bronze":
            bronze_share += weight
        elif cluster_material == "gold":
            gold_share += weight

    used_fallback = False
    if max(bronze_mass, gold_mass) < 0.20:
        material_label = fallback_label
        bronze_mass = float(fallback_bronze)
        gold_mass = float(fallback_gold)
        used_fallback = True
    elif abs(bronze_mass - gold_mass) < 0.035:
        material_label = "bronze" if bronze_share >= gold_share else "gold"
    else:
        material_label = "bronze" if bronze_mass > gold_mass else "gold"

    return {
        "material_label": material_label,
        "bronze_score": float(np.clip(bronze_mass, 0.0, 1.0)),
        "gold_score": float(np.clip(gold_mass, 0.0, 1.0)),
        "chosen_k": int(chosen_k),
        "k2_gain": float(k2_gain),
        "bronze_share": float(np.clip(bronze_share, 0.0, 1.0)),
        "gold_share": float(np.clip(gold_share, 0.0, 1.0)),
        "used_fallback": bool(used_fallback),
    }


def material_family_from_inner_hsv(inner_hsv: tuple[int, int, int] | list[int]) -> str:
    """Map inner HSV to a denomination family hint (`bronze`/`gold`/`unknown`)."""
    if not isinstance(inner_hsv, (tuple, list)) or len(inner_hsv) < 3:
        return "unknown"
    h = int(inner_hsv[0])
    s = int(inner_hsv[1])
    v = int(inner_hsv[2])

    bronze_score = bronze_score_from_inner_hsv(h, s, v)
    gold_score = gold_score_from_inner_hsv(h, s, v)
    if max(bronze_score, gold_score) >= 0.30 and abs(bronze_score - gold_score) >= 0.02:
        return "bronze" if bronze_score > gold_score else "gold"

    if s >= 45:
        if h <= 15:
            return "bronze"
        if h >= 16:
            return "gold"
    return "unknown"


def coin_kmeans_radial_structure(hsv_pixels: np.ndarray, radial_norm: np.ndarray) -> dict[str, float | bool]:
    """Check whether coin colors form two radially separated clusters.

    Features combine hue direction (`cos/sin`), saturation and value. We cluster
    into two groups and then evaluate if those groups align with inner/outer
    radial layout.
    """
    n = int(hsv_pixels.shape[0])
    if n < 120:
        return {"ok": False, "score": 0.0, "radial_sep": 0.0, "agreement": 0.0, "balance": 0.0}

    h = hsv_pixels[:, 0].astype(np.float32) * (2.0 * np.pi / 180.0)
    s = hsv_pixels[:, 1].astype(np.float32) / 255.0
    v = hsv_pixels[:, 2].astype(np.float32) / 255.0

    w = np.clip(s, 0.2, 1.0)
    feats = np.stack([np.cos(h) * w, np.sin(h) * w, s, 0.35 * v], axis=1).astype(np.float32)

    if n > 4000:
        rng = np.random.default_rng(123)
        idx = rng.choice(n, size=4000, replace=False)
        feats_fit = feats[idx]
    else:
        feats_fit = feats

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.02)
    cv2.setRNGSeed(1234)
    _, _, centers = cv2.kmeans(feats_fit, 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

    d0 = np.sum((feats - centers[0]) ** 2, axis=1)
    d1 = np.sum((feats - centers[1]) ** 2, axis=1)
    labels = (d1 < d0).astype(np.int32)

    cnt0 = int(np.sum(labels == 0))
    cnt1 = int(np.sum(labels == 1))
    balance = float(min(cnt0, cnt1) / max(n, 1))
    if balance < 0.08:
        return {
            "ok": False,
            "score": float(np.clip(balance / 0.08, 0.0, 1.0)),
            "radial_sep": 0.0,
            "agreement": 0.0,
            "balance": balance,
        }

    r0 = float(np.mean(radial_norm[labels == 0]))
    r1 = float(np.mean(radial_norm[labels == 1]))
    radial_sep = abs(r0 - r1)

    outer_id = 0 if r0 > r1 else 1
    inner_mean = min(r0, r1)
    outer_mean = max(r0, r1)
    radial_cut = 0.5 * (inner_mean + outer_mean)

    pred_outer = radial_norm >= radial_cut
    lbl_outer = labels == outer_id
    agreement = float(np.mean(pred_outer == lbl_outer))

    score = (
        0.45 * np.clip(radial_sep / 0.22, 0.0, 1.0)
        + 0.35 * np.clip((agreement - 0.5) / 0.4, 0.0, 1.0)
        + 0.20 * np.clip(balance / 0.22, 0.0, 1.0)
    )
    ok = (radial_sep >= 0.13) and (agreement >= 0.66) and (balance >= 0.10)
    return {
        "ok": bool(ok),
        "score": float(np.clip(score, 0.0, 1.0)),
        "radial_sep": float(radial_sep),
        "agreement": float(agreement),
        "balance": float(balance),
    }


def coin_radial_step_score(
    hsv_pixels: np.ndarray,
    radial_norm: np.ndarray,
    split_radius: float,
    bins: int = 8,
) -> dict[str, float | bool]:
    """Measure strongest radial color step and its alignment to split radius.

    The coin is split into radial bins, each summarized by mean HSV-like
    features. We score the largest consecutive-bin jump and reward proximity
    to expected inner/border transition radius.
    """
    n = int(hsv_pixels.shape[0])
    if n < 120:
        return {"ok": False, "score": 0.0, "max_step": 0.0, "step_radius": float(split_radius)}

    h = hsv_pixels[:, 0].astype(np.float32) * (2.0 * np.pi / 180.0)
    s = hsv_pixels[:, 1].astype(np.float32) / 255.0
    v = hsv_pixels[:, 2].astype(np.float32) / 255.0
    feats = np.stack([np.cos(h) * s, np.sin(h) * s, s, 0.20 * v], axis=1).astype(np.float32)

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_vecs: list[np.ndarray | None] = []
    boundary_r: list[float] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (radial_norm >= lo) & (radial_norm <= hi)
        else:
            mask = (radial_norm >= lo) & (radial_norm < hi)
        if int(np.sum(mask)) < max(20, int(0.03 * n)):
            bin_vecs.append(None)
        else:
            bin_vecs.append(np.mean(feats[mask], axis=0))
        if i < bins - 1:
            boundary_r.append(float(edges[i + 1]))

    steps: list[float] = []
    for i in range(bins - 1):
        a = bin_vecs[i]
        b = bin_vecs[i + 1]
        if (a is None) or (b is None):
            steps.append(0.0)
        else:
            steps.append(float(np.linalg.norm(a - b)))

    if len(steps) == 0:
        return {"ok": False, "score": 0.0, "max_step": 0.0, "step_radius": float(split_radius)}

    idx = int(np.argmax(steps))
    max_step = float(steps[idx])
    step_radius = float(boundary_r[idx])
    dist_to_split = abs(step_radius - float(split_radius))

    strength = np.clip(max_step / 0.17, 0.0, 1.0)
    location = np.clip(1.0 - (dist_to_split / 0.28), 0.0, 1.0)
    score = float(0.65 * strength + 0.35 * location)
    ok = (max_step >= 0.10) and (dist_to_split <= 0.26)
    return {
        "ok": bool(ok),
        "score": float(np.clip(score, 0.0, 1.0)),
        "max_step": float(max_step),
        "step_radius": float(step_radius),
    }


def coin_edge_roughness_score(
    gray_u8: np.ndarray,
    x: int,
    y: int,
    r: int,
    n_bins: int = 72,
) -> dict[str, float]:
    """Estimate edge roughness around detected circumference.

    This is a shape prior: we collect edge pixels in an annulus around the
    detected radius, estimate per-angle radius medians, then compute IQR/median
    as normalized roughness.
    """
    if r < 8:
        return {"roughness": 0.0, "coverage": 0.0, "score": 0.0}

    h, w = gray_u8.shape[:2]
    x0 = max(0, int(x - 1.3 * r))
    x1 = min(w, int(x + 1.3 * r) + 1)
    y0 = max(0, int(y - 1.3 * r))
    y1 = min(h, int(y + 1.3 * r) + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return {"roughness": 0.0, "coverage": 0.0, "score": 0.0}

    roi = gray_u8[y0:y1, x0:x1]
    roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
    edges = cv2.Canny(roi_blur, 60, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) < 20:
        return {"roughness": 0.0, "coverage": 0.0, "score": 0.0}

    xs = xs.astype(np.float32) + float(x0)
    ys = ys.astype(np.float32) + float(y0)
    dx = xs - float(x)
    dy = ys - float(y)
    dist = np.sqrt(dx * dx + dy * dy)

    mask = (dist >= 0.62 * float(r)) & (dist <= 1.20 * float(r))
    if int(np.sum(mask)) < 18:
        return {"roughness": 0.0, "coverage": 0.0, "score": 0.0}

    dist = dist[mask]
    ang = np.arctan2(dy[mask], dx[mask])
    ang[ang < 0.0] += 2.0 * np.pi

    bins = np.full((n_bins,), np.nan, dtype=np.float32)
    idx = np.floor((ang / (2.0 * np.pi)) * n_bins).astype(np.int32)
    idx = np.clip(idx, 0, n_bins - 1)
    for b in range(n_bins):
        vals = dist[idx == b]
        if vals.size > 0:
            bins[b] = float(np.median(vals))

    valid = np.isfinite(bins)
    coverage = float(np.mean(valid))
    if int(np.sum(valid)) < max(14, int(0.30 * n_bins)):
        return {"roughness": 0.0, "coverage": coverage, "score": 0.0}

    vals = bins[valid].astype(np.float32)
    med = float(np.median(vals))
    q1, q3 = np.percentile(vals, [25, 75])
    roughness = float((q3 - q1) / (med + 1e-6))

    score = np.clip((roughness - 0.040) / 0.032, 0.0, 1.0)
    score *= np.clip((coverage - 0.35) / 0.45, 0.0, 1.0)
    return {
        "roughness": float(roughness),
        "coverage": float(coverage),
        "score": float(np.clip(score, 0.0, 1.0)),
    }


def radius_spread_metric(circles_int: np.ndarray | None) -> float:
    if circles_int is None or len(circles_int) == 0:
        return 1e9
    r = circles_int[:, 2].astype(np.float32)
    med = float(np.median(r))
    if med <= 1e-6:
        return 1e9
    q1, q3 = np.percentile(r, [25, 75])
    iqr = float(q3 - q1)
    return iqr / med


def draw_and_analyze_circle_inner_border_colors(
    image_bgr: np.ndarray,
    circles: np.ndarray | None,
    border_ratio: float = 0.24,
    sat_delta_threshold: float | None = None,
    bimetal_mode: str = "hybrid",
    material_mode: str = "lab_proto",
) -> tuple[np.ndarray, list[dict]]:
    """Core decision algorithm for per-coin material and type labeling.

    Returns
    -------
    tuple[np.ndarray, list[dict]]
        - RGB debug image with rendered labels.
        - List of rich per-coin diagnostic dictionaries.

    Notes
    -----
    The algorithm is intentionally two-pass:
    - pass 1 computes raw features/scores for each coin,
    - pass 2 applies global + local thresholds to classify each coin.
    """
    bimetal_mode = str(bimetal_mode).strip().lower()
    if bimetal_mode != "hybrid":
        raise ValueError("bimetal_mode must be: 'hybrid'")
    material_mode = str(material_mode).strip().lower()
    valid_material_modes = {"hsv", "hsv_kmeans", "lab_proto"}
    if material_mode not in valid_material_modes:
        raise ValueError("material_mode must be one of: 'hsv', 'hsv_kmeans', 'lab_proto'")

    output_bgr = image_bgr.copy()
    stats: list[dict] = []
    if circles is None:
        return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), stats

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray_shape = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    circles_float = np.asarray(circles[0, :], dtype=np.float32)
    h_img, w_img = image_bgr.shape[:2]

    rows: list[dict] = []
    color_deltas_hybrid: list[float] = []

    # Pass 1: compute raw per-coin features, independent from global threshold.
    for idx, (x_raw, y_raw, r_raw) in enumerate(circles_float):
        x = int(round(float(x_raw)))
        y = int(round(float(y_raw)))
        r = float(max(1.0, float(r_raw)))
        r_draw = int(max(1, round(r)))
        inner_r = int(max(1, round(r * (1.0 - border_ratio))))
        split_radius = float(inner_r) / max(float(r_draw), 1.0)

        outer_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        inner_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.circle(outer_mask, (x, y), r_draw, 255, -1)
        cv2.circle(inner_mask, (x, y), inner_r, 255, -1)
        border_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))

        pix_full_hsv = hsv[outer_mask > 0]
        pix_inner_hsv = hsv[inner_mask > 0]
        pix_border_hsv = hsv[border_mask > 0]
        pix_material_hsv = material_pixels_from_stable_ring(hsv, x, y, r_draw, ring_inner_ratio=0.45, ring_outer_ratio=0.80)

        full_hsv = mean_hsv_from_pixels(pix_full_hsv)
        inner_hsv = mean_hsv_from_pixels(pix_inner_hsv)
        border_hsv = mean_hsv_from_pixels(pix_border_hsv)
        material_hsv = mean_hsv_from_pixels(pix_material_hsv)
        material_lab = mean_lab_from_hsv_pixels(pix_material_hsv)
        hsv_kmeans_material = material_from_hsv_kmeans_auto(pix_material_hsv, material_hsv)

        similarity = hsv_similarity_score(inner_hsv, border_hsv)
        sat_delta = abs(float(inner_hsv[1]) - float(border_hsv[1]))
        hue_delta = hue_circular_delta_cv(int(inner_hsv[0]), int(border_hsv[0]))
        val_delta = abs(float(inner_hsv[2]) - float(border_hsv[2]))

        sat_mean = 0.5 * (float(inner_hsv[1]) + float(border_hsv[1]))
        sat_conf = float(np.clip(sat_mean / 70.0, 0.0, 1.0))
        hue_term = hue_delta * sat_conf
        value_term = 0.15 * val_delta
        color_delta = float(sat_delta + hue_term + value_term)

        color_deltas_hybrid.append(color_delta)

        ys, xs = np.where(outer_mask > 0)
        if len(xs) > 0:
            radial_norm = (
                np.sqrt((xs.astype(np.float32) - float(x)) ** 2 + (ys.astype(np.float32) - float(y)) ** 2)
                / max(float(r_draw), 1e-6)
            )
            radial_norm = np.clip(radial_norm, 0.0, 1.0)
            coin_hsv = hsv[ys, xs]
            radial_kmeans = coin_kmeans_radial_structure(coin_hsv, radial_norm)
            radial_step = coin_radial_step_score(coin_hsv, radial_norm, split_radius=split_radius, bins=8)
        else:
            radial_kmeans = {"ok": False, "score": 0.0, "radial_sep": 0.0, "agreement": 0.0, "balance": 0.0}
            radial_step = {"ok": False, "score": 0.0, "max_step": 0.0, "step_radius": split_radius}

        edge_shape = coin_edge_roughness_score(gray_shape, x=x, y=y, r=r_draw, n_bins=72)
        rows.append(
            {
                "id": idx,
                "x": x,
                "y": y,
                "r": int(r_draw),
                "r_subpx": float(r),
                "inner_r": inner_r,
                "full_hsv": full_hsv,
                "inner_hsv": inner_hsv,
                "border_hsv": border_hsv,
                "material_hsv": material_hsv,
                "material_lab": material_lab,
                "full_lab": (0.0, 0.0, 0.0),
                "inner_lab": (0.0, 0.0, 0.0),
                "border_lab": (0.0, 0.0, 0.0),
                "similarity": float(similarity),
                "sat_delta": float(sat_delta),
                "hue_delta": float(hue_delta),
                "val_delta": float(val_delta),
                "sat_conf": float(sat_conf),
                "hue_term": float(hue_term),
                "value_term": float(value_term),
                "color_delta": float(color_delta),
                "lab_delta": 0.0,
                "mean_warm_delta": 0.0,
                "mean_color_delta": 0.0,
                "material_label_hsv_kmeans": str(hsv_kmeans_material["material_label"]),
                "bronze_score_hsv_kmeans": float(hsv_kmeans_material["bronze_score"]),
                "gold_score_hsv_kmeans": float(hsv_kmeans_material["gold_score"]),
                "color_kmeans_k": int(hsv_kmeans_material["chosen_k"]),
                "color_kmeans_k2_gain": float(hsv_kmeans_material["k2_gain"]),
                "color_kmeans_bronze_share": float(hsv_kmeans_material["bronze_share"]),
                "color_kmeans_gold_share": float(hsv_kmeans_material["gold_share"]),
                "color_kmeans_used_fallback": bool(hsv_kmeans_material["used_fallback"]),
                "kmeans_radial_ok": bool(radial_kmeans["ok"]),
                "kmeans_radial_score": float(radial_kmeans["score"]),
                "kmeans_radial_sep": float(radial_kmeans["radial_sep"]),
                "kmeans_radial_agreement": float(radial_kmeans["agreement"]),
                "kmeans_radial_balance": float(radial_kmeans["balance"]),
                "step_ok": bool(radial_step["ok"]),
                "step_score": float(radial_step["score"]),
                "step_max": float(radial_step["max_step"]),
                "step_radius": float(radial_step["step_radius"]),
                "edge_roughness": float(edge_shape["roughness"]),
                "edge_coverage": float(edge_shape["coverage"]),
                "gold_flower_score": float(edge_shape["score"]),
            }
        )

    # Global threshold derived from all coins in current image.
    outlier_threshold_hybrid = choose_dynamic_sat_delta_threshold(color_deltas_hybrid, default=18.0)
    if sat_delta_threshold is not None:
        outlier_threshold_hybrid = float(sat_delta_threshold)

    # Global absolute thresholds on fused color delta:
    # - `abs_lo`: strongly uniform colors,
    # - `abs_mid/hi/vhi`: progressively stronger two-tone evidence.
    hybrid_abs_mid = 16.0
    hybrid_abs_hi = 22.0
    hybrid_abs_vhi = 30.0
    hybrid_abs_lo = 10.0

    enrich_rows_with_lab_material_labels(rows)

    # Pass 2: final deterministic decision with fused evidence.
    for row in rows:
        x = row["x"]
        y = row["y"]
        r = row["r"]
        inner_r = row["inner_r"]
        r_draw = int(max(1, int(r)))
        full_hsv = row["full_hsv"]
        inner_hsv = row["inner_hsv"]
        border_hsv = row["border_hsv"]
        material_hsv = row["material_hsv"]

        color_delta = float(row["color_delta"])
        sat_delta = float(row["sat_delta"])
        hue_delta = float(row["hue_delta"])
        bronze_veto_applied = False

        outlier_threshold = outlier_threshold_hybrid
        abs_mid = hybrid_abs_mid
        abs_hi = hybrid_abs_hi
        abs_vhi = hybrid_abs_vhi
        abs_lo = hybrid_abs_lo

        outlier_vote = color_delta >= outlier_threshold
        mid_delta = color_delta >= abs_mid
        strong_delta = color_delta >= abs_hi
        very_strong_delta = color_delta >= abs_vhi

        kmeans_ok = bool(row["kmeans_radial_ok"])
        step_ok = bool(row["step_ok"])
        kmeans_support = bool(
            kmeans_ok
            and (float(row["kmeans_radial_agreement"]) >= 0.64)
            and (float(row["kmeans_radial_balance"]) >= 0.12)
        )
        step_support = bool(step_ok and (float(row["step_score"]) >= 0.75))
        structure_votes = int(kmeans_support) + int(step_support)

        strong_color_evidence = bool(
            (sat_delta >= 22.0) or (hue_delta >= 10.0 and sat_delta >= 10.0) or (color_delta >= abs_vhi)
        )
        evidence = int(strong_delta) + int(outlier_vote) + structure_votes + int(strong_color_evidence)

        # Decision lattice: strongest, most reliable branches first.
        # Decision logic is ordered from highest-confidence bimetal evidence
        # to strongest one-color evidence, and finally uncertain fallback.
        if strong_color_evidence and very_strong_delta and (outlier_vote or kmeans_support):
            detector_type = "bi-metal-like"
        elif strong_color_evidence and strong_delta and ((outlier_vote and structure_votes >= 1) or kmeans_support):
            detector_type = "bi-metal-like"
        elif strong_color_evidence and mid_delta and outlier_vote and (kmeans_support and step_support):
            detector_type = "bi-metal-like"
        elif (not outlier_vote) and (color_delta < 20.0) and (sat_delta <= 18.0) and (hue_delta <= 8.0):
            detector_type = "one-color-like"
        elif (color_delta <= 14.0) and (sat_delta <= 12.0) and (hue_delta <= 10.0):
            detector_type = "one-color-like"
        elif (color_delta <= abs_lo) and (structure_votes <= 1):
            detector_type = "one-color-like"
        elif (not strong_color_evidence) and (not outlier_vote) and (color_delta <= (abs_hi + 2.0)):
            detector_type = "one-color-like"
        elif (
            (not outlier_vote)
            and (not kmeans_support)
            and strong_color_evidence
            and (int(inner_hsv[0]) <= 15)
            and (int(inner_hsv[1]) >= 120)
        ):
            detector_type = "one-color-like"
        else:
            detector_type = "uncertain"

        inner_h = int(inner_hsv[0])
        inner_s = int(inner_hsv[1])
        inner_v = int(inner_hsv[2])
        border_h = int(border_hsv[0])
        border_s = int(border_hsv[1])
        kmeans_score = float(row["kmeans_radial_score"])
        step_score = float(row["step_score"])
        sat_mean = 0.5 * (float(inner_s) + float(border_s))

        promote_uncertain_to_bimetal = bool(
            (detector_type == "uncertain")
            and (float(r) >= 34.0)
            and (sat_mean < 60.0)
            and (val_delta >= 8.0)
            and (color_delta >= 22.0)
            and ((kmeans_score >= 0.62) or (step_score >= 0.55))
        )
        if promote_uncertain_to_bimetal:
            detector_type = "bi-metal-like"

        false_bimetal_copper = bool(
            (detector_type == "bi-metal-like")
            and (inner_h <= 18)
            and (border_h <= 18)
            and (inner_s >= 140)
            and (border_s >= 95)
            and (kmeans_score < 0.85)
        )
        false_bimetal_warm_uniform = bool(
            (detector_type == "bi-metal-like")
            and (inner_h >= 19)
            and (border_h >= 19)
            and (hue_delta <= 2.0)
            and (min(inner_s, border_s) >= 100)
            and (kmeans_score < 0.85)
            and (sat_delta < 30.0)
            and (color_delta < 30.0)
        )
        # Conservative vetoes to reduce common false positives.
        if false_bimetal_copper or false_bimetal_warm_uniform:
            detector_type = "one-color-like"
            bronze_veto_applied = True

        if detector_type == "bi-metal-like":
            bimetal_euro_label = label_bimetal_euro_from_saturation(inner_s, border_s)
            final_label = bimetal_euro_label
            bronze_score = 0.0
            bronze_score_hsv = 0.0
            bronze_score_lab = 0.0
            bronze_score_hybrid = 0.0
            gold_score_hsv = 0.0
            gold_score_lab = 0.0
            gold_score_hybrid = 0.0
            bronze_score_hsv_kmeans = 0.0
            gold_score_hsv_kmeans = 0.0
            material_label_hsv = "n/a"
            material_label_lab = "n/a"
            material_label_hybrid = "n/a"
            material_label_hsv_kmeans = "n/a"
            material_label_lab_scene = "n/a"
            material_label_lab_proto = "n/a"
            material_label = "n/a"
            color_kmeans_k = 0
            color_kmeans_k2_gain = 0.0
            color_kmeans_bronze_share = 0.0
            color_kmeans_gold_share = 0.0
            color_kmeans_used_fallback = False
            material_lab_scene_conf = 0.0
            material_lab_proto_conf = 0.0
        else:
            bimetal_euro_label = "n/a"

            material_h = int(material_hsv[0])
            material_s = int(material_hsv[1])
            material_v = int(material_hsv[2])

            bronze_score_hsv = bronze_score_from_inner_hsv(material_h, material_s, material_v)
            gold_score_hsv = gold_score_from_inner_hsv(material_h, material_s, material_v)
            bronze_score_lab = bronze_score_hsv
            gold_score_lab = gold_score_hsv
            bronze_score_hybrid = bronze_score_hsv
            gold_score_hybrid = gold_score_hsv
            material_label_hsv = label_material_from_inner_hsv(material_h, material_s, material_v)
            material_label_lab = material_label_hsv
            material_label_hybrid = fallback_material_from_inner_hsv(material_h, material_s, material_v)
            bronze_score_hsv_kmeans = float(row.get("bronze_score_hsv_kmeans", bronze_score_hsv))
            gold_score_hsv_kmeans = float(row.get("gold_score_hsv_kmeans", gold_score_hsv))
            material_label_hsv_kmeans = str(row.get("material_label_hsv_kmeans", material_label_hybrid))
            material_label_lab_scene = str(row.get("material_label_lab_scene", material_label_hsv_kmeans))
            material_lab_scene_conf = float(row.get("material_lab_scene_conf", 0.0))
            material_label_lab_proto = str(row.get("material_label_lab_proto", material_label_hsv_kmeans))
            material_lab_proto_conf = float(row.get("material_lab_proto_conf", 0.0))
            color_kmeans_k = int(row.get("color_kmeans_k", 1))
            color_kmeans_k2_gain = float(row.get("color_kmeans_k2_gain", 0.0))
            color_kmeans_bronze_share = float(row.get("color_kmeans_bronze_share", 0.0))
            color_kmeans_gold_share = float(row.get("color_kmeans_gold_share", 0.0))
            color_kmeans_used_fallback = bool(row.get("color_kmeans_used_fallback", False))

            material_label, bronze_score = pick_material_label_by_mode(
                material_mode=material_mode,
                material_label_hybrid=material_label_hybrid,
                material_label_hsv_kmeans=material_label_hsv_kmeans,
                material_label_lab_proto=material_label_lab_proto,
                bronze_score_hsv=bronze_score_hsv,
                bronze_score_hsv_kmeans=bronze_score_hsv_kmeans,
            )

            if detector_type == "uncertain":
                final_label = "uncertain"
            elif material_label == "borderline":
                final_label = "one-color-like/borderline"
            else:
                final_label = f"one-color-like/{material_label}"

        full_bgr = cv2.cvtColor(np.uint8([[[*full_hsv]]]), cv2.COLOR_HSV2BGR)[0, 0]
        inner_bgr = cv2.cvtColor(np.uint8([[[*inner_hsv]]]), cv2.COLOR_HSV2BGR)[0, 0]
        border_bgr = cv2.cvtColor(np.uint8([[[*border_hsv]]]), cv2.COLOR_HSV2BGR)[0, 0]

        if detector_type == "bi-metal-like":
            cv2.circle(output_bgr, (x, y), r_draw, tuple(int(c) for c in border_bgr), -1)
            cv2.circle(output_bgr, (x, y), inner_r, tuple(int(c) for c in inner_bgr), -1)
        else:
            cv2.circle(output_bgr, (x, y), r_draw, tuple(int(c) for c in full_bgr), -1)

        cv2.circle(output_bgr, (x, y), r_draw, (0, 0, 0), 2)
        if detector_type == "bi-metal-like":
            cv2.circle(output_bgr, (x, y), inner_r, (0, 0, 0), 2)

        text = str(row["id"] + 1)
        if detector_type == "bi-metal-like":
            if bimetal_euro_label == "2-euro-like":
                short_label = "(2)"
            elif bimetal_euro_label == "1-euro-like":
                short_label = "(1)"
            else:
                short_label = "(?)"
        else:
            if material_label == "gold":
                short_label = "G"
            elif material_label == "bronze":
                short_label = "B"
            else:
                short_label = "?"

        text_fg = (0, 220, 255) if detector_type == "uncertain" else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        num_scale = 0.55
        lbl_scale = 0.45
        num_size, _ = cv2.getTextSize(text, font, num_scale, 1)
        lbl_size, _ = cv2.getTextSize(short_label, font, lbl_scale, 1)
        num_org = (int(x) - (num_size[0] // 2), int(y) - 2)
        lbl_org = (int(x) - (lbl_size[0] // 2), int(y) + 14)
        cv2.putText(output_bgr, text, num_org, font, num_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output_bgr, text, num_org, font, num_scale, text_fg, 1, cv2.LINE_AA)
        cv2.putText(output_bgr, short_label, lbl_org, font, lbl_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output_bgr, short_label, lbl_org, font, lbl_scale, text_fg, 1, cv2.LINE_AA)

        row["color_delta_threshold"] = float(outlier_threshold)
        row["abs_threshold_mid"] = float(abs_mid)
        row["abs_threshold_hi"] = float(abs_hi)
        row["abs_threshold_vhi"] = float(abs_vhi)
        row["decision_evidence"] = int(evidence)
        row["bronze_score"] = float(bronze_score)
        row["bronze_score_hsv"] = float(bronze_score_hsv)
        row["bronze_score_lab"] = float(bronze_score_lab)
        row["bronze_score_hybrid"] = float(bronze_score_hybrid)
        row["gold_score_hsv"] = float(gold_score_hsv)
        row["gold_score_lab"] = float(gold_score_lab)
        row["gold_score_hybrid"] = float(gold_score_hybrid)
        row["bronze_score_hsv_kmeans"] = float(bronze_score_hsv_kmeans)
        row["gold_score_hsv_kmeans"] = float(gold_score_hsv_kmeans)
        row["material_label_hsv"] = material_label_hsv
        row["material_label_lab"] = material_label_lab
        row["material_label_hybrid"] = material_label_hybrid
        row["material_label_hsv_kmeans"] = material_label_hsv_kmeans
        row["material_label_lab_scene"] = material_label_lab_scene
        row["material_lab_scene_conf"] = float(material_lab_scene_conf)
        row["material_label_lab_proto"] = material_label_lab_proto
        row["material_lab_proto_conf"] = float(material_lab_proto_conf)
        row["material_label"] = material_label
        row["color_kmeans_k"] = int(color_kmeans_k)
        row["color_kmeans_k2_gain"] = float(color_kmeans_k2_gain)
        row["color_kmeans_bronze_share"] = float(color_kmeans_bronze_share)
        row["color_kmeans_gold_share"] = float(color_kmeans_gold_share)
        row["color_kmeans_used_fallback"] = bool(color_kmeans_used_fallback)
        row["detector_type"] = detector_type
        row["bronze_veto"] = bool(bronze_veto_applied)
        row["bimetal_euro_label"] = bimetal_euro_label
        row["bimetal_mode"] = bimetal_mode
        row["material_mode"] = material_mode
        row["short_label"] = short_label
        row["label"] = final_label
        row["type"] = detector_type
        stats.append(row)

    return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), stats
