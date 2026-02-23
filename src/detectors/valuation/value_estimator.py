"""Coin denomination estimation from per-coin split/color stats.

This module maps color-analysis outputs to euro denominations with a hybrid
approach:
- family inference (`bronze`, `gold`, `bimetal`, `unknown`),
- radius clustering inside each family,
- optional global scale estimation (px/mm),
- probabilistic scoring over candidate denominations.
"""

from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np

from .value_estimation_result import ValueEstimationResult


def _coin_marker_token(index: int) -> str:
    """Convert zero-based index into Excel-like marker tokens: A, B, ..., Z, AA, AB..."""
    n = max(0, int(index))
    out: list[str] = []
    while True:
        n, rem = divmod(n, 26)
        out.append(chr(ord("A") + rem))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(out))


def _clamp_value(x: float, lo: float, hi: float) -> float:
    """Clamp helper used by handcrafted HSV family priors."""
    return lo if x < lo else hi if x > hi else x


def _bronze_score_from_inner_hsv(h: int, s: int, v: int) -> float:
    """Bronze-like confidence from HSV inner-region statistics."""
    h_term = _clamp_value((18.0 - float(h)) / 3.0, 0.0, 1.0)
    sat_conf = _clamp_value((float(s) - 45.0) / 75.0, 0.0, 1.0)
    val_conf = _clamp_value((float(v) - 35.0) / 90.0, 0.0, 1.0)
    conf = 0.65 * sat_conf + 0.35 * val_conf
    return float(np.clip(h_term * max(0.35, conf), 0.0, 1.0))


def _gold_score_from_inner_hsv(h: int, s: int, v: int) -> float:
    """Gold-like confidence from HSV inner-region statistics."""
    hue_center = 22.0
    hue_width = 8.5
    h_term = 1.0 - _clamp_value(abs(float(h) - hue_center) / hue_width, 0.0, 1.0)
    sat_conf = _clamp_value((float(s) - 40.0) / 90.0, 0.0, 1.0)
    val_conf = _clamp_value((float(v) - 55.0) / 110.0, 0.0, 1.0)
    conf = 0.60 * sat_conf + 0.40 * val_conf
    return float(np.clip(h_term * max(0.30, conf), 0.0, 1.0))


def _material_family_from_inner_hsv(inner_hsv: tuple[int, int, int] | list[int]) -> str:
    """Convert raw HSV tuple to a coarse family hint."""
    if not isinstance(inner_hsv, (tuple, list)) or len(inner_hsv) < 3:
        return "unknown"

    h = int(inner_hsv[0])
    s = int(inner_hsv[1])
    v = int(inner_hsv[2])

    bronze_score = _bronze_score_from_inner_hsv(h, s, v)
    gold_score = _gold_score_from_inner_hsv(h, s, v)
    if max(bronze_score, gold_score) >= 0.30 and abs(bronze_score - gold_score) >= 0.02:
        return "bronze" if bronze_score > gold_score else "gold"

    if s >= 45:
        if h <= 15:
            return "bronze"
        if h >= 16:
            return "gold"
    return "unknown"


class ValueEstimator:
    """Static denomination estimator with explainable intermediate artifacts."""

    EURO_DIAMETER_MM: dict[str, float] = {
        "1c": 16.25,
        "2c": 18.75,
        "5c": 21.25,
        "10c": 19.75,
        "20c": 22.25,
        "50c": 24.25,
        "1e": 23.25,
        "2e": 25.75,
    }

    FAMILY_TO_DENOMS: dict[str, list[str]] = {
        "bronze": ["1c", "2c", "5c"],
        "gold": ["10c", "20c", "50c"],
        "bimetal": ["1e", "2e"],
        "unknown": ["1c", "2c", "5c", "10c", "20c", "50c"],
    }

    DENOM_TEXT: dict[str, str] = {
        "1c": "1c",
        "2c": "2c",
        "5c": "5c",
        "10c": "10c",
        "20c": "20c",
        "50c": "50c",
        "1e": "1EUR",
        "2e": "2EUR",
    }

    DENOM_TO_CENTS: dict[str, int] = {
        "1c": 1,
        "2c": 2,
        "5c": 5,
        "10c": 10,
        "20c": 20,
        "50c": 50,
        "1e": 100,
        "2e": 200,
    }

    DENOM_PRINT_ORDER: list[str] = ["1c", "2c", "5c", "10c", "20c", "50c", "1e", "2e"]

    @classmethod
    def estimate_from_stats(cls, rows: list[dict]) -> ValueEstimationResult:
        """Run full denomination estimation from split-analysis rows."""
        predictions, scale_info, family_models = cls._estimate_coin_values_from_stats(rows)
        counts, total_cents = cls.summarize_prediction_totals(predictions)
        return ValueEstimationResult(
            predictions=predictions,
            scale_info=scale_info,
            family_models=family_models,
            counts=counts,
            total_cents=total_cents,
        )

    @classmethod
    def coin_family_from_row(cls, row: dict) -> str:
        """Infer denomination family for a single analyzed coin row."""
        if row.get("type") == "bi-metal-like":
            return "bimetal"

        material = row.get("material_label")
        if material in ("bronze", "gold"):
            return material

        inner_hsv = row.get("inner_hsv")
        material_hint = _material_family_from_inner_hsv(inner_hsv) if isinstance(inner_hsv, (tuple, list)) else "unknown"
        if material_hint in ("bronze", "gold"):
            return material_hint

        return "unknown"

    @classmethod
    def summarize_prediction_totals(cls, predictions: dict[int, dict]) -> tuple[dict[str, int], int]:
        """Aggregate per-coin best labels into denomination counts and cents."""
        counts = {d: 0 for d in cls.DENOM_PRINT_ORDER}
        total_cents = 0
        for pred in predictions.values():
            denom = pred.get("best_denom")
            if denom not in cls.DENOM_TO_CENTS:
                continue
            counts[denom] += 1
            total_cents += int(cls.DENOM_TO_CENTS[denom])
        return counts, int(total_cents)

    @classmethod
    def prob_string(cls, prob_map: dict[str, float]) -> str:
        """Format probability map in descending order for debug display."""
        ordered = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
        return ", ".join(f"{cls.DENOM_TEXT[k]}={100.0 * v:.1f}%" for k, v in ordered)

    @classmethod
    def draw_coin_value_labels(
        cls,
        image_bgr: np.ndarray,
        rows: list[dict],
        predictions: dict[int, dict],
    ) -> np.ndarray:
        """Render family-colored circles plus marker tokens on image."""
        out = image_bgr.copy()
        family_colors = {
            "bronze": (40, 120, 230),
            "gold": (0, 200, 255),
            "bimetal": (255, 180, 40),
            "unknown": (180, 180, 180),
        }

        for row in rows:
            coin_id = int(row["id"])
            pred = predictions.get(coin_id)
            if pred is None:
                continue

            x = int(row["x"])
            y = int(row["y"])
            r = int(row["r"])
            family = pred["family"]
            color = family_colors.get(family, (180, 180, 180))

            cv2.circle(out, (x, y), r, color, 2)
            cv2.circle(out, (x, y), 2, (0, 0, 0), 3)
            marker = _coin_marker_token(coin_id)
            marker_radius = max(8, min(16, int(0.34 * max(1, r))))
            cv2.circle(out, (x, y), marker_radius, (15, 15, 15), -1)
            cv2.circle(out, (x, y), marker_radius, color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.52 if len(marker) == 1 else 0.42
            thickness = 1
            text_size, _ = cv2.getTextSize(marker, font, scale, thickness)
            tx = int(x) - (text_size[0] // 2)
            ty = int(y) + (text_size[1] // 2)
            cv2.putText(out, marker, (tx, ty), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, marker, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    @classmethod
    def _kmeans_1d_sorted(cls, values: list[float], k: int) -> tuple[np.ndarray, np.ndarray]:
        """1D KMeans with sorted centers and remapped labels."""
        vals = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        n = int(vals.shape[0])
        if n == 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        k = int(max(1, min(k, n)))
        if k == 1:
            center = np.array([float(np.mean(vals[:, 0]))], dtype=np.float32)
            labels = np.zeros((n,), dtype=np.int32)
            return labels, center

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.02)
        cv2.setRNGSeed(42)
        _, labels, centers = cv2.kmeans(vals, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

        centers = centers[:, 0]
        order = np.argsort(centers)
        centers_sorted = centers[order]
        remap = np.zeros((k,), dtype=np.int32)
        for rank, old_idx in enumerate(order):
            remap[int(old_idx)] = int(rank)

        labels_flat = labels[:, 0].astype(np.int32)
        labels_sorted = remap[labels_flat]
        return labels_sorted, centers_sorted.astype(np.float32)

    @classmethod
    def _kmeans_1d_with_stats(cls, values: list[float], k: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """1D KMeans plus inertia and cluster sizes."""
        labels, centers = cls._kmeans_1d_sorted(values, k)
        vals = np.asarray(values, dtype=np.float64)

        if vals.size == 0:
            return labels, centers, 0.0, np.zeros((0,), dtype=np.int32)
        if centers.size == 0:
            return labels, centers, 0.0, np.zeros((0,), dtype=np.int32)

        labels_i = labels.astype(np.int32)
        pred = centers[labels_i].astype(np.float64)
        inertia = float(np.sum((vals - pred) ** 2))
        counts = np.bincount(labels_i, minlength=int(centers.size)).astype(np.int32)
        return labels_i, centers.astype(np.float32), inertia, counts

    @classmethod
    def _choose_k_for_family(cls, values: list[float], max_k: int, family: str) -> tuple[int, dict]:
        """Choose number of clusters with BIC-like objective + tiny-cluster penalty."""
        vals = np.asarray(values, dtype=np.float64)
        n = int(vals.size)
        if n <= 1:
            return 1, {"reason": "n<=1", "evaluated": []}

        uniq = int(np.unique(np.round(vals, 2)).size)
        max_k = int(max(1, min(max_k, n, uniq)))
        if max_k <= 1:
            return 1, {"reason": "max_k<=1", "evaluated": []}

        min_count_abs = 1 if n < 6 else 2
        min_count_frac = 0.14 if family in ("gold", "bronze") else 0.10

        rows = []
        for k in range(1, max_k + 1):
            _, centers, inertia, counts = cls._kmeans_1d_with_stats(vals.tolist(), k)
            sigma2 = max(float(inertia) / max(n, 1), 1e-6)
            dof = max(1, 2 * k - 1)
            bic = float(n * np.log(sigma2) + dof * np.log(max(n, 2)))

            min_count = int(np.min(counts)) if counts.size > 0 else 0
            min_needed = max(int(min_count_abs), int(np.ceil(min_count_frac * n)))
            tiny = max(0, min_needed - min_count)
            if tiny > 0:
                bic += float(4.0 * tiny)

            rows.append(
                {
                    "k": int(k),
                    "bic": float(bic),
                    "inertia": float(inertia),
                    "min_count": int(min_count),
                    "counts": [int(c) for c in counts.tolist()],
                }
            )

        best = min(rows, key=lambda r: r["bic"])
        best_k = int(best["k"])
        if best_k > 1:
            prev = [r for r in rows if r["k"] == (best_k - 1)][0]
            rel_gain = float((prev["inertia"] - best["inertia"]) / (prev["inertia"] + 1e-9))
            min_gain = {"gold": 0.12, "bronze": 0.10, "bimetal": 0.08, "unknown": 0.14}.get(family, 0.10)
            if rel_gain < min_gain:
                best_k = int(best_k - 1)

        return best_k, {"reason": "bic+min_cluster", "evaluated": rows, "chosen_k": int(best_k)}

    @classmethod
    def _best_denom_subset_for_centers(
        cls,
        sorted_centers_px: np.ndarray,
        candidate_denoms: list[str],
        px_per_mm: float | None,
    ) -> list[str]:
        """Match sorted cluster centers to best denomination subset."""
        k = int(len(sorted_centers_px))
        if k <= 0:
            return []
        if k >= len(candidate_denoms):
            return candidate_denoms[:]

        centers = np.asarray(sorted_centers_px, dtype=np.float64)
        best_score = 1e18
        best: list[str] | None = None

        for comb in combinations(candidate_denoms, k):
            half_mm = np.array([0.5 * cls.EURO_DIAMETER_MM[d] for d in comb], dtype=np.float64)
            if px_per_mm is not None and px_per_mm > 0:
                expected_px = half_mm * float(px_per_mm)
                denom = np.mean(expected_px ** 2) + 1e-9
                score = float(np.mean((centers - expected_px) ** 2) / denom)
            else:
                c = centers / (np.median(centers) + 1e-9)
                e = half_mm / (np.median(half_mm) + 1e-9)
                score = float(np.mean((c - e) ** 2))
            if score < best_score:
                best_score = score
                best = list(comb)

        return best if best is not None else candidate_denoms[:k]

    @classmethod
    def _estimate_scale_from_all_radii_voting(cls, rows: list[dict]) -> tuple[float | None, dict]:
        """Robustly estimate image scale by consensus across all radius hypotheses."""
        scale_candidates: list[float] = []
        for row in rows:
            r_px = float(max(1.0, float(row.get("r_subpx", row.get("r", 1.0)))))
            family = cls.coin_family_from_row(row)
            denom_pool = cls.FAMILY_TO_DENOMS.get(family, cls.FAMILY_TO_DENOMS["unknown"])
            for denom in denom_pool:
                d_mm = cls.EURO_DIAMETER_MM[denom]
                scale_candidates.append((2.0 * r_px) / d_mm)

        if len(scale_candidates) == 0:
            return None, {"support": 0, "total": 0}

        arr = np.asarray(scale_candidates, dtype=np.float64)
        tol = 0.18
        diffs = np.abs(arr[:, None] - arr[None, :])
        support_counts = np.sum(diffs <= tol, axis=1)
        best_idx = int(np.argmax(support_counts))
        support_mask = diffs[best_idx] <= tol
        robust_scale = float(np.median(arr[support_mask]))
        return robust_scale, {"support": int(np.sum(support_mask)), "total": int(arr.size), "window_tol": float(tol)}

    @classmethod
    def _estimate_px_per_mm(cls, rows: list[dict]) -> tuple[float | None, dict]:
        """Estimate pixel-per-mm using bimetal references and robust fallback."""
        ref_scales: list[float] = []
        for row in rows:
            if row.get("type") != "bi-metal-like":
                continue
            label = row.get("bimetal_euro_label", "")
            if label == "1-euro-like":
                denom = "1e"
            elif label == "2-euro-like":
                denom = "2e"
            else:
                continue
            r_px = float(max(1.0, float(row.get("r_subpx", row.get("r", 1.0)))))
            ref_scales.append((2.0 * r_px) / cls.EURO_DIAMETER_MM[denom])

        fallback_scale, fallback_dbg = cls._estimate_scale_from_all_radii_voting(rows)
        if len(ref_scales) >= 2:
            ref_arr = np.asarray(ref_scales, dtype=np.float64)
            ref_med = float(np.median(ref_arr))
            ref_mad = float(np.median(np.abs(ref_arr - ref_med)))
            if ref_mad > 1e-6:
                keep_mask = np.abs(ref_arr - ref_med) <= (2.0 * ref_mad + 0.10)
                kept = ref_arr[keep_mask]
                if kept.size > 0:
                    ref_med = float(np.median(kept))
                kept_count = int(kept.size)
            else:
                kept_count = int(ref_arr.size)

            if fallback_scale is not None:
                rel_gap = abs(ref_med - float(fallback_scale)) / max(ref_med, 1e-6)
                if rel_gap >= 0.04:
                    blend_w_ref = 0.80
                    blended = float(blend_w_ref * ref_med + (1.0 - blend_w_ref) * float(fallback_scale))
                    return blended, {
                        "method": "bimetal_reference_blend",
                        "count": int(len(ref_scales)),
                        "kept_count": kept_count,
                        "raw_scales": [float(x) for x in ref_scales],
                        "ref_median": float(ref_med),
                        "fallback_scale": float(fallback_scale),
                        "blend_w_ref": float(blend_w_ref),
                        "relative_gap": float(rel_gap),
                        **fallback_dbg,
                    }

            return float(ref_med), {
                "method": "bimetal_reference",
                "count": int(len(ref_scales)),
                "kept_count": kept_count,
                "raw_scales": [float(x) for x in ref_scales],
            }
        if len(ref_scales) == 1:
            ref = float(ref_scales[0])
            info = {"method": "single_bimetal_reference", "count": 1, "raw_scales": [float(ref)]}
            if fallback_scale is not None:
                info["fallback_scale"] = float(fallback_scale)
                info.update(fallback_dbg)
            return ref, info
        if fallback_scale is not None:
            return float(fallback_scale), {"method": "all_radii_voting", **fallback_dbg}
        return None, {"method": "none", "count": 0}

    @classmethod
    def _build_family_radius_models(cls, rows: list[dict], px_per_mm: float | None) -> dict:
        """Build per-family radius clustering models and denom mappings."""
        models = {}
        for family, denom_pool in cls.FAMILY_TO_DENOMS.items():
            if family == "bimetal":
                continue
            fam_rows = [row for row in rows if cls.coin_family_from_row(row) == family]
            if len(fam_rows) == 0:
                continue

            radii = [float(row["r"]) for row in fam_rows]
            max_k = min(len(denom_pool), len(radii))
            k, k_dbg = cls._choose_k_for_family(radii, max_k=max_k, family=family)
            labels, centers, _, counts = cls._kmeans_1d_with_stats(radii, k)
            subset = cls._best_denom_subset_for_centers(centers, denom_pool, px_per_mm=px_per_mm)

            cluster_to_denom = {int(i): subset[i] for i in range(len(subset))}
            coin_to_cluster = {int(fam_rows[i]["id"]): int(labels[i]) for i in range(len(fam_rows))}
            models[family] = {
                "k": int(k),
                "centers_px": [float(c) for c in centers.tolist()],
                "cluster_counts": [int(c) for c in counts.tolist()],
                "subset_denoms": subset,
                "cluster_to_denom": cluster_to_denom,
                "coin_to_cluster": coin_to_cluster,
                "k_debug": k_dbg,
            }
        return models

    @classmethod
    def _normalize_scores(cls, score_map: dict[str, float]) -> dict[str, float]:
        """Normalize positive scores into a probability-like distribution."""
        vals = np.array([max(1e-12, float(v)) for v in score_map.values()], dtype=np.float64)
        total = float(np.sum(vals))
        if total <= 0:
            n = max(1, len(score_map))
            return {k: 1.0 / n for k in score_map.keys()}
        return {k: float(max(1e-12, score_map[k]) / total) for k in score_map.keys()}

    @classmethod
    def _estimate_coin_values_from_stats(cls, rows: list[dict]) -> tuple[dict[int, dict], dict, dict]:
        """Core denomination inference with scale, clustering and Bayesian-like fusion."""
        px_per_mm, scale_info = cls._estimate_px_per_mm(rows)
        family_models = cls._build_family_radius_models(rows, px_per_mm=px_per_mm)
        predictions: dict[int, dict] = {}

        for row in rows:
            coin_id = int(row["id"])
            family = cls.coin_family_from_row(row)

            if family == "bimetal":
                bm = row.get("bimetal_euro_label", "")
                if bm == "1-euro-like":
                    probs = {"1e": 1.0, "2e": 0.0}
                    best_denom = "1e"
                elif bm == "2-euro-like":
                    probs = {"1e": 0.0, "2e": 1.0}
                    best_denom = "2e"
                else:
                    probs = {"1e": 0.5, "2e": 0.5}
                    best_denom = "1e"
                predictions[coin_id] = {
                    "coin_id": int(coin_id),
                    "family": family,
                    "cluster_idx": None,
                    "estimated_diameter_mm": None,
                    "probs": probs,
                    "best_denom": best_denom,
                    "best_label": cls.DENOM_TEXT[best_denom],
                    "best_prob": float(probs[best_denom]),
                }
                continue

            denom_pool = cls.FAMILY_TO_DENOMS.get(family, cls.FAMILY_TO_DENOMS["unknown"])
            model = family_models.get(family, {})
            coin_to_cluster = model.get("coin_to_cluster", {})
            cluster_to_denom = model.get("cluster_to_denom", {})
            cluster_idx = coin_to_cluster.get(coin_id)

            centers_px = np.asarray(model.get("centers_px", []), dtype=np.float64)
            denom_to_center = {}
            for c_idx, d_name in cluster_to_denom.items():
                ci = int(c_idx)
                if 0 <= ci < int(centers_px.size):
                    denom_to_center[d_name] = float(centers_px[ci])

            r_px = float(row["r"])
            d_est_mm = None if px_per_mm is None else (2.0 * r_px / float(px_per_mm))

            # Combine independent-ish priors: physical size, cluster mapping,
            # cluster center proximity, and optional shape priors.
            raw_scores = {}
            for denom in denom_pool:
                d_ref = cls.EURO_DIAMETER_MM[denom]
                if d_est_mm is None:
                    size_like = 1.0
                else:
                    sigma_map = {"gold": 0.48, "bronze": 0.62, "unknown": 0.85}
                    sigma_mm = float(sigma_map.get(family, 0.80))
                    z = (float(d_est_mm) - float(d_ref)) / sigma_mm
                    size_like = float(np.exp(-0.5 * z * z))

                k_model = int(model.get("k", 0))
                full_k = int(len(denom_pool))
                if cluster_idx is None:
                    cluster_prior = 1.0
                else:
                    mapped = cluster_to_denom.get(int(cluster_idx))
                    if mapped == denom:
                        cluster_prior = 1.0
                    elif k_model >= full_k:
                        strict_map = {"gold": 0.08, "bronze": 0.10, "unknown": 0.30}
                        cluster_prior = float(strict_map.get(family, 0.25))
                    elif k_model >= 2:
                        partial_map = {"gold": 0.46, "bronze": 0.38, "unknown": 0.55}
                        cluster_prior = float(partial_map.get(family, 0.45))
                    else:
                        cluster_prior = 0.75

                if denom in denom_to_center:
                    c_px = float(denom_to_center[denom])
                    sigma_px = max(0.80, 0.055 * c_px)
                    zc = (r_px - c_px) / sigma_px
                    center_prior = float(np.exp(-0.5 * zc * zc))
                elif k_model >= full_k:
                    strict_miss = {"gold": 0.12, "bronze": 0.14, "unknown": 0.35}
                    center_prior = float(strict_miss.get(family, 0.25))
                elif k_model >= 2:
                    partial_miss = {"gold": 0.70, "bronze": 0.66, "unknown": 0.80}
                    center_prior = float(partial_miss.get(family, 0.65))
                else:
                    center_prior = 1.0

                shape_prior = 1.0
                raw_scores[denom] = float(max(1e-9, size_like * cluster_prior * center_prior * shape_prior))

            probs = cls._normalize_scores(raw_scores)
            best_denom = max(probs.keys(), key=lambda d: probs[d])
            predictions[coin_id] = {
                "coin_id": int(coin_id),
                "family": family,
                "cluster_idx": None if cluster_idx is None else int(cluster_idx),
                "estimated_diameter_mm": None if d_est_mm is None else float(d_est_mm),
                "probs": probs,
                "best_denom": best_denom,
                "best_label": cls.DENOM_TEXT[best_denom],
                "best_prob": float(probs[best_denom]),
            }

        scale_info = {**scale_info, "px_per_mm": None if px_per_mm is None else float(px_per_mm)}
        return predictions, scale_info, family_models
