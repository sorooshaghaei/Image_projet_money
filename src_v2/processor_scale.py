from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from .coin_metadata import COIN_DIAMETER_MM


class ScaleValueClassifier:
    """Global per-image px/mm fitting with soft color constraints."""

    def __init__(
        self,
        *,
        color_mismatch_penalty: float = 0.18,
        outlier_rel_penalty: float = 0.05,
    ) -> None:
        self._color_mismatch_penalty = float(color_mismatch_penalty)
        self._outlier_rel_penalty = float(outlier_rel_penalty)

    def classify(
        self,
        radii_px: Sequence[float],
        candidate_denoms_per_coin: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[List[int], List[float], float]:
        n = len(radii_px)
        if n == 0:
            return [], [], 1.0

        all_denoms = sorted(COIN_DIAMETER_MM.keys())
        if candidate_denoms_per_coin is None or len(candidate_denoms_per_coin) != n:
            candidate_denoms_per_coin = [all_denoms for _ in range(n)]

        diam_px = [max(1e-6, float(r) * 2.0) for r in radii_px]
        hypotheses = self._build_scale_hypotheses(diam_px, candidate_denoms_per_coin, all_denoms)

        best_scale = float(hypotheses[0])
        best_labels: List[int] = [all_denoms[0]] * n
        best_rel_errors: List[float] = [0.0] * n
        best_loss = float("inf")

        for scale in hypotheses:
            labels, rel_errors, loss = self._score_scale_hypothesis(
                scale=float(scale),
                diam_px=diam_px,
                candidate_denoms_per_coin=candidate_denoms_per_coin,
                all_denoms=all_denoms,
            )
            if loss < best_loss:
                best_loss = float(loss)
                best_scale = float(scale)
                best_labels = labels
                best_rel_errors = rel_errors

        refined = self._least_squares_refine_scale(diam_px, best_labels)
        if refined is not None:
            labels, rel_errors, loss = self._score_scale_hypothesis(
                scale=float(refined),
                diam_px=diam_px,
                candidate_denoms_per_coin=candidate_denoms_per_coin,
                all_denoms=all_denoms,
            )
            if loss <= best_loss:
                best_scale = float(refined)
                best_labels = labels
                best_rel_errors = rel_errors

        return best_labels, best_rel_errors, best_scale

    def _build_scale_hypotheses(
        self,
        diam_px: Sequence[float],
        candidate_denoms_per_coin: Sequence[Sequence[int]],
        all_denoms: Sequence[int],
    ) -> np.ndarray:
        hypotheses: List[float] = []
        for idx, d_px in enumerate(diam_px):
            candidates = list(candidate_denoms_per_coin[idx]) if idx < len(candidate_denoms_per_coin) else []
            if not candidates:
                candidates = list(all_denoms)
            for denom in candidates:
                mm = float(COIN_DIAMETER_MM.get(int(denom), 0.0))
                if mm <= 0.0:
                    continue
                hypotheses.append(float(d_px) / mm)

        if not hypotheses:
            mm_vals = [float(COIN_DIAMETER_MM[d]) for d in all_denoms]
            hypotheses = [float(np.median(diam_px) / np.median(mm_vals))]

        hyp = np.asarray(hypotheses, dtype=float)
        if hyp.size > 8:
            lo, hi = np.percentile(hyp, [5, 95])
            clipped = hyp[(hyp >= lo) & (hyp <= hi)]
            if clipped.size > 0:
                hyp = clipped

        return np.unique(np.round(hyp, 4))

    def _least_squares_refine_scale(self, diam_px: Sequence[float], labels: Sequence[int]) -> Optional[float]:
        mm = np.asarray([float(COIN_DIAMETER_MM[int(d)]) for d in labels], dtype=float)
        dpx = np.asarray(diam_px, dtype=float)
        den = float(np.sum(mm * mm))
        if den <= 1e-9:
            return None
        num = float(np.sum(dpx * mm))
        return num / den

    def _score_scale_hypothesis(
        self,
        *,
        scale: float,
        diam_px: Sequence[float],
        candidate_denoms_per_coin: Sequence[Sequence[int]],
        all_denoms: Sequence[int],
    ) -> Tuple[List[int], List[float], float]:
        labels: List[int] = []
        rel_errors: List[float] = []
        losses: List[float] = []

        safe_scale = max(1e-6, float(scale))
        for i, dpx in enumerate(diam_px):
            allowed = self._allowed_denoms_for_coin(i, candidate_denoms_per_coin, all_denoms)

            best_denom = int(all_denoms[0])
            best_rel = float("inf")
            best_loss = float("inf")
            for denom in all_denoms:
                expected = safe_scale * float(COIN_DIAMETER_MM[int(denom)])
                rel = abs(float(dpx) - expected) / max(expected, 1e-6)

                mismatch_penalty = 0.0 if int(denom) in allowed else self._color_mismatch_penalty
                outlier_penalty = self._outlier_rel_penalty * max(0.0, rel - 0.15)
                loss = rel + mismatch_penalty + outlier_penalty

                if loss < best_loss:
                    best_loss = float(loss)
                    best_rel = float(rel)
                    best_denom = int(denom)

            labels.append(best_denom)
            rel_errors.append(best_rel)
            losses.append(best_loss)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        rel_std = float(np.std(rel_errors)) if rel_errors else 0.0
        return labels, rel_errors, mean_loss + 0.10 * rel_std

    def _allowed_denoms_for_coin(
        self,
        idx: int,
        candidate_denoms_per_coin: Sequence[Sequence[int]],
        all_denoms: Sequence[int],
    ) -> Set[int]:
        if idx >= len(candidate_denoms_per_coin):
            return set(int(d) for d in all_denoms)
        allowed = set(int(d) for d in candidate_denoms_per_coin[idx])
        return allowed if allowed else set(int(d) for d in all_denoms)
