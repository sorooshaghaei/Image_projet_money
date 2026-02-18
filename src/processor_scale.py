from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from .coin_metadata import COIN_DIAMETER_MM


class ScaleValueClassifier:
    """Assign denominations by fitting one shared px/mm scale for all detected coins."""

    def __init__(self, *, color_mismatch_penalty: float = 0.10) -> None:
        self._color_mismatch_penalty = float(color_mismatch_penalty)

    def classify(
        self,
        radii_px: Sequence[float],
        candidate_denoms_per_coin: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[List[int], List[float], float]:
        """
        Fit one global scale (px per mm diameter) and assign each coin to the closest real euro diameter.
        Color-derived candidates are used as a soft penalty, not a hard constraint.
        """
        n = len(radii_px)
        if n == 0:
            return [], [], 1.0

        all_denoms = sorted(COIN_DIAMETER_MM.keys())
        if candidate_denoms_per_coin is None or len(candidate_denoms_per_coin) != n:
            candidate_denoms_per_coin = [all_denoms for _ in range(n)]

        diameters_px = [max(1e-6, float(r) * 2.0) for r in radii_px]
        hypotheses = self._build_scale_hypotheses(diameters_px, candidate_denoms_per_coin, all_denoms)

        best_scale = float(hypotheses[0])
        best_labels: List[int] = [all_denoms[0]] * n
        best_errors: List[float] = [0.0] * n
        best_score = float("inf")
        for scale in hypotheses:
            labels, rel_errors, score = self._score_scale_hypothesis(
                scale=float(scale),
                diameters_px=diameters_px,
                candidate_denoms_per_coin=candidate_denoms_per_coin,
                all_denoms=all_denoms,
            )
            if score < best_score:
                best_score = score
                best_scale = float(scale)
                best_labels = labels
                best_errors = rel_errors

        refined_scale = self._least_squares_refine_scale(diameters_px, best_labels)
        if refined_scale is not None:
            labels, rel_errors, score = self._score_scale_hypothesis(
                scale=refined_scale,
                diameters_px=diameters_px,
                candidate_denoms_per_coin=candidate_denoms_per_coin,
                all_denoms=all_denoms,
            )
            if score <= best_score:
                best_score = score
                best_scale = refined_scale
                best_labels = labels
                best_errors = rel_errors

        return best_labels, best_errors, best_scale

    def _build_scale_hypotheses(
        self,
        diameters_px: Sequence[float],
        candidate_denoms_per_coin: Sequence[Sequence[int]],
        all_denoms: Sequence[int],
    ) -> np.ndarray:
        hypotheses: List[float] = []
        for i, d_px in enumerate(diameters_px):
            candidates = list(candidate_denoms_per_coin[i]) or list(all_denoms)
            for denom in candidates:
                mm = COIN_DIAMETER_MM.get(int(denom))
                if mm is None or mm <= 0:
                    continue
                hypotheses.append(float(d_px) / mm)

        if not hypotheses:
            mm_values = [COIN_DIAMETER_MM[d] for d in all_denoms]
            hypotheses = [float(np.median(diameters_px) / np.median(mm_values))]

        hyp = np.asarray(hypotheses, dtype=float)
        if hyp.size > 6:
            lo, hi = np.percentile(hyp, [8, 92])
            hyp = hyp[(hyp >= lo) & (hyp <= hi)]
            if hyp.size == 0:
                hyp = np.asarray(hypotheses, dtype=float)
        return np.unique(np.round(hyp, 4))

    def _least_squares_refine_scale(self, diameters_px: Sequence[float], labels: Sequence[int]) -> Optional[float]:
        # Weighted least-squares refinement after discrete hypothesis selection.
        mm_from_labels = [COIN_DIAMETER_MM[d] for d in labels]
        num = float(np.sum(np.asarray(diameters_px) * np.asarray(mm_from_labels)))
        den = float(np.sum(np.asarray(mm_from_labels) * np.asarray(mm_from_labels)))
        if den <= 1e-9:
            return None
        return num / den

    def _score_scale_hypothesis(
        self,
        scale: float,
        diameters_px: Sequence[float],
        candidate_denoms_per_coin: Sequence[Sequence[int]],
        all_denoms: Sequence[int],
    ) -> Tuple[List[int], List[float], float]:
        """Score one global scale by assigning each coin to the lowest penalized relative error."""
        labels: List[int] = []
        rel_errors: List[float] = []
        losses: List[float] = []

        safe_scale = max(1e-6, float(scale))
        for i, d_px in enumerate(diameters_px):
            allowed = self._allowed_denoms_for_coin(i, candidate_denoms_per_coin, all_denoms)

            best_denom = int(all_denoms[0])
            best_loss = float("inf")
            best_rel = float("inf")
            for denom in all_denoms:
                expected = safe_scale * COIN_DIAMETER_MM[int(denom)]
                rel = abs(float(d_px) - expected) / max(expected, 1e-6)
                penalty = 0.0 if int(denom) in allowed else self._color_mismatch_penalty
                loss = rel + penalty
                if loss < best_loss:
                    best_loss = float(loss)
                    best_rel = float(rel)
                    best_denom = int(denom)

            labels.append(best_denom)
            rel_errors.append(best_rel)
            losses.append(best_loss)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        return labels, rel_errors, mean_loss

    def _allowed_denoms_for_coin(
        self,
        idx: int,
        candidate_denoms_per_coin: Sequence[Sequence[int]],
        all_denoms: Sequence[int],
    ) -> Set[int]:
        if idx >= len(candidate_denoms_per_coin):
            return set(all_denoms)
        allowed = set(int(d) for d in candidate_denoms_per_coin[idx])
        return allowed if allowed else set(all_denoms)
