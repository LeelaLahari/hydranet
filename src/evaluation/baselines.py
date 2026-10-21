"""The baseline the CNN must beat: HAND-threshold flood mapping (Wing et al. 2018-style)."""
import numpy as np

def hand_threshold_extent(hand: np.ndarray, level: float) -> np.ndarray:
    return (hand <= level).astype(np.float32)

def hand_quantile_levels(hand: np.ndarray, quantiles=(0.55, 0.65, 0.75, 0.85)):
    return {q: float(np.quantile(hand, q)) for q in quantiles}
