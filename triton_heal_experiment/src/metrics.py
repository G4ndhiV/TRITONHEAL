"""Classification metrics for kernel safety verification."""
import numpy as np
import pandas as pd
from scipy import stats


def confusion_counts(gt_unsafe: np.ndarray, pred_unsafe: np.ndarray) -> dict:
    tp = int(np.sum(gt_unsafe & pred_unsafe))
    fn = int(np.sum(gt_unsafe & ~pred_unsafe))
    fp = int(np.sum(~gt_unsafe & pred_unsafe))
    tn = int(np.sum(~gt_unsafe & ~pred_unsafe))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def msr_from_df(sub: pd.DataFrame) -> float:
    gt = sub["label"].values == "unsafe"
    pred = ~sub["pred_safe"].values.astype(bool)
    c = confusion_counts(gt, pred)
    if c["tp"] + c["fn"] == 0:
        return float("nan")
    return c["tp"] / (c["tp"] + c["fn"])


def fnr_from_df(sub: pd.DataFrame) -> float:
    gt = sub["label"].values == "unsafe"
    pred = ~sub["pred_safe"].values.astype(bool)
    c = confusion_counts(gt, pred)
    if c["fn"] + c["tp"] == 0:
        return float("nan")
    return c["fn"] / (c["fn"] + c["tp"])


def f1_from_df(sub: pd.DataFrame) -> float:
    gt = sub["label"].values == "unsafe"
    pred = ~sub["pred_safe"].values.astype(bool)
    c = confusion_counts(gt, pred)
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))
