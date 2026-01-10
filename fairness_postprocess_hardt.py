"""
Hardt, Price, Srebro (2016) — Post-processing for fairness:
- Equalized Odds:  Y_hat ⟂ A | Y  (equal TPR and FPR across groups)
- Equal Opportunity: equal TPR across groups (only among Y=1)

This implementation is a practical, reusable extractor of the methodology described in:
"Equality of Opportunity in Supervised Learning" (Hardt et al., 2016).

It supports:
1) Post-processing a *real-valued score* : learn per-group randomized thresholding
   (implemented as a distribution over thresholds per group via linear programming).
2) Post-processing a *binary predictor*: learn per-group randomized "flip" probabilities
   depending on (A, Y_pred) via linear programming.

Dependencies: numpy, pandas, scipy (linprog). Optional: scikit-learn for demos.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import linprog


Method = Literal["equalized_odds", "equal_opportunity"]


def _to_numpy_1d(x, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _validate_binary(y: np.ndarray, name: str) -> np.ndarray:
    uniq = np.unique(y)
    if not set(uniq).issubset({0, 1}):
        raise ValueError(f"{name} must be binary in {{0,1}}. Got unique values: {uniq}")
    return y.astype(int)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return TP, FP, TN, FN counts."""
    y_true = _validate_binary(_to_numpy_1d(y_true, "y_true"), "y_true")
    y_pred = _validate_binary(_to_numpy_1d(y_pred, "y_pred"), "y_pred")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def rates_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    pos = tp + fn
    neg = tn + fp
    tpr = tp / pos if pos else np.nan
    fpr = fp / neg if neg else np.nan
    sel = (tp + fp) / (pos + neg) if (pos + neg) else np.nan
    acc = (tp + tn) / (pos + neg) if (pos + neg) else np.nan
    return {"TPR": tpr, "FPR": fpr, "SelectionRate": sel, "Accuracy": acc}


def fairness_report(y_true, y_pred, a) -> pd.DataFrame:
    """Per-group confusion + rates. Works for any hashable group values."""
    y_true = _validate_binary(_to_numpy_1d(y_true, "y_true"), "y_true")
    y_pred = _validate_binary(_to_numpy_1d(y_pred, "y_pred"), "y_pred")
    a = _to_numpy_1d(a, "a")

    rows = []
    for g in pd.unique(a):
        mask = (a == g)
        cc = confusion_counts(y_true[mask], y_pred[mask])
        rr = rates_from_counts(cc["TP"], cc["FP"], cc["TN"], cc["FN"])
        rows.append({"group": g, "n": int(mask.sum()), **cc, **rr})
    df = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)

    # Add "gaps" (max-min) where it makes sense.
    for col in ["TPR", "FPR", "SelectionRate", "Accuracy"]:
        if df[col].notna().all():
            df[f"{col}_gap(max-min)"] = float(df[col].max() - df[col].min())
        else:
            df[f"{col}_gap(max-min)"] = np.nan
    return df


def _threshold_grid(scores: np.ndarray, n_thresholds: int = 200) -> np.ndarray:
    """Quantile-based thresholds + infinities for all-negative/all-positive."""
    scores = _to_numpy_1d(scores, "scores").astype(float)
    # quantiles include endpoints
    qs = np.linspace(0.0, 1.0, max(2, int(n_thresholds)))
    thr = np.quantile(scores, qs)
    thr = np.unique(thr)
    # add -inf / +inf to allow always-positive / always-negative classifiers
    thr = np.concatenate(([-np.inf], thr, [np.inf]))
    # prediction rule is score > t
    return thr


@dataclass
class ScorePostProcessor:
    """
    Post-process a real-valued score R into a fair binary decision \hat{Y}
    using randomized thresholding per group, fitted by linear programming.

    The LP chooses, for each group g, a distribution over thresholds (w_{g,k})
    to minimize expected cost, subject to:
      - Equalized Odds: equal FPR and equal TPR across groups
      - Equal Opportunity: equal TPR across groups

    Notes:
    - The resulting predictor is randomized (as in Hardt et al. 2016) and guarantees the
      constraint in expectation under the fitted data distribution.
    - If you need deterministic outputs, you can use predict_proba() and pick your own
      threshold, but that may break the fairness guarantees.
    """

    method: Method = "equalized_odds"
    cost_fp: float = 1.0
    cost_fn: float = 1.0
    n_thresholds: int = 200
    random_state: Optional[int] = 0

    # fitted attributes
    groups_: Optional[np.ndarray] = None
    thresholds_: Optional[np.ndarray] = None
    weights_: Optional[Dict[Hashable, np.ndarray]] = None  # per group, length K

    def fit(self, y_true, scores, a) -> "ScorePostProcessor":
        y_true = _validate_binary(_to_numpy_1d(y_true, "y_true"), "y_true")
        scores = _to_numpy_1d(scores, "scores").astype(float)
        a = _to_numpy_1d(a, "a")

        groups = pd.unique(a)
        groups = np.array(list(groups), dtype=object)
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups in sensitive attribute a.")

        thresholds = _threshold_grid(scores, n_thresholds=self.n_thresholds)
        K = len(thresholds)
        G = len(groups)

        # Precompute confusion counts per (group, threshold)
        # For each group g and threshold t: y_pred = (score > t)
        # Store TP, FP, FN counts (TN unused in objective/constraints)
        TP = np.zeros((G, K), dtype=float)
        FP = np.zeros((G, K), dtype=float)
        FN = np.zeros((G, K), dtype=float)
        pos = np.zeros(G, dtype=float)
        neg = np.zeros(G, dtype=float)

        for gi, g in enumerate(groups):
            mask = (a == g)
            y_g = y_true[mask]
            s_g = scores[mask]
            pos[gi] = float(np.sum(y_g == 1))
            neg[gi] = float(np.sum(y_g == 0))
            if pos[gi] == 0 or neg[gi] == 0:
                raise ValueError(
                    f"Group {g!r} has pos={pos[gi]} neg={neg[gi]}. "
                    "Equalized-odds constraints require both positives and negatives per group."
                )

            # vectorized thresholds
            # preds shape: (K, n_g)
            preds = (s_g[None, :] > thresholds[:, None]).astype(int)
            # compute counts for each threshold
            TP[gi, :] = preds @ (y_g == 1).astype(int)
            FP[gi, :] = preds @ (y_g == 0).astype(int)
            # FN = positives - TP
            FN[gi, :] = pos[gi] - TP[gi, :]

        # Linear program
        # Variables: w_{g,k} for all g,k  => length G*K
        # Constraints:
        #   (1) for each g: sum_k w_{g,k} = 1
        #   (2) fairness equalities vs reference group 0
        # Objective: minimize sum_{g,k} w_{g,k} * (cost_fp*FP + cost_fn*FN)
        c = (self.cost_fp * FP + self.cost_fn * FN).reshape(-1)

        A_eq = []
        b_eq = []

        # sum-to-1 per group
        for gi in range(G):
            row = np.zeros(G * K, dtype=float)
            row[gi * K : (gi + 1) * K] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)

        ref = 0
        for gi in range(1, G):
            # Equalized odds => match FPR and TPR
            if self.method == "equalized_odds":
                # neg[ref] * FP_g - neg[g] * FP_ref = 0
                row = np.zeros(G * K, dtype=float)
                row[gi * K : (gi + 1) * K] = neg[ref] * FP[gi, :]
                row[ref * K : (ref + 1) * K] = -neg[gi] * FP[ref, :]
                A_eq.append(row)
                b_eq.append(0.0)

            # Both methods constrain TPR:
            # pos[ref] * TP_g - pos[g] * TP_ref = 0
            row = np.zeros(G * K, dtype=float)
            row[gi * K : (gi + 1) * K] = pos[ref] * TP[gi, :]
            row[ref * K : (ref + 1) * K] = -pos[gi] * TP[ref, :]
            A_eq.append(row)
            b_eq.append(0.0)

        A_eq = np.vstack(A_eq)
        b_eq = np.asarray(b_eq, dtype=float)

        bounds = [(0.0, 1.0)] * (G * K)

        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if not res.success:
            raise RuntimeError(
                f"Linear program failed: {res.message}. "
                "Try increasing n_thresholds, or check that each group has both positives and negatives."
            )

        w = res.x.reshape(G, K)
        weights = {groups[gi]: w[gi, :].copy() for gi in range(G)}

        self.groups_ = groups
        self.thresholds_ = thresholds
        self.weights_ = weights
        return self

    def predict_proba(self, scores, a) -> np.ndarray:
        """Return P(\hat{Y}=1 | score, group) implied by randomized thresholds."""
        if self.thresholds_ is None or self.weights_ is None:
            raise RuntimeError("Call fit() first.")
        scores = _to_numpy_1d(scores, "scores").astype(float)
        a = _to_numpy_1d(a, "a")
        out = np.zeros_like(scores, dtype=float)

        thresholds = self.thresholds_
        for g in pd.unique(a):
            if g not in self.weights_:
                raise ValueError(f"Unknown group {g!r} at predict time; seen groups={list(self.weights_.keys())}")
        for g, w in self.weights_.items():
            mask = (a == g)
            if not np.any(mask):
                continue
            # For each sample, prob positive = sum_k w_k * I(score > t_k)
            # Vectorized: indicators shape (n_mask, K)
            ind = (scores[mask][:, None] > thresholds[None, :]).astype(float)
            out[mask] = ind @ w
        # numerical safety
        out = np.clip(out, 0.0, 1.0)
        return out

    def predict(self, scores, a) -> np.ndarray:
        """Sample a fair binary prediction using the learned randomized rule."""
        p = self.predict_proba(scores, a)
        rng = np.random.default_rng(self.random_state)
        return (rng.random(size=p.shape[0]) < p).astype(int)

    def explain(self) -> Dict[str, Any]:
        """Return a compact explanation of learned per-group randomization."""
        if self.thresholds_ is None or self.weights_ is None or self.groups_ is None:
            raise RuntimeError("Call fit() first.")
        info = {"method": self.method, "cost_fp": self.cost_fp, "cost_fn": self.cost_fn, "n_thresholds": self.n_thresholds}
        groups_info = {}
        thr = self.thresholds_
        for g, w in self.weights_.items():
            # show only non-trivial weights
            nz = np.where(w > 1e-6)[0]
            groups_info[g] = [{"threshold": float(thr[i]), "weight": float(w[i])} for i in nz]
        info["threshold_mixtures"] = groups_info
        return info


@dataclass
class BinaryPostProcessor:
    """
    Post-process an existing binary predictor Y_pred into a fair predictor \hat{Y}
    using randomized flipping depending only on (A, Y_pred).

    This is the direct implementation of Hardt et al.'s LP for a derived predictor
    from a binary predictor (4 variables for 2 groups; 2*G variables in general).
    """

    method: Method = "equalized_odds"
    cost_fp: float = 1.0
    cost_fn: float = 1.0
    random_state: Optional[int] = 0

    groups_: Optional[np.ndarray] = None
    flip_probs_: Optional[Dict[Hashable, Tuple[float, float]]] = None  # (p0, p1) for y_pred=0/1

    def fit(self, y_true, y_pred, a) -> "BinaryPostProcessor":
        y_true = _validate_binary(_to_numpy_1d(y_true, "y_true"), "y_true")
        y_pred = _validate_binary(_to_numpy_1d(y_pred, "y_pred"), "y_pred")
        a = _to_numpy_1d(a, "a")

        groups = pd.unique(a)
        groups = np.array(list(groups), dtype=object)
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups in sensitive attribute a.")
        G = len(groups)

        # counts per group for each (y_true, y_pred)
        # For each group g:
        #   among negatives (Y=0): n00 = count(Y=0, pred=0), n01 = count(Y=0, pred=1)
        #   among positives (Y=1): n10 = count(Y=1, pred=0), n11 = count(Y=1, pred=1)
        n00 = np.zeros(G, dtype=float)
        n01 = np.zeros(G, dtype=float)
        n10 = np.zeros(G, dtype=float)
        n11 = np.zeros(G, dtype=float)
        pos = np.zeros(G, dtype=float)
        neg = np.zeros(G, dtype=float)

        for gi, g in enumerate(groups):
            m = (a == g)
            yt = y_true[m]
            yp = y_pred[m]
            n00[gi] = float(np.sum((yt == 0) & (yp == 0)))
            n01[gi] = float(np.sum((yt == 0) & (yp == 1)))
            n10[gi] = float(np.sum((yt == 1) & (yp == 0)))
            n11[gi] = float(np.sum((yt == 1) & (yp == 1)))
            pos[gi] = float(np.sum(yt == 1))
            neg[gi] = float(np.sum(yt == 0))
            if pos[gi] == 0 or neg[gi] == 0:
                raise ValueError(f"Group {g!r} has pos={pos[gi]} neg={neg[gi]} — cannot enforce constraints.")

        # Variables: p_{g,0} = P(out=1 | group=g, y_pred=0)
        #           p_{g,1} = P(out=1 | group=g, y_pred=1)
        # Length = 2*G, bounds [0,1]
        # Expected confusion per group:
        #   FP_g = p_g0*n00 + p_g1*n01
        #   TP_g = p_g0*n10 + p_g1*n11
        #   FN_g = (1-p_g0)*n10 + (1-p_g1)*n11 = pos_g - TP_g
        # Objective: sum_g (cost_fp*FP_g + cost_fn*FN_g)
        c = np.zeros(2 * G, dtype=float)
        for gi in range(G):
            # cost_fp*FP_g + cost_fn*(pos_g - TP_g)
            # = cost_fp*(p0*n00 + p1*n01) + cost_fn*(pos - (p0*n10 + p1*n11))
            # coefficients on p0, p1:
            c[2 * gi + 0] = self.cost_fp * n00[gi] - self.cost_fn * n10[gi]
            c[2 * gi + 1] = self.cost_fp * n01[gi] - self.cost_fn * n11[gi]
            # constant part cost_fn*pos ignored

        A_eq = []
        b_eq = []
        ref = 0
        for gi in range(1, G):
            if self.method == "equalized_odds":
                # neg[ref]*FP_g - neg[g]*FP_ref = 0
                row = np.zeros(2 * G, dtype=float)
                row[2 * gi + 0] = neg[ref] * n00[gi]
                row[2 * gi + 1] = neg[ref] * n01[gi]
                row[2 * ref + 0] = -neg[gi] * n00[ref]
                row[2 * ref + 1] = -neg[gi] * n01[ref]
                A_eq.append(row); b_eq.append(0.0)

            # pos[ref]*TP_g - pos[g]*TP_ref = 0
            row = np.zeros(2 * G, dtype=float)
            row[2 * gi + 0] = pos[ref] * n10[gi]
            row[2 * gi + 1] = pos[ref] * n11[gi]
            row[2 * ref + 0] = -pos[gi] * n10[ref]
            row[2 * ref + 1] = -pos[gi] * n11[ref]
            A_eq.append(row); b_eq.append(0.0)

        A_eq = np.vstack(A_eq) if A_eq else None
        b_eq = np.asarray(b_eq, dtype=float) if b_eq else None
        bounds = [(0.0, 1.0)] * (2 * G)

        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"Linear program failed: {res.message}")

        x = res.x
        flip_probs = {}
        for gi, g in enumerate(groups):
            p0 = float(x[2 * gi + 0])
            p1 = float(x[2 * gi + 1])
            flip_probs[g] = (p0, p1)

        self.groups_ = groups
        self.flip_probs_ = flip_probs
        return self

    def predict_proba(self, y_pred, a) -> np.ndarray:
        if self.flip_probs_ is None:
            raise RuntimeError("Call fit() first.")
        y_pred = _validate_binary(_to_numpy_1d(y_pred, "y_pred"), "y_pred")
        a = _to_numpy_1d(a, "a")
        out = np.zeros_like(y_pred, dtype=float)
        for g in pd.unique(a):
            if g not in self.flip_probs_:
                raise ValueError(f"Unknown group {g!r} at predict time; seen groups={list(self.flip_probs_.keys())}")
        for g, (p0, p1) in self.flip_probs_.items():
            m = (a == g)
            if not np.any(m):
                continue
            out[m] = np.where(y_pred[m] == 0, p0, p1)
        return np.clip(out, 0.0, 1.0)

    def predict(self, y_pred, a) -> np.ndarray:
        p = self.predict_proba(y_pred, a)
        rng = np.random.default_rng(self.random_state)
        return (rng.random(size=p.shape[0]) < p).astype(int)

    def explain(self) -> Dict[str, Any]:
        if self.flip_probs_ is None:
            raise RuntimeError("Call fit() first.")
        return {
            "method": self.method,
            "cost_fp": self.cost_fp,
            "cost_fn": self.cost_fn,
            "flip_probs": {str(g): {"P(out=1|pred=0)": p0, "P(out=1|pred=1)": p1} for g, (p0, p1) in self.flip_probs_.items()},
        }
