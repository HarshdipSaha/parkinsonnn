"""L2 logistic regression with standardisation, serialised to JSON (no pickle in the submission)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import SCALAR_NAMES
from .features import RADIOMICS_NAMES

DEFAULT_NAMES = list(RADIOMICS_NAMES) + list(SCALAR_NAMES)


class LRModel:
    def __init__(self, C: float = 1.0, names=None):
        self.C = C
        self.names = list(names) if names is not None else DEFAULT_NAMES
        self.mean = self.scale = self.coef = None
        self.intercept = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, np.float64)
        self.mean = X.mean(0); self.scale = X.std(0) + 1e-6
        clf = LogisticRegression(C=self.C, max_iter=2000).fit((X - self.mean) / self.scale, y)
        self.coef = clf.coef_.ravel(); self.intercept = float(clf.intercept_[0])
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, np.float64) - self.mean) / self.scale
        return Z @ self.coef + self.intercept

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision(X)))

    def save(self, path):
        Path(path).write_text(json.dumps({"C": self.C, "names": self.names, "mean": self.mean.tolist(),
                                          "scale": self.scale.tolist(), "coef": self.coef.tolist(),
                                          "intercept": self.intercept}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        m = cls(C=d["C"], names=d["names"])
        m.mean, m.scale = np.array(d["mean"]), np.array(d["scale"])
        m.coef, m.intercept = np.array(d["coef"]), float(d["intercept"])
        return m


def feature_matrix(cache_dir: Path, uids) -> np.ndarray:
    rows = []
    for u in uids:
        z = np.load(Path(cache_dir) / f"{u}.npz")
        rows.append(np.concatenate([z["radiomics"], z["scalars"]]))
    return np.stack(rows).astype(np.float32)
