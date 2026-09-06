"""PCA (striatal patch) + standardised hand-crafted features -> L2 logistic regression; numpy-only inference, npz on disk."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class FeatModel:
    def __init__(self, n_comp: int = 40, C: float = 0.1, seed: int = 0):
        self.n_comp, self.C, self.seed = int(n_comp), float(C), int(seed)
        self.patch_mean = self.components = None          # PCA of the striatal patch
        self.mean = self.scale = self.coef = None           # standardiser + logistic weights over [features, pca scores]
        self.intercept = 0.0
        self.score_mean = self.score_cov_inv = None         # typicality: Mahalanobis statistics of the training PCA scores

    # ---- fitting (sklearn only here) ----
    def fit(self, F: np.ndarray, P: np.ndarray, y: np.ndarray):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        F, P = np.asarray(F, np.float64), np.asarray(P, np.float32)
        pca = PCA(n_components=self.n_comp, svd_solver="randomized", random_state=self.seed).fit(P)
        self.patch_mean, self.components = pca.mean_.astype(np.float32), pca.components_.astype(np.float32)
        sc = (P - self.patch_mean) @ self.components.T
        self.score_mean = sc.mean(0).astype(np.float64)
        cov = np.cov(sc.astype(np.float64), rowvar=False) + 1e-3 * np.eye(sc.shape[1])
        self.score_cov_inv = np.linalg.inv(cov)
        X = self._design(F, P)
        self.mean = X.mean(0); self.scale = X.std(0) + 1e-6
        clf = LogisticRegression(C=self.C, max_iter=5000).fit((X - self.mean) / self.scale, y)
        self.coef = clf.coef_.ravel(); self.intercept = float(clf.intercept_[0])
        return self

    def _design(self, F, P):
        scores = (np.asarray(P, np.float32) - self.patch_mean) @ self.components.T
        return np.concatenate([np.asarray(F, np.float64), scores.astype(np.float64)], axis=1)

    def decision(self, F, P):
        Z = (self._design(F, P) - self.mean) / self.scale
        return Z @ self.coef + self.intercept

    def predict_proba(self, F, P):
        return 1.0 / (1.0 + np.exp(-self.decision(F, P)))

    def typicality(self, P):
        """Mahalanobis distance of each scan's PCA scores to the training distribution, divided by sqrt(n_comp)
        (so a typical training scan is ~1). Training statistics only: per-sample, never transductive."""
        if self.score_mean is None:
            return np.full(len(P), np.nan)
        sc = (np.asarray(P, np.float32) - self.patch_mean) @ self.components.T - self.score_mean
        d2 = np.einsum("ij,jk,ik->i", sc, self.score_cov_inv, sc)
        return np.sqrt(np.maximum(d2, 0) / sc.shape[1])

    # ---- persistence ----
    def save(self, path):
        np.savez_compressed(path, n_comp=self.n_comp, C=self.C, seed=self.seed, patch_mean=self.patch_mean, components=self.components,
                            mean=self.mean, scale=self.scale, coef=self.coef, intercept=self.intercept,
                            **({} if self.score_mean is None else {"score_mean": self.score_mean, "score_cov_inv": self.score_cov_inv}))

    @classmethod
    def load(cls, path):
        d = np.load(Path(path))
        m = cls(int(d["n_comp"]), float(d["C"]), int(d["seed"]))
        m.patch_mean, m.components = d["patch_mean"], d["components"]
        m.mean, m.scale, m.coef, m.intercept = d["mean"], d["scale"], d["coef"], float(d["intercept"])
        if "score_mean" in d.files:
            m.score_mean, m.score_cov_inv = d["score_mean"], d["score_cov_inv"]
        return m
