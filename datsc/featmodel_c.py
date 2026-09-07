"""Approach C feature model: standardised descriptors + striatal-patch PCA scores -> a logistic on [linear features + random
Fourier features] (RBF-kernel logistic with a linear pass-through) or a small MLP, both serialised to npz and evaluated with numpy only (no pickle in the zip).
Keeps the typicality (Mahalanobis) statistics of the training PCA scores like the run-4 feature model."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class FeatModelC:
    def __init__(self, kind: str = "rff", n_comp: int = 40, C: float = 0.3, gamma: float = 0.5, n_rff: int = 512,
                 hidden=(64, 32), alpha: float = 1e-2, seed: int = 0):
        assert kind in ("rff", "mlp", "linear")
        self.kind, self.n_comp, self.C, self.gamma, self.n_rff = kind, int(n_comp), float(C), float(gamma), int(n_rff)
        self.hidden, self.alpha, self.seed = tuple(int(h) for h in hidden), float(alpha), int(seed)
        self.patch_mean = self.components = None            # PCA of the striatal patch
        self.mean = self.scale = None                        # standardiser over [features, pca scores]
        self.W = self.b = None                               # rff projection
        self.coef = None; self.intercept = 0.0               # logistic weights (rff / linear)
        self.layers = None                                   # mlp: list of (W, b)
        self.score_mean = self.score_cov_inv = None          # typicality statistics

    # ---- fitting (sklearn only here) ----
    def _pca(self, P):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.n_comp, svd_solver="randomized", random_state=self.seed).fit(P)
        self.patch_mean, self.components = pca.mean_.astype(np.float32), pca.components_.astype(np.float32)
        sc = (P - self.patch_mean) @ self.components.T
        self.score_mean = sc.mean(0).astype(np.float64)
        self.score_cov_inv = np.linalg.inv(np.cov(sc.astype(np.float64), rowvar=False) + 1e-3 * np.eye(sc.shape[1]))

    def _design(self, F, P):
        sc = (np.asarray(P, np.float32) - self.patch_mean) @ self.components.T
        return np.concatenate([np.asarray(F, np.float64), sc.astype(np.float64)], axis=1)

    def _phi(self, Z):
        if self.kind == "rff":                            # linear pass-through + random Fourier features (RBF kernel approx.)
            return np.concatenate([Z, np.sqrt(2.0 / self.n_rff) * np.cos(Z @ self.W + self.b)], axis=1)
        return Z

    def fit(self, F: np.ndarray, P: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        F, P = np.asarray(F, np.float64), np.asarray(P, np.float32)
        self._pca(P)
        X = self._design(F, P)
        self.mean = X.mean(0); self.scale = X.std(0) + 1e-6
        Z = (X - self.mean) / self.scale
        rng = np.random.default_rng(self.seed)
        if self.kind == "rff":
            self.W = (rng.normal(0, 1, (Z.shape[1], self.n_rff)) * np.sqrt(2 * self.gamma / Z.shape[1])).astype(np.float64)
            self.b = rng.uniform(0, 2 * np.pi, self.n_rff)
        if self.kind in ("rff", "linear"):
            clf = LogisticRegression(C=self.C, max_iter=5000).fit(self._phi(Z), y)
            self.coef = clf.coef_.ravel(); self.intercept = float(clf.intercept_[0])
        else:
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(hidden_layer_sizes=self.hidden, activation="tanh", alpha=self.alpha, max_iter=600,
                                random_state=self.seed, early_stopping=False).fit(Z, y)
            self.layers = [(w.astype(np.float64), b.astype(np.float64)) for w, b in zip(clf.coefs_, clf.intercepts_)]
        return self

    # ---- numpy inference ----
    def decision(self, F, P):
        Z = (self._design(F, P) - self.mean) / self.scale
        if self.kind == "mlp":
            h = Z
            for k, (w, b) in enumerate(self.layers):
                h = h @ w + b
                if k < len(self.layers) - 1:
                    h = np.tanh(h)
            return h.ravel()
        return self._phi(Z) @ self.coef + self.intercept

    def predict_proba(self, F, P):
        return 1.0 / (1.0 + np.exp(-self.decision(F, P)))

    def typicality(self, P):
        sc = (np.asarray(P, np.float32) - self.patch_mean) @ self.components.T - self.score_mean
        d2 = np.einsum("ij,jk,ik->i", sc, self.score_cov_inv, sc)
        return np.sqrt(np.maximum(d2, 0) / sc.shape[1])

    # ---- persistence ----
    def save(self, path):
        d = dict(kind=np.array(self.kind), n_comp=self.n_comp, C=self.C, gamma=self.gamma, n_rff=self.n_rff, hidden=np.array(self.hidden), alpha=self.alpha, seed=self.seed,
                 patch_mean=self.patch_mean, components=self.components, mean=self.mean, scale=self.scale,
                 score_mean=self.score_mean, score_cov_inv=self.score_cov_inv)
        if self.kind == "mlp":
            for k, (w, b) in enumerate(self.layers):
                d[f"W{k}"] = w; d[f"b{k}"] = b
            d["n_layers"] = len(self.layers)
        else:
            d["coef"] = self.coef; d["intercept"] = self.intercept
            if self.kind == "rff":
                d["W"] = self.W; d["b"] = self.b
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path):
        d = np.load(Path(path))
        m = cls(str(d["kind"]), int(d["n_comp"]), float(d["C"]), float(d["gamma"]), int(d["n_rff"]), tuple(int(h) for h in d["hidden"]), float(d["alpha"]), int(d["seed"]))
        m.patch_mean, m.components, m.mean, m.scale = d["patch_mean"], d["components"], d["mean"], d["scale"]
        m.score_mean, m.score_cov_inv = d["score_mean"], d["score_cov_inv"]
        if m.kind == "mlp":
            m.layers = [(d[f"W{k}"], d[f"b{k}"]) for k in range(int(d["n_layers"]))]
        else:
            m.coef, m.intercept = d["coef"], float(d["intercept"])
            if m.kind == "rff":
                m.W, m.b = d["W"], d["b"]
        return m
