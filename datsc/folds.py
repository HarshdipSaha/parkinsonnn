"""Repeat-patient audit via coarse embeddings, then stratified group K-fold."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedGroupKFold


def embed(crops: np.ndarray, size: int = 16) -> np.ndarray:
    """crops [N, X, Y, Z] -> L2-normalised [N, size^3] embeddings."""
    out = []
    for c in crops:
        f = [size / s for s in c.shape]
        e = zoom(c.astype(np.float32), f, order=1).ravel()
        e = e - e.mean()
        out.append(e / (np.linalg.norm(e) + 1e-9))
    return np.stack(out)


def group_near_duplicates(emb: np.ndarray, thresh: float = 0.98) -> np.ndarray:
    n = len(emb)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sim = emb @ emb.T
    ii, jj = np.where(np.triu(sim, k=1) > thresh)
    for i, j in zip(ii, jj):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def assign_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(y), -1, dtype=int)
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
        folds[val_idx] = k
    assert (folds >= 0).all()
    return folds
