"""Exactly three fitted numbers: blend weight w, temperature T, prior-mix lambda (plus the prior pi)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-4


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_blend_weight(p_cnn, p_lr, y, grid=np.linspace(0, 1, 21)):
    return float(min(grid, key=lambda w: logloss(y, w * p_cnn + (1 - w) * p_lr)))


def fit_temperature(z, y):
    res = minimize_scalar(lambda T: logloss(y, sigmoid(z / T)), bounds=(0.25, 10.0), method="bounded")
    return float(res.x)


@dataclass(frozen=True)
class Calibration:
    w: float
    T: float
    lam: float
    pi: float

    def apply(self, p_cnn, p_lr):
        p_cnn = np.asarray(p_cnn, float); p_lr = np.asarray(p_lr, float)
        blend = self.w * p_cnn + (1 - self.w) * p_lr
        cal = sigmoid(logit(blend) / self.T)
        return np.clip((1 - self.lam) * cal + self.lam * self.pi, EPS, 1 - EPS)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def fit(p_cnn, p_lr, y, lam: float, pi: float) -> Calibration:
    w = fit_blend_weight(p_cnn, p_lr, y)
    T = fit_temperature(logit(w * p_cnn + (1 - w) * p_lr), y)
    return Calibration(w=w, T=T, lam=lam, pi=pi)
