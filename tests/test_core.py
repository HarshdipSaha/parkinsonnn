"""Unit tests for config, io, preprocess, features, folds, lrmodel, model, augment, calibrate, train."""
import json
import numpy as np
import pandas as pd
import pytest
import torch

from datsc import augment as ag, calibrate as cb, config, features as ft, folds, io, preprocess as pp
from datsc.config import CROP_SHAPE, N_SCALARS
from datsc.lrmodel import LRModel
from datsc.model import SmallResNet3D, count_params
from datsc.train import train_fold


# ---------------- config ----------------
def test_constants():
    assert config.SPACING_MM == 3.0 and config.CROP_SHAPE == (72, 72, 48) and config.N_SCALARS == 8


def test_find_data_dir_prefers_env(tmp_path, monkeypatch):
    (tmp_path / "train_labels_x.csv").write_text("uid,is_pathologic\nabc,1.0\n")
    (tmp_path / "niftis").mkdir()
    (tmp_path / "niftis" / "abc.nii").write_bytes(b"")
    monkeypatch.setenv("DATSC_DATA_DIR", str(tmp_path))
    d = config.find_data_dir()
    assert d.root == tmp_path and d.labels_csv.name == "train_labels_x.csv" and d.niftis_dir == tmp_path / "niftis"


def test_find_nifti_path_handles_both_extensions(tmp_path):
    (tmp_path / "u1.nii").write_bytes(b""); (tmp_path / "u2.nii.gz").write_bytes(b"")
    assert config.find_nifti_path(tmp_path, "u1").name == "u1.nii"
    assert config.find_nifti_path(tmp_path, "u2").name == "u2.nii.gz"
    assert config.find_nifti_path(tmp_path, "u3") is None


# ---------------- io ----------------
def test_load_canonical(synth_nifti):
    vol, spacing, native_shape = io.load_canonical(synth_nifti(shape=(64, 64, 40)))
    assert vol.dtype == np.float32 and vol.shape == (64, 64, 40) and native_shape == (64, 64, 40)
    assert np.allclose(spacing, (2.46, 2.46, 2.46))


def test_resample_iso(synth):
    vol, sp = synth(shape=(60, 60, 30), spacing=(3.9, 3.9, 3.9))
    out = io.resample_iso(vol, sp, target_mm=3.0)
    assert out.shape == (78, 78, 39) and out.dtype == np.float32 and out.max() > 0


def test_load_iso(synth_nifti):
    vol, sp = io.load_iso(synth_nifti(shape=(48, 48, 24), spacing=(2.0, 2.0, 4.0)))
    assert vol.shape == (32, 32, 32) and sp == (2.0, 2.0, 4.0)


# ---------------- preprocess ----------------
def _iso(synth, **kw):
    vol, sp = synth(**kw)
    return io.resample_iso(vol, sp), sp


def test_brain_mask(synth):
    vol, _ = _iso(synth)
    m = pp.brain_mask(vol)
    assert m.dtype == bool and 0.15 < m.mean() < 0.6 and pp._largest_component(m).sum() == m.sum()


def test_brain_mask_ultra_low(synth):
    vol, _ = _iso(synth, levels=8)
    assert 0.1 < pp.brain_mask(vol).mean() < 0.7


def test_normalize_p75(synth):
    vol, _ = _iso(synth)
    m = pp.brain_mask(vol)
    out, p75 = pp.normalize_p75(vol, m)
    assert p75 > 0 and np.isclose(np.percentile(out[m], 75), 1.0, atol=1e-4)


def test_normalize_fallback():
    vol = np.zeros((20, 20, 20), np.float32); vol[10, 10, 10] = 5.0
    out, p75 = pp.normalize_p75(vol, np.ones_like(vol, bool))
    assert p75 > 0 and np.isfinite(out).all()


def test_anchor_between_blobs(synth):
    vol, _ = _iso(synth, shape=(96, 96, 60))
    m = pp.brain_mask(vol); norm, _ = pp.normalize_p75(vol, m)
    (cx, cy, cz), fallback = pp.find_anchor(norm, m)
    c = np.array(norm.shape) / 2
    assert not fallback and abs(cx - c[0]) < 3 and abs(cy - c[1]) < 3 and abs(cz - c[2]) < 3


def test_anchor_fallback_flat():
    vol = np.ones((60, 60, 40), np.float32)
    (cx, cy, cz), fallback = pp.find_anchor(vol, np.ones_like(vol, bool))
    assert fallback and 0 <= cx < 60 and 0 <= cy < 60 and 0 <= cz < 40


def test_crop_zero_pad():
    crop = pp.crop_at(np.ones((30, 30, 20), np.float32), (2, 2, 2))
    assert crop.shape == CROP_SHAPE and crop[0, 0, 0] == 0.0 and crop[36, 36, 24] == 1.0


@pytest.mark.parametrize("kw", [{}, {"levels": 8}, {"shape": (128, 128, 33), "spacing": (3.9,) * 3},
                                {"shape": (128, 128, 128), "spacing": (2.3,) * 3, "scale": 60000.0}])
def test_preprocess_end_to_end(synth_nifti, kw):
    out = pp.preprocess(synth_nifti(**kw))
    assert out["crop"].shape == CROP_SHAPE and out["crop"].dtype == np.float16 and out["crop_com"].shape == CROP_SHAPE
    assert out["scalars"].shape == (N_SCALARS,) and np.isfinite(out["scalars"]).all() and out["crop"].max() > 0


def test_preprocess_deterministic(synth_nifti):
    p = synth_nifti(); a, b = pp.preprocess(p), pp.preprocess(p)
    assert np.array_equal(a["crop"], b["crop"]) and np.array_equal(a["scalars"], b["scalars"])


# ---------------- features ----------------
def test_radiomics_shape(synth_nifti):
    f = ft.hotspot_radiomics(pp.preprocess(synth_nifti())["crop"].astype(np.float32))
    assert f.shape == (ft.N_RADIOMICS,) and np.isfinite(f).all()


def test_radiomics_asymmetry(synth_nifti):
    sym = ft.hotspot_radiomics(pp.preprocess(synth_nifti("s1"))["crop"].astype(np.float32))
    asym = ft.hotspot_radiomics(pp.preprocess(synth_nifti("s2", asym=0.6))["crop"].astype(np.float32))
    assert asym[ft.RADIOMICS_NAMES.index("asym_mean")] > sym[ft.RADIOMICS_NAMES.index("asym_mean")] + 0.1


def test_radiomics_dim_striatum(synth_nifti):
    hi = ft.hotspot_radiomics(pp.preprocess(synth_nifti("h", striatum_amp=4.0))["crop"].astype(np.float32))
    lo = ft.hotspot_radiomics(pp.preprocess(synth_nifti("l", striatum_amp=1.5))["crop"].astype(np.float32))
    assert lo[ft.RADIOMICS_NAMES.index("min_side_p90")] < hi[ft.RADIOMICS_NAMES.index("min_side_p90")]


# ---------------- folds ----------------
def test_near_duplicates_same_group():
    rng = np.random.default_rng(0)
    base = rng.random((4, 72, 72, 48)).astype(np.float32)
    crops = np.concatenate([base, base[:1] + 1e-3 * rng.random((1, 72, 72, 48)).astype(np.float32)])
    emb = folds.embed(crops)
    assert emb.shape == (5, 16 ** 3)
    g = folds.group_near_duplicates(emb, thresh=0.98)
    assert g[0] == g[4] and len(set(g[:4])) == 4


def test_stratified_group_folds():
    y = np.array([0, 1] * 50); groups = np.arange(100); groups[1] = 0
    f = folds.assign_folds(y, groups, n_splits=5, seed=0)
    assert set(f) == set(range(5)) and f[0] == f[1]
    for k in range(5):
        assert 0.3 < y[f == k].mean() < 0.7


# ---------------- lrmodel ----------------
def test_lr_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6)); y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    m = LRModel(C=1.0).fit(X, y); p = m.predict_proba(X)
    assert ((p > 0.5) == y).mean() > 0.9
    m.save(tmp_path / "lr.json"); m2 = LRModel.load(tmp_path / "lr.json")
    assert np.allclose(m2.predict_proba(X), p) and json.load(open(tmp_path / "lr.json"))["names"] == m.names


# ---------------- model ----------------
def test_model_shape_and_params():
    m = SmallResNet3D(n_scalars=8)
    assert m(torch.zeros(2, 1, 72, 72, 48), torch.zeros(2, 8)).shape == (2,)
    assert 0.6e6 < count_params(m) < 1.3e6


# ---------------- augment ----------------
def test_augment_ops():
    torch.manual_seed(0); x = torch.rand(3, 1, 72, 72, 48)
    for fn in (ag.random_blur, ag.random_noise, ag.random_quantize, ag.random_scale_jitter, ag.random_affine, ag.random_flip):
        y = fn(x.clone()); assert y.shape == x.shape and torch.isfinite(y).all(), fn.__name__
    assert torch.unique(ag.quantize(x, k=8)[0]).numel() <= 9
    assert torch.equal(ag.random_flip(x, p=1.0), x.flip(dims=[2]))
    assert ag.train_augment(x).shape == x.shape


# ---------------- calibrate ----------------
def test_temperature():
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 2000)
    logit = (y * 2 - 1) * 0.5 + rng.normal(0, 1.0, 2000); over = logit * 3.0
    T = cb.fit_temperature(over, y)
    assert 2.0 < T < 4.5 and cb.logloss(y, cb.sigmoid(over / T)) < cb.logloss(y, cb.sigmoid(over))


def test_blend_weight():
    rng = np.random.default_rng(1); y = rng.integers(0, 2, 1000)
    good = np.clip(y * 0.8 + 0.1 + rng.normal(0, 0.05, 1000), 0.01, 0.99); bad = np.clip(rng.random(1000), 0.01, 0.99)
    assert cb.fit_blend_weight(good, bad, y) >= 0.8


def test_calibration_roundtrip(tmp_path):
    c = cb.Calibration(w=0.7, T=1.5, lam=0.02, pi=0.55)
    p = c.apply(np.array([0.9, 0.1]), np.array([0.8, 0.2]))
    assert (p > 0).all() and (p < 1).all() and p[0] > p[1]
    c.save(tmp_path / "c.json"); assert cb.Calibration.load(tmp_path / "c.json") == c


# ---------------- train ----------------
def test_train_fold(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir(); rng = np.random.default_rng(0); rows = []
    for i in range(12):
        y = i % 2; crop = (rng.random((72, 72, 48)) + y * 0.5).astype(np.float16)
        np.savez(cache / f"u{i}.npz", crop=crop, crop_com=crop, scalars=rng.random(8).astype(np.float32),
                 radiomics=rng.random(14).astype(np.float32), anchor=np.zeros(3, np.float32))
        rows.append({"uid": f"u{i}", "is_pathologic": float(y), "group": i, "fold": i % 3})
    pd.DataFrame(rows).to_csv(cache / "folds.csv", index=False)
    out = tmp_path / "art"
    res = train_fold(cache, fold=0, seed=1, out_dir=out, epochs=1, batch_size=4, device="cpu", num_workers=0)
    assert (out / "cnn_f0_s1.pt").exists() and len(pd.read_csv(out / "oof_f0_s1.csv")) == 4
    assert 0 <= res["best_val_logloss"] < 5 and json.load(open(out / "cnn_f0_s1.json"))["fold"] == 0
