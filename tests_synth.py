"""Synthetic DaT-like volumes (no pytest deps) - used by tests/conftest.py and by main.py --selftest."""
import numpy as np
import nibabel as nib


def make_synthetic_volume(shape=(96, 96, 60), spacing=(2.46, 2.46, 2.46), levels=None,
                          seed=0, striatum_amp=4.0, asym=0.0, scale=1000.0):
    """Synthetic DaT-like head: ellipsoid background + two bright striatal blobs.

    asym in [0,1]: fraction by which the RIGHT blob is dimmed (simulates pathology).
    levels: if set, quantise to this many integer grey levels (ultra-low-count regime).
    Returns (array[x,y,z] float32, spacing).
    """
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    c = np.array(shape) / 2.0
    r = np.array(shape) / 2.6
    ell = (((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2) <= 1.0
    vol = np.where(ell, 1.0, 0.0) + rng.normal(0, 0.08, size=shape)
    sig_vox = 6.0 / np.array(spacing)  # 6 mm blobs
    off_vox = 9.0 / spacing[0]         # blobs 9 mm left/right of midline
    for side, amp in ((-1, striatum_amp), (+1, striatum_amp * (1.0 - asym))):
        cx = c[0] + side * off_vox
        blob = np.exp(-(((x - cx) / sig_vox[0]) ** 2 + ((y - c[1]) / sig_vox[1]) ** 2
                        + ((z - c[2]) / sig_vox[2]) ** 2) / 2.0)
        vol += amp * blob
    vol = np.clip(vol, 0, None)
    if levels is not None:
        vol = np.round(vol / vol.max() * (levels - 1))
    else:
        vol = vol * scale
    return vol.astype(np.float32), tuple(float(s) for s in spacing)


def write_nifti(path, vol, spacing):
    affine = np.diag(list(spacing) + [1.0])
    nib.save(nib.Nifti1Image(vol, affine), str(path))
    return path
