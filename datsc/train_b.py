"""Run-6b input spec: tri-planar view tensor for `TriPlanarNet`, drop-in for the run-3 trainer/inference contract.

`InputSpecB(x, dz)` -> [B, V, n_slabs, hw, hw]; `variants(x)` yields (identity + affine boxes) x z-shift x mirror exactly like
`InputSpec`, and `flip()` is the anatomically correct mirror: axial/coronal images flip their X axis, the two sagittal stacks
swap (the right stack is the left window taken on the mirrored volume, so the swap is exact). `make_spec(cfg)` builds the right spec class from a manifest/meta dict (arm "tri" -> InputSpecB, else InputSpec).
"""
from __future__ import annotations

import torch

from .data_v3 import to_slabs
from .data_b import to_slabs_sagittal
from .train_v3 import InputSpec

DEFAULT_VIEWS = ("axial", "coronal", "sagL", "sagR")


class InputSpecB:
    arm = "tri"

    def __init__(self, arm: str = "tri", n_slabs: int = 3, slab_mm: float = 8.0, hw: int = 160, slab_mode: str = "mean",
                 z_center: float = 33.5, y_center: float = 57.3, x_left: float = 36.8, x_right: float = 58.2,
                 views=DEFAULT_VIEWS, tta_z=(0,), crop=None, tta_affine: int = 0, **_ignored):
        assert arm == "tri"
        self.n_slabs, self.slab_mm, self.hw, self.slab_mode = int(n_slabs), float(slab_mm), int(hw), slab_mode
        self.z_center, self.y_center, self.x_left, self.x_right = float(z_center), float(y_center), float(x_left), float(x_right)
        self.views, self.tta_z, self.crop, self.tta_affine = tuple(views), tuple(tta_z), crop, int(tta_affine or 0)
        self._aff = InputSpec("3d", tta_affine=self.tta_affine)          # reuse the shared reproducible affine boxes

    # ---- one view ----
    def _view(self, x, name: str, dz: int):
        k = dict(n_slabs=self.n_slabs, slab_mm=self.slab_mm, hw=self.hw, mode=self.slab_mode, crop=self.crop)
        if name == "axial":
            return to_slabs(x, z_center=self.z_center + dz, plane="axial", **k)
        if name == "coronal":
            return to_slabs(x, z_center=self.y_center + dz, plane="coronal", **k)
        if name == "sagL":
            return to_slabs_sagittal(x, x_center=self.x_left, **k)
        if name == "sagR":                                              # the left window applied to the mirrored volume:
            return to_slabs_sagittal(x.flip(dims=[2]), x_center=self.x_left, **k)   # exact mirror of sagL, so flip == swap
        raise ValueError(name)

    def __call__(self, x, dz: int = 0):
        return torch.stack([self._view(x, v, dz) for v in self.views], dim=1)       # [B, V, C, hw, hw]

    def variants(self, x):
        boxes = [x] + (list(self._aff._affine_boxes(x)) if self.tta_affine else [])
        for b in boxes:
            for dz in self.tta_z:
                inp = self(b, dz)
                yield inp
                yield self.flip(inp)

    def flip(self, inp):
        """Mirror: axial/coronal views flip their last axis (X); sagittal-left <-> sagittal-right swap."""
        out = []
        for i, v in enumerate(self.views):
            if v == "sagL" and "sagR" in self.views:
                out.append(inp[:, self.views.index("sagR")])
            elif v == "sagR" and "sagL" in self.views:
                out.append(inp[:, self.views.index("sagL")])
            elif v in ("sagL", "sagR"):
                out.append(inp[:, i])                                              # lone sagittal view: unchanged
            else:
                out.append(inp[:, i].flip(dims=[-1]))
        return torch.stack(out, dim=1)

    def to_dict(self):
        return dict(arm="tri", n_slabs=self.n_slabs, slab_mm=self.slab_mm, hw=self.hw, slab_mode=self.slab_mode,
                    z_center=self.z_center, y_center=self.y_center, x_left=self.x_left, x_right=self.x_right,
                    views=list(self.views), tta_z=list(self.tta_z), crop=self.crop, tta_affine=self.tta_affine)


def make_spec(cfg: dict):
    """Spec object from a meta/manifest dict: arm 'tri' -> InputSpecB, '2d'/'3d' -> InputSpec."""
    cfg = dict(cfg)
    if cfg.get("arm") == "tri":
        return InputSpecB(**cfg)
    return InputSpec(**cfg)
