"""One-off: make datsc.models_v3.build_model dispatch 3D arms with model_kw['arch'] == 'se' to datsc.models_a (backward compatible)."""
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "datsc" / "models_v3.py"
s = p.read_text(encoding="utf-8")
old = '''def build_model(arm: str, n_scalars: int = 0, **kw) -> nn.Module:
    if arm == "3d":
        return ResNet3D(n_scalars=n_scalars, **{k: v for k, v in kw.items() if k in ("widths", "dropout", "stem_stride")})'''
new = '''def build_model(arm: str, n_scalars: int = 0, **kw) -> nn.Module:
    if arm == "3d":
        if kw.get("arch") == "se":                       # run 6a: SE / stochastic-depth striatal-crop family (datsc.models_a)
            from .models_a import build_3d_a
            return build_3d_a(n_scalars=n_scalars, **kw)
        return ResNet3D(n_scalars=n_scalars, **{k: v for k, v in kw.items() if k in ("widths", "dropout", "stem_stride")})'''
assert old in s, "build_model anchor not found"
if "arch" not in s.split("def build_model")[1]:
    p.write_text(s.replace(old, new, 1), encoding="utf-8"); print("build_model: arch dispatch added")
else:
    print("build_model: already patched")
