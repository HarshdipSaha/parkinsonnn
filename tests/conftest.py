import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests_synth import make_synthetic_volume, write_nifti  # noqa: E402


@pytest.fixture
def synth():
    return make_synthetic_volume


@pytest.fixture
def synth_nifti(tmp_path):
    def _make(name="abc12345", **kw):
        vol, sp = make_synthetic_volume(**kw)
        return write_nifti(tmp_path / f"{name}.nii.gz", vol, sp)
    return _make
