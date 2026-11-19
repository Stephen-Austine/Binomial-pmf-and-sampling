from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from octave_text import load_octave_text

DATA = Path(__file__).resolve().parents[1] / "data" / "simpson_data.mat"


def test_scalars_and_ranges():
    v = load_octave_text(DATA)
    assert v["data"].shape == (1000, 1)
    assert v["p"] == pytest.approx(0.8)
    assert (v["n1"], v["n2"], v["n3"]) == (10, 25, 50)
    assert np.array_equal(v["x3"], np.arange(51))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_stored_pmf_matches_scipy(k):
    v = load_octave_text(DATA)
    n, x, y = int(v[f"n{k}"]), v[f"x{k}"], v[f"y{k}"].ravel()
    assert np.allclose(y, stats.binom.pmf(x, n, v["p"]))
    assert y.sum() == pytest.approx(1.0)


def test_sample_summary():
    d = load_octave_text(DATA)["data"].ravel()
    assert d.mean() == pytest.approx(2.965, abs=1e-3)
    assert d.std(ddof=1) == pytest.approx(2.033, abs=1e-3)


def test_unsupported_type_raises(tmp_path):
    bad = tmp_path / "bad.mat"
    bad.write_text("# name: s\n# type: string\n# elements: 1\nhello\n")
    with pytest.raises(ValueError):
        load_octave_text(bad)
