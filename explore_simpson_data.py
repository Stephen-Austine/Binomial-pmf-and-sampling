"""Explore ``simpson_data.mat``: binomial PMFs and a 1,000-point sample.

The workspace contains
  * ``n1, n2, n3`` = 10, 25, 50 and ``p`` = 0.8 (binomial parameters),
  * ``x1, x2, x3`` (ranges 0..n) and ``y1, y2, y3`` (the PMF values at those x),
  * ``data`` - a 1000 x 1 vector of sampled values.

This script confirms the PMFs against ``scipy.stats.binom`` and summarises the
sample.

    python explore_simpson_data.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from octave_text import load_octave_text

HERE = Path(__file__).parent


def main() -> None:
    w = load_octave_text(HERE / "data/simpson_data.mat")
    p = w["p"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, colour in zip((1, 2, 3), ("#1f4e79", "#2e8b57", "#c0392b"), strict=False):
        n, x, y = int(w[f"n{k}"]), w[f"x{k}"], w[f"y{k}"].ravel()
        ok = np.allclose(y, stats.binom.pmf(x, n, p))
        print(
            f"Binomial(n={n}, p={p}): stored PMF matches scipy -> {ok}; "
            f"sum = {y.sum():.6f}"
        )
        ax1.plot(x, y, "o-", ms=3, color=colour, label=f"n = {n}")
    ax1.set(title=f"Binomial PMF, p = {p}", xlabel="k (successes)", ylabel="P(X = k)")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)

    d = w["data"].ravel()
    mean, sd = d.mean(), d.std(ddof=1)
    print(
        f"\nsample: n = {d.size}, mean = {mean:.3f}, sd = {sd:.3f}, "
        f"min = {d.min():.2f}, max = {d.max():.2f}"
    )
    print(f"skewness = {stats.skew(d):.3f}, excess kurtosis = {stats.kurtosis(d):.3f}")
    print(f"D'Agostino normality test p-value = {stats.normaltest(d).pvalue:.3f}")

    ax2.hist(
        d,
        bins=35,
        density=True,
        color="#9bb7d4",
        edgecolor="white",
        label="sample (n = 1000)",
    )
    grid = np.linspace(d.min(), d.max(), 300)
    ax2.plot(
        grid,
        stats.norm.pdf(grid, mean, sd),
        color="#c0392b",
        lw=2,
        label=f"Normal(μ={mean:.2f}, σ={sd:.2f})",
    )
    ax2.set(
        title="Sampled data vs. fitted normal density", xlabel="value", ylabel="density"
    )
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = HERE / "figures/binomial_and_sample.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
