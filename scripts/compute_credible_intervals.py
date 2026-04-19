#!/usr/bin/env python3
"""
Bayesian credible intervals for per-(system, environment, SNR) metric cells
using a Student-t likelihood with weakly-informative priors, sampled via
NUTS in NumPyro.

Why Student-t rather than Normal? Per-cell PESQ/CSIG/CBAK/COVL distributions
exhibit heavier-than-Normal tails, particularly at low SNR, where occasional
catastrophic-failure files (e.g. babble overlapping with silent speech
frames) pull the lower tail down. A Student-t likelihood with unknown
degrees-of-freedom parameter absorbs such outliers without requiring data
exclusion.

Usage:
    python scripts/compute_credible_intervals.py \\
        --results-csv results/VOICEBANK_DEMAND/<run-dir>/results.csv \\
        --out results/credible_intervals.csv [--num-samples 4000 --num-warmup 1000]

Output columns:
    system, noise, snr_db, metric, n,
    mean, ci_low, ci_high, nu_mean
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

METRICS = ["PESQ", "CSIG", "CBAK", "COVL"]


def student_t_model(y):
    """
    y_i | mu, sigma, nu ~ StudentT(nu, mu, sigma)
    mu     ~ Normal(2.5, 1.5)   # weakly centered on the metric mid-range
    sigma  ~ HalfNormal(1.0)    # weakly informative
    nu     ~ Gamma(2.0, 0.1)    # mean 20, heavy mass in [3, 60]
    """
    mu    = numpyro.sample("mu",    dist.Normal(2.5, 1.5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    nu    = numpyro.sample("nu",    dist.Gamma(2.0, 0.1))
    numpyro.sample("y", dist.StudentT(df=nu, loc=mu, scale=sigma), obs=y)


def fit_cell(y, num_warmup, num_samples, rng_key):
    """Run NUTS on one cell and return summary stats."""
    y = np.asarray(y, dtype=np.float32)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 2:
        return dict(n=n, mean=float(np.mean(y)) if n else np.nan,
                    ci_low=np.nan, ci_high=np.nan, nu_mean=np.nan)
    kernel = NUTS(student_t_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=1, progress_bar=False)
    mcmc.run(rng_key, y=jnp.asarray(y))
    samples = mcmc.get_samples()
    mu_samples = samples["mu"]
    return dict(
        n=int(n),
        mean=float(jnp.mean(mu_samples)),
        ci_low=float(jnp.percentile(mu_samples, 2.5)),
        ci_high=float(jnp.percentile(mu_samples, 97.5)),
        nu_mean=float(jnp.mean(samples["nu"])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", required=True,
                    help="Per-file results.csv with columns (file, noise, snr_db, system, <metric>...)")
    ap.add_argument("--out", required=True, help="Output path for credible intervals CSV.")
    ap.add_argument("--num-warmup",  type=int, default=1000)
    ap.add_argument("--num-samples", type=int, default=4000)
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Only compute system-level aggregate CIs (no env x snr breakdown).")
    args = ap.parse_args()

    df = pd.read_csv(args.results_csv)
    # Normalize column names.
    cols = {c.lower(): c for c in df.columns}
    required = {"system", "noise", "snr_db"}
    missing = required - set(cols.keys())
    if missing:
        sys.exit(f"Missing required columns {missing} in {args.results_csv}")
    metric_cols = [m for m in METRICS if m in df.columns]
    if not metric_cols:
        sys.exit(f"No metric columns found among {METRICS} in {args.results_csv}")

    rng = jax.random.PRNGKey(0)
    rows = []

    # Aggregate: per (system, metric) across all files.
    for sys_ in sorted(df["system"].unique()):
        for m in metric_cols:
            sub = df[df["system"] == sys_][m].dropna().values
            rng, sk = jax.random.split(rng)
            stats = fit_cell(sub, args.num_warmup, args.num_samples, sk)
            rows.append(dict(system=sys_, noise="ALL", snr_db=-1.0, metric=m, **stats))
            print(f"[agg] {sys_:15s} {m:5s} n={stats['n']:4d}  "
                  f"mean={stats['mean']:.3f}  CI=[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]  "
                  f"nu={stats['nu_mean']:.1f}", flush=True)

    if not args.aggregate_only:
        # Per (system, noise, snr, metric).
        for sys_ in sorted(df["system"].unique()):
            for noise in sorted(df["noise"].unique()):
                for snr in sorted(df["snr_db"].unique()):
                    cell = df[(df["system"] == sys_) & (df["noise"] == noise) & (df["snr_db"] == snr)]
                    for m in metric_cols:
                        vals = cell[m].dropna().values
                        if len(vals) == 0:
                            continue
                        rng, sk = jax.random.split(rng)
                        stats = fit_cell(vals, args.num_warmup, args.num_samples, sk)
                        rows.append(dict(system=sys_, noise=noise, snr_db=snr, metric=m, **stats))
            print(f"[cells] completed {sys_}", flush=True)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nWritten {len(out_df)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
