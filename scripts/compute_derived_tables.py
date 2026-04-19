#!/usr/bin/env python3
"""
Derived tables for the IEEE OJ-SP 2026 revision.

Produces the four tables that the manuscript derives from the raw per-file
results.csv:
    tab:factorial         aggregate per-system mean + 95% CI per metric
    tab:per-env-eum-delta  EUM contribution per environment
                           (AIDA-2 minus pure SEM, averaged over SNR)
    tab:effects-by-snr     main effects of theta and beta + theta*beta
                           interaction, stratified by input SNR
    tab:per-regime-optimum argmax over {AIDA-2, beta-only, theta-only,
                           pure-Wiener} at alpha=0.5 per (env, SNR)

Aggregate 95% credible intervals are computed as mean +/- 1.96*sd/sqrt(N);
at N=824 the Student-t posterior is effectively Normal, so this is the
large-N limit of the Student-t NUTS method used for per-cell intervals.

Usage:
    python scripts/compute_derived_tables.py \\
        --results-csv results/VOICEBANK_DEMAND/<run-dir>/results.csv \\
        --out-dir results/VOICEBANK_DEMAND/<run-dir>/

Outputs four CSVs in --out-dir:
    tab_factorial_with_ci.csv
    tab_per_env_eum_delta.csv
    tab_effects_by_snr.csv
    tab_per_regime_optimum.csv
"""

import argparse
import csv
import os
from collections import defaultdict
from math import sqrt


METRICS = ("PESQ", "CSIG", "CBAK", "COVL")

# Private->public system name map for the canonical 9-system factorial
SYSTEM_DISPLAY = [
    ("Unprocessed",   "Unprocessed"),
    ("SEM_uFB",       "SEM (uFB)"),
    ("SEM",           "SEM"),
    ("AIDA2_thetauFB","theta-only (uFB)"),
    ("AIDA2_betauFB", "beta-only (uFB)"),
    ("AIDA2_thetaWFB","theta-only"),
    ("AIDA2_betaWFB", "beta-only"),
    ("SEM_lit_uFB",   "AIDA-2 (uFB)"),
    ("SEM_lit",       "AIDA-2"),
]

# Four-way regime table uses only the alpha=0.5 family
REGIME_SYSTEMS = {
    "A": "SEM_lit",        # AIDA-2 full
    "B": "AIDA2_betaWFB",  # beta-only (WFB)
    "T": "AIDA2_thetaWFB", # theta-only (WFB)
    "W": "SEM",            # pure Wiener
}


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_std(xs):
    n = len(xs)
    mu = sum(xs) / n
    if n > 1:
        sd = sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    else:
        sd = 0.0
    return mu, sd, n


def ci95(xs):
    mu, sd, n = mean_std(xs)
    se = sd / sqrt(n) if n > 1 else 0.0
    return mu, mu - 1.96 * se, mu + 1.96 * se


def table_factorial_with_ci(rows, out_path):
    vals = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in METRICS:
            vals[r["system"]][m].append(float(r[m]))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["system"]
        for m in METRICS:
            header += [f"{m}_mean", f"{m}_ci_low", f"{m}_ci_high"]
        w.writerow(header)
        for key, disp in SYSTEM_DISPLAY:
            row = [disp]
            for m in METRICS:
                mu, lo, hi = ci95(vals[key][m])
                row += [f"{mu:.4f}", f"{lo:.4f}", f"{hi:.4f}"]
            w.writerow(row)


def table_per_env_eum_delta(rows, out_path):
    # Δ = AIDA-2 (SEM_lit at α=0.5, θ=2.5, β=0.25) - pure SEM (α=0.5, θ=0, β=0)
    per_env = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        sys_ = r["system"]
        env = r["noise"]
        for m in METRICS:
            per_env[sys_][env][m].append(float(r[m]))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["environment", "dPESQ", "dCSIG", "dCBAK", "dCOVL"])
        for env in sorted(per_env["SEM_lit"].keys()):
            d = []
            for m in METRICS:
                aida = mean_std(per_env["SEM_lit"][env][m])[0]
                sem  = mean_std(per_env["SEM"][env][m])[0]
                d.append(aida - sem)
            w.writerow([env] + [f"{v:+.4f}" for v in d])


def table_effects_by_snr(rows, out_path):
    # At alpha=0.5, compute main effects and interaction stratified by SNR.
    #   theta-alone: AIDA2_thetaWFB(theta=2.5, beta=0) - SEM(theta=0, beta=0)
    #   beta-alone : AIDA2_betaWFB(theta=0, beta=0.25) - SEM(theta=0, beta=0)
    #   theta*beta interaction: AIDA-2 full - theta-only - beta-only + SEM
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        sys_ = r["system"]
        snr = float(r["snr_db"])
        for m in METRICS:
            per[sys_][snr][m].append(float(r[m]))
    snrs = sorted(set(float(r["snr_db"]) for r in rows))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "effect"] + [f"{snr}dB" for snr in snrs])
        for m in METRICS:
            theta_row = ["theta_alone"]
            beta_row  = ["beta_alone"]
            ix_row    = ["theta_x_beta"]
            for snr in snrs:
                mu_sem   = mean_std(per["SEM"][snr][m])[0]
                mu_theta = mean_std(per["AIDA2_thetaWFB"][snr][m])[0]
                mu_beta  = mean_std(per["AIDA2_betaWFB"][snr][m])[0]
                mu_aida  = mean_std(per["SEM_lit"][snr][m])[0]
                theta_row.append(f"{mu_theta - mu_sem:+.4f}")
                beta_row.append(f"{mu_beta - mu_sem:+.4f}")
                # 2-way interaction (AIDA2 - theta-only - beta-only + SEM)
                ix_row.append(f"{mu_aida - mu_theta - mu_beta + mu_sem:+.4f}")
            w.writerow([m] + theta_row[1:])
            w.writerow([m] + beta_row[1:])
            w.writerow([m] + ix_row[1:])


def table_per_regime_optimum(rows, out_path):
    # For each (env, snr) cell at alpha=0.5 compute mean PESQ of each of the
    # four regime systems, return argmax label A/B/T/W.
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        per[r["system"]][r["noise"]][float(r["snr_db"])].append(float(r["PESQ"]))
    envs = sorted(per["SEM_lit"].keys())
    snrs = sorted(set(s for e in per["SEM_lit"].values() for s in e.keys()))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["environment"] + [f"{snr}dB" for snr in snrs])
        for env in envs:
            row = [env]
            for snr in snrs:
                means = {
                    lbl: mean_std(per[sys_][env][snr])[0]
                    for lbl, sys_ in REGIME_SYSTEMS.items()
                }
                winner = max(means, key=means.get)
                row.append(winner)
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_rows(args.results_csv)

    outputs = [
        ("tab_factorial_with_ci.csv",   table_factorial_with_ci),
        ("tab_per_env_eum_delta.csv",   table_per_env_eum_delta),
        ("tab_effects_by_snr.csv",      table_effects_by_snr),
        ("tab_per_regime_optimum.csv",  table_per_regime_optimum),
    ]
    for fname, fn in outputs:
        out = os.path.join(args.out_dir, fname)
        fn(rows, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
