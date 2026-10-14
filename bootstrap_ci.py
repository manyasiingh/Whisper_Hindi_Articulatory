#!/usr/bin/env python
"""Utterance-level bootstrap confidence intervals for the Hindi Whisper study.

Inputs (in --results dir):
  test_hypotheses.json   {model_name: [hypothesis, ...]}   (written by train_and_eval.py)
  test_references.json   [reference, ...]                  (see README for how to export)

Outputs (same dir):
  wer_cer_ci.csv, contrast_errors_ci.csv, asymmetry_ci.csv,
  paired_improvement_ci.csv, contrast_errors_ci.png

Resampling unit = utterance. The SAME resamples are used for every model, so
comparisons between models (zero-shot vs fine-tuned) are paired.
Runs on CPU in about a minute; no GPU needed.
"""
import argparse, json, os
import numpy as np, pandas as pd, jiwer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from articulatory import normalize_hi, align_chars, DIRECTED, ASYMMETRY_PAIRS

DNAMES = list(DIRECTED)


def articulatory_counts(refs, hyps):
    """Per-utterance counts: tot[u,k] = ref chars in contrast k, hit[u,k] = swapped for partner."""
    n = len(refs)
    tot = np.zeros((n, len(DNAMES)))
    hit = np.zeros_like(tot)
    for u, (r, h) in enumerate(zip(refs, hyps)):
        r = normalize_hi(r).replace(" ", "")
        h = normalize_hi(h).replace(" ", "")
        for op, rc, hc in align_chars(r, h):
            if op == "I":
                continue
            for k, name in enumerate(DNAMES):
                mp = DIRECTED[name]
                if rc in mp:
                    tot[u, k] += 1
                    if op == "S" and hc == mp[rc]:
                        hit[u, k] += 1
    return tot, hit


def wer_cer_counts(refs, hyps):
    """Per-utterance (errors, ref_units) for words and characters."""
    n = len(refs)
    out = np.zeros((n, 4))  # w_err, w_ref, c_err, c_ref
    for u, (r, h) in enumerate(zip(refs, hyps)):
        r, h = normalize_hi(r), normalize_hi(h)
        if not r:
            continue
        if not h:  # empty hypothesis: everything deleted
            out[u] = [len(r.split()), len(r.split()), len(r), len(r)]
            continue
        w = jiwer.process_words(r, h)
        c = jiwer.process_characters(r, h)
        out[u] = [w.substitutions + w.deletions + w.insertions,
                  w.substitutions + w.deletions + w.hits,
                  c.substitutions + c.deletions + c.insertions,
                  c.substitutions + c.deletions + c.hits]
    return out


def ci(x, axis=0):
    return np.nanpercentile(x, [2.5, 97.5], axis=axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline", default="zero-shot")
    a = ap.parse_args()

    hyps_by_model = json.load(open(os.path.join(a.results, "test_hypotheses.json")))
    refs = json.load(open(os.path.join(a.results, "test_references.json")))
    n = len(refs)
    for m, h in hyps_by_model.items():
        assert len(h) == n, f"{m}: {len(h)} hypotheses vs {n} references (filtering mismatch?)"
    models = list(hyps_by_model)

    rng = np.random.default_rng(a.seed)
    M = rng.multinomial(n, [1 / n] * n, size=a.n_boot).astype(float)  # (n_boot, n) resample multiplicities

    wc, tot, hit = {}, {}, {}
    for m in models:
        print(f"counting {m} ...")
        wc[m] = wer_cer_counts(refs, hyps_by_model[m])
        tot[m], hit[m] = articulatory_counts(refs, hyps_by_model[m])

    # ---- WER / CER ----
    rows, boot_wc = [], {}
    for m in models:
        full = wc[m].sum(0)
        b = M @ wc[m]
        bw, bc = 100 * b[:, 0] / b[:, 1], 100 * b[:, 2] / b[:, 3]
        boot_wc[m] = (bw, bc)
        for name, pt, bs in [("WER", 100 * full[0] / full[1], bw), ("CER", 100 * full[2] / full[3], bc)]:
            lo, hi = ci(bs)
            rows.append(dict(model=m, metric=name, value=pt, ci_lo=lo, ci_hi=hi))
    pd.DataFrame(rows).round(2).to_csv(os.path.join(a.results, "wer_cer_ci.csv"), index=False)

    # ---- per-contrast rates ----
    rate_pt, rate_boot, rows = {}, {}, []
    for m in models:
        b_tot, b_hit = M @ tot[m], M @ hit[m]
        rate_boot[m] = 100 * b_hit / np.where(b_tot == 0, np.nan, b_tot)
        rate_pt[m] = 100 * hit[m].sum(0) / tot[m].sum(0)
        lo, hi = ci(rate_boot[m])
        for k, name in enumerate(DNAMES):
            rows.append(dict(model=m, contrast=name, n_ref=int(tot[m][:, k].sum()),
                             n_events=int(hit[m][:, k].sum()),
                             rate_pct=rate_pt[m][k], ci_lo=lo[k], ci_hi=hi[k]))
    contrast_df = pd.DataFrame(rows).round(2)
    contrast_df.to_csv(os.path.join(a.results, "contrast_errors_ci.csv"), index=False)

    # ---- asymmetry (forward vs reverse) ----
    rows = []
    for m in models:
        for fwd, rev in ASYMMETRY_PAIRS:
            i, j = DNAMES.index(fwd), DNAMES.index(rev)
            gap = rate_boot[m][:, i] - rate_boot[m][:, j]
            ratio = rate_boot[m][:, i] / np.where(rate_boot[m][:, j] > 0, rate_boot[m][:, j], np.nan)
            glo, ghi = ci(gap)
            rlo, rhi = ci(ratio)
            rows.append(dict(model=m, forward=fwd, reverse=rev,
                             fwd_rate=rate_pt[m][i], rev_rate=rate_pt[m][j],
                             gap_pp=rate_pt[m][i] - rate_pt[m][j], gap_ci_lo=glo, gap_ci_hi=ghi,
                             ratio=rate_pt[m][i] / rate_pt[m][j] if rate_pt[m][j] > 0 else np.nan,
                             ratio_ci_lo=rlo, ratio_ci_hi=rhi,
                             ratio_valid_frac=float(np.mean(~np.isnan(ratio))),
                             sig_asymmetry=bool(glo > 0 or ghi < 0)))
    asym_df = pd.DataFrame(rows).round(3)
    asym_df.to_csv(os.path.join(a.results, "asymmetry_ci.csv"), index=False)

    # ---- paired improvement vs baseline ----
    rows = []
    if a.baseline in models:
        for m in [x for x in models if x != a.baseline]:
            d = rate_boot[a.baseline] - rate_boot[m]
            lo, hi = ci(d)
            for k, name in enumerate(DNAMES):
                rows.append(dict(model=m, vs=a.baseline, metric=name,
                                 improvement_pp=rate_pt[a.baseline][k] - rate_pt[m][k],
                                 ci_lo=lo[k], ci_hi=hi[k], significant=bool(lo[k] > 0 or hi[k] < 0)))
            for idx, nm in [(0, "WER"), (1, "CER")]:
                dd = boot_wc[a.baseline][idx] - boot_wc[m][idx]
                lo2, hi2 = ci(dd)
                fb, fm = wc[a.baseline].sum(0), wc[m].sum(0)
                pt = (100 * fb[2 * idx] / fb[2 * idx + 1]) - (100 * fm[2 * idx] / fm[2 * idx + 1])
                rows.append(dict(model=m, vs=a.baseline, metric=nm, improvement_pp=pt,
                                 ci_lo=lo2, ci_hi=hi2, significant=bool(lo2 > 0 or hi2 < 0)))
    paired_df = pd.DataFrame(rows).round(2)
    paired_df.to_csv(os.path.join(a.results, "paired_improvement_ci.csv"), index=False)

    # ---- plot ----
    x = np.arange(len(DNAMES)); w = 0.8 / len(models)
    plt.figure(figsize=(11, 4.8))
    for q, m in enumerate(models):
        sub = contrast_df[contrast_df.model == m].set_index("contrast").loc[DNAMES]
        yerr = np.vstack([sub.rate_pct - sub.ci_lo, sub.ci_hi - sub.rate_pct])
        plt.bar(x + q * w, sub.rate_pct, w, yerr=yerr, capsize=3, label=m)
    plt.xticks(x + w * (len(models) - 1) / 2, DNAMES, rotation=30, ha="right")
    plt.ylabel("% of reference chars replaced by contrast partner")
    plt.title("Articulatory contrast errors (95% utterance-level bootstrap CI)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(a.results, "contrast_errors_ci.png"), dpi=200)

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\n== WER/CER =="); print(pd.read_csv(os.path.join(a.results, "wer_cer_ci.csv")))
    print("\n== Asymmetry =="); print(asym_df[["model", "forward", "reverse", "fwd_rate", "rev_rate",
                                              "gap_pp", "gap_ci_lo", "gap_ci_hi", "ratio",
                                              "ratio_ci_lo", "ratio_ci_hi", "sig_asymmetry"]])
    print("\n== Paired improvement =="); print(paired_df)


if __name__ == "__main__":
    main()
