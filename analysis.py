"""Validation-driven bin size vs fixed bin size 16 against a LightGBM decoy attacker (OpenML Adult, ID 1590)."""
import json
import os
import sys
import tempfile
import time

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import requests
from scipy.io import arff
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
SEEDS = [42, 43, 44]
API = "https://www.openml.org/api/v1/json/data/1590"
TARGET_RATE = 0.10          # attacker top-1 identification target
BASE_BIN = 16               # fixed baseline bin size
BIN_SIZES = [2, 4, 8, 16, 32, 64]
REPS = 10                   # bins per record (top-1 rate and per-record shares)
CHUNK = 1000                # records scored per batch (bounds memory)
MAX_ROWS = 200_000          # subsample cap (Adult has ~49k rows)


def get(url):
    """HTTP GET with timeout=60; retry with exponential backoff on 429/5xx and network errors."""
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if attempt == 4:
                raise
            print(f"  download error ({e}); retrying in {2 ** attempt}s")
            time.sleep(2 ** attempt)


def _decode(v):
    """Nominal value -> str; bytes decoded; missing -> '?' category."""
    if isinstance(v, bytes):
        return v.decode()
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "?"
    return str(v)


def load():
    """Fetch the Adult ARFF file named by the OpenML API and return (DataFrame, target column)."""
    desc = get(API).json()["data_set_description"]
    print(f"Downloading {desc['name']} from {desc['url']}")
    path = os.path.join(tempfile.gettempdir(), "adult.arff")
    with open(path, "w", encoding="utf-8") as f:
        f.write(get(desc["url"]).text)
    raw, _ = arff.loadarff(path)
    df = pd.DataFrame(raw)
    for c in df.columns:  # decode nominal columns (any string-like dtype, incl. pandas 3 'str')
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.Series([_decode(v) for v in df[c].tolist()], index=df.index, dtype=object)
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED)
    return df.reset_index(drop=True), desc["default_target_attribute"]


def build_matrix(df):
    """Raw-coded float matrix (categoricals as integer codes) plus a per-column spec (n_categories or 0)."""
    M = np.zeros((len(df), df.shape[1]))
    spec = []
    for j, c in enumerate(df.columns):
        if not pd.api.types.is_numeric_dtype(df[c]):
            cat = pd.Categorical(df[c].astype(object))
            M[:, j] = cat.codes
            spec.append((j, len(cat.categories)))
        else:
            M[:, j] = df[c].astype(float).values
            spec.append((j, 0))
    return M, spec


def encode(X, spec):
    """One-hot encode categorical codes; numeric columns are passed through."""
    parts = [np.eye(k, dtype=np.float32)[X[:, j].astype(int)] if k else X[:, [j]].astype(np.float32)
             for j, k in spec]
    return np.hstack(parts)


def decoys(pool, shape, rng):
    """Random population decoys: each attribute is drawn independently from its marginal in the pool.
    This breaks inter-attribute correlations, which is the signal the attacker can exploit."""
    idx = rng.integers(0, len(pool), size=shape + (pool.shape[1],))
    return pool[idx, np.arange(pool.shape[1])]


def train_attacker(tr, spec, rng, seed):
    """LightGBM 'is-real' classifier: real train records (label 1) vs decoys (label 0), 1:1."""
    neg = decoys(tr, (len(tr),), rng)
    X = encode(np.vstack([tr, neg]), spec)
    y = np.r_[np.ones(len(tr)), np.zeros(len(neg))]
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                               n_jobs=-1, verbose=-1)
    model.fit(X, y)
    return model


def top1_hits(model, recs, pool, spec, B, rng):
    """hits[r, i] = True if the attacker's top-1 pick in rep r's bin for record i is the real record.
    Each bin = real record in a random slot + (B-1) decoys; ties resolved by argmax (first slot)."""
    n = len(recs)
    hits = np.zeros((REPS, n), dtype=bool)
    for r in range(REPS):
        for s in range(0, n, CHUNK):
            m = recs[s:s + CHUNK]
            k = len(m)
            bins = decoys(pool, (k, B), rng)              # (k, B, p)
            pos = rng.integers(0, B, size=k)              # slot of the real record
            bins[np.arange(k), pos] = m
            scores = model.predict_proba(encode(bins.reshape(k * B, -1), spec))[:, 1].reshape(k, B)
            hits[r, s:s + k] = scores.argmax(1) == pos
    return hits


def main():
    t0 = time.time()
    df, target = load()
    M, spec = build_matrix(df)
    t = list(df.columns).index(target)
    # ONE split: 20% test; the other 80% -> 60% attacker-train + 20% validation (stratified on income)
    rest, te = train_test_split(M, test_size=0.2, stratify=M[:, t], random_state=SEED)
    tr, va = train_test_split(rest, test_size=0.25, stratify=rest[:, t], random_state=SEED)
    print(f"rows={len(M)} train={len(tr)} validation={len(va)} test={len(te)}")

    out = {k: [] for k in ["base", "imp", "base_share", "imp_share", "chosen", "unmet", "val"]}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        print(f"seed {seed}: training LightGBM attacker")
        model = train_attacker(tr, spec, rng, seed)
        # bin size selection uses validation records only
        val = {B: float(top1_hits(model, va, tr, spec, B, rng).mean()) for B in BIN_SIZES}
        ok = [B for B in BIN_SIZES if val[B] <= TARGET_RATE]
        chosen = min(ok) if ok else BIN_SIZES[-1]
        print(f"  validation top-1: {({B: round(v, 4) for B, v in val.items()})} -> chosen bin {chosen}")
        # test set used once per seed, for both models with the same attacker and function
        base_h = top1_hits(model, te, tr, spec, BASE_BIN, rng)
        imp_h = top1_hits(model, te, tr, spec, chosen, rng)
        out["base"].append(float(base_h.mean()))
        out["imp"].append(float(imp_h.mean()))
        out["base_share"].append(float((base_h.mean(0) <= TARGET_RATE).mean()))
        out["imp_share"].append(float((imp_h.mean(0) <= TARGET_RATE).mean()))
        out["chosen"].append(chosen)
        out["unmet"].append(not ok)
        out["val"].append(val)
        print(f"  test top-1: baseline({BASE_BIN})={out['base'][-1]:.4f} improved({chosen})={out['imp'][-1]:.4f}")

    b, i = np.array(out["base"]), np.array(out["imp"])
    if not (np.isfinite(b).all() and np.isfinite(i).all()):
        sys.exit("Non-finite metrics; aborting")
    mean_val = {B: float(np.mean([v[B] for v in out["val"]])) for B in BIN_SIZES}

    metrics = {
        "baseline_score": float(b.mean()), "improved_score": float(i.mean()),
        "improvement": float(b.mean() - i.mean()),
        "baseline_score_std": float(b.std()), "improved_score_std": float(i.std()),
        "baseline_bin_size": float(BASE_BIN), "improved_bin_size": float(np.mean(out["chosen"])),
        "baseline_share_meeting_target": float(np.mean(out["base_share"])),
        "improved_share_meeting_target": float(np.mean(out["imp_share"])),
        "seeds_target_unmet": float(sum(out["unmet"])),
        "chance_rate_fixed_bin": 1.0 / BASE_BIN,
    }

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([f"baseline ({BASE_BIN})", f"improved (avg {metrics['improved_bin_size']:.0f})"],
           [b.mean(), i.mean()], yerr=[b.std(), i.std()], capsize=4)
    ax.set_ylabel("test top-1 identification rate (lower = better)")
    fig.tight_layout(); fig.savefig("figures/test_identification_rate.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(BIN_SIZES, [mean_val[B] for B in BIN_SIZES], "o-", label="validation top-1 (mean over seeds)")
    ax.axhline(TARGET_RATE, color="grey", ls="--", label="target 0.10")
    ax.set_xscale("log", base=2); ax.set_xlabel("bin size"); ax.set_ylabel("top-1 rate")
    ax.legend(); fig.tight_layout(); fig.savefig("figures/validation_curve.png", dpi=120); plt.close(fig)

    results = {
        "question": "Can a validation-driven rule for choosing the bin size give a lower attacker identification "
                    "rate at lower storage cost than a fixed bin size of 16 under the same attacker?",
        "method": "Stand-in experiment on OpenML Adult (ID 1590). Records split 60/20/20 (train/validation/test), "
                  "one-hot encoded. A LightGBM attacker learns real records vs random population decoys and, "
                  "for each bin (real record + decoys), picks the top-scoring member. Same attacker in all conditions.",
        "dataset": {"name": "Adult (OpenML 1590)", "source_url": "https://www.openml.org/d/1590",
                    "n_rows": int(len(M)), "n_train": int(len(tr)), "n_test": int(len(te)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "test top-1 attacker identification rate",
        "higher_is_better": False,
        "baseline": f"Fixed bin size {BASE_BIN} (common default, not from the paper) with the same LightGBM attacker.",
        "improved": f"Bin size chosen as the smallest in {BIN_SIZES} whose validation top-1 rate is <= {TARGET_RATE}; "
                    "falls back to the largest size if none qualifies.",
        "metrics": metrics,
        "notes": "Stand-in for the paper's ledger, not a re-implementation. Decoys are drawn attribute-by-attribute "
                 "from population marginals, so the attacker exploits broken correlations (e.g. education vs "
                 "education-num). Bin-size selection uses a held-out validation split (not k-fold CV) because the "
                 "rule is defined on validation top-1; the test set is used once per seed. Attacker hyper-parameters "
                 "are fixed, not tuned. Std is population std over seeds 42-44 (attacker and bin randomness; one "
                 "data split). Storage overhead is the bin size. Fix: nominal-column detection now uses "
                 "is_numeric_dtype instead of dtype==object, so it works with pandas 3 string dtypes.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"baseline={b.mean():.4f} improved={i.mean():.4f} improvement={metrics['improvement']:.4f} "
          f"(elapsed {time.time() - t0:.0f}s); wrote results.json")


if __name__ == "__main__":
    main()
