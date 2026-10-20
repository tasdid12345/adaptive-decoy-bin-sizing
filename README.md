# Adaptive decoy bin sizing (stand-in experiment on OpenML Adult, ID 1590)

Compares a validation-driven bin-size rule against a fixed bin size of 16 under the same LightGBM decoy attacker. Records are split 60/20/20 (train/validation/test) with a fixed seed (42); the attacker is trained on the train split, bin size is selected on validation top-1 identification rate, and the test split is used once per seed (42, 43, 44).

Run:

```
pip install -r requirements.txt
python analysis.py
```

Outputs: `results.json`, `figures/test_identification_rate.png`, `figures/validation_curve.png`.

Latest fix: categorical-column detection now uses `pd.api.types.is_numeric_dtype` rather than `dtype == object`, because pandas 3.x loads string columns with a `str` dtype. The method, data, baseline, and split are unchanged.

## Results (latest run)

On Adult (OpenML 1590) test data, a validation-selected bin size cut top-1 attacker identification from 88.9% (fixed bin size 16, chosen without data) to 70.6%, but it chose bin size 64, so storage cost rose rather than fell.

| Metric | Value |
|---|---|
| baseline_score | 0.8893 |
| improved_score | 0.7055 |
| improvement | 0.1838 |
| baseline_score_std | 0.002398 |
| improved_score_std | 0.004477 |
| baseline_bin_size | 16 |
| improved_bin_size | 64 |
| baseline_share_meeting_target | 0.004982 |
| improved_share_meeting_target | 0.08936 |
| seeds_target_unmet | 3 |
| chance_rate_fixed_bin | 0.0625 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in for the paper's ledger, not a re-implementation. Decoys are drawn attribute-by-attribute from population marginals, so the attacker exploits broken correlations (e.g. education vs education-num). Bin-size selection uses a held-out validation split (not k-fold CV) because the rule is defined on validation top-1; the test set is used once per seed. Attacker hyper-parameters are fixed, not tuned. Std is population std over seeds 42-44 (attacker and bin randomness; one data split). Storage overhead is the bin size. Fix: nominal-column detection now uses is_numeric_dtype instead of dtype==object, so it works with pandas 3 string dtypes.

## Reference

This project is motivated by the following paper:

> C Cachin, FX Wicht (2025). *Toxic Decoys: A Path to Scaling Privacy-Preserving Cryptocurrencies*. Cryptology ePrint Archive. https://eprint.iacr.org/2025/1124

Paper author: Prof. Christian Cachin, University of Bern (christian.cachin@unibe.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
