# Validation-driven bin size vs fixed bin size 16 against a decoy attacker (Adult stand-in)

Motivation: "Toxic Decoys: A Path to Scaling Privacy-Preserving Cryptocurrencies" (2025) studies how decoys and anonymity-set (bin) size trade storage cost against how well an observer can single out the real record. Choosing that bin size is a design decision, and a fixed default may waste storage or leave records exposed. This repository tests a simple alternative, choosing the bin size from validation data, on a public tabular dataset. It is a stand-in experiment, not a re-implementation of the paper's ledger.

Question: Can a validation-driven rule for choosing the bin size give a lower attacker identification rate at lower storage cost than a fixed bin size of 16 under the same attacker?

Dataset: OpenML Adult census data (ID 1590, https://www.openml.org/d/1590), downloaded at run time. This is not the paper's own dataset. Records are split 60/20/20 into train, validation and test, with categorical features one-hot encoded.

Baseline vs improved:
- Baseline: fixed bin size of 16, a common default not taken from the paper.
- Improved: the smallest bin size in {2, 4, 8, 16, 32, 64} whose validation top-1 identification rate is at most 0.10.
- Both use the same LightGBM attacker (real records vs random population decoys), the same data and the same evaluation function.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to `results.json` (charts go to `figures/`).

## Reference

This project is motivated by the following paper:

> C Cachin, FX Wicht (2025). *Toxic Decoys: A Path to Scaling Privacy-Preserving Cryptocurrencies*. Cryptology ePrint Archive. https://eprint.iacr.org/2025/1124

Paper author: Prof. Christian Cachin, University of Bern (christian.cachin@unibe.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
