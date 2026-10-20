# Adaptive decoy bin sizing (stand-in experiment on OpenML Adult, ID 1590)

Compares a validation-driven bin-size rule against a fixed bin size of 16 under the same LightGBM decoy attacker. Records are split 60/20/20 (train/validation/test) with a fixed seed (42); the attacker is trained on the train split, bin size is selected on validation top-1 identification rate, and the test split is used once per seed (42, 43, 44).

Run:

```
pip install -r requirements.txt
python analysis.py
```

Outputs: `results.json`, `figures/test_identification_rate.png`, `figures/validation_curve.png`.

Latest fix: categorical-column detection now uses `pd.api.types.is_numeric_dtype` rather than `dtype == object`, because pandas 3.x loads string columns with a `str` dtype. The method, data, baseline, and split are unchanged.

## Reference

This project is motivated by the following paper:

> C Cachin, FX Wicht (2025). *Toxic Decoys: A Path to Scaling Privacy-Preserving Cryptocurrencies*. Cryptology ePrint Archive. https://eprint.iacr.org/2025/1124

Paper author: Prof. Christian Cachin, University of Bern (christian.cachin@unibe.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
