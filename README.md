# FREUID Fraud Detection

Trimester 9 term project (B.Sc. Honours, Data Science and Artificial
Intelligence, Indian Institute of Technology Guwahati) built around
participation in **The FREUID Challenge 2026 (IJCAI-ECAI)** — a Kaggle
competition hosted by the Microblink Fraud Lab on binary fraud detection
for identity document images.

- **Author:** Maheshwar Mishra (Roll No. 23035010423)
- **Competition:** [kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai](https://kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai)
- **Term project report:** [`report/main.tex`](report/main.tex)
- **Reproducibility package** (frozen submission, Docker artifact, technical
  report for competition organizers): a separate repository,
  [freuid-challenge-2026-reproducibility](https://github.com/MaheshwarMahi/freuid-challenge-2026-reproducibility)

This repository holds the development notebooks and the academic term
project report. It is not the prize-eligibility reproducibility package —
that is intentionally kept in the separate repository linked above, frozen
at a specific commit as required by the competition rules.

---

## Task and Metric

Given an identity document image, and metadata fields `is_digital`
(fully digital vs. recaptured) and `type` (document format, e.g.
`USA/DL`) available at training time only, predict a continuous fraud
score in `[0, 1]`. Performance is measured by the **FREUID Score**
(lower is better):

```
g_audet = 1 - AuDET
g_apcer = 1 - APCER@1%BPCER
FREUID  = 1 - (2 * g_audet * g_apcer) / (g_audet + g_apcer)
```

AuDET is the area under the Detection Error Trade-off curve (overall
ranking quality); APCER@1%BPCER is the attack-miss rate at the fixed
operating point where the genuine-sample false-accept rate is 1%. The
harmonic-mean structure penalizes a model that is strong overall but weak
specifically at that operating point.

---

## Repository Structure

```
freuid-fraud-detection/
├── notebooks/
│   ├── freuid_challenge_v1.ipynb
│   ├── freuid_challenge_v1.1.ipynb
│   ├── freuid_challenge_v2.ipynb
│   ├── freuid_challenge_v3.ipynb
│   ├── freuid_challenge_v4.ipynb
│   ├── freuid_challenge_v5.ipynb
│   ├── freuid_challenge_v6.ipynb
│   ├── freuid_challenge_v7.ipynb
│   ├── freuid_challenge_v8.ipynb
│   └── freuid_challenge_v9.ipynb
├── report/
│   ├── main.tex
│   ├── refs.bib
│   ├── IEEEbib.bst
│   └── spconf.sty
├── requirements.txt
├── .gitattributes
└── .gitignore
```

---

## Notebook Versioning Trail

| Version | Purpose | Public LB FREUID |
|---|---|---|
| v1 | Train fold 0 — failed (`NameError: get_calibrator`) | — |
| v1.1 | Completed v1; first submitted checkpoint | 0.27886 |
| v2 | Full-resolution (320²) production training run | timed out |
| v3 | Diagnostics: label/metadata leakage check, `type_idx`/`is_digital` reliance check, train/val near-duplicate check (perceptual hashing), dataset-release sanity check, per-document-type breakdown, visual near-duplicate review, low-level pipeline-fingerprint check, Grad-CAM attention check | — |
| **v4** | **Frozen, submitted model** — resumes fold 0/fold 1 from v2 checkpoints, adds face-region occlusion augmentation (targeting the shortcut identified via Grad-CAM in v3) and early stopping | **0.22090** |
| v5 | Diagnostic: `is_digital` test-time flag ablation (hardcoded 0 vs. 1) | 0.21941 |
| v6 | Diagnostic: Grad-CAM re-check of learned attention regions | — |
| v7 | Diagnostic: per-document-type pipeline/noise statistics | — |
| v8 | Mitigation attempt: noise/sharpness-equalizing fine-tune warm-started from v4 (only fold 0 completed; fold 1 fell back to unchanged v4 weights) — inconclusive, **not adopted** | 0.22086 |
| v9 | Full public + private test-set inference using the frozen v4 checkpoints (`submission_ensemble_v4_FINAL.csv`) | 0.21868 |

v4 remains the frozen, submitted model throughout. v5–v8 are diagnostic or
mitigation-attempt notebooks, not replacements for v4; v9 is an inference
run, not a training iteration.

---

## Results

| Submission | Public LB FREUID ↓ | Private LB FREUID ↓ |
|---|---|---|
| Prize-eligible final (`submission_ensemble_v4.csv`) | 0.22090 | — |
| Full-set inference (`submission_ensemble_v4_FINAL.csv`, v9) | 0.21868 | **0.46683** |

Local validation FREUID at export time was near-ceiling for both folds
(fold 0: 0.000274 at epoch 17; fold 1: 0.000683 at epoch 8) — a roughly
three-orders-of-magnitude gap against the public leaderboard score, and a
further degradation on the much larger private set (134,997 images,
~17× the public set, including two document types unseen during training).

---

## Key Finding: Diagnosed Shortcut Learning

Rather than treating the local-validation-to-leaderboard gap as an
unexplained cost of the competition, a structured, falsifiable diagnostic
process (see `report/main.tex`, Section 5) identified the root cause:

- **Grad-CAM (v6)** showed the model's attention concentrating almost
  entirely on the face-photo region for fraud-labeled images, and diffuse
  elsewhere for genuine images — consistent with a narrow face-region
  anomaly detector rather than a general document-fraud detector.
- **Ruled out:** metadata leakage (v5: only a noise-level 0.00149 LB
  delta), train/validation image duplication (false positives from shared
  templates), global pipeline fingerprints (file size/JPEG quality
  identical across classes), and a suspected submission pipeline bug
  (logs confirmed real predictions on all 142,818 rows, zero placeholder
  fallbacks).
- **Open, unresolved finding (v7):** `MAURITIUS/ID`, the only ID-card-format
  document type in the dataset, showed statistically distinct noise,
  sharpness, and intensity characteristics from every other type,
  independent of label — a capture-pipeline fingerprint not corrected
  before the competition's code-freeze deadline.
- The **private leaderboard score confirms the diagnosis**: a face-shortcut
  weakness that was only partially mitigated (face occlusion applied with
  probability 0.3 during training) shows through clearly at ~17× scale and
  on genuinely unseen document types.

Full details, methodology, and discussion are in the term project report.

---

## Building the Report

```bash
cd report
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Requires `spconf.sty` and `IEEEbib.bst` (included in `report/`) alongside a
standard TeX Live installation. Produces `main.pdf`.

---

## Environment

See `requirements.txt` for Python dependencies. Notebooks were developed
and run on Kaggle (T4 GPU); key libraries include PyTorch, `timm`
(`tf_efficientnet_b4` backbone, Apache-2.0), `albumentations`, and
`scikit-learn`.

---

## Acknowledgements

Dataset and competition: Microblink Fraud Lab, via Kaggle. Backbone
architecture and ImageNet-pretrained weights via `timm`
(Tan and Le, *EfficientNet*, ICML 2019). Full citation list in
`report/refs.bib`.
