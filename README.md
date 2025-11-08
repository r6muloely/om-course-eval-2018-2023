# Perceived Quality Without Grade Inflation (2018–2023)
Replication package for a longitudinal Operations Management evaluation using fuzzy logic and text mining.

Deterministic Python pipeline reproducing all figures and tables from de-identified 2018–2023 SIRS, prompt-aware sentiment, and grade data.

---

## 1) Repository layout

```
OM_System_Analysis_Codepack/
├─ run_pipeline.py
├─ environment.yml
├─ src/
│  ├─ utils.py
│  ├─ data_loading.py
│  ├─ sirs_metrics.py
│  ├─ sentiment.py
│  ├─ models/
│  │  └─ fuzzy_model.py
│  ├─ visualization/
│  │  └─ plots.py
│  └─ pipelines/
│     └─ build_all.py
├─ data/raw/
│  ├─ sirs_quant.csv
│  ├─ comments.csv
│  └─ grades.csv
├─ outputs/
│  ├─ figures/
│  └─ tables/
└─ paper_tables/
```
---

## 2) Environment

Conda (recommended)

```bash
conda env create -f environment.yml
conda activate om-course-env
python -c "import nltk; nltk.download('vader_lexicon')"
```

Alternatively (virtualenv)

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\Activate.ps1  # Windows (PowerShell)
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## 3) Raw data (expected schemas)

All files are UTF-8 CSVs at the section_id × term level (Spring/Fall). section_id encodes term/year (e.g., F2023-28, S2019-largeclass).

### data/raw/sirs_quant.csv (SIRS item means on the 1–5 scale)
Required columns:
- section_id, year, term
- clarity — The instructor was prepared for class and presented the material in an organized manner.
- fairness — The instructor assigned grades fairly.
- overall — I rate the overall quality of the course as.
- teaching_effectiveness — I rate the teaching effectiveness of the instructor as.

### data/raw/comments.csv (one row per comment)
Required columns:
- section_id, year, term
- prompt_type ∈ {best, change, growth, other}  (enables by-prompt sentiment plots)
- comment_text  (de-identified text)

Note: sentiment scores/labels (VADER compound in −1..1; positive/neutral/negative) are computed in the pipeline.

### data/raw/grades.csv (section–term grade distributions)
Required columns:
- section_id, year, term, student_count
- mean_gpa, median_gpa, sd_gpa
- pct_A, pct_B, pct_C, pct_D, pct_F  (proportions in [0,1])
- Optional: pct_W, pct_B+, pct_C+ (B+/C+ collapsed in the main figure)

Large-class alignment. Sections 01–10 meet together and are evaluated on one SIRS form each term. To align units, those grade rows are summed and recorded as a single large-class record (SYYYY-largeclass / FYYYY-largeclass).

---

## 4) Run the pipeline

From the repository root:

```bash
python run_pipeline.py
```

Build complete. See outputs/ and paper_tables/.

---

## 5) Outputs

Figures → outputs/figures/

- sirs_trend_teaching_overall_1to5.(png|pdf) — teaching effectiveness and course quality on 1–5
- sirs_trends.(png|pdf) — normalized clarity/fairness/course quality/teaching effectiveness
- sentiment_trend_by_semester.(png|pdf) — counts by term
- sentiment_trend_share_by_semester.(png|pdf) — percentage shares by term
- sentiment_trend_share_prompt_{best|change|growth|other}.(png|pdf) — per-prompt (if prompt_type)
- sentiment_trend_share_prompts_multipanel.(png|pdf) — 2×2 grid (if prompts present)
- fuzzy_sensitivity_scatter.(png|pdf) — baseline vs shifted composite
- grade_distribution_collapsed.(png|pdf) (+ optional ..._uncollapsed.(png|pdf))

Tables → outputs/tables/

- sirs_trends.csv — term means for clarity_norm, fairness_norm, overall_norm, teaching_effectiveness_norm
- sentiment_by_section.csv — comment-level sentiment labels/scores
- sentiment_counts_by_semester.csv — aggregate sentiment counts per term
- sentiment_counts_by_prompt.csv — counts per term × prompt (if prompt_type)
- topics_top_words.csv — topic → top words (if topic modeling is enabled)
- fuzzy_effectiveness_by_section.csv — inputs + effectiveness_score per section-term
- fuzzy_sensitivity.csv — baseline vs shifted composite (Δ, r, share(|Δ|>0.02))
- grades_summary.csv — GPA and letter-grade summaries

Key tables are duplicated to paper_tables/ for manuscript inclusion.

---

## 6) Method notes required to run

- SIRS normalization: 1–5 → [0,1] via (x − 1)/4
- Term means: unweighted across SIRS forms in a term (typical section); ALL/largeclass terms contribute that single form’s value
- Sentiment: VADER compound in [−1,1]; labels: ≥ 0.05 positive, ≤ −0.05 negative, else neutral
- Composite: Mamdani FIS (triangular sets; centroid defuzzification) over clarity_norm, fairness_norm, sentiment_mean
- Sensitivity: membership vertices shifted by ±0.05; report r, median |Δ|, share(|Δ|>0.02)
- Grades: main figure collapses B+/C+ → B/C; W is shown separately and excluded from GPA

---

## 7) Troubleshooting

- NLTK VADER not found → run: python -c "import nltk; nltk.download('vader_lexicon')" once
- UnicodeDecodeError → ensure CSVs are UTF-8; if needed, set encoding='latin-1' in the loader
- Legend overlaps → use right-margin legends with bbox_to_anchor=(1.02, 0.5) and fig.subplots_adjust(right=0.84)
- Pandas FutureWarnings (observed=False) → harmless in pinned versions
- Large-class duplicates → verify sections 01–10 were summed into a single …-largeclass record per term

---

## 8) License and citation

- Code: MIT License
- Data: CC BY 4.0 (or CC BY-NC 4.0 for non-commercial reuse)

If you use this package, please cite the paper and the archive DOI:

Ely, R. N. (2025). Perceived Quality Without Grade Inflation: Replication Package (2018–2023). DOI: 10.5281/zenodo.xxxxxxx.

---

## 9) Archive (Zenodo)

A versioned snapshot of this repository (including de-identified data) is archived on Zenodo:

DOI: 10.5281/zenodo.xxxxxxx

---

### Replace these placeholders before publishing

- Replace 10.5281/zenodo.xxxxxxx with the DOI minted by Zenodo when you publish the release.
- Ensure the environment name om-course-env matches the name in environment.yml. If different, update both places.
- If you prefer not to support virtualenv, remove that small alternative block.
