# om-course-eval-2018-2023
Replication package for “Perceived Quality Without Grade Inflation (2018–2023)”. Includes Python code (conda env), de-identified data (sirs_quant.csv, comments.csv, grades.csv), and a deterministic pipeline (run_pipeline.py) to reproduce all figures/tables. Code: MIT; data: CC-BY 4.0.

{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww12720\viewh16500\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Perceived Quality Without Grade Inflation \'97 Code & De-identified Data\
\
This repository contains the code, configuration, and de-identified data used to reproduce all figures and tables in:\
\
**Perceived Quality Without Grade Inflation: A Five-Year Operations Management Evaluation Across Delivery Modes Using Fuzzy Logic and Text Mining (2018\'962023)**\
\
Given the same inputs and environment, running `python run_pipeline.py` regenerates all artifacts under `outputs/` and copies key tables to `paper_tables/` for manuscript use.\
\
---\
\
## 1) Repository layout\
\
OM_System_Analysis_Codepack/\
\uc0\u9500 \u9472  run_pipeline.py\
\uc0\u9500 \u9472  environment.yml\
\uc0\u9500 \u9472  src/\
\uc0\u9474  \u9500 \u9472  utils.py\
\uc0\u9474  \u9500 \u9472  data_loading.py\
\uc0\u9474  \u9500 \u9472  sirs_metrics.py\
\uc0\u9474  \u9500 \u9472  sentiment.py\
\uc0\u9474  \u9500 \u9472  models/fuzzy_model.py\
\uc0\u9474  \u9500 \u9472  visualization/plots.py\
\uc0\u9474  \u9492 \u9472  pipelines/build_all.py\
\uc0\u9500 \u9472  data/raw/\
\uc0\u9474  \u9500 \u9472  sirs_quant.csv\
\uc0\u9474  \u9500 \u9472  comments.csv\
\uc0\u9474  \u9492 \u9472  grades.csv\
\uc0\u9500 \u9472  outputs/\{figures|tables\}/\
\uc0\u9492 \u9472  paper_tables/\
---\
\
## 2) Environment\
\
**Conda (recommended)**\
\
```bash\
conda env create -f environment.yml\
conda activate om-course-env\
python -c "import nltk; nltk.download('vader_lexicon')"\
```\
\
**Alternatively (virtualenv)**\
\
```bash\
python -m venv .venv\
source .venv/bin/activate  # macOS/Linux\
# .\\.venv\\Scripts\\Activate.ps1  # Windows (PowerShell)\
pip install -r requirements.txt\
python -c "import nltk; nltk.download('vader_lexicon')"\
```\
---\
\
## 3) Raw data (expected schemas)\
\
All files are UTF-8 CSVs at the **`section_id \'d7 term`** level (Spring/Fall). `section_id` encodes term/year (e.g., `F2023-28`, `S2019-largeclass`).\
\
### `data/raw/sirs_quant.csv` (SIRS item means on the 1\'965 scale)\
Required columns:\
- `section_id`, `year`, `term`\
- `clarity` \'97 The instructor was prepared for class and presented the material in an organized manner.\
- `fairness` \'97 The instructor assigned grades fairly.\
- `overall` \'97 I rate the overall quality of the course as.\
- `teaching_effectiveness` \'97 I rate the teaching effectiveness of the instructor as.\
\
### `data/raw/comments.csv` (one row per comment)\
Required columns:\
- `section_id`, `year`, `term`\
- `prompt_type` \uc0\u8712  \{`best`, `change`, `growth`, `other`\}  (enables by-prompt sentiment plots)\
- `comment_text`  (de-identified text)\
\
_Note:_ sentiment scores/labels (VADER compound in \uc0\u8722 1..1; positive/neutral/negative) are **computed in the pipeline**.\
\
### `data/raw/grades.csv` (section\'96term grade distributions)\
Required columns:\
- `section_id`, `year`, `term`, `student_count`\
- `mean_gpa`, `median_gpa`, `sd_gpa`\
- `pct_A`, `pct_B`, `pct_C`, `pct_D`, `pct_F`  (proportions in `[0,1]`)\
- Optional: `pct_W`, `pct_B+`, `pct_C+` (B+/C+ collapsed in the main figure)\
\
> **Large-class alignment.** Sections 01\'9610 meet together and are evaluated on one SIRS form each term. To align units, those grade rows are **summed** and recorded as a single large-class record (`SYYYY-largeclass` / `FYYYY-largeclass`).\
\
## 4) Run the pipeline\
\
From the repository root:\
\
```bash\
python run_pipeline.py\
\
Build complete. See outputs/ and paper_tables/.\
\
::contentReference[oaicite:0]\{index=0\}\
\
```\
\
\
5) Outputs\
Figures \uc0\u8594  outputs/figures/\
\
sirs_trend_teaching_overall_1to5.(png|pdf) \'97 teaching effectiveness & course quality on 1\'965\
\
sirs_trends.(png|pdf) \'97 normalized clarity/fairness/course quality/teaching effectiveness\
\
sentiment_trend_by_semester.(png|pdf) \'97 counts by term\
\
sentiment_trend_share_by_semester.(png|pdf) \'97 percentage shares by term\
\
sentiment_trend_share_prompt_\{best|change|growth|other\}.(png|pdf) \'97 per-prompt (if prompt_type)\
\
sentiment_trend_share_prompts_multipanel.(png|pdf) \'97 2\'d72 grid (if prompts present)\
\
fuzzy_sensitivity_scatter.(png|pdf) \'97 baseline vs shifted composite\
\
grade_distribution_collapsed.(png|pdf) (+ optional ..._uncollapsed.(png|pdf))\
\
Tables \uc0\u8594  outputs/tables/\
\
sirs_trends.csv \'97 term means for clarity_norm, fairness_norm, overall_norm, teaching_effectiveness_norm\
\
sentiment_by_section.csv \'97 comment-level sentiment labels/scores\
\
sentiment_counts_by_semester.csv \'97 aggregate sentiment counts per term\
\
sentiment_counts_by_prompt.csv \'97 counts per term \'d7 prompt (if prompt_type)\
\
topics_top_words.csv \'97 topic \uc0\u8594  top words (if topic modeling is enabled)\
\
fuzzy_effectiveness_by_section.csv \'97 inputs + effectiveness_score per section-term\
\
fuzzy_sensitivity.csv \'97 baseline vs shifted composite (\uc0\u916 , r, share(|\u916 |>0.02))\
\
grades_summary.csv \'97 GPA and letter-grade summaries\
\
Key tables are duplicated to paper_tables/ for manuscript inclusion.\
\
6) Method notes required to run\
\
SIRS normalization: 1\'965 \uc0\u8594  [0,1] via (x \u8722  1)/4\
\
Term means: unweighted across SIRS forms in a term (typical section); \'93ALL/largeclass\'94 terms contribute that single form\'92s value\
\
Sentiment: VADER compound in [\uc0\u8722 1,1]; labels: \u8805  0.05 positive, \u8804  \u8722 0.05 negative, else neutral\
\
Composite: Mamdani FIS (triangular sets; centroid defuzzification) over clarity_norm, fairness_norm, sentiment_mean\
\
Sensitivity: membership vertices shifted by \'b10.05; report r, median |\uc0\u916 |, share(|\u916 |>0.02)\
\
Grades: main figure collapses B+/C+ \uc0\u8594  B/C; W is shown separately and excluded from GPA\
\
7) Troubleshooting\
\
NLTK VADER not found \uc0\u8594  run python -c "import nltk; nltk.download('vader_lexicon')" once\
\
UnicodeDecodeError \uc0\u8594  ensure CSVs are UTF-8; if needed, set encoding='latin-1' in the loader\
\
Legend overlaps \uc0\u8594  right-margin legends with bbox_to_anchor=(1.02, 0.5) and fig.subplots_adjust(right=0.84)\
\
Pandas FutureWarnings (observed=False) \uc0\u8594  harmless in pinned versions\
\
Large-class duplicates \uc0\u8594  verify sections 01\'9610 were summed into a single \'85-largeclass record per term\
\
8) License & citation\
\
Code: MIT License\
\
Data: CC-BY 4.0 (or CC-BY-NC if you prefer non-commercial reuse)\
\
If you use this package, please cite the paper and the archive DOI:\
\
Ely, R. N. (2025). Perceived Quality Without Grade Inflation: Replication Package (2018\'962023). DOI: 10.5281/zenodo.xxxxxxx.\
\
\
9) Archive (Zenodo)\
\
A versioned snapshot of this repository (including de-identified data) is archived on Zenodo: DOI: 10.5281/zenodo.xxxxxxx.\
\
\
Replace these placeholders:\
- `om-course-env` with the actual environment name in your `environment.yml` (if different).\
- `10.5281/zenodo.xxxxxxx` with the DOI Zenodo gives you after you publish the release.\
- If you prefer not to support `venv`, delete that small alternative block.\
\
If you\'92d rather keep the README shorter, we can strip it down to sections 1\'965 and 8\'969, but the version above will make reviewers\'92 lives easy and keeps your Appendix nice and lean.\
::contentReference[oaicite:0]\{index=0\}\
\
}
