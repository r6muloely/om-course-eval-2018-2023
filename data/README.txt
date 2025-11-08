Data directory — de-identified course-level records (2018–2023)

Contents
---------
raw/                Original input CSVs used in the analysis
  ├─ sirs_quant.csv   SIRS item means on 1–5 scale (clarity, fairness, overall, teaching_effectiveness)
  ├─ comments.csv     Free-text comments with prompt_type ∈ {best, change, growth, other}
  └─ grades.csv       Grade distributions (A–F, GPA summaries)

Structure
----------
All files are UTF-8 CSVs at the section_id × term level (Spring/Fall).
Each record represents one course section per term.

Identifiers
-----------
section_id encodes term and section (e.g., F2023-28, S2019-largeclass)
year in YYYY format
term ∈ {Spring, Fall}

Notes
-----
- Sentiment (VADER compound −1..1; pos/neu/neg) is computed in the Python pipeline.
- Normalization: SIRS 1–5 → [0,1] via (x − 1)/4.
- Sections 01–10 are aggregated as single “largeclass” entries per term.
- All data are de-identified; no personally identifiable information is present.

License
--------
Data: CC BY 4.0 (or CC BY-NC 4.0 for non-commercial reuse)

Citation
--------
Ely, R. N. (2025). Perceived Quality Without Grade Inflation: Replication Package (2018–2023). DOI: 10.5281/zenodo.17560637

