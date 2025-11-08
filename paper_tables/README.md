# Paper Tables

Purpose
-------
This folder contains a curated subset of output tables used directly in the manuscript and appendix.
Each file here is a copy of a corresponding CSV generated under outputs/tables/.

Relationship to outputs/
------------------------
- The pipeline (run_pipeline.py) regenerates all results under outputs/.
- After each build, key tables are copied into paper_tables/ for stable referencing in the paper.
- Files in this folder are not meant to be edited manually; they mirror finalized outputs at the time of writing.

Typical contents
----------------
- sirs_trends.csv — normalized SIRS item means by term
- fuzzy_effectiveness_by_section.csv — composite effectiveness scores
- grades_summary.csv — GPA and grade distribution summaries
- sentiment_counts_by_prompt.csv — by-prompt sentiment aggregation
- sentiment_counts_by_semester.csv — term-level sentiment totals

Notes
-----
If rerunning the pipeline, the paper_tables/ folder will be refreshed automatically.
For full reproducibility or intermediate diagnostics, refer to outputs/tables/.

License
-------
Data: CC BY 4.0 (or CC BY-NC 4.0 for non-commercial reuse)
