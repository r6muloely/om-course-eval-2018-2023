from pathlib import Path
import pandas as pd

from src.utils import seed_everything
from src.data_loading import load_config, load_all
from src.sirs_metrics import compute_section_level_metrics, aggregate_trends
from src.sentiment import score_sentiment
from src.topics import fit_topics, top_words_per_topic
from src.models.fuzzy_model import score_effectiveness, score_effectiveness_shifted
from src.grade_analysis import merge_with_grades, summarize_grade_alignment
from src.pipelines.validate_inputs import ensure_output_dirs, validate_inputs
from src.visualization.plots import (
    line_trends,
    plot_teaching_and_overall_1to5,
    sentiment_trend_by_semester,
    sentiment_trend_share_by_semester,
    grade_distribution_by_semester,
    fuzzy_sensitivity_scatter,
    sentiment_trend_share_by_prompt,      # per-prompt shares (single-panel)
    sentiment_trend_share_multipanel,   # optional 2×2 grid; uncomment only if you added it
)


def build_all():
    ensure_output_dirs()
    seed_everything(42)
    cfg = load_config()
    validate_inputs(cfg, strict=True)

    # Load de-identified, section–term–linked inputs
    sirs, comments, grades = load_all(cfg)

    # Handy column aliases used later
    sec_col = cfg['columns']['section']
    ycol    = cfg['columns']['year']
    tcol    = cfg['columns']['term']
    textcol = cfg['columns']['comment_text']

    # ---------------------------
    # SIRS metrics
    # ---------------------------
    sirs2  = compute_section_level_metrics(sirs, cfg)
    # optional: quick visibility while we finalize TE wiring
    # print("DEBUG sirs2 columns:", sirs2.columns.tolist())

    # Safety guard: create TE normalized if the normalizer didn't
    if ('teaching_effectiveness' in sirs2.columns
        and 'teaching_effectiveness_norm' not in sirs2.columns):
        smin, smax = cfg['scales']['sirs_min'], cfg['scales']['sirs_max']
        sirs2['teaching_effectiveness_norm'] = (
            sirs2['teaching_effectiveness'].astype(float) - smin
        ) / (smax - smin)
        # print("DEBUG: created teaching_effectiveness_norm in build_all")

    trends = aggregate_trends(sirs2, cfg)
    trends.to_csv('outputs/tables/sirs_trends.csv', index=False)
    print("SIRS: trends table written.")

    # Plots for SIRS trends (both built from the same 'trends' table)
    line_trends(trends, outdir='outputs/figures')                         # normalized series figure
    plot_teaching_and_overall_1to5(trends, cfg, outdir='outputs/figures') # 1–5 series for TE and Overall



    # ---------------------------
    # Sentiment (all comments)
    # ---------------------------
    sent = score_sentiment(comments, cfg)

    # Export per-comment sentiment with year/term for downstream pivots
    sent[[sec_col, ycol, tcol, 'sent_compound', 'sent_label']].to_csv(
        'outputs/tables/sentiment_by_section.csv', index=False
    )

    # Per-semester counts for all comments (positive / neutral / negative)
    order_map = {'Spring': 1, 'Fall': 2, 'S': 1, 'F': 2}
    tmp = sent.copy()
    tmp['term'] = tmp[tcol].astype(str)
    tmp['term_order'] = tmp['term'].map(order_map)

    counts = (
        tmp.groupby([ycol, 'term', 'sent_label'])
           .size().reset_index(name='n')
           .pivot(index=[ycol, 'term'], columns='sent_label', values='n')
           .fillna(0)
           .reset_index()
    )
    counts['term_order'] = counts['term'].map(order_map)
    counts = counts.sort_values([ycol, 'term_order']).drop(columns=['term_order']).reset_index(drop=True)



    # Ensure expected sentiment columns even if a bucket is absent in a term
    for c in ['positive', 'neutral', 'negative']:
        if c not in counts.columns:
            counts[c] = 0

    counts.to_csv('outputs/tables/sentiment_counts_by_semester.csv', index=False)

    # Figures (counts and percentage shares)
    sentiment_trend_by_semester(counts, outdir='outputs/figures')
    sentiment_trend_share_by_semester(counts, outdir='outputs/figures')
    print("Sentiment: all-comments counts/shares written.")


    # ---------------------------
    # Per-prompt sentiment counts + shares (best / change / growth / other)
    # ---------------------------
    if 'prompt_type' in sent.columns:
        tmp_p = sent.copy()

        # Normalize prompt labels to the canonical set
        tmp_p['prompt_type'] = (
            tmp_p['prompt_type']
                .astype(str).str.strip().str.lower()
                .replace({
                    'best prompt': 'best',
                    'do differently': 'change',
                    'changes': 'change',
                    'growth/learning': 'growth',
                })
        )

        # Chronological order helper (if you want the CSV sorted)
        order_map = {'Spring': 1, 'Fall': 2, 'S': 1, 'F': 2}
        tmp_p['term_order'] = tmp_p[tcol].map(order_map)

        # Aggregate to term × prompt_type × sentiment counts
        counts_by_prompt = (
            tmp_p.groupby([ycol, tcol, 'prompt_type', 'sent_label'])
                 .size().reset_index(name='n')
                 .pivot(index=[ycol, tcol, 'prompt_type'], columns='sent_label', values='n')
                 .fillna(0)
                 .reset_index()
        )

        # Optional: sort for deterministic CSV output
        counts_by_prompt['term_order'] = counts_by_prompt[tcol].map(order_map)
        counts_by_prompt = (counts_by_prompt
                            .sort_values([ycol, 'term_order', 'prompt_type'])
                            .drop(columns=['term_order'])
                            .reset_index(drop=True))

        counts_by_prompt.to_csv('outputs/tables/sentiment_counts_by_prompt.csv', index=False)

        # Emit one chart per prompt present
        available = counts_by_prompt['prompt_type'].dropna().unique().tolist()
        for p in ['best', 'change', 'growth', 'other']:
            if p in available:
                sentiment_trend_share_by_prompt(
                    counts_by_prompt, outdir='outputs/figures', prompt=p
                )

        # 2×2 multi-panel comparing prompts (if at least one is present)
        if available:
            panel_prompts = [p for p in ['best','change','growth','other'] if p in available][:4]
            sentiment_trend_share_multipanel(
                counts_by_prompt, prompts=panel_prompts, outdir='outputs/figures'
            )

        print("Sentiment: by-prompt counts/shares written.")


    # ---------------------------
    # Topics (robust to empty/noisy text)
    # ---------------------------
    try:
        # keep only needed columns; coerce to string, strip, drop trivially short
        valid = comments[[sec_col, ycol, tcol, textcol]].copy()
        valid[textcol] = valid[textcol].astype(str).str.strip()
        valid = valid[valid[textcol].str.len() >= 2]  # drop "", "N/A", etc.

        if valid.empty:
            print("Topics: no eligible comments after filtering; skipping.")
            pd.DataFrame({'topic': [], 'top_words': []}).to_csv(
                'outputs/tables/topics_top_words.csv', index=False
            )
        else:
            # fit LDA over TF-IDF per your fit_topics(valid, cfg)
            lda, W, H, feature_names = fit_topics(valid, cfg)
            topics = top_words_per_topic(H, feature_names, n_top=12)
            topics.to_csv('outputs/tables/topics_top_words.csv', index=False)
            print(f"Topics: fitted on {len(valid)} comments; wrote topics_top_words.csv")
            print(f"Topics: table written.")
    except ValueError as e:
        # common sklearn error when, after filtering, the vocabulary is empty
        if 'empty vocabulary' in str(e).lower():
            print("Topics: empty vocabulary after filtering; writing empty table.")
            pd.DataFrame({'topic': [], 'top_words': []}).to_csv(
                'outputs/tables/topics_top_words.csv', index=False
            )
        else:
            # re-raise unexpected ValueErrors so they are visible
            raise

    except Exception as e:
        print(f"Topics step skipped due to error: {e}")
        pd.DataFrame({'topic': [], 'top_words': []}).to_csv(
            'outputs/tables/topics_top_words.csv', index=False
        )
        print("Topics: empty or skipped; empty table written.")



    # ---------------------------
    # Merge mean sentiment to SIRS (section–term level)
    # ---------------------------
    sent_mean = (
        sent.groupby([sec_col, ycol, tcol])['sent_compound']
            .agg(sentiment_mean='mean', n_comments='size')
            .reset_index()
    )

    sirs_sent = (
        sirs2.merge(sent_mean, on=[sec_col, ycol, tcol], how='left')
             .fillna({'sentiment_mean': 0.0, 'n_comments': 0})
    )


    # ---------------------------
    # Fuzzy effectiveness
    # ---------------------------
    sirs_fuzzy = score_effectiveness(sirs_sent, 'clarity_norm', 'fairness_norm', 'sentiment_mean')

    # Optional sanity checks (leave in or remove once stable)
    assert sirs_fuzzy['clarity_norm'].between(0, 1).all()
    assert sirs_fuzzy['fairness_norm'].between(0, 1).all()
    assert sirs_fuzzy['effectiveness_score'].between(0, 1).all()

    # Compose a tidy export with the full key; include n_comments if present
    base_cols = [sec_col, ycol, tcol, 'clarity_norm', 'fairness_norm', 'sentiment_mean', 'effectiveness_score']
    if 'n_comments' in sirs_fuzzy.columns:
        base_cols.insert(3, 'n_comments')  # after term keys
    fe_out = sirs_fuzzy[base_cols].copy()

    # Optional human-friendly rounding (keeps upstream precision intact)
    for c in ['clarity_norm', 'fairness_norm', 'sentiment_mean', 'effectiveness_score']:
        fe_out[c] = fe_out[c].astype(float).round(3)

    fe_out.to_csv('outputs/tables/fuzzy_effectiveness_by_section.csv', index=False)
    print("Fuzzy: effectiveness table written.")

    # ---------------------------
    # Sensitivity analysis: shifted membership cutpoints (delta=0.05)
    # ---------------------------
    delta = 0.05
    thr   = 0.02  # tolerance for |Δ| exceedance

    sirs_fuzzy_shift = score_effectiveness_shifted(
        sirs_sent,
        clarity_col='clarity_norm',
        fairness_col='fairness_norm',
        sent_col='sentiment_mean',
        delta=delta
    )

    # Merge baseline and shifted on the section–term key
    sens = (
        sirs_fuzzy[[sec_col, ycol, tcol, 'effectiveness_score']]
        .merge(
            sirs_fuzzy_shift[[sec_col, ycol, tcol, 'effectiveness_score_shifted']],
            on=[sec_col, ycol, tcol], how='inner'
        )
        .rename(columns={'effectiveness_score': 'effectiveness_score_baseline'})
    )

    # Differences and basic sanity
    sens['diff'] = sens['effectiveness_score_shifted'] - sens['effectiveness_score_baseline']
    # (optional) assert ranges to catch any unexpected values
    assert sens['effectiveness_score_baseline'].between(0, 1).all()
    assert sens['effectiveness_score_shifted'].between(0, 1).all()

    # Summary stats (printed to console)
    if sens[['effectiveness_score_baseline','effectiveness_score_shifted']].notna().all(axis=None) and len(sens) > 1:
        r = sens[['effectiveness_score_baseline','effectiveness_score_shifted']].corr().iloc[0,1]
    else:
        r = float('nan')
    med_abs = sens['diff'].abs().median()
    share_gt_thr = float((sens['diff'].abs() > thr).mean())
    pct_gt_thr   = 100.0 * share_gt_thr
    print(f"Sensitivity (delta={delta}): r={r:.3f}, median|Δ|={med_abs:.3f}, share(|Δ|>{thr:.2f})={share_gt_thr:.3f} ({pct_gt_thr:.1f}%)")

    # Save the sensitivity table and figure
    sens_out = sens[[sec_col, ycol, tcol,
                     'effectiveness_score_baseline',
                     'effectiveness_score_shifted',
                     'diff']].copy()
    # human-friendly rounding for the CSV (upstream precision remains in `sens`)
    for c in ['effectiveness_score_baseline','effectiveness_score_shifted','diff']:
        sens_out[c] = sens_out[c].astype(float).round(3)

    sens_out.to_csv('outputs/tables/fuzzy_sensitivity.csv', index=False)
    fuzzy_sensitivity_scatter(sens, outdir='outputs/figures')  # pass full-precision frame to the plotter
    print(f"Sensitivity: delta=0.05 → r={r:.3f}, median|Δ|={med_abs:.3f}, share(|Δ|>{thr:.2f})={pct_gt_thr:.1f}%.")


    # ---------------------------
    # Outlier snapshot (lowest 3 / highest 3 sections)
    # ---------------------------
    try:
        k = 3  # how many lows/highs to show

        # lowest/highest by effectiveness
        low  = sirs_fuzzy.nsmallest(k, 'effectiveness_score')
        high = sirs_fuzzy.nlargest(k,  'effectiveness_score')
        focus = pd.concat([low, high], ignore_index=True).drop_duplicates(subset=[sec_col])

        # comment counts by sentiment for those sections (use full key for safety)
        sent_counts = (
            sent.groupby([sec_col, ycol, tcol, 'sent_label'])
                .size().reset_index(name='n')
                .pivot(index=[sec_col, ycol, tcol], columns='sent_label', values='n')
                .reset_index()
        )
        # guarantee expected columns even if a bucket is absent
        for col in ['positive', 'neutral', 'negative']:
            if col not in sent_counts.columns:
                sent_counts[col] = 0
        sent_counts.rename(columns={'positive':'pos','neutral':'neu','negative':'neg'}, inplace=True)

        # join counts + add year/term from SIRS keys
        key_cols = [sec_col, ycol, tcol]
        sirs_keys = sirs2[key_cols].drop_duplicates()

        snap = (
            focus[[sec_col, ycol, tcol, 'clarity_norm','fairness_norm','sentiment_mean','effectiveness_score']]
            .merge(sent_counts[[sec_col, ycol, tcol, 'pos','neu','neg']], on=[sec_col, ycol, tcol], how='left')
            .merge(sirs_keys, on=[sec_col, ycol, tcol], how='left')  # keeps alignment explicit
            .fillna({'pos': 0, 'neu': 0, 'neg': 0})
        )

        # total comments for small-N context
        snap['n_comments'] = snap[['pos','neu','neg']].sum(axis=1).astype(int)

        # tiny interpretation line for each row
        def _explain(r):
            drivers = []
            if r['clarity_norm']   < 0.75: drivers.append('lower clarity')
            if r['fairness_norm']  < 0.75: drivers.append('lower fairness')
            if r['sentiment_mean'] < -0.02: drivers.append('more negative comments')
            if not drivers and r['effectiveness_score'] < 0.70:
                drivers.append('mixed inputs')
            if not drivers:
                return 'High effectiveness: strong clarity/fairness and positive sentiment.'
            return 'Lower effectiveness driven by ' + ', '.join(drivers) + '.'

        snap['interpretation'] = snap.apply(_explain, axis=1)

        # lows first (ascending), then highs (descending) — your concat already puts lows first
        snap = snap.sort_values(['effectiveness_score']).reset_index(drop=True)

        # columns & rounding
        export_cols = [
            sec_col, ycol, tcol,
            'clarity_norm','fairness_norm','sentiment_mean','effectiveness_score',
            'pos','neu','neg','n_comments','interpretation'
        ]
        snap = snap[export_cols]
        snap[['clarity_norm','fairness_norm','sentiment_mean','effectiveness_score']] = (
            snap[['clarity_norm','fairness_norm','sentiment_mean','effectiveness_score']].round(3)
        )

        # save CSV + Markdown and copy CSV to paper_tables
        snap_csv = 'outputs/tables/outlier_snapshot.csv'
        snap_md  = 'outputs/tables/outlier_snapshot.md'
        snap.to_csv(snap_csv, index=False)

        # markdown table (handy to paste in the appendix)
        md = snap.rename(columns={
            sec_col:'section_id', ycol:'year', tcol:'term',
            'clarity_norm':'Clarity','fairness_norm':'Fairness',
            'sentiment_mean':'Sentiment','effectiveness_score':'Effectiveness',
            'pos':'Pos','neu':'Neu','neg':'Neg','n_comments':'N'
        })
        md_str = '| ' + ' | '.join(md.columns) + ' |\n'
        md_str += '| ' + ' | '.join(['---']*len(md.columns)) + ' |\n'
        for _, r in md.iterrows():
            md_str += '| ' + ' | '.join(str(x) for x in r.values) + ' |\n'
        with open(snap_md, 'w', encoding='utf-8') as f:
            f.write(md_str)

        # also place the CSV in paper_tables
        Path('paper_tables/outlier_snapshot.csv').write_bytes(Path(snap_csv).read_bytes())

    except Exception as e:
        print(f"Outlier snapshot skipped due to: {e}")


    # ---------------------------
    # Grades alignment (descriptives)
    # ---------------------------
    merged = merge_with_grades(sirs_fuzzy, grades, cfg)
    grade_sum = summarize_grade_alignment(merged, cfg)
    grade_sum.to_csv('outputs/tables/grades_summary.csv', index=False)

    # Optional: make sure we can copy to paper_tables later
    Path('paper_tables').mkdir(parents=True, exist_ok=True)
    Path('paper_tables/grades_summary.csv').write_bytes(Path('outputs/tables/grades_summary.csv').read_bytes())

    # ---------------------------
    # Grade distributions figure(s)
    # ---------------------------
    # Light integrity checks (log only; do not abort)
    if 'student_count' not in grades.columns:
        print("Warning: 'student_count' missing in grades; counts plot may be zero.")
    # ensure percent columns numeric if present
    for c in ['pct_A','pct_B','pct_C','pct_D','pct_F','pct_W','pct_B+','pct_C+']:
        if c in grades.columns:
            grades[c] = pd.to_numeric(grades[c], errors='coerce').fillna(0.0)

    # Main figure (collapsed B+/C+ → B/C)
    grade_distribution_by_semester(grades, outdir='outputs/figures', collapse_plus=True)
    print("Grades: summary and distributions written.")

    # Appendix figure with B+ and C+ separated (only if those columns exist)
    if any(c in grades.columns for c in ['pct_B+','pct_C+']):
        grade_distribution_by_semester(grades, outdir='outputs/figures', collapse_plus=False)



    # ---------------------------
    # Figures
    # ---------------------------
    line_trends(trends, outdir='outputs/figures')
    plot_teaching_and_overall_1to5(trends, cfg, outdir='outputs/figures')



    # ---------------------------
    # Copy key tables to paper_tables
    # ---------------------------
    Path('paper_tables').mkdir(parents=True, exist_ok=True)
    key_tables = [
        'sirs_trends.csv',
        'topics_top_words.csv',
        'fuzzy_effectiveness_by_section.csv',
        'fuzzy_sensitivity.csv',              # sensitivity diagnostics
        'grades_summary.csv',
        'sentiment_counts_by_semester.csv',
        'sentiment_counts_by_prompt.csv',     # per-prompt counts (if present)
        'outlier_snapshot.csv',               # outlier snapshot (if present)
    ]
    for fname in key_tables:
        src = Path('outputs/tables') / fname
        dst = Path('paper_tables') / fname
        if src.exists():
            dst.write_bytes(src.read_bytes())

    print("Build complete. See outputs/ and paper_tables/.")


