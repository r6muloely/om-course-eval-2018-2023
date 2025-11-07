import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def line_trends(trends: pd.DataFrame, outdir: str) -> None:
    """
    Plot normalized SIRS trends by term. Expects columns:
      - 'year', 'term'
      - one or more metric columns ending with '_norm'
    Saves PNG and PDF to `outdir` as 'sirs_trends'.
    """
    # Ensure output directory exists
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Validate and prepare data
    if not {'year', 'term'}.issubset(trends.columns):
        raise ValueError("`trends` must include 'year' and 'term' columns.")
    df = trends.copy()
    if not any(c.endswith('_norm') for c in df.columns):
        raise ValueError("`trends` must include at least one column ending with '_norm'.")

    # Enforce chronological order: Spring before Fall within each year
    order_map = {'spring': 1, 'fall': 2, 's': 1, 'f': 2}
    df['__order__'] = df['term'].astype(str).str.lower().map(lambda t: order_map.get(t, 99))
    df = df.sort_values(['year', '__order__']).drop(columns='__order__').reset_index(drop=True)

    # Build x-axis and labels
    x = np.arange(len(df))
    labels = [f"{t} {int(y)}" for y, t in zip(df['year'], df['term'])]

    # Draw lines (distinct markers)
    fig = plt.figure(figsize=(10, 4.2))
    markers = ['o', '^', 's', 'D']  # cycle if >3 *_norm series
    series = [c for c in df.columns if c.endswith('_norm')]

    # NEW: explicit legend labels to harmonize naming across figures
    label_map = {
        'clarity_norm': 'Clarity',
        'fairness_norm': 'Fairness',
        'overall_norm': 'Course quality',               # force rename from "Overall"
        'teaching_effectiveness_norm': 'Teaching effectiveness',
    }

    for i, col in enumerate(series):
        key = col.strip()  # normalize whitespace
        label = label_map.get(key, key.replace('_norm', '').replace('_', ' ').title())
        plt.plot(x, df[col], marker=markers[i % len(markers)], label=label)


    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Normalized score')
    plt.title('SIRS normalized trends')
    plt.grid(axis='y', alpha=0.15)

    # Legend (semi-transparent box so it never occludes lines)
    leg = plt.legend(title='SIRS metric', loc='best', frameon=True)
    leg.get_frame().set_alpha(0.85)
    leg.get_frame().set_facecolor('white')

    # Save outputs
    fig.savefig(f"{outdir}/sirs_trends.png", dpi=200, bbox_inches='tight')
    fig.savefig(f"{outdir}/sirs_trends.pdf",  bbox_inches='tight')
    plt.close(fig)


def plot_teaching_and_overall_1to5(trends: pd.DataFrame, cfg: dict, outdir: str) -> None:
    """
    Plot the 1–5 series for teaching effectiveness and course quality
    derived from the normalized columns in `trends`.
    Denormalization uses cfg['scales']['sirs_min'] and cfg['scales']['sirs_max'].
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    df = trends.copy()
    order = {'Spring':1,'Fall':2,'S':1,'F':2}
    df['__order__'] = df['term'].map(order)
    df = df.sort_values(['year','__order__']).drop(columns='__order__').reset_index(drop=True)

    if 'teaching_effectiveness_norm' not in df.columns or 'overall_norm' not in df.columns:
        raise ValueError("trends must contain 'teaching_effectiveness_norm' and 'overall_norm'.")

    smin = cfg['scales']['sirs_min']
    smax = cfg['scales']['sirs_max']
    span = (smax - smin)

    # correct back-transform: x = smin + norm * (smax - smin)
    df['te_1to5']      = smin + df['teaching_effectiveness_norm'].astype(float) * span
    df['overall_1to5'] = smin + df['overall_norm'].astype(float) * span

    x = np.arange(len(df))
    labels = [f"{t} {int(y)}" for y, t in zip(df['year'], df['term'])]

    fig = plt.figure(figsize=(10, 4.8))
    plt.plot(x, df['te_1to5'],      marker='o', label='Teaching effectiveness')
    plt.plot(x, df['overall_1to5'], marker='s', label='Course quality')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Average score (1–5 scale)')
    plt.title('SIRS trend: teaching effectiveness and course quality (1–5)')

    leg = plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    plt.tight_layout()
    plt.subplots_adjust(right=0.84)

    fig.savefig(f"{outdir}/sirs_trend_teaching_overall_1to5.png", dpi=200, bbox_inches='tight')
    fig.savefig(f"{outdir}/sirs_trend_teaching_overall_1to5.pdf",  bbox_inches='tight')
    plt.close(fig)



def sentiment_trend_by_semester(counts_df: pd.DataFrame, outdir: str) -> None:
    """
    Plot raw counts of positive/neutral/negative comments by term.
    Expects columns: ['year','term','positive','neutral','negative'].
    Saves PNG/PDF to outdir as 'sentiment_trend_by_semester.*'.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Validate and copy
    required = {'year', 'term'}
    if not required.issubset(counts_df.columns):
        raise ValueError("counts_df must include 'year' and 'term' columns.")
    df = counts_df.copy()

    # Ensure sentiment columns exist even if a label never appears in a term
    for c in ['positive', 'neutral', 'negative']:
        if c not in df.columns:
            df[c] = 0

    # Chronological order: Spring -> Fall within each year
    order_map = {'spring': 1, 'fall': 2, 's': 1, 'f': 2}
    df['__order__'] = df['term'].astype(str).str.lower().map(lambda t: order_map.get(t, 99))
    df = df.sort_values(['year', '__order__']).drop(columns='__order__')

    # X and labels
    x = np.arange(len(df))
    labels = [f"{t} {int(y)}" for y, t in zip(df['year'], df['term'])]

    # Plot
    fig = plt.figure(figsize=(10, 4.2))
    # plot with distinct markers
    plt.plot(x, df['positive'], marker='o', label='Positive')   # circle
    plt.plot(x, df['neutral'],  marker='^', label='Mixed')      # triangle
    plt.plot(x, df['negative'], marker='s', label='Negative')   # square
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Number of comments')
    plt.title('Sentiment trends in student feedback (counts)')
    plt.grid(axis='y', alpha=0.15)
    leg = plt.legend(title='Sentiment',
                 loc='center left', bbox_to_anchor=(1.02, 0.5),
                 frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    plt.tight_layout()                 # first let matplotlib lay things out
    plt.subplots_adjust(right=0.84)    # then give the legend some room on the right

    # Save
    fig.savefig(f"{outdir}/sentiment_trend_by_semester.png", dpi=200, bbox_inches='tight')
    fig.savefig(f"{outdir}/sentiment_trend_by_semester.pdf",  bbox_inches='tight')
    plt.close(fig)


def sentiment_trend_share_by_semester(counts_df: pd.DataFrame, outdir: str) -> None:
    """
    Plot percentage shares of positive/neutral/negative comments by term.
    Expects columns: ['year','term','positive','neutral','negative'].
    Saves PNG/PDF as 'sentiment_trend_share_by_semester.*'.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Validate and copy
    if not {'year', 'term'}.issubset(counts_df.columns):
        raise ValueError("counts_df must include 'year' and 'term' columns.")
    df = counts_df.copy()

    # Ensure sentiment columns exist even if absent in some terms
    for col in ['positive', 'neutral', 'negative']:
        if col not in df.columns:
            df[col] = 0

    # Chronological order: Spring → Fall within each year
    order_map = {'spring': 1, 'fall': 2, 's': 1, 'f': 2}
    df['__order__'] = df['term'].astype(str).str.lower().map(lambda t: order_map.get(t, 99))
    df = df.sort_values(['year', '__order__']).drop(columns='__order__').reset_index(drop=True)

    # Shares
    totals = df[['positive', 'neutral', 'negative']].sum(axis=1).replace(0, np.nan)
    share = df[['positive', 'neutral', 'negative']].div(totals, axis=0) * 100.0

    # X and labels
    x = np.arange(len(df))
    labels = [f"{t} {int(y)}" for y, t in zip(df['year'], df['term'])]

    # Plot
    fig = plt.figure(figsize=(10, 4.2))
    plt.plot(x, share['positive'], marker='o', label='Positive')
    plt.plot(x, share['neutral'],  marker='^', label='Mixed')
    plt.plot(x, share['negative'], marker='s', label='Negative')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Share of comments (%)')
    plt.title('Sentiment shares in student feedback')
    plt.grid(axis='y', alpha=0.15)
    leg = plt.legend(title='Sentiment',
                 loc='center left', bbox_to_anchor=(1.02, 0.5),
                 frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    plt.tight_layout()                 # first let matplotlib lay things out
    plt.subplots_adjust(right=0.84)    # then give the legend some room on the right

    # Save
    fig.savefig(f"{outdir}/sentiment_trend_share_by_semester.png", dpi=200, bbox_inches='tight')
    fig.savefig(f"{outdir}/sentiment_trend_share_by_semester.pdf",  bbox_inches='tight')
    plt.close(fig)


def fuzzy_sensitivity_scatter(sens_df: pd.DataFrame, outdir: str) -> None:
    """
    Baseline vs shifted effectiveness scatter with 45° line.
    Expects columns: ['effectiveness_score_baseline','effectiveness_score_shifted'].
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    required = {'effectiveness_score_baseline', 'effectiveness_score_shifted'}
    if not required.issubset(sens_df.columns):
        missing = required - set(sens_df.columns)
        raise ValueError(f"sens_df missing required columns: {missing}")

    x = sens_df['effectiveness_score_baseline'].to_numpy()
    y = sens_df['effectiveness_score_shifted'].to_numpy()

    fig = plt.figure(figsize=(5.5, 5.0))
    plt.scatter(x, y, s=24, alpha=0.7)
    lim_lo = float(min(x.min(), y.min(), 0.0))
    lim_hi = float(max(x.max(), y.max(), 1.0))
    plt.plot([lim_lo, lim_hi], [lim_lo, lim_hi], linestyle='--')
    plt.xlabel('Baseline effectiveness')
    plt.ylabel('Shifted effectiveness')
    plt.title('Fuzzy effectiveness sensitivity (±δ cutpoints)')
    plt.xlim(lim_lo, lim_hi)
    plt.ylim(lim_lo, lim_hi)
    fig.savefig(f"{outdir}/fuzzy_sensitivity_scatter.png", dpi=200, bbox_inches='tight')
    fig.savefig(f"{outdir}/fuzzy_sensitivity_scatter.pdf",  bbox_inches='tight')
    plt.close(fig)


def grade_distribution_by_semester(grades_df: pd.DataFrame, outdir: str, collapse_plus: bool = True) -> None:
    """
    Two-panel figure: counts (top) and percentages (bottom) by semester.
    Expects section-level rows with: ['year','term','student_count'] and grade shares in pct_*
    e.g., ['pct_A','pct_B','pct_C','pct_D','pct_F','pct_W'] (optionally with 'pct_B+','pct_C+').
    If collapse_plus=True, B+→B and C+→C before aggregation.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Basic validation
    req = {'year', 'term', 'student_count'}
    if not req.issubset(grades_df.columns):
        missing = req - set(grades_df.columns)
        raise ValueError(f"grades_df is missing required columns: {missing}")

    g = grades_df.copy()

    # Identify grade share columns
    share_cols = [c for c in g.columns if c.startswith('pct_')]
    if not share_cols:
        raise ValueError("No grade-share columns (pct_*) found in grades_df.")

    # Collapse plus categories if requested
    if collapse_plus:
        if 'pct_B+' in g.columns:
            g['pct_B'] = g.get('pct_B', 0) + g['pct_B+']
            g.drop(columns=['pct_B+'], inplace=True)
        if 'pct_C+' in g.columns:
            g['pct_C'] = g.get('pct_C', 0) + g['pct_C+']
            g.drop(columns=['pct_C+'], inplace=True)

    # Final plotting order (keep only those present)
    grades_order = [c for c in ['pct_A','pct_B','pct_C','pct_D','pct_F','pct_W'] if c in g.columns]
    if not grades_order:
        raise ValueError("No recognized grade-share columns among pct_A,B,C,D,F,W.")

    # Ensure numeric
    for c in grades_order:
        g[c] = pd.to_numeric(g[c], errors='coerce').fillna(0.0)
    g['student_count'] = pd.to_numeric(g['student_count'], errors='coerce').fillna(0).astype(int)

    # Chronological label and sort key
    order_map = {'Spring': 1, 'Fall': 2, 'S': 1, 'F': 2}
    g['term_order'] = g['term'].map(order_map)

    # Convert shares to counts per section, then aggregate to term
    agg = g[['year', 'term', 'term_order', 'student_count'] + grades_order].copy()
    for c in grades_order:
        agg[c] = agg[c] * agg['student_count']  # shares are decimals in [0,1]

    counts = (
        agg.groupby(['year','term','term_order'], as_index=False)[grades_order + ['student_count']]
           .sum()
           .sort_values(['year','term_order'])
           .reset_index(drop=True)
    )

    # Percentages by semester (only over the grade columns, not including student_count)
    totals = counts[grades_order].sum(axis=1).replace(0, np.nan)
    shares = counts.copy()
    for c in grades_order:
        shares[c] = (shares[c] / totals) * 100.0

    # X axis
    labels = counts.apply(lambda r: f"{r['term']} {int(r['year'])}", axis=1)
    x = np.arange(len(labels))

    # Figure
    fig = plt.figure(figsize=(9, 10))

    # Top: counts (stacked)
    ax1 = fig.add_subplot(2, 1, 1)
    bottom = np.zeros(len(x))
    for c in grades_order:
        ax1.bar(x, counts[c].values, bottom=bottom, label=c.replace('pct_', ''))
        bottom += counts[c].values
    ax1.set_title("Grade distribution by semester (counts)")
    ax1.set_ylabel("Number of students")
    ax1.set_xticks(x, labels, rotation=45, ha='right')
    ax1.legend(title="Letter grade", ncol=min(6, len(grades_order)))

    # Bottom: percentages (stacked)
    ax2 = fig.add_subplot(2, 1, 2)
    bottom = np.zeros(len(x))
    for c in grades_order:
        ax2.bar(x, shares[c].values, bottom=bottom, label=c.replace('pct_', ''))
        bottom += shares[c].values
    ax2.set_title("Grade distribution by semester (percentages)")
    ax2.set_ylabel("Percentage of students")
    ax2.set_xticks(x, labels, rotation=45, ha='right')

    fig.tight_layout()
    name = 'grade_distribution_collapsed' if collapse_plus else 'grade_distribution_uncollapsed'
    fig.savefig(f"{outdir}/{name}.png", dpi=200, bbox_inches='tight')
    fig.savefig(f"{outdir}/{name}.pdf", bbox_inches='tight')
    plt.close(fig)


# --- 2×2 multi-panel of per-prompt sentiment shares ---
def sentiment_trend_share_multipanel(
    counts_df: pd.DataFrame,
    prompts,
    outdir: str,
    filename: str = "sentiment_trend_share_prompts_multipanel",
) -> None:
    """
    Draw a 2×2 grid of sentiment share-by-term line charts for multiple prompt types.

    Parameters
    ----------
    counts_df : DataFrame
        Columns required: ['year','term','prompt_type','positive','neutral','negative'].
    prompts : iterable of str
        Prompt labels to include (e.g., ['best','change','growth','other']).
    outdir : str
        Output directory for the figure.
    filename : str
        Base filename (without extension) for the saved figure.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Validate and normalize labels
    required = {'year', 'term', 'prompt_type'}
    if not required.issubset(counts_df.columns):
        missing = required - set(counts_df.columns)
        raise ValueError(f"counts_df missing required columns: {missing}")

    df = counts_df.copy()
    df['prompt'] = df['prompt_type'].astype(str).str.strip().str.lower()
    want = [str(p).strip().lower() for p in prompts]
    present = [p for p in want if p in df['prompt'].unique()]
    if not present:
        print(f"[sentiment_trend_share_multipanel] None of the requested prompts {prompts} present; skipping.")
        return

    # Layout: up to 4 panels in 2×2
    n = min(len(present), 4)
    rows = 2 if n > 2 else 1
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.2 * rows), squeeze=False)

    order_map = {'spring': 1, 'fall': 2, 's': 1, 'f': 2}

    for idx, p in enumerate(present[:4]):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        sub = df[df['prompt'] == p].copy()
        # Ensure sentiment columns exist
        for col in ['positive', 'neutral', 'negative']:
            if col not in sub.columns:
                sub[col] = 0

        # Chronological order and sorted copy
        sub['__order__'] = sub['term'].astype(str).str.lower().map(lambda t: order_map.get(t, 99))
        sub_sorted = sub.sort_values(['year', '__order__']).drop(columns='__order__').reset_index(drop=True)

        # Shares
        totals = sub_sorted[['positive','neutral','negative']].sum(axis=1).replace(0, np.nan)
        share = sub_sorted[['positive','neutral','negative']].div(totals, axis=0) * 100.0

        # X and labels
        x = np.arange(len(sub_sorted))
        labels = [f"{t} {int(y)}" for y, t in zip(sub_sorted['year'], sub_sorted['term'])]

        # Plot lines
        ax.plot(x, share['positive'], marker='o', label='Positive')
        ax.plot(x, share['neutral'],  marker='^', label='Mixed')
        ax.plot(x, share['negative'], marker='s', label='Negative')
        ax.set_title(f"{p.capitalize()} prompt")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Share of comments (%)')
        ax.grid(axis='y', alpha=0.15)
        if idx == 0:
            leg = ax.legend(title='Sentiment',
                    loc='center left', bbox_to_anchor=(1.02, 0.5),
                    frameon=True)
            leg.get_frame().set_alpha(0.9)
            leg.get_frame().set_facecolor('white')
        fig.tight_layout()
        fig.subplots_adjust(right=0.86)    # widen the canvas for the legend column

        plt.grid(axis='both', alpha=0.12, linestyle='-')
        # or keep only y-grid as you have now; your current style is already clean.




    # Hide any unused panels
    total_panels = rows * cols
    for j in range(n, total_panels):
        r, c = divmod(j, cols)
        axes[r][c].axis('off')

    fig.tight_layout()
    out = Path(outdir)
    fig.savefig(out / f"{filename}.png", dpi=200, bbox_inches='tight')
    fig.savefig(out / f"{filename}.pdf",  bbox_inches='tight')
    plt.close(fig)


def sentiment_trend_share_by_prompt(counts_df: pd.DataFrame, outdir: str, prompt: str) -> None:
    """
    Plot per-prompt sentiment shares by term.
    Expects columns: ['year','term','prompt_type','positive','neutral','negative'].
    Saves PNG/PDF as 'sentiment_trend_share_prompt_<prompt>.*'.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    required = {'year', 'term', 'prompt_type'}
    if not required.issubset(counts_df.columns):
        missing = required - set(counts_df.columns)
        raise ValueError(f"counts_df missing required columns: {missing}")

    # Work on a copy; normalize prompt label and filter
    df = counts_df.copy()
    df['prompt_type'] = df['prompt_type'].astype(str).str.strip().str.lower()
    prompt_norm = str(prompt).strip().lower()
    df = df[df['prompt_type'] == prompt_norm].copy()
    if df.empty:
        print(f"[sentiment_trend_share_by_prompt] No rows for prompt_type='{prompt}'; skipping.")
        return

    # Ensure sentiment columns exist
    for col in ['positive', 'neutral', 'negative']:
        if col not in df.columns:
            df[col] = 0

    # Chronological order: Spring → Fall within each year
    order_map = {'spring': 1, 'fall': 2, 's': 1, 'f': 2}
    df['__order__'] = df['term'].astype(str).str.lower().map(lambda t: order_map.get(t, 99))
    df = df.sort_values(['year', '__order__']).drop(columns='__order__').reset_index(drop=True)

    # Shares
    totals = df[['positive','neutral','negative']].sum(axis=1).replace(0, np.nan)
    share = df[['positive','neutral','negative']].div(totals, axis=0) * 100.0

    # X and labels
    x = np.arange(len(df))
    labels = [f"{t} {int(y)}" for y, t in zip(df['year'], df['term'])]

    # Plot
    fig = plt.figure(figsize=(10, 4.2))
    plt.plot(x, share['positive'].values, marker='o', label='Positive')
    plt.plot(x, share['neutral'].values,  marker='^', label='Mixed')
    plt.plot(x, share['negative'].values, marker='s', label='Negative')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Share of comments (%)')
    plt.title(f"Sentiment by prompt: {prompt_norm}")
    plt.grid(axis='y', alpha=0.15)
    leg = plt.legend(title='Sentiment',
                 loc='center left', bbox_to_anchor=(1.02, 0.5),
                 frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    plt.tight_layout()                 # first let matplotlib lay things out
    plt.subplots_adjust(right=0.84)    # then give the legend some room on the right

    # Save
    out = Path(outdir)
    fig.savefig(out / f"sentiment_trend_share_prompt_{prompt_norm}.png", dpi=200, bbox_inches='tight')
    fig.savefig(out / f"sentiment_trend_share_prompt_{prompt_norm}.pdf",  bbox_inches='tight')
    plt.close(fig)






