from pathlib import Path
import pandas as pd

REQUIRED_GRADES = ["student_count","mean_gpa","median_gpa","sd_gpa",
                   "pct_A","pct_B","pct_C","pct_D","pct_F","pct_W"]

def ensure_output_dirs():
    for d in ["outputs/tables", "outputs/figures", "paper_tables"]:
        Path(d).mkdir(parents=True, exist_ok=True)

def _check_columns(df, need, name):
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}. Got: {list(df.columns)}")

def validate_inputs(cfg, strict=True):
    sirs_path = Path(cfg["data"]["sirs_quant"])
    com_path  = Path(cfg["data"]["comments"])
    grd_path  = Path(cfg["data"]["grades"])
    for p in [sirs_path, com_path, grd_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    sirs = pd.read_csv(sirs_path)
    com  = pd.read_csv(com_path)
    grd  = pd.read_csv(grd_path)

    c = cfg["columns"]
    keys = [c["section"], c["year"], c["term"]]

    _check_columns(sirs, keys + [c["clarity"], c["fairness"], c["overall"]], "sirs_quant")
    _check_columns(com,  keys + [c["comment_text"]], "comments")
    _check_columns(grd,  keys + REQUIRED_GRADES, "grades")

    s_dup = int(sirs.duplicated(keys).sum())
    g_dup = int(grd.duplicated(keys).sum())
    if s_dup or g_dup:
        msg = []
        if s_dup: msg.append(f"sirs_quant has {s_dup} duplicate key rows on {keys}")
        if g_dup: msg.append(f"grades has {g_dup} duplicate key rows on {keys}")
        raise ValueError("; ".join(msg))

    smin, smax = float(cfg["scales"]["sirs_min"]), float(cfg["scales"]["sirs_max"])
    out_ranges = {}
    for col in [c["clarity"], c["fairness"], c["overall"]]:
        mn, mx = sirs[col].min(skipna=True), sirs[col].max(skipna=True)
        bad = int(((sirs[col] < smin) | (sirs[col] > smax)).sum())
        out_ranges[col] = dict(min=float(mn), max=float(mx), out_of_range=bad)
        if strict and bad > 0:
            raise ValueError(f"{col} has {bad} values outside [{smin},{smax}] (min={mn}, max={mx})")

    print("SIRS:", sirs.shape, list(sirs.columns)[:10])
    print("Comments:", com.shape, list(com.columns)[:10])
    print("Grades:", grd.shape, list(grd.columns)[:12])
    print("SIRS ranges:", out_ranges)
    print("Preflight OK.")
