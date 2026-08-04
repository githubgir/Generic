"""Summarize and compare pandas DataFrames.

Public API:
    summarize(df)                          -> Summary (dict) describing one DataFrame
    summary.display()                      -> DisplaySummary: dict of readable DataFrames
    compare(df1, df2)                      -> dict comparing two DataFrames
    distance(summary1, summary2)           -> single float in [0, 1], 0 = identical
    nearest_matches(summary, reference)    -> reference entries ranked by distance
    assess_coverage(df, positive, negative)-> verdict on whether df needs a new test
    generate_sample(summary, n, seed)      -> DataFrame synthesized from a summary

distance() and everything built on it (nearest_matches, assess_coverage) work on
the dicts returned by summarize() rather than on raw DataFrames. That's what
makes them cheap to run against a reference set of hundreds of persisted
summaries in production, without needing to keep the original reference
DataFrames around. Summary is a plain dict subclass, so all of that keeps
working on summarize() output exactly as before - it only adds .display().
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import norm

# A finer grid than you'd want to *display* is kept here because distance()
# approximates a Wasserstein distance from these quantiles alone (summaries
# don't retain the raw values).
DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
DEFAULT_TOP_N = 10


# ------------------------------------------------------------------ #
# Single-DataFrame summary
# ------------------------------------------------------------------ #

def _summarize_index_level(values: pd.Index, name) -> dict:
    return {
        "name": name,
        "dtype": str(values.dtype),
        "n_missing": int(values.isna().sum()),
        "nunique": int(values.nunique(dropna=True)),
        "has_duplicates": bool(values.has_duplicates),
        "is_monotonic_increasing": bool(values.is_monotonic_increasing),
    }


def summarize_index(index: pd.Index) -> dict:
    """Names/dtypes/uniqueness of an Index or MultiIndex, per level."""
    if isinstance(index, pd.MultiIndex):
        levels = [
            _summarize_index_level(index.get_level_values(i), name)
            for i, name in enumerate(index.names)
        ]
    else:
        levels = [_summarize_index_level(index, index.name)]
    return {
        "nlevels": index.nlevels,
        "length": len(index),
        "levels": levels,
    }


def _column_stats(s: pd.Series, quantiles=DEFAULT_QUANTILES) -> dict:
    n = len(s)
    is_numeric = pd.api.types.is_numeric_dtype(s)
    is_float = pd.api.types.is_float_dtype(s)
    is_datetime = pd.api.types.is_datetime64_any_dtype(s)

    n_missing = int(s.isna().sum())
    stats = {
        "dtype": str(s.dtype),
        "n_missing": n_missing,
        "pct_missing": n_missing / n if n else np.nan,
        "nunique": int(s.nunique(dropna=True)),
    }

    # Display-only range for datetime columns; deliberately not folded into
    # the numeric branch below (mean/std/quantiles) - the numeric/categorical
    # split drives distance()/generate_sample()'s branching elsewhere, and
    # datetime arithmetic doesn't behave the same way as plain float math.
    non_null_dt = s.dropna()
    stats["dt_min"] = non_null_dt.min() if is_datetime and len(non_null_dt) else pd.NaT
    stats["dt_max"] = non_null_dt.max() if is_datetime and len(non_null_dt) else pd.NaT

    if is_float:
        inf_mask = np.isinf(s.to_numpy(dtype=float, na_value=0.0))
        n_inf = int(inf_mask.sum())
    else:
        n_inf = 0
    stats["n_inf"] = n_inf
    stats["pct_inf"] = n_inf / n if n else np.nan

    if is_numeric:
        finite = s[np.isfinite(s.astype(float))].astype(float)  # bool dtype breaks quantile()
        n_zero = int((finite == 0).sum())
        n_negative = int((finite < 0).sum())
        n_positive = int((finite > 0).sum())
        n_above_1 = int((finite.abs() > 1).sum())
        stats.update(
            n_zero=n_zero, pct_zero=n_zero / n if n else np.nan,
            n_negative=n_negative, pct_negative=n_negative / n if n else np.nan,
            n_positive=n_positive, pct_positive=n_positive / n if n else np.nan,
            n_above_1=n_above_1, pct_above_1=n_above_1 / n if n else np.nan,
            mean=float(finite.mean()) if len(finite) else np.nan,
            median=float(finite.median()) if len(finite) else np.nan,
            std=float(finite.std()) if len(finite) else np.nan,
            min=float(finite.min()) if len(finite) else np.nan,
            max=float(finite.max()) if len(finite) else np.nan,
        )
        for q in quantiles:
            stats[f"q{q}"] = float(finite.quantile(q)) if len(finite) else np.nan
    else:
        stats.update(
            n_zero=np.nan, pct_zero=np.nan,
            n_negative=np.nan, pct_negative=np.nan,
            n_positive=np.nan, pct_positive=np.nan,
            n_above_1=np.nan, pct_above_1=np.nan,
            mean=np.nan, median=np.nan, std=np.nan, min=np.nan, max=np.nan,
        )
        for q in quantiles:
            stats[f"q{q}"] = np.nan

    return stats


def column_stats(df: pd.DataFrame, quantiles=DEFAULT_QUANTILES) -> pd.DataFrame:
    """One row per column with dtype, missing/inf/zero/sign/quantile stats."""
    return pd.DataFrame(
        {col: _column_stats(df[col], quantiles) for col in df.columns}
    ).T


def top_values(df: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> dict:
    """Top-N value_counts per column."""
    return {col: df[col].value_counts().head(top_n) for col in df.columns}


def _rank_encode_column(s: pd.Series) -> pd.Series:
    """Map a column to a numeric proxy suitable for rank correlation: passed
    through as-is if numeric, or - for categorical/bool columns - each
    category mapped to the midpoint of its cumulative-frequency interval
    (e.g. a category covering the most-frequent 30% of rows lands at 0.15).
    NaNs stay NaN either way. This lets a single corr(method="spearman")
    call produce one association matrix spanning numeric and categorical
    columns alike."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    counts = s.value_counts()  # sorted descending by frequency, NaN excluded
    total = counts.sum()
    if total == 0:
        return pd.Series(np.nan, index=s.index)
    midpoints = (counts.cumsum() - counts / 2) / total
    return s.map(midpoints)


def associations(df: pd.DataFrame) -> pd.DataFrame:
    """Unified rank-based association matrix spanning numeric, categorical,
    and boolean columns (datetime columns are excluded). For a pair of
    numeric columns this is exactly their Spearman correlation; categorical
    columns participate via the frequency-rank encoding in
    _rank_encode_column.

    Caveat: for a *nominal* categorical column (no inherent order), this only
    detects a relationship with another column to the extent that category
    frequency happens to align with that column's values - e.g. it can miss
    a real "group b has much higher values than group a" effect if b isn't
    also the more/less frequent group. It's still a useful drift signal for
    distance() (a shift in this matrix means *something* about the joint
    structure changed), but generate_sample() does not rely on it for
    categorical<->numeric dependence - see conditional_numeric_stats()."""
    supported = df.select_dtypes(exclude=["datetime", "datetimetz"])
    encoded = pd.DataFrame({c: _rank_encode_column(supported[c]) for c in supported.columns})
    return encoded.corr(method="spearman")


def correlations(df: pd.DataFrame) -> dict:
    """Pearson/Spearman matrices for numeric columns, plus a unified
    association matrix spanning numeric and categorical columns."""
    numeric = df.select_dtypes(include=np.number)
    return {
        "pearson": numeric.corr(method="pearson"),
        "spearman": numeric.corr(method="spearman"),
        "association": associations(df),
    }


def conditional_numeric_stats(
    df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    quantiles=DEFAULT_QUANTILES,
) -> dict:
    """Per-category numeric summaries: {categorical_col: {category: column_stats
    of the numeric columns restricted to rows where categorical_col == category}}.

    This is what actually captures categorical<->numeric dependence for
    generate_sample() - e.g. a nominal group whose numeric values run
    systematically higher or lower than the rest - which associations()'s
    rank-encoding trick can't reliably represent (see its docstring). Only
    the categorical column's top-N categories get their own stats; rarer
    categories fall back to the unconditional marginal at generation time.

    Note this multiplies summary size by roughly
    n_categorical_cols * top_n * n_numeric_cols - worth keeping in mind if
    you're persisting many of these summaries.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    result = {}
    if not numeric_cols or not categorical_cols:
        return result
    for cat_col in categorical_cols:
        per_category = {}
        for category in df[cat_col].value_counts().head(top_n).index:
            subset = df.loc[df[cat_col] == category, numeric_cols]
            if len(subset):
                per_category[category] = column_stats(subset, quantiles=quantiles)
        if per_category:
            result[cat_col] = per_category
    return result


class Summary(dict):
    """dict returned by summarize(). A drop-in dict everywhere else in this
    module (summary["columns"], summary["correlations"]["pearson"], etc. all
    keep working exactly as before - this only adds a .display() method for
    a human-readable table view)."""

    def display(self, decimals: int = 3) -> "DisplaySummary":
        return display_summary(self, decimals=decimals)


def summarize(
    df: pd.DataFrame,
    quantiles=DEFAULT_QUANTILES,
    top_n: int = DEFAULT_TOP_N,
) -> Summary:
    """Full summary of a single DataFrame."""
    return Summary({
        "shape": df.shape,
        "index": summarize_index(df.index),
        "columns": column_stats(df, quantiles=quantiles),
        "top_values": top_values(df, top_n=top_n),
        "correlations": correlations(df),
        "conditional_numeric": conditional_numeric_stats(df, top_n=top_n, quantiles=quantiles),
    })


class DisplaySummary(dict):
    """dict of {name: DataFrame} returned by display_summary() / Summary.display().
    A plain dict everywhere (displayed["columns"], displayed["top_values"], ...)
    but renders every table automatically when it's the last expression in a
    Jupyter cell, or via print()/str() in a plain console."""

    def _repr_html_(self) -> str:
        parts = []
        for name, table in self.items():
            parts.append(f"<h4>{name}</h4>")
            parts.append(table.to_html() if isinstance(table, pd.DataFrame) else f"<pre>{table}</pre>")
        return "".join(parts)

    def __repr__(self) -> str:
        parts = [f"=== {name} ===\n{table}" for name, table in self.items()]
        return "\n\n".join(parts)


def display_summary(summary: dict, decimals: int = 3) -> DisplaySummary:
    """Human-readable view of a summarize() output, as a dict of DataFrames:

    - "overview": row/column counts by type, overall missing %.
    - "columns": one row per column - type, dtype, missing/zero/negative/
      positive/>1 % (numeric), nunique, mean/median/std/quantiles/min/max
      (numeric), top category + its % (categorical), date range (datetime).
    - "top_values": every column's top-N value counts in one tidy long table
      (columns, rank, value, count, pct), instead of a dict of Series.
    - "correlations": the unified numeric+categorical association matrix,
      rounded.

    Works on any dict shaped like a summarize() output, including ones
    deserialized from storage that lost the Summary class identity - you
    don't need a live Summary instance to call this.
    """
    cols = summary["columns"]
    n = summary["shape"][0]
    q_fields = _quantile_fields(cols.columns)

    def _col_kind(row) -> str:
        if str(row["dtype"]).startswith("datetime"):
            return "datetime"
        return "numeric" if _is_numeric_row(row) else "categorical"

    kinds = {c: _col_kind(cols.loc[c]) for c in cols.index}
    overview = pd.DataFrame({
        "metric": ["rows", "columns", "numeric_columns", "categorical_columns", "datetime_columns", "overall_missing_pct"],
        "value": [
            n,
            len(cols),
            sum(k == "numeric" for k in kinds.values()),
            sum(k == "categorical" for k in kinds.values()),
            sum(k == "datetime" for k in kinds.values()),
            round(float(cols["pct_missing"].mean()) * 100, decimals) if len(cols) else 0.0,
        ],
    })

    def _pct(row, field):
        v = row.get(field)
        return round(float(v) * 100, decimals) if v is not None and not pd.isna(v) else np.nan

    def _num(row, field):
        v = row.get(field)
        return round(float(v), decimals) if v is not None and not pd.isna(v) else np.nan

    rows = []
    for c in cols.index:
        row = cols.loc[c]
        kind = kinds[c]
        is_num = kind == "numeric"
        top = summary["top_values"].get(c)
        non_null = round(n * (1 - row["pct_missing"])) if n and not pd.isna(row["pct_missing"]) else 0
        top_value, top_value_pct = np.nan, np.nan
        if top is not None and len(top) and non_null:
            top_value = top.index[0]
            top_value_pct = round(float(top.iloc[0]) / non_null * 100, decimals)

        entry = {
            "column": c,
            "type": kind,
            "dtype": row["dtype"],
            "missing_pct": _pct(row, "pct_missing"),
            "nunique": row["nunique"],
            "mean": _num(row, "mean") if is_num else np.nan,
            "median": _num(row, "median") if is_num else np.nan,
            "std": _num(row, "std") if is_num else np.nan,
            "min": _num(row, "min") if is_num else np.nan,
        }
        for level, name in q_fields:
            entry[f"q{level}"] = _num(row, name) if is_num else np.nan
        entry.update({
            "max": _num(row, "max") if is_num else np.nan,
            "zero_pct": _pct(row, "pct_zero") if is_num else np.nan,
            "negative_pct": _pct(row, "pct_negative") if is_num else np.nan,
            "positive_pct": _pct(row, "pct_positive") if is_num else np.nan,
            "above_1_pct": _pct(row, "pct_above_1") if is_num else np.nan,
            "inf_pct": _pct(row, "pct_inf"),
            "top_value": top_value,
            "top_value_pct": top_value_pct,
            "date_min": row.get("dt_min", pd.NaT) if kind == "datetime" else pd.NaT,
            "date_max": row.get("dt_max", pd.NaT) if kind == "datetime" else pd.NaT,
        })
        rows.append(entry)
    columns_table = pd.DataFrame(rows).set_index("column")

    top_values_rows = []
    for c, counts in summary["top_values"].items():
        row = cols.loc[c]
        non_null = round(n * (1 - row["pct_missing"])) if n and not pd.isna(row["pct_missing"]) else 0
        for rank, (value, count) in enumerate(counts.items(), start=1):
            top_values_rows.append({
                "column": c,
                "rank": rank,
                "value": value,
                "count": int(count),
                "pct": round(float(count) / non_null * 100, decimals) if non_null else np.nan,
            })
    top_values_table = pd.DataFrame(top_values_rows, columns=["column", "rank", "value", "count", "pct"])

    return DisplaySummary({
        "overview": overview,
        "columns": columns_table,
        "top_values": top_values_table,
        "correlations": summary["correlations"]["association"].round(decimals),
    })


# ------------------------------------------------------------------ #
# Two-DataFrame comparison
# ------------------------------------------------------------------ #

def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def overlap(a, b) -> dict:
    """Set overlap between two iterables (columns or index values)."""
    set_a, set_b = set(a), set(b)
    return {
        "common": sorted(set_a & set_b, key=str),
        "only_in_first": sorted(set_a - set_b, key=str),
        "only_in_second": sorted(set_b - set_a, key=str),
        "jaccard": _jaccard(set_a, set_b),
    }


def compare_columns(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    quantiles=DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Side-by-side stats for columns common to both DataFrames."""
    stats1 = column_stats(df1, quantiles=quantiles)
    stats2 = column_stats(df2, quantiles=quantiles)
    common = [c for c in df1.columns if c in set(df2.columns)]

    numeric_fields = [c for c in stats1.columns if c not in ("dtype",)]
    rows = {}
    for col in common:
        row = {"dtype_1": stats1.loc[col, "dtype"], "dtype_2": stats2.loc[col, "dtype"]}
        row["dtype_match"] = row["dtype_1"] == row["dtype_2"]
        for field in numeric_fields:
            v1, v2 = stats1.loc[col, field], stats2.loc[col, field]
            row[f"{field}_1"] = v1
            row[f"{field}_2"] = v2
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                row[f"{field}_diff"] = v2 - v1
        rows[col] = row
    return pd.DataFrame(rows).T


def _is_numeric_row(row: pd.Series) -> bool:
    return not pd.isna(row.get("mean"))


def _quantile_fields(index_like) -> list:
    """(level, field_name) pairs for the q<level> fields in a column_stats
    row/frame, sorted by level - e.g. [(0.05, "q0.05"), (0.1, "q0.1"), ...]."""
    return sorted(
        (float(c[1:]), c) for c in index_like
        if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()
    )


def _numeric_distribution_distance(row1: pd.Series, row2: pd.Series) -> float:
    """Wasserstein-ish distance approximated from quantiles alone (no raw values)."""
    q_cols = [name for _, name in _quantile_fields(row1.index)]
    diffs = [abs(row1[c] - row2[c]) for c in q_cols if not (pd.isna(row1[c]) or pd.isna(row2[c]))]
    if not diffs:
        return np.nan
    mean_abs_diff = float(np.mean(diffs))

    pooled_std = float(np.sqrt(np.nanmean([row1["std"] ** 2, row2["std"] ** 2])))
    if pooled_std == 0:
        return 0.0 if mean_abs_diff == 0 else 1.0
    d = mean_abs_diff / pooled_std
    return d / (1 + d)  # squash to [0, 1)


def _categorical_distribution_distance(top1: pd.Series, top2: pd.Series, n1: int, n2: int) -> float:
    """Jensen-Shannon divergence approximated from top-N value counts + an 'other' bucket."""
    if n1 == 0 or n2 == 0:
        return np.nan
    categories = top1.index.union(top2.index)
    p = (top1.reindex(categories, fill_value=0) / n1).to_numpy(dtype=float)
    q = (top2.reindex(categories, fill_value=0) / n2).to_numpy(dtype=float)
    other_p, other_q = max(0.0, 1 - p.sum()), max(0.0, 1 - q.sum())
    p, q = np.append(p, other_p), np.append(q, other_q)
    return float(jensenshannon(p, q, base=2))  # already in [0, 1]


def _correlation_distance(corr1: pd.DataFrame, corr2: pd.DataFrame) -> float:
    common = corr1.columns.intersection(corr2.columns)
    if len(common) < 2:
        return np.nan
    c1, c2 = corr1.loc[common, common].to_numpy(), corr2.loc[common, common].to_numpy()
    iu = np.triu_indices_from(c1, k=1)
    diff = np.abs(c1[iu] - c2[iu])
    if np.all(np.isnan(diff)):
        return np.nan
    return float(np.nanmean(diff)) / 2  # |diff| in [0, 2] -> [0, 1]


def _index_schema_distance(index1: dict, index2: dict) -> float:
    """Distance between index *schemas* (names/dtypes/nlevels) - summaries don't
    retain raw index values, so this compares structure rather than overlap."""
    if index1["nlevels"] != index2["nlevels"]:
        return 1.0
    mismatches = [
        0.0 if (l1["name"] == l2["name"] and l1["dtype"] == l2["dtype"]) else 1.0
        for l1, l2 in zip(index1["levels"], index2["levels"])
    ]
    return float(np.mean(mismatches))


def distance(
    summary1: dict,
    summary2: dict,
    weights: dict | None = None,
) -> float:
    """Single proximity/distance score in [0, 1] between two summarize() outputs;
    0 means very similar.

    Blends: column-set overlap, index schema match, per-column value
    distribution distance (numeric via quantile/Wasserstein approximation,
    categorical via Jensen-Shannon over top values), and association-structure
    distance (the unified numeric+categorical association matrix, so this
    catches categorical<->numeric or categorical<->categorical drift too, not
    just numeric<->numeric). Works entirely off summaries - no raw DataFrame
    access needed.
    """
    default_weights = {"columns": 1.0, "index": 1.0, "distributions": 2.0, "correlation": 1.0}
    weights = {**default_weights, **(weights or {})}

    cols1, cols2 = summary1["columns"], summary2["columns"]
    columns_dist = 1 - _jaccard(set(cols1.index), set(cols2.index))
    index_dist = _index_schema_distance(summary1["index"], summary2["index"])

    common_cols = [c for c in cols1.index if c in set(cols2.index)]
    n1, n2 = summary1["shape"][0], summary2["shape"][0]

    col_distances = []
    for c in common_cols:
        row1, row2 = cols1.loc[c], cols2.loc[c]
        if _is_numeric_row(row1) and _is_numeric_row(row2):
            col_distances.append(_numeric_distribution_distance(row1, row2))
        elif not _is_numeric_row(row1) and not _is_numeric_row(row2):
            top1 = summary1["top_values"].get(c, pd.Series(dtype=int))
            top2 = summary2["top_values"].get(c, pd.Series(dtype=int))
            non_null1 = round(n1 * (1 - row1["pct_missing"])) if n1 else 0
            non_null2 = round(n2 * (1 - row2["pct_missing"])) if n2 else 0
            col_distances.append(_categorical_distribution_distance(top1, top2, non_null1, non_null2))
        else:
            col_distances.append(1.0)  # numeric vs non-numeric: not comparable
    distributions_dist = float(np.nanmean(col_distances)) if col_distances and not np.all(np.isnan(col_distances)) else np.nan

    correlation_dist = _correlation_distance(summary1["correlations"]["association"], summary2["correlations"]["association"])

    components = {
        "columns": columns_dist,
        "index": index_dist,
        "distributions": distributions_dist,
        "correlation": correlation_dist,
    }
    valid = {k: v for k, v in components.items() if not np.isnan(v)}
    if not valid:
        return np.nan
    total_weight = sum(weights[k] for k in valid)
    return float(sum(weights[k] * v for k, v in valid.items()) / total_weight)


def compare(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str = "df1",
    name2: str = "df2",
    quantiles=DEFAULT_QUANTILES,
) -> dict:
    """Full comparison of two DataFrames: overlaps, per-column diffs, distance."""
    try:
        index_overlap = overlap(df1.index, df2.index)
    except TypeError:
        index_overlap = None  # unhashable index values

    summary1, summary2 = summarize(df1, quantiles=quantiles), summarize(df2, quantiles=quantiles)
    return {
        "names": (name1, name2),
        "shape": {name1: df1.shape, name2: df2.shape},
        "columns_overlap": overlap(df1.columns, df2.columns),
        "index_overlap": index_overlap,
        "index_summary": {name1: summarize_index(df1.index), name2: summarize_index(df2.index)},
        "column_comparison": compare_columns(df1, df2, quantiles=quantiles),
        "distance": distance(summary1, summary2),
    }


# ------------------------------------------------------------------ #
# Test-coverage assessment: is a DataFrame "like" something already tested?
# ------------------------------------------------------------------ #

def nearest_matches(summary: dict, reference: dict, k: int = 5) -> list:
    """Rank {name: summary} reference entries by distance to `summary`, closest first."""
    scored = [(name, distance(summary, ref_summary)) for name, ref_summary in reference.items()]
    scored = [(name, d) for name, d in scored if not np.isnan(d)]
    scored.sort(key=lambda item: item[1])
    return scored[:k]


def _schema_gap(summary: dict, positive: dict) -> list:
    """Columns whose (name, dtype) was never seen in any positive reference summary."""
    seen = {(col, row["dtype"]) for ref in positive.values() for col, row in ref["columns"].iterrows()}
    return [col for col, row in summary["columns"].iterrows() if (col, row["dtype"]) not in seen]


def _index_schema_seen(summary: dict, positive: dict) -> bool:
    return any(_index_schema_distance(summary["index"], ref["index"]) == 0.0 for ref in positive.values())


def calibrate_thresholds(positive: dict, sample_size: int = 200) -> dict:
    """Inner/outer distance thresholds from pairwise distances within the positive
    set, used as a fallback verdict boundary when no negative examples are given.

    O(n^2) in len(positive) (capped via `sample_size`) - compute this once per
    reference set (e.g. whenever the reference set is (re)persisted) and pass
    the result into assess_coverage(..., thresholds=...) rather than letting
    every call recompute it.
    """
    names = list(positive)
    if len(names) > sample_size:
        rng = np.random.default_rng(0)
        names = list(rng.choice(names, size=sample_size, replace=False))
    pairwise = [distance(positive[a], positive[b]) for a, b in itertools.combinations(names, 2)]
    pairwise = [d for d in pairwise if not np.isnan(d)]
    if not pairwise:
        return {"inner": 0.0, "outer": 0.0}
    return {"inner": float(np.percentile(pairwise, 90)), "outer": float(np.percentile(pairwise, 99))}


def assess_coverage(
    df_or_summary,
    positive: dict,
    negative: dict | None = None,
    k: int = 5,
    margin: float = 0.05,
    thresholds: dict | None = None,
) -> dict:
    """Decide whether a DataFrame is well-represented by existing unit-test coverage.

    positive: {name: summarize(df)} for every DataFrame your unit tests exercise.
    negative: optional {name: summarize(df)} for DataFrames confirmed to need a
        test but not yet covered. When given, the verdict compares distance to
        the single nearest positive vs. single nearest negative match - this
        self-calibrates to however tight or loose each cluster is, and stays
        robust even when positives vastly outnumber negatives, since only the
        closest example of *each* class is used (not a neighborhood vote).
        When omitted, falls back to a threshold derived from the spread of
        pairwise distances *within* the positive set itself (90th/99th
        percentile) - a reasonable bootstrap until negative examples exist.
    thresholds: precomputed output of calibrate_thresholds(positive), used only
        in the no-negatives fallback path. Pass this in for repeated/production
        calls - recomputing it per call is O(len(positive)^2) and, with a
        reference set in the hundreds, far slower than everything else this
        function does combined. If omitted, it is computed on the fly.

    A schema gap (a column, or its dtype, never seen in any positive example;
    or an index name/dtype never seen) always forces "add_test", regardless
    of distance - that represents a code path with literally zero coverage.

    Returns a report dict with "verdict" ("covered" / "borderline" /
    "add_test"), a human-readable "reason", the nearest positive (and
    negative, if given) match, and the top-k matches for inspection.
    """
    summary = summarize(df_or_summary) if isinstance(df_or_summary, pd.DataFrame) else df_or_summary

    schema_gap_columns = _schema_gap(summary, positive)
    index_gap = not _index_schema_seen(summary, positive)

    top_matches = nearest_matches(summary, positive, k=k)
    best_positive = top_matches[0] if top_matches else (None, np.nan)

    report = {
        "schema_gap_columns": schema_gap_columns,
        "index_schema_gap": index_gap,
        "nearest_positive": {"name": best_positive[0], "distance": best_positive[1]},
        "top_matches": top_matches,
    }

    if schema_gap_columns or index_gap:
        report["verdict"] = "add_test"
        report["reason"] = (
            "columns not seen in any tested case: " + ", ".join(map(str, schema_gap_columns))
            if schema_gap_columns else "index name/dtype not seen in any tested case"
        )
        return report

    if negative:
        neg_matches = nearest_matches(summary, negative, k=k)
        best_negative = neg_matches[0] if neg_matches else (None, np.nan)
        report["nearest_negative"] = {"name": best_negative[0], "distance": best_negative[1]}
        report["top_negative_matches"] = neg_matches

        if np.isnan(best_positive[1]) or np.isnan(best_negative[1]):
            report["verdict"] = "borderline"
            report["reason"] = "could not compute a comparable distance to both reference sets"
            return report

        gap = best_negative[1] - best_positive[1]
        if abs(gap) <= margin:
            report["verdict"] = "borderline"
            report["reason"] = (
                f"about equally close to tested case '{best_positive[0]}' "
                f"and untested case '{best_negative[0]}' - worth a human look"
            )
        elif gap > 0:
            report["verdict"] = "covered"
            report["reason"] = f"closer to tested case '{best_positive[0]}' than to any untested case"
        else:
            report["verdict"] = "add_test"
            report["reason"] = f"closer to untested case '{best_negative[0]}' than to any tested case"
        return report

    if thresholds is None:
        thresholds = calibrate_thresholds(positive)
    report["thresholds"] = thresholds
    if np.isnan(best_positive[1]):
        report["verdict"] = "borderline"
        report["reason"] = "no comparable positive reference case found"
    elif best_positive[1] <= thresholds["inner"]:
        report["verdict"] = "covered"
        report["reason"] = f"within the typical spread of tested cases (distance {best_positive[1]:.3f} <= {thresholds['inner']:.3f})"
    elif best_positive[1] <= thresholds["outer"]:
        report["verdict"] = "borderline"
        report["reason"] = f"somewhat outside the typical spread of tested cases (distance {best_positive[1]:.3f})"
    else:
        report["verdict"] = "add_test"
        report["reason"] = f"far outside the typical spread of tested cases (distance {best_positive[1]:.3f} > {thresholds['outer']:.3f})"
    return report


# ------------------------------------------------------------------ #
# Sample generation: synthesize a DataFrame that resembles a summary
# ------------------------------------------------------------------ #

def _nearest_psd_correlation(corr: np.ndarray) -> np.ndarray:
    """Project a possibly-invalid correlation matrix (NaNs from zero-variance
    columns, float rounding drift) onto the nearest valid one via eigenvalue
    clipping, so Cholesky decomposition below never fails."""
    corr = np.nan_to_num(corr, nan=0.0)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-8, None)
    reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    scale = np.sqrt(np.diag(reconstructed))
    reconstructed = reconstructed / np.outer(scale, scale)
    np.fill_diagonal(reconstructed, 1.0)
    return reconstructed


def _numeric_marginal_sample(row: pd.Series, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF sample from a numeric column's stored min/quantiles/max,
    via piecewise-linear interpolation of the empirical quantile function."""
    q_items = [(level, row[name]) for level, name in _quantile_fields(row.index)]
    xp = [0.0] + [q for q, _ in q_items] + [1.0]
    fp = [row["min"]] + [v for _, v in q_items] + [row["max"]]
    fp = np.maximum.accumulate(fp)  # guard against rounding-induced non-monotonicity
    return np.interp(u, xp, fp)


def _categorical_marginal_sample(top: pd.Series, nunique: int, non_null: int, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF-style sample from a categorical column's top-N value counts.
    Probability mass beyond the stored top-N is spread evenly across synthetic
    placeholder categories sized to make up the recorded nunique, since their
    individual frequencies were never captured (same "other" bucket idea used
    for the Jensen-Shannon term in distance())."""
    categories = list(top.index)
    probs = list((top / non_null).to_numpy()) if non_null else []
    n_other = max(0, nunique - len(categories))
    remaining = max(0.0, 1 - sum(probs))
    if n_other > 0:
        categories += [f"__other_{i}__" for i in range(n_other)]
        probs += [remaining / n_other] * n_other
    elif remaining > 0 and categories:
        probs[-1] += remaining  # no room for new categories; pad the last one
    if not categories:
        return np.full(len(u), np.nan, dtype=object)
    boundaries = np.cumsum(probs)
    boundaries[-1] = 1.0  # guard against float drift
    idx = np.clip(np.searchsorted(boundaries, u, side="right"), 0, len(categories) - 1)
    return np.array(categories, dtype=object)[idx]


def _best_categorical_predictor(numeric_col: str, overall_std: float, conditional_numeric: dict) -> str | None:
    """Pick whichever categorical column's top-category means for `numeric_col`
    vary the most relative to its overall std - i.e. splitting by that column
    reveals a real group effect worth modeling, versus noise. None if nothing
    clears a small bar (or `conditional_numeric` has nothing for this column)."""
    best_col, best_score = None, 0.0
    if not overall_std or pd.isna(overall_std) or overall_std <= 0:
        return None
    for cat_col, per_category in conditional_numeric.items():
        means = [
            stats_df.loc[numeric_col, "mean"] for stats_df in per_category.values()
            if numeric_col in stats_df.index and not pd.isna(stats_df.loc[numeric_col, "mean"])
        ]
        if len(means) >= 2:
            score = float(np.std(means)) / overall_std
            if score > best_score:
                best_col, best_score = cat_col, score
    return best_col if best_score > 0.05 else None


def _numeric_sample_conditional(row_default: pd.Series, u: np.ndarray, realized_category: np.ndarray, per_category: dict) -> np.ndarray:
    """Like _numeric_marginal_sample, but rows whose realized categorical
    value has its own conditional stats are sampled from that group's
    quantile function instead of the unconditional marginal - reusing the
    same u so numeric<->numeric rank correlation is preserved as well as a
    lossy summary reasonably allows. Rows whose category fell outside the
    stored top-N fall back to the marginal."""
    values = np.empty(len(u))
    filled = np.zeros(len(u), dtype=bool)
    for category, stats_df in per_category.items():
        if row_default.name not in stats_df.index:
            continue
        mask = realized_category == category
        if mask.any():
            values[mask] = _numeric_marginal_sample(stats_df.loc[row_default.name], u[mask])
            filled |= mask
    if not filled.all():
        values[~filled] = _numeric_marginal_sample(row_default, u[~filled])
    return values


def generate_sample(summary: dict, n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Synthesize a DataFrame that approximately matches a summarize() output:
    per-column marginals (from stored min/quantiles/max for numeric columns,
    top-N value counts for categorical/boolean columns), numeric<->numeric
    dependency via a Gaussian copula built from the stored correlation
    structure, and categorical<->numeric dependency (e.g. a group whose
    values run systematically higher/lower) via conditional_numeric_stats().

    This is necessarily approximate - summaries are lossy by design - and
    intended for test fixtures / negative examples for assess_coverage(),
    not as a substitute for real data:
    - categorical columns are sampled independently of each other -
      categorical<->categorical dependence isn't modeled;
    - missing/inf ratios are injected independently per column, not jointly
      correlated with other columns' missingness;
    - categorical values beyond the stored top-N are represented by
      synthetic placeholder categories with equal assumed probability;
    - datetime columns aren't reconstructed (summarize() doesn't capture
      numeric stats for them yet) - they come back all-NaT;
    - the row index is a plain RangeIndex, not a reconstruction of the
      original index's values.

    A good sanity check after generating:
    `distance(summary, summarize(generate_sample(summary)))` should be small.
    """
    rng = np.random.default_rng(seed)
    cols = summary["columns"]
    n = summary["shape"][0] if n is None else n
    columns = list(cols.index)
    numeric_cols = [c for c in columns if _is_numeric_row(cols.loc[c])]

    assoc = summary["correlations"]["association"]
    u_by_col = {}
    if len(numeric_cols) >= 2 and n > 0:
        rho_spearman = assoc.loc[numeric_cols, numeric_cols].to_numpy()
        rho_gaussian = _nearest_psd_correlation(2 * np.sin(np.pi * np.clip(rho_spearman, -1, 1) / 6))
        z = rng.standard_normal((n, len(numeric_cols))) @ np.linalg.cholesky(rho_gaussian).T
        u = norm.cdf(z)
        u_by_col = {c: u[:, i] for i, c in enumerate(numeric_cols)}
    for c in numeric_cols:
        if c not in u_by_col:
            u_by_col[c] = rng.uniform(0, 1, n)

    # Categorical (and datetime-placeholder) columns first: numeric columns
    # may need their realized values to sample conditionally below.
    data = {}
    for c in columns:
        row = cols.loc[c]
        dtype = row["dtype"]
        if dtype.startswith("datetime"):
            data[c] = pd.Series(pd.NaT, index=range(n), dtype=dtype)
        elif c not in numeric_cols:
            top = summary["top_values"].get(c, pd.Series(dtype=int))
            non_null = round(n * (1 - row["pct_missing"])) if n else 0
            data[c] = _categorical_marginal_sample(top, int(row["nunique"]), non_null, rng.uniform(0, 1, n))

    conditional_numeric = summary.get("conditional_numeric", {})
    for c in numeric_cols:
        row = cols.loc[c]
        predictor = _best_categorical_predictor(c, row.get("std"), conditional_numeric)
        if predictor is not None and predictor in data:
            values = _numeric_sample_conditional(row, u_by_col[c], np.asarray(data[predictor]), conditional_numeric[predictor])
        else:
            values = _numeric_marginal_sample(row, u_by_col[c])
        dtype = row["dtype"]
        if dtype == "bool":
            values = values >= 0.5
        elif dtype.startswith("int") or dtype.startswith("uint"):
            values = np.round(values)
        data[c] = values

    df = pd.DataFrame(data, index=pd.RangeIndex(n), columns=columns)

    for c in columns:
        row = cols.loc[c]
        if n == 0:
            continue
        n_missing = min(round(n * row["pct_missing"]) if not pd.isna(row["pct_missing"]) else 0, n)
        missing_idx = rng.choice(n, size=n_missing, replace=False) if n_missing else np.array([], dtype=int)
        if n_missing:
            df.loc[missing_idx, c] = pd.NaT if row["dtype"].startswith("datetime") else np.nan

        if _is_numeric_row(row) and not pd.isna(row.get("pct_inf")) and row["pct_inf"]:
            n_inf = min(round(n * row["pct_inf"]), n - n_missing)
            if n_inf > 0:
                remaining = np.setdiff1d(np.arange(n), missing_idx)
                inf_idx = rng.choice(remaining, size=n_inf, replace=False)
                df.loc[inf_idx, c] = rng.choice([np.inf, -np.inf], size=n_inf)

    for c in columns:
        dtype = cols.loc[c, "dtype"]
        try:
            if dtype.startswith("int") or dtype.startswith("uint"):
                df[c] = df[c].astype("Int64" if df[c].isna().any() else dtype)
            elif dtype == "bool":
                df[c] = df[c].astype("boolean" if df[c].isna().any() else "bool")
            elif not dtype.startswith("datetime"):
                df[c] = df[c].astype(dtype)
        except (TypeError, ValueError):
            pass  # best-effort dtype restoration; leave as generated if it doesn't fit

    return df
