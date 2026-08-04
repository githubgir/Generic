"""Summarize and compare pandas DataFrames.

Public API:
    summarize(df)                          -> dict describing one DataFrame
    compare(df1, df2)                      -> dict comparing two DataFrames
    distance(summary1, summary2)           -> single float in [0, 1], 0 = identical
    nearest_matches(summary, reference)    -> reference entries ranked by distance
    assess_coverage(df, positive, negative)-> verdict on whether df needs a new test

distance() and everything built on it (nearest_matches, assess_coverage) work on
the dicts returned by summarize() rather than on raw DataFrames. That's what
makes them cheap to run against a reference set of hundreds of persisted
summaries in production, without needing to keep the original reference
DataFrames around.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

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

    n_missing = int(s.isna().sum())
    stats = {
        "dtype": str(s.dtype),
        "n_missing": n_missing,
        "pct_missing": n_missing / n if n else np.nan,
        "nunique": int(s.nunique(dropna=True)),
    }

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
        )
        for q in quantiles:
            stats[f"q{q}"] = float(finite.quantile(q)) if len(finite) else np.nan
    else:
        stats.update(
            n_zero=np.nan, pct_zero=np.nan,
            n_negative=np.nan, pct_negative=np.nan,
            n_positive=np.nan, pct_positive=np.nan,
            n_above_1=np.nan, pct_above_1=np.nan,
            mean=np.nan, median=np.nan, std=np.nan,
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


def correlations(df: pd.DataFrame) -> dict:
    """Pearson and Spearman correlation matrices for numeric columns."""
    numeric = df.select_dtypes(include=np.number)
    return {
        "pearson": numeric.corr(method="pearson"),
        "spearman": numeric.corr(method="spearman"),
    }


def summarize(
    df: pd.DataFrame,
    quantiles=DEFAULT_QUANTILES,
    top_n: int = DEFAULT_TOP_N,
) -> dict:
    """Full summary of a single DataFrame."""
    return {
        "shape": df.shape,
        "index": summarize_index(df.index),
        "columns": column_stats(df, quantiles=quantiles),
        "top_values": top_values(df, top_n=top_n),
        "correlations": correlations(df),
    }


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


def _numeric_distribution_distance(row1: pd.Series, row2: pd.Series) -> float:
    """Wasserstein-ish distance approximated from quantiles alone (no raw values)."""
    q_cols = [c for c in row1.index if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()]
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
    categorical via Jensen-Shannon over top values), and correlation-structure
    distance. Works entirely off summaries - no raw DataFrame access needed.
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
    distributions_dist = float(np.nanmean(col_distances)) if col_distances else np.nan

    correlation_dist = _correlation_distance(summary1["correlations"]["pearson"], summary2["correlations"]["pearson"])

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
