"""Summarize and compare pandas DataFrames.

Public API:
    summarize(df)                -> dict describing one DataFrame
    compare(df1, df2)            -> dict comparing two DataFrames
    distance(df1, df2)           -> single float in [0, 1], 0 = identical
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

DEFAULT_QUANTILES = (0.25, 0.5, 0.75)
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
        finite = s[np.isfinite(s.astype(float))]
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


def _numeric_distribution_distance(s1: pd.Series, s2: pd.Series) -> float:
    a, b = s1.dropna().to_numpy(dtype=float), s2.dropna().to_numpy(dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    pooled_std = np.std(np.concatenate([a, b]))
    if pooled_std == 0:
        return 0.0 if np.array_equal(np.unique(a), np.unique(b)) else 1.0
    d = wasserstein_distance(a, b) / pooled_std
    return d / (1 + d)  # squash to [0, 1)


def _categorical_distribution_distance(s1: pd.Series, s2: pd.Series) -> float:
    vc1, vc2 = s1.value_counts(normalize=True), s2.value_counts(normalize=True)
    categories = vc1.index.union(vc2.index)
    p = vc1.reindex(categories, fill_value=0.0).to_numpy()
    q = vc2.reindex(categories, fill_value=0.0).to_numpy()
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    return float(jensenshannon(p, q, base=2))  # already in [0, 1]


def _correlation_distance(df1: pd.DataFrame, df2: pd.DataFrame, common_numeric: list) -> float:
    if len(common_numeric) < 2:
        return np.nan
    c1 = df1[common_numeric].corr(method="pearson").to_numpy()
    c2 = df2[common_numeric].corr(method="pearson").to_numpy()
    iu = np.triu_indices_from(c1, k=1)
    diff = np.abs(c1[iu] - c2[iu])
    return float(np.nanmean(diff)) / 2  # |diff| in [0, 2] -> [0, 1]


def distance(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    weights: dict | None = None,
) -> float:
    """Single proximity/distance score in [0, 1]; 0 means very similar.

    Blends: column-set overlap, index-value overlap, per-column value
    distribution distance (numeric via Wasserstein, categorical via
    Jensen-Shannon), and correlation-structure distance.
    """
    default_weights = {"columns": 1.0, "index": 1.0, "distributions": 2.0, "correlation": 1.0}
    weights = {**default_weights, **(weights or {})}

    columns_dist = 1 - _jaccard(set(df1.columns), set(df2.columns))

    try:
        index_dist = 1 - _jaccard(set(df1.index), set(df2.index))
    except TypeError:
        index_dist = np.nan  # unhashable index values

    common_cols = [c for c in df1.columns if c in set(df2.columns)]
    common_numeric = [c for c in common_cols if pd.api.types.is_numeric_dtype(df1[c]) and pd.api.types.is_numeric_dtype(df2[c])]
    common_other = [c for c in common_cols if c not in common_numeric]

    col_distances = []
    for c in common_numeric:
        col_distances.append(_numeric_distribution_distance(df1[c], df2[c]))
    for c in common_other:
        col_distances.append(_categorical_distribution_distance(df1[c], df2[c]))
    distributions_dist = float(np.nanmean(col_distances)) if col_distances else np.nan

    correlation_dist = _correlation_distance(df1, df2, common_numeric)

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

    return {
        "names": (name1, name2),
        "shape": {name1: df1.shape, name2: df2.shape},
        "columns_overlap": overlap(df1.columns, df2.columns),
        "index_overlap": index_overlap,
        "index_summary": {name1: summarize_index(df1.index), name2: summarize_index(df2.index)},
        "column_comparison": compare_columns(df1, df2, quantiles=quantiles),
        "distance": distance(df1, df2),
    }
