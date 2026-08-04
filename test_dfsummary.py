"""Unit tests for dfsummary, exercised against DataFrames with a variety
of column/index shapes (numeric, categorical, bool, datetime, missing/inf,
MultiIndex, empty, unhashable index)."""
import itertools

import numpy as np
import pandas as pd
import pytest

import dfsummary as dfs


# ------------------------------------------------------------------ #
# Sample DataFrames
# ------------------------------------------------------------------ #

def make_numeric_df(seed=0, n=200, loc=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "int_col": rng.integers(-10, 10, n),
        "float_col": rng.normal(loc, scale, n),
        "ratio_col": rng.uniform(0, 2, n),  # exercises the >1 ratio stat
    })


def make_categorical_df(seed=0, n=150, categories=("a", "b", "c")):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "cat_col": rng.choice(categories, n),
        "bool_col": rng.choice([True, False], n),
    })


def make_missing_inf_df(n=100):
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, n)
    vals[:10] = np.nan
    vals[10:15] = np.inf
    vals[15:18] = -np.inf
    return pd.DataFrame({"messy": vals})


def make_datetime_df(n=50):
    return pd.DataFrame({
        "date_col": pd.date_range("2020-01-01", periods=n, freq="D"),
        "value": np.arange(n, dtype=float),
    })


def make_multiindex_df(n_groups=3, n_sub=4):
    idx = pd.MultiIndex.from_product(
        [[f"g{i}" for i in range(n_groups)], range(n_sub)],
        names=["group", "sub"],
    )
    rng = np.random.default_rng(0)
    return pd.DataFrame({"value": rng.normal(0, 1, n_groups * n_sub)}, index=idx)


def make_empty_df():
    return pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=object)})


def make_unhashable_index_df():
    df = pd.DataFrame({"v": [1, 2, 3]})
    df.index = pd.Index([[0], [1], [2]], dtype=object)
    return df


SAMPLE_DFS = {
    "numeric": make_numeric_df(),
    "categorical": make_categorical_df(),
    "missing_inf": make_missing_inf_df(),
    "datetime": make_datetime_df(),
    "multiindex": make_multiindex_df(),
    "empty": make_empty_df(),
}


# ------------------------------------------------------------------ #
# summarize()
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name", SAMPLE_DFS)
def test_summarize_runs_and_has_expected_keys(name):
    df = SAMPLE_DFS[name]
    result = dfs.summarize(df)
    assert set(result) == {"shape", "index", "columns", "top_values", "correlations"}
    assert result["shape"] == df.shape
    assert list(result["columns"].index) == list(df.columns)


def test_column_stats_counts_missing_and_inf_correctly():
    df = make_missing_inf_df()
    stats = dfs.column_stats(df).loc["messy"]
    assert stats["n_missing"] == 10
    assert stats["n_inf"] == 8  # 5 +inf, 3 -inf
    assert stats["pct_missing"] == pytest.approx(10 / 100)
    assert stats["pct_inf"] == pytest.approx(8 / 100)


def test_column_stats_sign_and_above_one_counts():
    df = pd.DataFrame({"x": [-2.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 3.0]})
    stats = dfs.column_stats(df).loc["x"]
    assert stats["n_negative"] == 2
    assert stats["n_zero"] == 2
    assert stats["n_positive"] == 4
    assert stats["n_above_1"] == 3  # abs() > 1: -2.0, 1.5, 3.0


def test_top_values_respects_top_n():
    df = pd.DataFrame({"cat": ["a"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d"] * 1})
    top = dfs.top_values(df, top_n=2)
    assert list(top["cat"].index) == ["a", "b"]
    assert top["cat"].iloc[0] == 5


def test_correlations_only_include_numeric_columns():
    df = make_numeric_df()
    corr = dfs.correlations(df)
    assert set(corr["pearson"].columns) == {"int_col", "float_col", "ratio_col"}
    assert corr["pearson"].shape == (3, 3)
    assert set(corr["spearman"].columns) == set(corr["pearson"].columns)


def test_summarize_index_reports_each_multiindex_level():
    df = make_multiindex_df()
    idx_summary = dfs.summarize_index(df.index)
    assert idx_summary["nlevels"] == 2
    assert [lvl["name"] for lvl in idx_summary["levels"]] == ["group", "sub"]
    assert idx_summary["levels"][0]["dtype"] == "str"
    assert idx_summary["levels"][1]["nunique"] == 4


# ------------------------------------------------------------------ #
# compare() / distance()
# ------------------------------------------------------------------ #

def test_compare_identical_dataframe_has_zero_distance():
    df = make_numeric_df()
    result = dfs.compare(df, df.copy())
    assert result["distance"] == pytest.approx(0.0, abs=1e-9)
    assert result["columns_overlap"]["jaccard"] == 1.0


def test_compare_disjoint_columns_reports_full_overlap_gap():
    df1, df2 = make_numeric_df(), make_categorical_df()
    result = dfs.compare(df1, df2)
    assert result["columns_overlap"]["common"] == []
    assert set(result["columns_overlap"]["only_in_first"]) == set(df1.columns)
    assert set(result["columns_overlap"]["only_in_second"]) == set(df2.columns)
    assert result["columns_overlap"]["jaccard"] == 0.0
    assert result["column_comparison"].empty


def test_column_comparison_flags_dtype_mismatch_and_diffs():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1.0, 2.0, 3.5]})
    comparison = dfs.compare_columns(df1, df2)
    assert comparison.loc["a", "dtype_match"] == False
    assert comparison.loc["a", "mean_diff"] == pytest.approx(df2["a"].mean() - df1["a"].mean())


def test_distance_is_symmetric_across_all_sample_pairs():
    for a, b in itertools.combinations(SAMPLE_DFS.values(), 2):
        d_ab, d_ba = dfs.distance(a, b), dfs.distance(b, a)
        if np.isnan(d_ab):
            assert np.isnan(d_ba)
        else:
            assert d_ab == pytest.approx(d_ba)


def test_distance_within_bounds_for_all_sample_pairs():
    for a, b in itertools.combinations(SAMPLE_DFS.values(), 2):
        d = dfs.distance(a, b)
        assert np.isnan(d) or 0.0 <= d <= 1.0


def test_distance_increases_with_distribution_shift():
    base = make_numeric_df(seed=1)
    similar = make_numeric_df(seed=2)               # same distribution, different sample
    shifted = make_numeric_df(seed=3, loc=5.0)       # shifted mean

    d_similar = dfs.distance(base, similar)
    d_shifted = dfs.distance(base, shifted)
    assert d_shifted > d_similar


def test_distance_detects_correlation_structure_change():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(0, 1, n)
    df_correlated = pd.DataFrame({"x": x, "y": x + rng.normal(0, 0.01, n)})
    df_uncorrelated = pd.DataFrame({"x": x, "y": rng.normal(0, 1, n)})

    d_same_corr = dfs.distance(df_correlated, df_correlated.copy())
    d_diff_corr = dfs.distance(df_correlated, df_uncorrelated)
    assert d_diff_corr > d_same_corr


def test_distance_handles_unhashable_index_without_crashing():
    df1, df2 = make_unhashable_index_df(), make_unhashable_index_df()
    d = dfs.distance(df1, df2)
    assert isinstance(d, float)


def test_compare_handles_unhashable_index_without_crashing():
    df1, df2 = make_unhashable_index_df(), make_unhashable_index_df()
    result = dfs.compare(df1, df2)
    assert result["index_overlap"] is None
