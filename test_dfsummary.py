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

SAMPLE_SUMMARIES = {name: dfs.summarize(df) for name, df in SAMPLE_DFS.items()}


# ------------------------------------------------------------------ #
# summarize()
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name", SAMPLE_DFS)
def test_summarize_runs_and_has_expected_keys(name):
    df = SAMPLE_DFS[name]
    result = dfs.summarize(df)
    assert set(result) == {"shape", "index", "columns", "top_values", "correlations", "conditional_numeric"}
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


def test_associations_matrix_spans_numeric_and_categorical_columns():
    rng = np.random.default_rng(0)
    n = 300
    group = rng.choice(["a", "b"], n)
    value = rng.normal(0, 1, n)
    df = pd.DataFrame({"group": group, "value": value})
    assoc = dfs.associations(df)
    assert set(assoc.columns) == {"group", "value"}
    assert assoc.loc["group", "group"] == pytest.approx(1.0)
    assert assoc.loc["value", "value"] == pytest.approx(1.0)


def test_conditional_numeric_stats_captures_group_mean_shift():
    rng = np.random.default_rng(0)
    n = 400
    group = rng.choice(["a", "b"], n)
    value = np.where(group == "a", rng.normal(0, 1, n), rng.normal(20, 1, n))
    df = pd.DataFrame({"group": group, "value": value})
    stats = dfs.conditional_numeric_stats(df)
    assert set(stats) == {"group"}
    means = {cat: s.loc["value", "mean"] for cat, s in stats["group"].items()}
    assert means["a"] == pytest.approx(0.0, abs=0.5)
    assert means["b"] == pytest.approx(20.0, abs=0.5)


def test_conditional_numeric_stats_empty_without_both_column_types():
    assert dfs.conditional_numeric_stats(make_numeric_df()) == {}
    only_categorical = pd.DataFrame({"cat": ["a", "b", "c"]})
    assert dfs.conditional_numeric_stats(only_categorical) == {}


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
    for a, b in itertools.combinations(SAMPLE_SUMMARIES.values(), 2):
        d_ab, d_ba = dfs.distance(a, b), dfs.distance(b, a)
        if np.isnan(d_ab):
            assert np.isnan(d_ba)
        else:
            assert d_ab == pytest.approx(d_ba)


def test_distance_within_bounds_for_all_sample_pairs():
    for a, b in itertools.combinations(SAMPLE_SUMMARIES.values(), 2):
        d = dfs.distance(a, b)
        assert np.isnan(d) or 0.0 <= d <= 1.0


def test_distance_increases_with_distribution_shift():
    base = dfs.summarize(make_numeric_df(seed=1))
    similar = dfs.summarize(make_numeric_df(seed=2))              # same distribution, different sample
    shifted = dfs.summarize(make_numeric_df(seed=3, loc=5.0))     # shifted mean

    d_similar = dfs.distance(base, similar)
    d_shifted = dfs.distance(base, shifted)
    assert d_shifted > d_similar


def test_distance_detects_correlation_structure_change():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(0, 1, n)
    df_correlated = pd.DataFrame({"x": x, "y": x + rng.normal(0, 0.01, n)})
    df_uncorrelated = pd.DataFrame({"x": x, "y": rng.normal(0, 1, n)})

    d_same_corr = dfs.distance(dfs.summarize(df_correlated), dfs.summarize(df_correlated.copy()))
    d_diff_corr = dfs.distance(dfs.summarize(df_correlated), dfs.summarize(df_uncorrelated))
    assert d_diff_corr > d_same_corr


def test_distance_handles_unhashable_index_without_crashing():
    summary1 = dfs.summarize(make_unhashable_index_df())
    summary2 = dfs.summarize(make_unhashable_index_df())
    d = dfs.distance(summary1, summary2)
    assert isinstance(d, float)


def test_compare_handles_unhashable_index_without_crashing():
    df1, df2 = make_unhashable_index_df(), make_unhashable_index_df()
    result = dfs.compare(df1, df2)
    assert result["index_overlap"] is None


# ------------------------------------------------------------------ #
# nearest_matches() / assess_coverage()
# ------------------------------------------------------------------ #

def test_nearest_matches_ranks_closest_first_and_respects_k():
    target = dfs.summarize(make_numeric_df(seed=100))
    reference = {
        "close": dfs.summarize(make_numeric_df(seed=101)),
        "far": dfs.summarize(make_numeric_df(seed=200, loc=10.0, scale=5.0)),
        "medium": dfs.summarize(make_numeric_df(seed=102, loc=1.0)),
    }
    ranked = dfs.nearest_matches(target, reference, k=2)
    assert len(ranked) == 2
    assert ranked[0][0] == "close"
    assert ranked[0][1] <= ranked[1][1]


def test_assess_coverage_flags_schema_gap_regardless_of_distance():
    positive = {"p1": dfs.summarize(make_numeric_df(seed=1))}
    # near-identical distribution, but with an extra column never seen in the positive set
    new_df = make_numeric_df(seed=1)
    new_df["extra_col"] = 1.0
    report = dfs.assess_coverage(new_df, positive)
    assert report["verdict"] == "add_test"
    assert "extra_col" in report["schema_gap_columns"]


def test_assess_coverage_flags_dtype_change_as_schema_gap():
    positive = {"p1": dfs.summarize(pd.DataFrame({"a": [1, 2, 3]}))}
    new_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})  # same column name, different dtype
    report = dfs.assess_coverage(new_df, positive)
    assert report["verdict"] == "add_test"
    assert "a" in report["schema_gap_columns"]


def test_assess_coverage_with_negatives_prefers_closer_class():
    positive = {f"p{i}": dfs.summarize(make_numeric_df(seed=i, loc=0.0)) for i in range(5)}
    negative = {f"n{i}": dfs.summarize(make_numeric_df(seed=50 + i, loc=8.0)) for i in range(5)}

    covered_case = make_numeric_df(seed=42, loc=0.1)   # close to the positive cluster
    gap_case = make_numeric_df(seed=43, loc=7.9)       # close to the negative cluster

    covered_report = dfs.assess_coverage(covered_case, positive, negative)
    gap_report = dfs.assess_coverage(gap_case, positive, negative)

    assert covered_report["verdict"] == "covered"
    assert gap_report["verdict"] == "add_test"


def test_assess_coverage_without_negatives_uses_self_calibrated_threshold():
    positive = {f"p{i}": dfs.summarize(make_numeric_df(seed=i, loc=0.0)) for i in range(10)}
    covered_case = make_numeric_df(seed=99, loc=0.05)   # within the positive spread
    outlier_case = make_numeric_df(seed=98, loc=50.0, scale=20.0)  # way outside it

    covered_report = dfs.assess_coverage(covered_case, positive)
    outlier_report = dfs.assess_coverage(outlier_case, positive)

    assert covered_report["verdict"] == "covered"
    assert outlier_report["verdict"] == "add_test"
    assert "thresholds" in outlier_report


def test_assess_coverage_accepts_precomputed_summary():
    positive = {"p1": dfs.summarize(make_numeric_df(seed=1))}
    summary = dfs.summarize(make_numeric_df(seed=1))
    report = dfs.assess_coverage(summary, positive)
    assert report["verdict"] == "covered"


# ------------------------------------------------------------------ #
# generate_sample()
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name", SAMPLE_DFS)
def test_generate_sample_matches_shape_and_round_trips_with_small_distance(name):
    df = SAMPLE_DFS[name]
    summary = dfs.summarize(df)
    generated = dfs.generate_sample(summary, seed=0)
    assert generated.shape == df.shape
    assert list(generated.columns) == list(df.columns)
    # generate_sample() always emits a plain RangeIndex (index reconstruction
    # is an accepted out-of-scope limitation - see its docstring), so exclude
    # the index component here: this check is about data fidelity, not index.
    d = dfs.distance(summary, dfs.summarize(generated), weights={"index": 0})
    assert np.isnan(d) or d < 0.15


def test_generate_sample_is_reproducible_with_same_seed():
    summary = dfs.summarize(make_numeric_df())
    a = dfs.generate_sample(summary, seed=7)
    b = dfs.generate_sample(summary, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_generate_sample_approximately_preserves_missing_ratio():
    df = make_missing_inf_df()
    summary = dfs.summarize(df)
    generated = dfs.generate_sample(summary, seed=3)
    original_ratio = df["messy"].isna().mean()
    generated_ratio = generated["messy"].isna().mean()
    assert generated_ratio == pytest.approx(original_ratio, abs=0.05)


def test_generate_sample_reproduces_categorical_group_mean_shift():
    rng = np.random.default_rng(0)
    n = 500
    group = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    value = np.select(
        [group == "a", group == "b", group == "c"],
        [rng.normal(0, 1, n), rng.normal(10, 1, n), rng.normal(-5, 2, n)],
    )
    df = pd.DataFrame({"group": group, "value": value})
    summary = dfs.summarize(df)
    generated = dfs.generate_sample(summary, seed=42)

    original_means = df.groupby("group")["value"].mean()
    generated_means = generated.groupby("group")["value"].mean()
    for group_name in original_means.index:
        assert generated_means[group_name] == pytest.approx(original_means[group_name], abs=1.5)


def test_generate_sample_with_custom_n():
    summary = dfs.summarize(make_numeric_df())
    generated = dfs.generate_sample(summary, n=25, seed=0)
    assert generated.shape[0] == 25


def test_generate_sample_handles_empty_summary():
    summary = dfs.summarize(make_empty_df())
    generated = dfs.generate_sample(summary, seed=0)
    assert generated.shape == (0, 2)
    assert list(generated.columns) == ["a", "b"]
