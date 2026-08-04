# dfsummary

A small, dependency-light (pandas / numpy / scipy) toolkit for:

- summarizing a single pandas DataFrame,
- comparing two DataFrames (schema, per-column stats, a single distance score),
- and assessing, at production call time, whether a new DataFrame is well
  represented by an existing suite of unit-test reference DataFrames.

No packaging — just drop `dfsummary.py` into your project.

## Quick start

```python
import dfsummary as dfs

summary = dfs.summarize(df)
summary["shape"]         # (n_rows, n_cols)
summary["index"]         # names/dtypes/nunique per index level (MultiIndex-aware)
summary["columns"]       # DataFrame: dtype, missing/inf/zero/negative/positive/>1
                          # counts & ratios, nunique, mean/median/std/quantiles
summary["top_values"]    # {column: top-10 value_counts}
summary["correlations"]  # {"pearson": ..., "spearman": ...} correlation matrices
```

## Comparing two DataFrames

```python
result = dfs.compare(df1, df2, name1="train", name2="test")
result["columns_overlap"]    # common / only_in_first / only_in_second / jaccard
result["index_overlap"]      # same, for index values (if hashable)
result["column_comparison"]  # side-by-side stats + diffs for shared columns
result["distance"]           # single float in [0, 1], 0 = identical
```

`distance()` itself works on `summarize()` output rather than raw DataFrames,
so it's cheap to run against many persisted reference summaries without
keeping the original data around:

```python
d = dfs.distance(dfs.summarize(df1), dfs.summarize(df2))
```

## Test-coverage assessment

Check whether a DataFrame you're about to run in production resembles what
your unit tests actually exercised:

```python
positive = {name: dfs.summarize(df) for name, df in your_unit_test_dataframes.items()}
thresholds = dfs.calibrate_thresholds(positive)  # O(n^2) - compute once, cache/persist it

report = dfs.assess_coverage(new_df, positive, thresholds=thresholds)
report["verdict"]        # "covered" / "borderline" / "add_test"
report["reason"]
report["top_matches"]    # closest reference cases, with distances
```

A schema gap (a column/dtype, or index name/dtype, never seen in `positive`)
always forces `"add_test"`, regardless of distance.

Optionally pass `negative={name: summarize(df), ...}` — a handful of
DataFrames confirmed to need a test but not yet covered — for a sharper
verdict: the new case is classified by whichever is closer, its nearest
*tested* match or its nearest *untested* match. This nearest-neighbor-per-class
rule self-adapts to how spread out each cluster is and needs no hand-picked
threshold; even 5-15 well-chosen negatives are enough to start.

## Design notes

- Numeric distribution distance is approximated from stored quantiles
  (Wasserstein-like); categorical distance from top-10 value counts plus an
  "other" bucket (Jensen-Shannon); correlation distance directly from the
  stored correlation matrices.
- Index comparison is schema-level (names/dtypes/levels), not raw value
  overlap — summaries don't retain the original index values.
- `calibrate_thresholds()` is O(n²) in reference-set size. Compute it once
  whenever the reference set is (re)persisted, not on every `assess_coverage`
  call.

## Tests

```
pytest test_dfsummary.py
```
