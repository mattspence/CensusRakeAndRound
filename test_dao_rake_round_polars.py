"""Tests for dao_rake_round.py (Polars version).

Run with:  pytest test_dao_rake_round.py

Three kinds of test:

1. Hand-worked examples for each building block.
2. A cross-check against `ref_*` functions below.  Those are a deliberately
   literal, row-by-row transcription of the SAS DATA steps (dictionaries,
   Python sorts, a retained counter) that share no code with the Polars
   implementation.  Both must give identical results on random data.
3. Invariants of the full pipeline on synthetic data.

None of this compares against real SAS output; see the note on tiebreakers
in the module docstring.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dao_rake_round_polars as m

DETAIL_KEYS = list(m.Config().detail_keys)
COUNTY_KEYS = list(m.Config().county_gqt_keys)
STATE_KEYS = list(m.Config().state_gqt_keys)


def evaluate(fn, values, *args, **kwargs) -> list:
    """Apply an expression helper such as m.coarsen to a list of numbers."""
    frame = pl.DataFrame({"x": values}, schema={"x": pl.Float64})
    out = frame.select(fn(pl.col("x"), *args, **kwargs))
    assert out.columns == ["x"]  # the helpers must keep the column's name
    return out["x"].to_list()


# ==========================================================================
# Reference implementation: literal transcription of the SAS code
# ==========================================================================


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def ref_round(x: float, unit: float = 1.0) -> float:
    """SAS ROUND via decimal arithmetic (independent of m.sas_round)."""
    q = (Decimal(repr(float(x))) / Decimal(repr(float(unit)))).quantize(
        Decimal(1), rounding=ROUND_HALF_UP  # decimal's HALF_UP is away from zero
    )
    return float(q) * unit


def ref_coarsen(x, base=5.0):
    if _missing(x):
        return x
    if x > 0:
        x = ref_round(x, base)
        if x == 0:
            x = 1.0
    return x


def _summary(rows, by, var):
    """proc summary nway; class &by; var &var; output sum= n=;"""
    sums, counts = {}, {}
    for r in rows:
        key = tuple(r[k] for k in by)
        sums.setdefault(key, None)
        counts.setdefault(key, 0)
        if not _missing(r[var]):
            sums[key] = (sums[key] or 0.0) + r[var]
            counts[key] += 1
    return sums, counts


def ref_rakeit(udsn, uvar, cgrp, cdsn, cvar, ovar):
    udsn = sorted(udsn, key=lambda r: tuple(r[k] for k in cgrp))
    control, _ = _summary(cdsn, cgrp, cvar)
    uncontrolled, records = _summary(udsn, cgrp, uvar)

    rake = {}
    for key in set(control) | set(uncontrolled):
        u = uncontrolled.get(key)
        c = control.get(key)
        rec = records.get(key)
        if u is None:
            u = 0
        if c is None:
            c = 0
        if rec is not None and rec > 0:
            dist = c / rec
        else:
            dist = 1
        if u > 0:
            rake[key] = (c / u, 0)
        else:
            rake[key] = (dist, 1)

    out = []
    for r in udsn:
        factor, flag = rake[tuple(r[k] for k in cgrp)]
        r = dict(r)
        if flag == 0:
            r[ovar] = float("nan") if _missing(r[uvar]) else r[uvar] * factor
        else:
            r[ovar] = factor
        out.append(r)
    return out


def ref_roundit(idsn, ivar, lst, ovar, seed=42):
    use_rules = all("age" in r and "gqt" in r for r in idsn)

    rounding = []
    for r in idsn:
        r = dict(r)
        r["age_uni"] = 1
        if use_rules:
            age, gqt = r["age"], r["gqt"]
            low = -math.inf if _missing(age) else age   # SAS: missing <= anything
            high = -math.inf if _missing(age) else age  # SAS: missing >= x is false
            if gqt == 1 and low <= 14:
                r["age_uni"] = 0
            if gqt == 2 and high >= 26:
                r["age_uni"] = 0
            if gqt == 3 and low <= 19:
                r["age_uni"] = 0
            if gqt == 5 and low <= 15:
                r["age_uni"] = 0
            if gqt == 5 and high >= 66:
                r["age_uni"] = 0
            if gqt == 6 and low <= 16:
                r["age_uni"] = 0
            if gqt == 6 and high >= 65:
                r["age_uni"] = 0
        r["integer"] = math.floor(r[ivar])
        r["mantissa"] = r[ivar] - r["integer"]
        rounding.append(r)

    rounding.sort(key=lambda r: tuple(r[k] for k in lst))

    total, _ = _summary(rounding, lst, ivar)
    integer, _ = _summary(rounding, lst, "integer")
    stuffpool = {
        key: ref_round(total[key] or 0) - (integer[key] or 0) for key in total
    }

    draws = np.random.Generator(np.random.PCG64(seed)).random(len(rounding))
    for r, d in zip(rounding, draws, strict=True):
        r["stuffpool"] = stuffpool[tuple(r[k] for k in lst)]
        r["tiebreaker"] = d

    # by &LIST descending mantissa descending integer descending tiebreaker
    rounding.sort(key=lambda r: -r["tiebreaker"])
    rounding.sort(key=lambda r: -r["integer"])
    rounding.sort(key=lambda r: -r["mantissa"])
    rounding.sort(key=lambda r: tuple(r[k] for k in lst))

    out, prev, counter = [], object(), 0
    for r in rounding:
        key = tuple(r[k] for k in lst)
        if key != prev:  # first.&LAST
            counter = 0
            prev = key
        if r["age_uni"] == 1 and counter < r["stuffpool"]:
            counter += 1
            r[ovar] = r["integer"] + 1
        else:
            r[ovar] = r["integer"]
        out.append(r)
    return out


def ref_pipeline(rows, v="blend_pop"):
    """prog03_dao_roundmethod8.sas, step by step."""
    rounded_input = sorted(rows, key=lambda r: tuple(r[k] for k in DETAIL_KEYS))

    sums, _ = _summary(rounded_input, COUNTY_KEYS, v)
    rounded_input_gqt = [
        {**dict(zip(COUNTY_KEYS, key, strict=True)), v: s} for key, s in sorted(sums.items())
    ]
    sums, _ = _summary(rounded_input_gqt, STATE_KEYS, v)
    rounded_input_stategqt = [
        {**dict(zip(STATE_KEYS, key, strict=True)), v: s} for key, s in sorted(sums.items())
    ]

    gqt_r = [{**r, v: ref_coarsen(r[v])} for r in rounded_input_gqt]
    gqt_rc = ref_rakeit(gqt_r, v, STATE_KEYS, rounded_input_stategqt, v, v)
    gqt_rcr = ref_roundit(gqt_rc, v, STATE_KEYS, v)
    gqt_rcr = [{k: r[k] for k in [*COUNTY_KEYS, v]} for r in gqt_rcr]
    gqt_rcr.sort(key=lambda r: tuple(r[k] for k in COUNTY_KEYS))

    base = [{**r, v: ref_coarsen(r[v])} for r in rounded_input]
    base_c = ref_rakeit(base, v, COUNTY_KEYS, gqt_rcr, v, v)
    base_cr = ref_roundit(base_c, v, COUNTY_KEYS, v)
    base_cr = [{k: r[k] for k in [*DETAIL_KEYS, v]} for r in base_cr]
    base_cr.sort(key=lambda r: tuple(r[k] for k in DETAIL_KEYS))
    return pl.DataFrame(base_cr), pl.DataFrame(gqt_rcr)


# ==========================================================================
# Synthetic data
# ==========================================================================


def make_detail(seed=0, states=(1, 2), counties=(1, 3, 5), max_age=85, races=4,
                oou_positive=False) -> pl.DataFrame:
    """Small detail file.  Values sit on a 1/64 grid so every sum is exact
    in floating point no matter what order it is added in."""
    rng = np.random.default_rng(seed)
    levels = {
        "nation": [1], "state": list(states), "county": list(counties),
        "age": list(range(max_age + 1)), "sex": [1, 2],
        "race31": list(range(1, races + 1)), "hisp": [1, 2], "gqt": list(range(8)),
    }
    df = pl.DataFrame({"nation": levels["nation"]})
    for name in DETAIL_KEYS[1:]:
        df = df.join(pl.DataFrame({name: levels[name]}), how="cross")
    df = df.sort(DETAIL_KEYS)
    n = df.height
    in_universe = m.age_universe_flag(df).to_numpy()
    scale = np.where(df["gqt"].to_numpy() == 0, 30.0, 1.5)
    val = rng.gamma(0.6, scale, n)
    val[rng.random(n) < 0.4] = 0.0
    # People only exist inside their GQ type's age universe ...
    val[~in_universe] = 0.0
    if oou_positive:
        # ... except for a few stray cells, as in the V2025 geocoding case.
        oou = np.flatnonzero(~in_universe)
        val[rng.choice(oou, 40, replace=False)] = rng.gamma(2.0, 2.0, 40)
    df = df.with_columns(pl.Series("blend_pop", np.round(val * 64) / 64))
    return df.sample(fraction=1.0, shuffle=True, seed=seed)  # shuffled


def group_sum(df: pl.DataFrame, keys, col="blend_pop") -> pl.DataFrame:
    return df.group_by(keys).agg(pl.col(col).sum()).sort(keys)


@pytest.fixture(scope="module")
def clean():
    return make_detail(seed=1)


@pytest.fixture(scope="module")
def clean_result(clean):
    return m.run_pipeline(clean)


# ==========================================================================
# 1. Hand-worked examples
# ==========================================================================


def test_sas_round_is_half_away_from_zero():
    x = [0.5, 1.5, 2.5, -0.5, -2.5, 2.4999, 0.0, None]
    assert evaluate(m.sas_round, x) == [1, 2, 3, -1, -3, 2, 0, None]
    assert evaluate(m.sas_round, [2.4, 2.5, 7.4, 7.5, 12, 13], 5) == [0, 5, 5, 10, 10, 15]


def test_sas_round_tolerates_floating_point_noise():
    noisy = [0.1 + 0.2 + 0.2]  # 0.5000000000000001
    assert evaluate(m.sas_round, noisy) == [1.0]
    assert evaluate(m.sas_round, [float(np.nextafter(2.5, 0))], 5) == [5.0]  # one ULP below
    assert evaluate(m.sas_round, [2.4999999], 5) == [0.0]                    # genuinely below


def test_coarsen():
    x = [0, 0.01, 2.4, 2.5, 7.4, 7.5, 12, 13, -3.2, None]
    assert evaluate(m.coarsen, x) == [0, 1, 1, 5, 5, 10, 10, 15, -3.2, None]
    assert evaluate(m.coarsen, [4.9, 5.1, 14.9], base=10) == [1, 10, 10]


def test_coarsen_replaces_the_column_in_with_columns():
    df = pl.DataFrame({"blend_pop": [0.4, 7.6], "other": [1, 2]})
    out = df.with_columns(m.coarsen(pl.col("blend_pop")))
    assert out.columns == ["blend_pop", "other"]
    assert out["blend_pop"].to_list() == [1.0, 10.0]


@pytest.mark.parametrize(
    "kind", ["integers", "negative", "int8", "uint64", "text", "floats", "wide range", "nulls"]
)
def test_sort_by_keys_matches_the_plain_stable_sort(kind):
    rng = np.random.default_rng(0)
    n = 5000
    a, b, c = rng.integers(0, 6, n), rng.integers(0, 40, n), rng.integers(0, 3, n)
    df = pl.DataFrame({"a": a, "b": b, "c": c, "row": np.arange(n)})  # many duplicate keys
    if kind == "negative":
        df = df.with_columns(pl.col("a") - 3, pl.col("b") * -7)
    elif kind == "int8":    # the span -105..105 does not fit in the column's own type
        df = df.with_columns(((pl.col("b") - 20) * 5).cast(pl.Int8), pl.col("c").cast(pl.Int8))
    elif kind == "uint64":  # values above the signed 64-bit range
        df = df.with_columns((pl.col("b").cast(pl.UInt64) + 2**63 + 5).alias("b"))
    elif kind == "text":    # "10" sorts before "9": text order, not numeric order
        df = df.with_columns(pl.col("b").cast(pl.String), pl.col("c").cast(pl.String))
    elif kind == "floats":
        df = df.with_columns(pl.col("a").cast(pl.Float64), pl.col("b") / 7)
    elif kind == "wide range":  # too wide to pack into 64 bits
        df = df.with_columns(pl.col("a") * 2**40, pl.col("b") * 2**40)
    elif kind == "nulls":
        df = df.with_columns(pl.when(pl.col("b") == 5).then(None).otherwise(pl.col("b")).alias("b"))
    keys = ["a", "b", "c"]
    assert_frame_equal(m.sort_by_keys(df, keys), df.sort(keys, maintain_order=True))
    assert m.sort_by_keys(df, keys).columns == df.columns
    assert_frame_equal(m.sort_by_keys(df.head(0), keys), df.head(0))


def test_rakeit_scales_to_control():
    df = pl.DataFrame({"g": ["a", "a", "b", "b", "c"], "x": [10.0, 30.0, 0.0, 0.0, 5.0]})
    control = pl.DataFrame({"g": ["a", "b"], "x": [80.0, 6.0]})
    out = m.rakeit(df, "x", ["g"], control=control, factor_col="rake")
    # a: 80/40 = 2.  b: total is 0, so 6 is spread evenly.  c: no control -> 0.
    assert out.columns == ["g", "x", "rake"]  # no working columns left behind
    assert out["x"].to_list() == [20.0, 60.0, 3.0, 3.0, 0.0]
    assert out["rake"].to_list() == [2.0, 2.0, 3.0, 3.0, 0.0]
    assert df["x"].to_list() == [10.0, 30.0, 0.0, 0.0, 5.0]  # input untouched


def test_rakeit_control_from_same_frame():
    df = pl.DataFrame({"g": [1, 1, 2], "x": [1.0, 3.0, 5.0], "target": [6.0, 6.0, 10.0]})
    out = m.rakeit(df, "x", ["g"], control_col="target", out_col="y")
    assert out["y"].to_list() == [3.0, 9.0, 10.0]  # controls: g1 = 12, g2 = 10
    assert out["x"].to_list() == [1.0, 3.0, 5.0]


def test_rakeit_keeps_row_order_and_accepts_integer_values():
    df = pl.DataFrame({"g": [2, 1, 2, 1], "x": [1, 3, 3, 1]})  # integer counts
    control = pl.DataFrame({"g": [1, 2], "x": [8, 2]})
    assert m.rakeit(df, "x", ["g"], control=control)["x"].to_list() == [0.5, 6.0, 1.5, 2.0]


def test_rakeit_control_keys_may_have_a_different_numeric_type():
    # e.g. a control read from SAS (every number a float) against integer keys
    df = pl.DataFrame({"g": [1, 1, 2], "x": [1.0, 1.0, 1.0]})
    control = pl.DataFrame({"g": [1.0, 2.0], "x": [4.0, 3.0]})
    assert m.rakeit(df, "x", ["g"], control=control)["x"].to_list() == [2.0, 2.0, 3.0]


def test_rakeit_missing_values_follow_sas():
    df = pl.DataFrame({"g": [1, 1, 2, 2], "x": [2.0, None, None, None]})
    control = pl.DataFrame({"g": [1, 2], "x": [10.0, 4.0]})
    out = m.rakeit(df, "x", ["g"], control=control)["x"].to_list()
    assert out[:2] == [10.0, None]  # missing times a factor stays missing
    assert out[2:] == [1.0, 1.0]    # all-missing group: n = 0 -> 1


def test_nan_is_treated_as_missing():
    df = pl.DataFrame({"g": [1, 1], "x": [2.0, float("nan")]})
    control = pl.DataFrame({"g": [1], "x": [10.0]})
    assert m.rakeit(df, "x", ["g"], control=control)["x"].to_list() == [10.0, None]
    assert m.roundit(df, "x", ["g"])["x"].to_list() == [2.0, None]


def test_rakeit_warns_on_unmatched_control(caplog):
    df = pl.DataFrame({"g": [1], "x": [2.0]})
    control = pl.DataFrame({"g": [1, 9], "x": [4.0, 7.0]})
    with caplog.at_level("WARNING"):
        out = m.rakeit(df, "x", ["g"], control=control)
    assert out.height == 1 and out["x"].to_list() == [4.0]
    assert "no rows to rake" in caplog.text


def test_roundit_greatest_mantissa():
    df = pl.DataFrame({"g": 1, "x": [1.6, 2.3, 3.1]})
    out = m.roundit(df, "x", ["g"])
    assert out["x"].to_list() == [2, 2, 3]
    assert out.columns == ["g", "x"]  # no working columns left behind


def test_roundit_half_total_rounds_up():
    # Totals of 0.5 and 2.5 round to 1 and 3 (SAS ROUND), not 0 and 2 (round-half-even).
    assert m.roundit(pl.DataFrame({"g": 1, "x": [0.25, 0.25]}), "x", ["g"])["x"].sum() == 1
    assert m.roundit(pl.DataFrame({"g": 1, "x": [1.25, 1.25]}), "x", ["g"])["x"].sum() == 3


def test_roundit_missing_value_does_not_take_the_added_person():
    # A missing value sorts last, as in a SAS descending sort, so 0.6 gets the 1.
    df = pl.DataFrame({"g": 1, "x": [None, 0.6, 0.4]})
    assert m.roundit(df, "x", ["g"])["x"].to_list() == [None, 1, 0]


def test_roundit_tie_goes_to_larger_integer():
    df = pl.DataFrame({"g": 1, "x": [1.5, 3.5]})
    assert m.roundit(df, "x", ["g"])["x"].to_list() == [1, 4]


def test_roundit_random_ties_are_seeded():
    df = pl.DataFrame({"g": 1, "x": [0.5] * 6})
    a = m.roundit(df, "x", ["g"], seed=42)["x"].to_list()
    b = m.roundit(df, "x", ["g"], seed=42)["x"].to_list()
    assert sum(a) == 3 and set(a) == {0, 1}
    assert a == b
    others = [m.roundit(df, "x", ["g"], seed=s)["x"].to_list() for s in range(20)]
    assert any(o != a for o in others)


def test_roundit_groups_are_independent_and_order_is_kept():
    df = pl.DataFrame({"g": [2, 1, 2, 1], "x": [0.7, 0.2, 0.3, 0.8]})
    assert m.roundit(df, "x", ["g"])["x"].to_list() == [1, 0, 0, 1]


def test_roundit_writes_a_new_column_when_asked():
    df = pl.DataFrame({"g": [1, 1], "x": [0.7, 0.3]})
    out = m.roundit(df, "x", ["g"], out_col="y")
    assert out.columns == ["g", "x", "y"]
    assert out["x"].to_list() == [0.7, 0.3] and out["y"].to_list() == [1, 0]


def test_roundit_negative_values_use_floor():
    df = pl.DataFrame({"g": 1, "x": [-2.3, 4.3]})  # floors -3 and 4, pool 1
    assert m.roundit(df, "x", ["g"])["x"].to_list() == [-2, 4]


def test_roundit_age_universe_redirects_stuffing():
    # Military (gqt 6): ages 16 and 65 are out of universe despite the
    # largest mantissas, so the two added persons go to ages 17 and 64.
    df = pl.DataFrame({"g": 1, "gqt": 6, "age": [16, 17, 64, 65],
                       "x": [0.9, 0.1, 0.1, 0.9]})
    assert m.roundit(df, "x", ["g"])["x"].to_list() == [0, 1, 1, 0]
    # Without an age column the rules are off.
    assert m.roundit(df.drop("age"), "x", ["g"])["x"].to_list() == [1, 0, 0, 1]


def test_roundit_out_of_universe_rows_keep_their_integer_part():
    df = pl.DataFrame({"g": 1, "gqt": 1, "age": [10, 30], "x": [3.6, 0.4]})
    assert m.roundit(df, "x", ["g"])["x"].to_list() == [3, 1]


def test_roundit_shortfall_is_reported(caplog):
    df = pl.DataFrame({"g": 1, "gqt": 2, "age": [30, 40], "x": [0.5, 0.5]})
    with caplog.at_level("WARNING"):
        out, groups = m.roundit(df, "x", ["g"], return_groups=True)
    assert out["x"].to_list() == [0, 0]
    cols = ["stuffpool", "eligible_rows", "stuffed", "shortfall"]
    assert list(groups.select(cols).row(0)) == [1, 0, 0, 1]
    assert "could not be placed" in caplog.text


@pytest.mark.parametrize("gqt,eligible_ages", [
    (0, range(0, 101)), (1, range(15, 101)), (2, range(0, 26)), (3, range(20, 101)),
    (4, range(0, 101)), (5, range(16, 66)), (6, range(17, 65)), (7, range(0, 101)),
])
def test_age_universe_table(gqt, eligible_ages):
    df = pl.DataFrame({"age": range(0, 101), "gqt": gqt})
    assert df.filter(m.age_universe_flag(df))["age"].to_list() == list(eligible_ages)


def test_age_universe_missing_age_is_below_every_threshold():
    df = pl.DataFrame({"age": [None] * 8, "gqt": range(8)},
                      schema={"age": pl.Int64, "gqt": pl.Int64})
    expected = [True, False, True, False, True, False, False, True]
    assert m.age_universe_flag(df).to_list() == expected


def test_age_universe_missing_gqt_is_eligible():
    df = pl.DataFrame({"age": [5, 90], "gqt": [None, None]},
                      schema={"age": pl.Int64, "gqt": pl.Int64})
    assert m.age_universe_flag(df).to_list() == [True, True]


def test_roundit_accepts_external_tiebreaker():
    df = pl.DataFrame({"g": 1, "x": [0.5, 0.5, 0.5, 0.5], "tb": [0.1, 0.9, 0.8, 0.2]})
    assert m.roundit(df, "x", ["g"], tiebreaker_col="tb")["x"].to_list() == [0, 1, 1, 0]


def test_missing_keys_are_rejected():
    df = pl.DataFrame({"g": [1, None], "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing value in a group key"):
        m.rakeit(df, "x", ["g"])
    with pytest.raises(ValueError, match="missing value in a group key"):
        m.roundit(df, "x", ["g"])


# ==========================================================================
# 2. Cross-check against the literal SAS transcription
# ==========================================================================


@pytest.mark.parametrize("seed", range(5))
def test_rakeit_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = 400
    x = np.round(rng.gamma(1.0, 5.0, n) * 64) / 64
    a, b = rng.integers(0, 6, n), rng.integers(0, 4, n)
    x[a == 0] = 0.0                       # groups with a zero total
    x[rng.random(n) < 0.03] = np.nan      # scattered missing values
    df = pl.DataFrame({"a": a, "b": b, "x": x})
    # Controls on the 1/64 grid too: their sums are then exact, so the
    # comparison does not depend on the order values are added in.
    control = pl.DataFrame({"a": rng.integers(0, 6, 60), "b": rng.integers(0, 5, 60),
                            "x": np.round(rng.gamma(2.0, 20.0, 60) * 64) / 64})
    control = control.filter(~((pl.col("a") == 5) & (pl.col("b") == 3)))  # one group uncontrolled
    control = control.filter(pl.col("b") < 4)                             # no control-only groups

    got = m.rakeit(df, "x", ["a", "b"], control=control, out_col="y")
    ref = ref_rakeit(df.to_dicts(), "x", ["a", "b"], control.to_dicts(), "x", "y")
    got = got.sort(["a", "b"], maintain_order=True)
    expected = np.array([r["y"] for r in ref], dtype=float)
    assert np.isnan(expected).any() and (expected == 0).any()
    np.testing.assert_array_equal(got["y"].to_numpy(), expected)  # null -> NaN


@pytest.mark.parametrize("seed", range(5))
def test_roundit_matches_reference(seed):
    rng = np.random.default_rng(100 + seed)
    n = 3000
    county, gqt = rng.integers(0, 5, n), rng.integers(0, 8, n)
    factor = 0.5 + rng.random(40)[county * 8 + gqt]  # one rake factor per group
    # Few distinct values per group, so mantissa and integer ties are common.
    df = pl.DataFrame({"county": county, "gqt": gqt, "age": rng.integers(0, 101, n),
                       "x": rng.choice([0.0, 1.0, 5.0, 10.0, 25.0], n) * factor})
    got = m.roundit(df, "x", ["county", "gqt"], out_col="y", seed=7)
    ref = pl.DataFrame(ref_roundit(df.to_dicts(), "x", ["county", "gqt"], "y", seed=7))
    cols = ["county", "gqt", "age", "x", "y"]
    assert_frame_equal(got.select(cols).sort(cols), ref.select(cols).sort(cols),
                       check_dtypes=False, check_exact=True)
    # The test is only meaningful if ties and out-of-universe rows occur.
    assert df.select(pl.struct("county", "gqt", "x").is_duplicated().mean()).item() > 0.5
    oou_with_fraction = (~m.age_universe_flag(df)) & (df["x"] % 1 > 0)
    assert oou_with_fraction.sum() > 100


@pytest.mark.parametrize("oou_positive", [False, True])
def test_pipeline_matches_reference(oou_positive):
    df = make_detail(seed=3, counties=(1, 3), max_age=70, races=2, oou_positive=oou_positive)
    result = m.run_pipeline(df)
    ref_detail, ref_county = ref_pipeline(df.to_dicts())
    assert_frame_equal(result.detail, ref_detail, check_dtypes=False, check_exact=True)
    np.testing.assert_array_equal(result.county_gqt["control"].to_numpy(),
                                  ref_county["blend_pop"].to_numpy())


# ==========================================================================
# 3. Pipeline invariants
# ==========================================================================


def test_output_has_same_cells_as_input(clean, clean_result):
    out = clean_result.detail
    assert out.columns == [*DETAIL_KEYS, "blend_pop"]
    assert_frame_equal(out.select(DETAIL_KEYS), clean.select(DETAIL_KEYS).sort(DETAIL_KEYS))


def test_output_is_non_negative_integers(clean_result):
    v = clean_result.detail["blend_pop"]
    assert v.dtype == pl.Int64 and v.null_count() == 0 and (v >= 0).all()


def test_state_gqt_totals_match_rounded_original(clean, clean_result):
    original = group_sum(clean, STATE_KEYS)
    final = group_sum(clean_result.detail, STATE_KEYS)
    target = original.select(m.sas_round(pl.col("blend_pop")))["blend_pop"]
    assert final["blend_pop"].to_list() == target.to_list()
    assert (clean_result.state_gqt_check["final_minus_target"] == 0).all()


def test_county_gqt_totals_match_stage1_controls(clean_result):
    c = clean_result.county_gqt
    assert c["final"].to_list() == c["control"].to_list()
    assert clean_result.stage1_groups["shortfall"].sum() == 0
    assert clean_result.stage2_groups["shortfall"].sum() == 0


def test_county_totals_are_not_preserved(clean, clean_result):
    keys = ["nation", "state", "county"]
    original = group_sum(clean, keys).select(m.sas_round(pl.col("blend_pop")))["blend_pop"]
    final = group_sum(clean_result.detail, keys)["blend_pop"]
    assert (final != original).any()


def test_zero_cells_stay_zero(clean, clean_result):
    original = clean.sort(DETAIL_KEYS)["blend_pop"]
    assert (clean_result.detail["blend_pop"].filter(original == 0) == 0).all()


def test_positive_cells_can_become_zero(clean, clean_result):
    original = clean.sort(DETAIL_KEYS)["blend_pop"]
    assert (clean_result.detail["blend_pop"].filter(original > 0) == 0).any()


def test_output_is_not_limited_to_multiples_of_base(clean_result):
    assert (clean_result.detail["blend_pop"] % 5 != 0).any()


def test_nobody_is_added_out_of_universe():
    df = make_detail(seed=5, oou_positive=True)
    result = m.run_pipeline(df)
    # Redo stage 2's rake to get the value each cell was floored from.
    control = result.county_gqt.select(*COUNTY_KEYS, pl.col("control").alias("blend_pop"))
    raked = m.rakeit(df.sort(DETAIL_KEYS).with_columns(m.coarsen(pl.col("blend_pop"))),
                     "blend_pop", COUNTY_KEYS, control=control)
    oou = result.detail.with_columns(raked["blend_pop"].alias("raked")).filter(
        ~m.age_universe_expr(result.detail.columns)
    )
    assert oou.height > 0
    assert (oou["blend_pop"] == oou["raked"].floor()).all()
    assert (oou["blend_pop"] > 0).any()  # integer parts survive


def test_pipeline_ignores_input_row_order(clean, clean_result):
    again = m.run_pipeline(clean.sample(fraction=1.0, shuffle=True, seed=99))
    assert_frame_equal(again.detail, clean_result.detail)


def test_pipeline_is_repeatable(clean, clean_result):
    assert_frame_equal(m.run_pipeline(clean).detail, clean_result.detail)


def test_seed_changes_tied_cells_but_not_totals(clean, clean_result):
    other = m.run_pipeline(clean, m.Config(seed=7))
    assert not other.detail["blend_pop"].equals(clean_result.detail["blend_pop"])
    assert_frame_equal(group_sum(other.detail, STATE_KEYS),
                       group_sum(clean_result.detail, STATE_KEYS))


def test_string_geography_codes(clean, clean_result):
    # FIPS-style text codes sort the same way here, so cells line up one to one.
    as_text = clean.with_columns(pl.col("state").cast(pl.String).str.zfill(2),
                                 pl.col("county").cast(pl.String).str.zfill(3))
    out = m.run_pipeline(as_text).detail
    assert out["state"].dtype == pl.String
    assert out["blend_pop"].to_list() == clean_result.detail["blend_pop"].to_list()


def test_float_keys_as_read_from_sas(clean, clean_result):
    # SAS stores every number as a float, so keys arrive as 1.0, 2.0, ...
    as_float = clean.with_columns(pl.col(DETAIL_KEYS).cast(pl.Float64))
    out = m.run_pipeline(as_float).detail
    assert out["blend_pop"].to_list() == clean_result.detail["blend_pop"].to_list()


def test_integer_counts_as_input(clean):
    counts = clean.with_columns(pl.col("blend_pop").round().cast(pl.Int64))
    out = m.run_pipeline(counts)
    assert (out.state_gqt_check["final_sum"] == out.state_gqt_check["original"]).all()


def test_missing_values_rejected_by_default(clean):
    for missing in (None, float("nan")):
        bad = clean.with_columns(
            pl.when(pl.int_range(pl.len()) == 0).then(missing).otherwise(pl.col("blend_pop"))
            .alias("blend_pop")
        )
        with pytest.raises(ValueError, match="missing value"):
            m.run_pipeline(bad)
        out = m.run_pipeline(bad, m.Config(allow_missing_values=True)).detail
        assert out["blend_pop"].null_count() == 1


def test_non_polars_input_gets_a_helpful_error():
    with pytest.raises(TypeError, match="from_pandas"):
        m.run_pipeline({"not": ["a", "polars", "frame"]})


def test_stats_table_shape(clean_result):
    s = clean_result.stats
    assert s.height == 12  # 2 levels x 3 universes x 2 variables
    expected = {"sum", "n", "mean", "std", "min", "max", "mode", "median", "p5", "p95"}
    assert expected <= set(s.columns)
    all_detail = s.filter((pl.col("level") == "detail") & (pl.col("universe") == "all")
                          & (pl.col("variable") == "diff"))
    assert all_detail["n"].item() == clean_result.detail.height


def test_describe_matches_sas_definitions():
    d = m._describe(np.array([1.0, 2.0, 2.0, 3.0, 10.0]))
    assert d["mode"] == 2 and d["median"] == 2 and d["n"] == 5
    assert d["std"] == pytest.approx(np.std([1, 2, 2, 3, 10], ddof=1))
    # QNTLDEF=5 with n*p = 0.25 and 4.75 (not integers): next order statistic up
    assert d["p5"] == 1 and d["p95"] == 10
    # n*p integer -> average of the two neighbours: n = 20, p = 0.05 -> (x1 + x2) / 2
    assert m._describe(np.arange(1.0, 21.0))["p5"] == 1.5


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_command_line_round_trip(tmp_path, clean, fmt):
    src = tmp_path / f"input.{fmt}"
    m.write_table(clean, src)
    cfg = tmp_path / "config.toml"
    cfg.write_text('seed = 42\nrounding_base = 5.0\n\n[age_universe]\n'
                   '1 = { out_at_or_below = 14 }\n2 = { out_at_or_above = 26 }\n'
                   '3 = { out_at_or_below = 19 }\n'
                   '5 = { out_at_or_below = 15, out_at_or_above = 66 }\n'
                   '6 = { out_at_or_below = 16, out_at_or_above = 65 }\n')
    out_dir = tmp_path / "out"
    assert m.main([str(src), "--out-dir", str(out_dir), "--format", fmt,
                   "--config", str(cfg)]) == 0
    out = m.read_table(out_dir / f"rounded_detail.{fmt}")
    assert_frame_equal(out, m.run_pipeline(clean).detail)
    for name in (f"county_gqt_controls.{fmt}", "state_gqt_check.csv", "difference_stats.csv",
                 "shortfall_groups.csv", "report.txt"):
        assert (out_dir / name).exists()


def test_csv_types_are_inferred_from_the_whole_file(tmp_path):
    # 150 whole numbers, then a fraction: the column must still be read as float.
    path = tmp_path / "late_fraction.csv"
    path.write_text("x\n" + "0\n" * 150 + "1.5\n")
    assert m.read_table(path)["x"].dtype == pl.Float64
