"""Tests for dao_rake_round.py.

Run with:  pytest test_dao_rake_round.py

Three kinds of test:

1. Hand-worked examples for each building block.
2. A cross-check against `ref_*` functions below.  Those are a deliberately
   literal, row-by-row transcription of the SAS DATA steps (dictionaries,
   Python sorts, a retained counter) that share no code with the vectorised
   implementation.  Both must give identical results on random data.
3. Invariants of the full pipeline on synthetic data.

None of this compares against real SAS output; see the note on tiebreakers
in the module docstring.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
import pytest

import dao_rake_round as m

DETAIL_KEYS = list(m.Config().detail_keys)
COUNTY_KEYS = list(m.Config().county_gqt_keys)
STATE_KEYS = list(m.Config().state_gqt_keys)


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
    return pd.DataFrame(base_cr), pd.DataFrame(gqt_rcr)


# ==========================================================================
# Synthetic data
# ==========================================================================


def make_detail(seed=0, states=(1, 2), counties=(1, 3, 5), max_age=85, races=4,
                oou_positive=False) -> pd.DataFrame:
    """Small detail file.  Values sit on a 1/64 grid so every sum is exact
    in floating point no matter what order it is added in."""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [[1], states, counties, range(max_age + 1), [1, 2],
         range(1, races + 1), [1, 2], range(8)],
        names=DETAIL_KEYS,
    )
    df = idx.to_frame(index=False)
    n = len(df)
    scale = np.where(df["gqt"] == 0, 30.0, 1.5)
    val = rng.gamma(0.6, scale, n)
    val[rng.random(n) < 0.4] = 0.0
    # People only exist inside their GQ type's age universe ...
    val[~m.age_universe_flag(df)] = 0.0
    if oou_positive:
        # ... except for a few stray cells, as in the V2025 geocoding case.
        oou = np.flatnonzero(~m.age_universe_flag(df))
        val[rng.choice(oou, 40, replace=False)] = rng.gamma(2.0, 2.0, 40)
    df["blend_pop"] = np.round(val * 64) / 64
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffled


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
    x = [0.5, 1.5, 2.5, -0.5, -2.5, 2.4999, 0.0]
    assert m.sas_round(x).tolist() == [1, 2, 3, -1, -3, 2, 0]
    assert m.sas_round([2.4, 2.5, 7.4, 7.5, 12, 13], 5).tolist() == [0, 5, 5, 10, 10, 15]


def test_sas_round_tolerates_floating_point_noise():
    assert m.sas_round(0.1 + 0.2 + 0.2) == 1.0          # 0.5000000000000001
    assert m.sas_round(np.nextafter(2.5, 0), 5) == 5.0  # one ULP below halfway
    assert m.sas_round(2.4999999, 5) == 0.0             # genuinely below


def test_coarsen():
    x = np.array([0, 0.01, 2.4, 2.5, 7.4, 7.5, 12, 13, -3.2, np.nan])
    expected = np.array([0, 1, 1, 5, 5, 10, 10, 15, -3.2, np.nan])
    np.testing.assert_array_equal(m.coarsen(x), expected)
    np.testing.assert_array_equal(m.coarsen([4.9, 5.1, 14.9], base=10), [1, 10, 10])


def test_rakeit_scales_to_control():
    df = pd.DataFrame({"g": ["a", "a", "b", "b", "c"], "x": [10.0, 30.0, 0.0, 0.0, 5.0]})
    control = pd.DataFrame({"g": ["a", "b"], "x": [80.0, 6.0]})
    out = m.rakeit(df, "x", ["g"], control=control, factor_col="rake")
    # a: 80/40 = 2.  b: total is 0, so 6 is spread evenly.  c: no control -> 0.
    assert out["x"].tolist() == [20.0, 60.0, 3.0, 3.0, 0.0]
    assert out["rake"].tolist() == [2.0, 2.0, 3.0, 3.0, 0.0]
    assert df["x"].tolist() == [10.0, 30.0, 0.0, 0.0, 5.0]  # input untouched


def test_rakeit_control_from_same_frame():
    df = pd.DataFrame({"g": [1, 1, 2], "x": [1.0, 3.0, 5.0], "target": [6.0, 6.0, 10.0]})
    out = m.rakeit(df, "x", ["g"], control_col="target", out_col="y")
    assert out["y"].tolist() == [3.0, 9.0, 10.0]  # controls: g1 = 12, g2 = 10


def test_rakeit_missing_values_follow_sas():
    df = pd.DataFrame({"g": [1, 1, 2, 2], "x": [2.0, np.nan, np.nan, np.nan]})
    control = pd.DataFrame({"g": [1, 2], "x": [10.0, 4.0]})
    out = m.rakeit(df, "x", ["g"], control=control)["x"].to_numpy()
    assert out[0] == 10.0 and np.isnan(out[1])  # missing times a factor stays missing
    assert out[2] == 1.0 and out[3] == 1.0      # all-missing group: n = 0 -> 1


def test_rakeit_warns_on_unmatched_control(caplog):
    df = pd.DataFrame({"g": [1], "x": [2.0]})
    control = pd.DataFrame({"g": [1, 9], "x": [4.0, 7.0]})
    with caplog.at_level("WARNING"):
        out = m.rakeit(df, "x", ["g"], control=control)
    assert len(out) == 1 and out["x"].tolist() == [4.0]
    assert "no rows to rake" in caplog.text


def test_roundit_greatest_mantissa():
    df = pd.DataFrame({"g": 1, "x": [1.6, 2.3, 3.1]})
    assert m.roundit(df, "x", ["g"])["x"].tolist() == [2, 2, 3]


def test_roundit_half_total_rounds_up():
    # Totals of 0.5 and 2.5 round to 1 and 3 (SAS ROUND), not 0 and 2 (round-half-even).
    assert m.roundit(pd.DataFrame({"g": 1, "x": [0.25, 0.25]}), "x", ["g"])["x"].sum() == 1
    assert m.roundit(pd.DataFrame({"g": 1, "x": [1.25, 1.25]}), "x", ["g"])["x"].sum() == 3


def test_roundit_tie_goes_to_larger_integer():
    df = pd.DataFrame({"g": 1, "x": [1.5, 3.5]})
    assert m.roundit(df, "x", ["g"])["x"].tolist() == [1, 4]


def test_roundit_random_ties_are_seeded():
    df = pd.DataFrame({"g": 1, "x": [0.5] * 6})
    a = m.roundit(df, "x", ["g"], seed=42)["x"]
    b = m.roundit(df, "x", ["g"], seed=42)["x"]
    assert a.sum() == 3 and set(a) == {0, 1}
    assert a.tolist() == b.tolist()
    others = [m.roundit(df, "x", ["g"], seed=s)["x"].tolist() for s in range(20)]
    assert any(o != a.tolist() for o in others)


def test_roundit_groups_are_independent_and_order_is_kept():
    df = pd.DataFrame({"g": [2, 1, 2, 1], "x": [0.7, 0.2, 0.3, 0.8]})
    assert m.roundit(df, "x", ["g"])["x"].tolist() == [1, 0, 0, 1]


def test_roundit_negative_values_use_floor():
    df = pd.DataFrame({"g": 1, "x": [-2.3, 4.3]})  # floors -3 and 4, pool 1
    assert m.roundit(df, "x", ["g"])["x"].tolist() == [-2, 4]


def test_roundit_age_universe_redirects_stuffing():
    # Military (gqt 6): ages 16 and 65 are out of universe despite the
    # largest mantissas, so the two added persons go to ages 17 and 64.
    df = pd.DataFrame({"g": 1, "gqt": 6, "age": [16, 17, 64, 65],
                       "x": [0.9, 0.1, 0.1, 0.9]})
    assert m.roundit(df, "x", ["g"])["x"].tolist() == [0, 1, 1, 0]
    # Without an age column the rules are off.
    assert m.roundit(df.drop(columns="age"), "x", ["g"])["x"].tolist() == [1, 0, 0, 1]


def test_roundit_out_of_universe_rows_keep_their_integer_part():
    df = pd.DataFrame({"g": 1, "gqt": 1, "age": [10, 30], "x": [3.6, 0.4]})
    assert m.roundit(df, "x", ["g"])["x"].tolist() == [3, 1]


def test_roundit_shortfall_is_reported(caplog):
    df = pd.DataFrame({"g": 1, "gqt": 2, "age": [30, 40], "x": [0.5, 0.5]})
    with caplog.at_level("WARNING"):
        out, groups = m.roundit(df, "x", ["g"], return_groups=True)
    assert out["x"].tolist() == [0, 0]
    cols = ["stuffpool", "eligible_rows", "stuffed", "shortfall"]
    assert groups.loc[0, cols].tolist() == [1, 0, 0, 1]
    assert "could not be placed" in caplog.text


@pytest.mark.parametrize("gqt,eligible_ages", [
    (0, range(0, 101)), (1, range(15, 101)), (2, range(0, 26)), (3, range(20, 101)),
    (4, range(0, 101)), (5, range(16, 66)), (6, range(17, 65)), (7, range(0, 101)),
])
def test_age_universe_table(gqt, eligible_ages):
    df = pd.DataFrame({"age": range(0, 101), "gqt": gqt})
    flag = m.age_universe_flag(df)
    assert df.loc[flag, "age"].tolist() == list(eligible_ages)


def test_age_universe_missing_age_is_below_every_threshold():
    df = pd.DataFrame({"age": [np.nan] * 8, "gqt": range(8)})
    assert m.age_universe_flag(df).tolist() == [True, False, True, False, True, False, False, True]


def test_roundit_accepts_external_tiebreaker():
    df = pd.DataFrame({"g": 1, "x": [0.5, 0.5, 0.5, 0.5], "tb": [0.1, 0.9, 0.8, 0.2]})
    assert m.roundit(df, "x", ["g"], tiebreaker_col="tb")["x"].tolist() == [0, 1, 1, 0]


def test_missing_keys_are_rejected():
    df = pd.DataFrame({"g": [1, np.nan], "x": [1.0, 2.0]})
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
    df = pd.DataFrame({"a": rng.integers(0, 6, n), "b": rng.integers(0, 4, n),
                       "x": np.round(rng.gamma(1.0, 5.0, n) * 64) / 64})
    df.loc[df["a"] == 0, "x"] = 0.0                      # groups with a zero total
    df.loc[rng.random(n) < 0.03, "x"] = np.nan           # scattered missing values
    # Controls on the 1/64 grid too: their sums are then exact, so the
    # comparison does not depend on the order values are added in.
    control = pd.DataFrame({"a": rng.integers(0, 6, 60), "b": rng.integers(0, 5, 60),
                            "x": np.round(rng.gamma(2.0, 20.0, 60) * 64) / 64})
    control = control[~((control["a"] == 5) & (control["b"] == 3))]  # one group uncontrolled
    control = control[control["b"] < 4]                  # no control-only groups

    got = m.rakeit(df, "x", ["a", "b"], control=control, out_col="y")
    ref = pd.DataFrame(ref_rakeit(df.to_dict("records"), "x", ["a", "b"],
                                  control.to_dict("records"), "x", "y"))
    got = got.sort_values(["a", "b"], kind="stable").reset_index(drop=True)
    np.testing.assert_array_equal(got["y"].to_numpy(), ref["y"].to_numpy())


@pytest.mark.parametrize("seed", range(5))
def test_roundit_matches_reference(seed):
    rng = np.random.default_rng(100 + seed)
    n = 3000
    df = pd.DataFrame({"county": rng.integers(0, 5, n), "gqt": rng.integers(0, 8, n),
                       "age": rng.integers(0, 101, n)})
    # Few distinct values per group, so mantissa and integer ties are common.
    factor = 0.5 + rng.random(40)[df["county"] * 8 + df["gqt"]]  # one rake factor per group
    df["x"] = rng.choice([0.0, 1.0, 5.0, 10.0, 25.0], n) * factor
    got = m.roundit(df, "x", ["county", "gqt"], out_col="y", seed=7)
    ref = pd.DataFrame(ref_roundit(df.to_dict("records"), "x", ["county", "gqt"], "y", seed=7))
    cols = ["county", "gqt", "age", "x", "y"]
    pd.testing.assert_frame_equal(
        got[cols].sort_values(cols).reset_index(drop=True),
        ref[cols].sort_values(cols).reset_index(drop=True),
        check_dtype=False,
    )
    # The test is only meaningful if ties and out-of-universe rows occur.
    tied = df.assign(f=np.floor(df["x"])).duplicated(["county", "gqt", "x"], keep=False)
    assert tied.mean() > 0.5
    assert (~m.age_universe_flag(df) & (got["x"] % 1 > 0)).sum() > 100


@pytest.mark.parametrize("oou_positive", [False, True])
def test_pipeline_matches_reference(oou_positive):
    df = make_detail(seed=3, counties=(1, 3), max_age=70, races=2, oou_positive=oou_positive)
    result = m.run_pipeline(df)
    ref_detail, ref_county = ref_pipeline(df.to_dict("records"))
    pd.testing.assert_frame_equal(result.detail, ref_detail, check_dtype=False)
    np.testing.assert_array_equal(result.county_gqt["control"].to_numpy(),
                                  ref_county["blend_pop"].to_numpy())


# ==========================================================================
# 3. Pipeline invariants
# ==========================================================================


def test_output_has_same_cells_as_input(clean, clean_result):
    out = clean_result.detail
    assert list(out.columns) == [*DETAIL_KEYS, "blend_pop"]
    expected = clean.sort_values(DETAIL_KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(out[DETAIL_KEYS], expected[DETAIL_KEYS])


def test_output_is_non_negative_integers(clean_result):
    v = clean_result.detail["blend_pop"]
    assert v.dtype == "int64" and (v >= 0).all()


def test_state_gqt_totals_match_rounded_original(clean, clean_result):
    original = clean.groupby(STATE_KEYS)["blend_pop"].sum()
    final = clean_result.detail.groupby(STATE_KEYS)["blend_pop"].sum()
    np.testing.assert_array_equal(final.to_numpy(), m.sas_round(original.to_numpy()))
    assert (clean_result.state_gqt_check["final_minus_target"] == 0).all()


def test_county_gqt_totals_match_stage1_controls(clean_result):
    c = clean_result.county_gqt
    np.testing.assert_array_equal(c["final"].to_numpy(), c["control"].to_numpy())
    assert clean_result.stage1_groups["shortfall"].sum() == 0
    assert clean_result.stage2_groups["shortfall"].sum() == 0


def test_county_totals_are_not_preserved(clean, clean_result):
    keys = ["nation", "state", "county"]
    original = clean.groupby(keys)["blend_pop"].sum()
    final = clean_result.detail.groupby(keys)["blend_pop"].sum()
    assert (final.to_numpy() != m.sas_round(original.to_numpy())).any()


def test_zero_cells_stay_zero(clean, clean_result):
    original = clean.sort_values(DETAIL_KEYS)["blend_pop"].to_numpy()
    assert (clean_result.detail["blend_pop"].to_numpy()[original == 0] == 0).all()


def test_positive_cells_can_become_zero(clean, clean_result):
    original = clean.sort_values(DETAIL_KEYS)["blend_pop"].to_numpy()
    assert (clean_result.detail["blend_pop"].to_numpy()[original > 0] == 0).any()


def test_output_is_not_limited_to_multiples_of_base(clean_result):
    assert (clean_result.detail["blend_pop"] % 5 != 0).any()


def test_nobody_is_added_out_of_universe():
    df = make_detail(seed=5, oou_positive=True)
    result = m.run_pipeline(df)
    out = result.detail
    oou = ~m.age_universe_flag(out)
    # Rerun stage 2's inputs to get the raked value each cell was floored from.
    cfg = m.Config()
    detail = df.sort_values(DETAIL_KEYS).reset_index(drop=True)
    control = result.county_gqt[COUNTY_KEYS].assign(blend_pop=result.county_gqt["control"])
    raked = m.rakeit(detail.assign(blend_pop=m.coarsen(detail["blend_pop"])),
                     "blend_pop", COUNTY_KEYS, control=control)["blend_pop"].to_numpy()
    assert (out["blend_pop"].to_numpy()[oou] == np.floor(raked[oou])).all()
    assert (out["blend_pop"].to_numpy()[oou] > 0).any()  # integer parts survive
    assert cfg.seed == 42


def test_pipeline_ignores_input_row_order(clean, clean_result):
    again = m.run_pipeline(clean.sample(frac=1.0, random_state=99))
    pd.testing.assert_frame_equal(again.detail, clean_result.detail)


def test_seed_changes_tied_cells_but_not_totals(clean, clean_result):
    other = m.run_pipeline(clean, m.Config(seed=7))
    assert not other.detail["blend_pop"].equals(clean_result.detail["blend_pop"])
    pd.testing.assert_frame_equal(
        other.detail.groupby(STATE_KEYS)["blend_pop"].sum().reset_index(),
        clean_result.detail.groupby(STATE_KEYS)["blend_pop"].sum().reset_index(),
    )


def test_missing_values_rejected_by_default(clean):
    bad = clean.copy()
    bad.loc[0, "blend_pop"] = np.nan
    with pytest.raises(ValueError, match="missing value"):
        m.run_pipeline(bad)
    out = m.run_pipeline(bad, m.Config(allow_missing_values=True)).detail
    assert out["blend_pop"].isna().sum() == 1


def test_stats_table_shape(clean_result):
    s = clean_result.stats
    assert len(s) == 12  # 2 levels x 3 universes x 2 variables
    expected = {"sum", "n", "mean", "std", "min", "max", "mode", "median", "p5", "p95"}
    assert expected <= set(s.columns)
    all_detail = s[(s.level == "detail") & (s.universe == "all") & (s.variable == "diff")].iloc[0]
    assert all_detail["n"] == len(clean_result.detail)


def test_describe_matches_sas_definitions():
    d = m._describe(np.array([1.0, 2.0, 2.0, 3.0, 10.0]))
    assert d["mode"] == 2 and d["median"] == 2 and d["n"] == 5
    assert d["std"] == pytest.approx(np.std([1, 2, 2, 3, 10], ddof=1))
    # QNTLDEF=5 with n*p = 0.25 and 4.75 (not integers): next order statistic up
    assert d["p5"] == 1 and d["p95"] == 10
    # n*p integer -> average of the two neighbours: n = 20, p = 0.05 -> (x1 + x2) / 2
    assert m._describe(np.arange(1.0, 21.0))["p5"] == 1.5


def test_command_line_round_trip(tmp_path, clean):
    src = tmp_path / "input.csv"
    clean.to_csv(src, index=False)
    cfg = tmp_path / "config.toml"
    cfg.write_text('seed = 42\nrounding_base = 5.0\n\n[age_universe]\n'
                   '1 = { out_at_or_below = 14 }\n2 = { out_at_or_above = 26 }\n'
                   '3 = { out_at_or_below = 19 }\n'
                   '5 = { out_at_or_below = 15, out_at_or_above = 66 }\n'
                   '6 = { out_at_or_below = 16, out_at_or_above = 65 }\n')
    assert m.main([str(src), "--out-dir", str(tmp_path / "out"), "--format", "csv",
                   "--config", str(cfg)]) == 0
    out = pd.read_csv(tmp_path / "out" / "rounded_detail.csv")
    pd.testing.assert_frame_equal(out, m.run_pipeline(clean).detail, check_dtype=False)
    for name in ("county_gqt_controls.csv", "state_gqt_check.csv", "difference_stats.csv",
                 "shortfall_groups.csv", "report.txt"):
        assert (tmp_path / "out" / name).exists()
