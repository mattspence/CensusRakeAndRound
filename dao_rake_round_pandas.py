"""Rake-and-round disclosure avoidance.

Python port of three SAS files:

    prog03_dao_roundmethod8.sas   main program        -> run_pipeline(), main()
    mod01_rakeit.sas              %rakeit             -> rakeit()
    mod02_roundit_revised2.sas    %roundit            -> roundit()

The procedure coarsens a population file (round every positive cell to the
nearest 5, never to zero), then rakes and integerizes it in two stages so the
output is whole numbers that sum to the rounded state x GQ-type totals of the
original file.

    Stage 1  county x GQT totals: coarsen -> rake to unrounded state x GQT
             totals -> integerize.  These become the stage 2 controls.
    Stage 2  detail cells: coarsen -> rake to the stage 1 controls ->
             integerize, with added persons kept inside each GQ type's age
             universe.

Usage
-----
    python dao_rake_round.py input.parquet --out-dir run1
    python dao_rake_round.py input.sas7bdat --out-dir run1 --config config.toml

or from Python:

    import pandas as pd
    from dao_rake_round import Config, run_pipeline
    result = run_pipeline(pd.read_parquet("input.parquet"), Config())
    result.detail            # the rounded file

Differences from SAS that affect results are listed under "Deviations from
SAS" at the bottom of this docstring; everything else follows the SAS code,
including where the SAS comments say something different.

Deviations from SAS
-------------------
1. Tiebreaker stream.  SAS uses RAND('UNIFORM') after CALL STREAMINIT(42).
   This port uses numpy's PCG64 seeded with 42.  The streams differ, so rows
   that tie on mantissa and integer part can receive the added 1 in a
   different order.  Group totals are unaffected.  For an exact comparison,
   export SAS's tiebreaker column and pass it to roundit(tiebreaker_col=...).
2. Missing keys.  SAS drops rows with a missing group key from every total
   and gives them a missing result.  This port raises ValueError.
3. Missing values.  SAS carries them through (and turns an all-missing rake
   group into 1s).  run_pipeline() raises unless Config.allow_missing_values
   is True, in which case it follows SAS.  rakeit() and roundit() always
   follow SAS.
4. Control groups with no rows to rake.  SAS emits an extra output row with
   the group keys, value 1 and every other column missing.  This port emits
   no such row and logs a warning.
5. Row order and helper columns.  rakeit() and roundit() return rows in the
   caller's order and do not add stuffpool / counter / age_uni columns.
   run_pipeline() sorts its outputs the way the SAS program does.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import tomllib

log = logging.getLogger("dao_rake_round")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Age-universe rules from mod02_roundit_revised2.sas.
# gqt code -> (out of universe when age <= this, out of universe when age >= this)
# None means no bound on that side.  GQ types not listed have no restriction.
DEFAULT_AGE_UNIVERSE: dict[int, tuple[float | None, float | None]] = {
    1: (14, None),  # eligible ages 15+
    2: (None, 26),  # eligible ages 0-25
    3: (19, None),  # eligible ages 20+
    5: (15, 66),    # eligible ages 16-65
    6: (16, 65),    # eligible ages 17-64 (military; changed from 66 on 7/23/2026)
}


@dataclass(frozen=True)
class Config:
    """Everything the SAS program hard-codes, in one place."""

    value_col: str = "blend_pop"
    detail_keys: tuple[str, ...] = (
        "nation", "state", "county", "age", "sex", "race31", "hisp", "gqt",
    )
    state_gqt_keys: tuple[str, ...] = ("nation", "state", "gqt")
    county_gqt_keys: tuple[str, ...] = ("nation", "state", "county", "gqt")
    age_col: str = "age"
    gqt_col: str = "gqt"

    # Coarsening: ROUND(x, rounding_base); positives that round to 0 become min_positive.
    rounding_base: float = 5.0
    min_positive: float = 1.0

    # Seed for the integerization tiebreaker (SAS: CALL STREAMINIT(42)).
    seed: int = 42

    age_universe: Mapping[int, tuple[float | None, float | None]] = field(
        default_factory=lambda: dict(DEFAULT_AGE_UNIVERSE)
    )

    # False: raise if the value column has missing values.  True: follow SAS.
    allow_missing_values: bool = False

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load a Config from a TOML file.  Unknown keys are an error."""
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        for name in ("detail_keys", "state_gqt_keys", "county_gqt_keys"):
            if name in raw:
                raw[name] = tuple(raw[name])
        if "age_universe" in raw:
            # TOML keys are strings; each value is a table such as
            # { out_at_or_below = 15, out_at_or_above = 66 }.
            raw["age_universe"] = {
                int(gqt): (rule.get("out_at_or_below"), rule.get("out_at_or_above"))
                for gqt, rule in raw["age_universe"].items()
            }
        return cls(**raw)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def sas_round(x, unit: float = 1.0) -> np.ndarray:
    """Round to the nearest multiple of `unit`, halves away from zero.

    Mirrors SAS ROUND(x, unit).  Python's round() and numpy.round() round
    halves to even, which is not what the SAS code does.

    SAS ROUND also "fuzzes": a value that is only approximately halfway
    (floating-point noise below the half) still rounds up.  The tolerance
    here, min(1e-6, 1e-12 * magnitude), is the one the SAS documentation
    uses to illustrate the behaviour; SAS does not publish the exact rule.
    """
    a = np.asarray(x, dtype="float64")
    q = np.abs(a) / unit
    with np.errstate(invalid="ignore"):
        r = np.floor(q + 0.5 + np.minimum(1e-6, q * 1e-12))
    return np.sign(a) * r * unit


def coarsen(values, base: float = 5.0, min_positive: float = 1.0) -> np.ndarray:
    """Round positive values to the nearest `base`; never let one round to zero.

    SAS:
        if blend_pop > 0 then do;
            blend_pop = round(blend_pop, 5);
            if blend_pop = 0 then blend_pop = 1;
        end;

    Zeros, negatives and missing values are returned unchanged.
    """
    v = np.asarray(values, dtype="float64")
    out = v.copy()
    positive = v > 0  # False for NaN, as in SAS (missing is not > 0)
    rounded = sas_round(v[positive], base)
    rounded[rounded == 0] = min_positive
    out[positive] = rounded
    return out


def _require_columns(df: pd.DataFrame, cols: Sequence[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{what}: missing column(s) {missing}")


def _require_complete_keys(df: pd.DataFrame, keys: Sequence[str], what: str) -> None:
    bad = df[list(keys)].isna().any(axis=1)
    if bad.any():
        raise ValueError(
            f"{what}: {int(bad.sum())} row(s) have a missing value in a group key "
            f"{list(keys)}. SAS would silently drop these rows from every total."
        )


def _group_ids(df: pd.DataFrame, keys: Sequence[str]) -> tuple[np.ndarray, int]:
    """Integer id per row for its key combination, numbered in sorted key order."""
    if len(df) == 0:
        return np.zeros(0, dtype=np.int64), 0
    gid = df.groupby(list(keys), sort=True, observed=True).ngroup().to_numpy()
    return gid, int(gid.max()) + 1


def _group_keys(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """One row per key combination, in the same order as _group_ids numbers them."""
    return (
        df.groupby(list(keys), sort=True, observed=True)
        .size()
        .index.to_frame(index=False)
    )


# --------------------------------------------------------------------------
# %rakeit
# --------------------------------------------------------------------------


def rakeit(
    df: pd.DataFrame,
    value_col: str,
    group_cols: Sequence[str],
    control: pd.DataFrame | None = None,
    control_col: str | None = None,
    out_col: str | None = None,
    factor_col: str | None = None,
) -> pd.DataFrame:
    """One-way rake: scale `value_col` within each group to a control total.

    SAS parameters:  UDSN=df  UVAR=value_col  CGRP=group_cols  CDSN=control
                     CVAR=control_col  OVAR=out_col
    `factor_col`, if given, receives the rake factor (SAS: rake_<CDSN>_<ITER>).

    For each group g with control total C, uncontrolled total U and n
    non-missing values:

        U > 0            out = value * C / U
        U <= 0, n > 0    out = C / n        (control spread evenly)
        U <= 0, n == 0   out = 1

    A group with no rows in `control` has C = 0.

    Returns a copy of `df`, same rows and order, with `out_col` set.
    """
    keys = list(group_cols)
    control = df if control is None else control
    control_col = control_col or value_col
    out_col = out_col or value_col
    _require_columns(df, [*keys, value_col], "rakeit input")
    _require_columns(control, [*keys, control_col], "rakeit control")
    _require_complete_keys(df, keys, "rakeit input")
    _require_complete_keys(control, keys, "rakeit control")

    x = df[value_col].to_numpy(dtype="float64")
    gid, n_groups = _group_ids(df, keys)

    # proc summary ... sum=uncontrolled n=records   (both skip missing values)
    present = ~np.isnan(x)
    unc_g = np.bincount(gid[present], weights=x[present], minlength=n_groups)
    n_g = np.bincount(gid[present], minlength=n_groups)

    # proc summary data=&CDSN ... sum=control ; a missing control becomes 0
    ctrl = control.groupby(keys, sort=True, observed=True)[control_col].sum()
    ctrl_row = (
        df[keys].join(ctrl.rename("_control"), on=keys)["_control"]
        .fillna(0.0)
        .to_numpy(dtype="float64")
    )

    unc = unc_g[gid]
    n = n_g[gid]
    scaled = unc > 0  # SAS flag = 0
    dist = np.where(n > 0, ctrl_row / np.where(n > 0, n, 1), 1.0)
    factor = np.where(scaled, ctrl_row / np.where(scaled, unc, 1.0), dist)
    out = np.where(scaled, x * factor, factor)

    _warn_unmatched_controls(df, keys, ctrl)

    new_cols = {out_col: out}
    if factor_col:
        new_cols[factor_col] = factor
    return df.assign(**new_cols)


def _warn_unmatched_controls(df: pd.DataFrame, keys: list[str], ctrl: pd.Series) -> None:
    """Warn about control groups that have no rows to receive the control."""
    if len(ctrl) == 0:
        return
    ctrl_keys = ctrl.index.to_frame(index=False)
    seen = df[keys].drop_duplicates()
    merged = ctrl_keys.merge(seen, on=keys, how="left", indicator=True)
    unmatched = (merged["_merge"] == "left_only").to_numpy()
    if unmatched.any():
        log.warning(
            "rakeit: %d control group(s) by %s have no rows to rake; "
            "their control total of %.6g is not allocated.",
            int(unmatched.sum()), keys, float(ctrl.to_numpy()[unmatched].sum()),
        )


# --------------------------------------------------------------------------
# %roundit
# --------------------------------------------------------------------------


def age_universe_flag(
    df: pd.DataFrame,
    age_col: str = "age",
    gqt_col: str = "gqt",
    rules: Mapping[int, tuple[float | None, float | None]] | None = None,
) -> np.ndarray:
    """Boolean array: True where a row may receive an added person.

    The rules apply only when BOTH `age_col` and `gqt_col` are columns of
    `df`; otherwise every row is eligible (as in the SAS macro).

    A missing age counts as below every threshold, as it does in SAS, so it
    is out of universe for any GQ type that has a lower bound.
    """
    eligible = np.ones(len(df), dtype=bool)
    if age_col not in df.columns or gqt_col not in df.columns:
        return eligible
    rules = DEFAULT_AGE_UNIVERSE if rules is None else rules
    age = pd.to_numeric(df[age_col]).to_numpy(dtype="float64")
    gqt = pd.to_numeric(df[gqt_col]).to_numpy(dtype="float64")
    age_low = np.where(np.isnan(age), -np.inf, age)  # SAS: missing <= anything
    for code, (out_at_or_below, out_at_or_above) in rules.items():
        in_type = gqt == code
        if out_at_or_below is not None:
            eligible &= ~(in_type & (age_low <= out_at_or_below))
        if out_at_or_above is not None:
            with np.errstate(invalid="ignore"):
                eligible &= ~(in_type & (age >= out_at_or_above))
    return eligible


def roundit(
    df: pd.DataFrame,
    value_col: str,
    group_cols: Sequence[str],
    out_col: str | None = None,
    *,
    seed: int = 42,
    age_col: str = "age",
    gqt_col: str = "gqt",
    age_universe: Mapping[int, tuple[float | None, float | None]] | None = None,
    tiebreaker_col: str | None = None,
    return_groups: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Greatest-mantissa rounding within groups.

    SAS parameters:  IDSN=df  IVAR=value_col  LIST=group_cols  OVAR=out_col

    Within each group:
      1. integer = floor(value), mantissa = value - integer
      2. stuffpool = ROUND(sum(value), 1) - sum(integer)
      3. order rows by mantissa desc, integer desc, random tiebreaker desc
      4. the first `stuffpool` ELIGIBLE rows in that order get integer + 1;
         every other row gets integer

    Eligibility comes from age_universe_flag(); ineligible rows are skipped
    without using up the pool.  If a group has fewer eligible rows than its
    pool, it ends short of its total (a warning is logged; SAS is silent).

    `tiebreaker_col` names a column of uniform draws to use instead of
    generating them, e.g. the tiebreaker exported from a SAS run.

    Returns a copy of `df`, same rows and order, with `out_col` set.  With
    return_groups=True also returns one row per group with its pool, number
    of eligible rows, number stuffed and shortfall.
    """
    keys = list(group_cols)
    out_col = out_col or value_col
    _require_columns(df, [*keys, value_col], "roundit input")
    _require_complete_keys(df, keys, "roundit input")

    x = df[value_col].to_numpy(dtype="float64")
    n_rows = len(df)
    gid, n_groups = _group_ids(df, keys)
    eligible = age_universe_flag(df, age_col, gqt_col, age_universe)

    integer = np.floor(x)
    mantissa = x - integer

    # proc summary ... sum=total integer ; missing sums become 0
    total_g = np.bincount(gid, weights=np.nan_to_num(x), minlength=n_groups)
    int_g = np.bincount(gid, weights=np.nan_to_num(integer), minlength=n_groups)
    pool_g = sas_round(total_g, 1.0) - int_g

    # SAS draws the tiebreaker row by row after sorting by LIST (a stable
    # sort), so draw i belongs to the i-th row in that order.
    if tiebreaker_col is not None:
        _require_columns(df, [tiebreaker_col], "roundit input")
        tie = df[tiebreaker_col].to_numpy(dtype="float64")
    else:
        rng = np.random.Generator(np.random.PCG64(seed))
        tie = np.empty(n_rows, dtype="float64")
        tie[np.argsort(gid, kind="stable")] = rng.random(n_rows)

    # proc sort; by &LIST descending mantissa descending integer descending tiebreaker;
    # (lexsort: last key is primary; NaN sorts last, as missing does in a
    # SAS descending sort)
    order = np.lexsort((-tie, -integer, -mantissa, gid))
    g_s = gid[order]
    elig_s = eligible[order]

    # Eligible rows ahead of each row within its group.  This equals the SAS
    # `counter` for as long as the pool has not run out.
    cum = np.cumsum(elig_s)
    starts = np.flatnonzero(np.r_[True, g_s[1:] != g_s[:-1]]) if n_rows else np.zeros(0, int)
    before = np.zeros(n_groups, dtype=np.int64)
    before[g_s[starts]] = cum[starts] - elig_s[starts]
    ahead = cum - before[g_s] - elig_s

    # if age_uni = 1 and counter < stuffpool then do; counter + 1; OVAR = integer + 1; end;
    with np.errstate(invalid="ignore"):
        stuff_s = elig_s & (ahead < pool_g[g_s])
    stuffed = np.zeros(n_rows, dtype=bool)
    stuffed[order] = stuff_s

    out = integer + stuffed

    stuffed_g = np.bincount(gid[stuffed], minlength=n_groups)
    shortfall_g = np.maximum(np.nan_to_num(pool_g) - stuffed_g, 0)
    if (shortfall_g > 0).any():
        log.warning(
            "roundit: %d group(s) by %s had fewer eligible rows than their stuffing "
            "pool; %d person(s) could not be placed and those groups end short.",
            int((shortfall_g > 0).sum()), keys, int(shortfall_g.sum()),
        )

    result = df.assign(**{out_col: out})
    if not return_groups:
        return result

    groups = _group_keys(df, keys)
    groups["total"] = total_g
    groups["integer_sum"] = int_g
    groups["stuffpool"] = pool_g
    groups["eligible_rows"] = np.bincount(gid[eligible], minlength=n_groups)
    groups["stuffed"] = stuffed_g
    groups["shortfall"] = shortfall_g
    return result, groups


# --------------------------------------------------------------------------
# Main program (prog03_dao_roundmethod8.sas)
# --------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Outputs of run_pipeline()."""

    detail: pd.DataFrame            # final file (SAS: base_round6_cr)
    county_gqt: pd.DataFrame        # stage 1: original, coarsened, raked, control
    state_gqt_check: pd.DataFrame   # do the outputs hit the state x GQT controls?
    stage1_groups: pd.DataFrame     # roundit diagnostics, one row per state x GQT
    stage2_groups: pd.DataFrame     # roundit diagnostics, one row per county x GQT
    stats: pd.DataFrame             # PROC MEANS equivalents on the differences
    compare: dict                   # PROC COMPARE equivalent, county x GQT level


def run_pipeline(df: pd.DataFrame, cfg: Config | None = None) -> PipelineResult:
    """Run both stages and the review checks on a detail file."""
    cfg = cfg or Config()
    v = cfg.value_col
    detail_keys = list(cfg.detail_keys)
    state_keys = list(cfg.state_gqt_keys)
    county_keys = list(cfg.county_gqt_keys)

    _require_columns(df, [*detail_keys, v], "input")
    _require_complete_keys(df, detail_keys, "input")
    if not pd.api.types.is_numeric_dtype(df[v]):
        raise TypeError(f"input: column {v!r} must be numeric")
    n_missing = int(df[v].isna().sum())
    if n_missing and not cfg.allow_missing_values:
        raise ValueError(
            f"input: {n_missing} missing value(s) in {v!r}. Set "
            "allow_missing_values=True to carry them through as SAS does."
        )
    if cfg.gqt_col in df.columns:
        pd.to_numeric(df[cfg.gqt_col])  # raises if GQ type is not numeric

    # ---- Preparation -------------------------------------------------------
    # proc sort data=rounded_input; by nation state county age sex race31 hisp gqt;
    detail = (
        df[[*detail_keys, v]]
        .sort_values(detail_keys, kind="stable")
        .reset_index(drop=True)
    )
    log.info("input: %d rows, total %s = %.4f", len(detail), v, detail[v].sum())

    # rounded_input_gqt: unrounded county x GQT totals
    county = detail.groupby(county_keys, sort=True, observed=True)[v].sum().reset_index()
    # rounded_input_stategqt: unrounded state x GQT totals (summed from the county totals)
    state = county.groupby(state_keys, sort=True, observed=True)[v].sum().reset_index()

    # ---- Stage 1: county x GQT controls -------------------------------------
    county_r = county.assign(**{v: coarsen(county[v], cfg.rounding_base, cfg.min_positive)})
    county_rc = rakeit(county_r, v, state_keys, control=state)
    # No age column here, so the age-universe rules are off.
    county_rcr, stage1_groups = roundit(
        county_rc, v, state_keys, seed=cfg.seed, age_col=cfg.age_col,
        gqt_col=cfg.gqt_col, age_universe=cfg.age_universe, return_groups=True,
    )
    county_rcr = county_rcr.sort_values(county_keys, kind="stable").reset_index(drop=True)
    log.info("stage 1: %d county x GQT controls", len(county_rcr))

    # ---- Stage 2: detail cells ----------------------------------------------
    detail_r = detail.assign(**{v: coarsen(detail[v], cfg.rounding_base, cfg.min_positive)})
    detail_c = rakeit(detail_r, v, county_keys, control=county_rcr)
    detail_cr, stage2_groups = roundit(
        detail_c, v, county_keys, seed=cfg.seed, age_col=cfg.age_col,
        gqt_col=cfg.gqt_col, age_universe=cfg.age_universe, return_groups=True,
    )
    # keep the nine columns; proc sort by the eight keys (already in that order)
    final = detail_cr[[*detail_keys, v]]
    if not final[v].isna().any():
        final = final.astype({v: "int64"})
    log.info("stage 2: %d detail rows, total %s = %d", len(final), v, final[v].sum())

    # ---- Review checks ------------------------------------------------------
    county_out = pd.DataFrame({
        **{k: county[k] for k in county_keys},
        "original": county[v],
        "coarsened": county_r[v],
        "raked": county_rc.sort_values(county_keys, kind="stable")[v].to_numpy(),
        "control": county_rcr[v],
        "final": final.groupby(county_keys, sort=True, observed=True)[v].sum().to_numpy(),
    })
    county_out["diff"] = county_out["final"] - county_out["original"]
    county_out["absdiff"] = county_out["diff"].abs()

    state_check = state.rename(columns={v: "original"})
    state_check["target"] = sas_round(state_check["original"], 1.0)
    by_state = county_out.groupby(state_keys, sort=True, observed=True)
    state_check["raked_county_sum"] = by_state["raked"].sum().to_numpy()
    state_check["control_sum"] = by_state["control"].sum().to_numpy()
    state_check["final_sum"] = by_state["final"].sum().to_numpy()
    state_check["final_minus_target"] = state_check["final_sum"] - state_check["target"]

    detail_diff = final[v].to_numpy(dtype="float64") - detail[v].to_numpy(dtype="float64")
    stats = pd.concat(
        [
            _difference_stats("county_gqt", county_out["diff"].to_numpy(),
                              pd.to_numeric(county_out[cfg.gqt_col]).to_numpy()),
            _difference_stats("detail", detail_diff,
                              pd.to_numeric(final[cfg.gqt_col]).to_numpy()),
        ],
        ignore_index=True,
    )

    unequal = county_out["diff"].to_numpy() != 0
    compare = {
        "level": "county_gqt",
        "rows_compared": int(len(county_out)),
        "rows_unequal": int(np.nansum(unequal)),
        "max_abs_diff": float(np.nanmax(county_out["absdiff"])) if len(county_out) else 0.0,
    }

    return PipelineResult(
        detail=final,
        county_gqt=county_out,
        state_gqt_check=state_check,
        stage1_groups=stage1_groups,
        stage2_groups=stage2_groups,
        stats=stats,
        compare=compare,
    )


def _describe(x: np.ndarray) -> dict:
    """PROC MEANS statistics: sum N mean std min max mode median p5 p95."""
    x = x[~np.isnan(x)]
    names = ["sum", "n", "mean", "std", "min", "max", "mode", "median", "p5", "p95"]
    if x.size == 0:
        return {**dict.fromkeys(names, np.nan), "n": 0}
    values, counts = np.unique(x, return_counts=True)
    # Lowest of the most frequent values; missing when no value repeats.
    # (Assumed to match PROC MEANS; its documentation does not spell this out.)
    mode = values[counts.argmax()] if counts.max() > 1 else np.nan
    # SAS default percentile definition (QNTLDEF=5): empirical distribution
    # function with averaging = numpy's "averaged_inverted_cdf".
    p5, p95 = np.percentile(x, [5, 95], method="averaged_inverted_cdf")
    return {
        "sum": x.sum(),
        "n": int(x.size),
        "mean": x.mean(),
        "std": x.std(ddof=1) if x.size > 1 else np.nan,
        "min": x.min(),
        "max": x.max(),
        "mode": mode,
        "median": np.median(x),
        "p5": p5,
        "p95": p95,
    }


def _difference_stats(level: str, diff: np.ndarray, gqt: np.ndarray) -> pd.DataFrame:
    """Stats on diff and |diff| for all rows, households (gqt = 0) and GQ (gqt ne 0)."""
    universes = {
        "all": np.ones(len(diff), dtype=bool),
        "household": gqt == 0,
        "group_quarters": gqt != 0,
    }
    rows = []
    for universe, mask in universes.items():
        for variable, values in (("diff", diff[mask]), ("absdiff", np.abs(diff[mask]))):
            rows.append({"level": level, "universe": universe, "variable": variable,
                         **_describe(values)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# File I/O and command line
# --------------------------------------------------------------------------


def read_table(path: str | Path) -> pd.DataFrame:
    """Read .parquet, .csv or .sas7bdat by file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".sas7bdat":
        df = pd.read_sas(path, format="sas7bdat", encoding="latin-1")
        # SAS names are case-insensitive; the config uses lower case.
        return df.rename(columns=str.lower)
    raise ValueError(f"Unsupported input type: {path.suffix!r} (use .parquet, .csv or .sas7bdat)")


def write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def write_outputs(result: PipelineResult, out_dir: Path, fmt: str = "parquet") -> None:
    """Write the rounded file, the stage 1 controls and the review tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(result.detail, out_dir / f"rounded_detail.{fmt}")
    write_table(result.county_gqt, out_dir / f"county_gqt_controls.{fmt}")
    result.state_gqt_check.to_csv(out_dir / "state_gqt_check.csv", index=False)
    result.stats.to_csv(out_dir / "difference_stats.csv", index=False)
    short = pd.concat(
        [result.stage1_groups.assign(stage=1), result.stage2_groups.assign(stage=2)],
        ignore_index=True,
    )
    short[short["shortfall"] > 0].to_csv(out_dir / "shortfall_groups.csv", index=False)


def summarize(result: PipelineResult) -> str:
    """Short text summary of the review checks."""
    sc = result.state_gqt_check
    off = int((sc["final_minus_target"] != 0).sum())
    lines = [
        f"Detail rows: {len(result.detail):,}",
        f"County x GQT groups: {len(result.county_gqt):,} "
        f"({result.compare['rows_unequal']:,} differ from the original; "
        f"max abs diff {result.compare['max_abs_diff']:.4f})",
        f"State x GQT groups: {len(sc):,} ({off} do not match their rounded original total)",
        f"Stage 1 shortfall: {int(result.stage1_groups['shortfall'].sum())} person(s)",
        f"Stage 2 shortfall: {int(result.stage2_groups['shortfall'].sum())} person(s)",
        "",
        result.stats.to_string(index=False, float_format=lambda f: f"{f:.4f}"),
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rake-and-round disclosure avoidance (port of prog03_dao_roundmethod8.sas)."
    )
    p.add_argument("input", type=Path, help="detail file: .parquet, .csv or .sas7bdat")
    p.add_argument("--out-dir", type=Path, required=True, help="folder for outputs")
    p.add_argument("--config", type=Path, help="TOML file overriding Config defaults")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                   help="format of the two large outputs (default: parquet)")
    p.add_argument("--base", type=float, help="rounding base (overrides config; SAS uses 5)")
    p.add_argument("--seed", type=int, help="tiebreaker seed (overrides config; SAS uses 42)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = Config.from_toml(args.config) if args.config else Config()
    overrides = {}
    if args.base is not None:
        overrides["rounding_base"] = args.base
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        cfg = replace(cfg, **overrides)

    df = read_table(args.input)
    result = run_pipeline(df, cfg)
    write_outputs(result, args.out_dir, args.format)

    report = summarize(result)
    (args.out_dir / "report.txt").write_text(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
