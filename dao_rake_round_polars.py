"""Rake-and-round disclosure avoidance (Polars version).

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
    python dao_rake_round_polars.py input.parquet --out-dir run1
    python dao_rake_round_polars.py input.sas7bdat --out-dir run1 --config config.toml

or from Python:

    import polars as pl
    from dao_rake_round_polars import Config, run_pipeline
    result = run_pipeline(pl.read_parquet("input.parquet"), Config())
    result.detail            # the rounded file

Everything is Polars.  numpy is used for two small things Polars has no
equivalent for: the seeded tiebreaker stream, and the PROC MEANS percentile
and mode definitions in the review statistics.

Missing values are Polars nulls.  NaN in the value column is treated as
missing and comes back as null.

Differences from SAS that affect results are listed below; everything else
follows the SAS code, including where the SAS comments say something
different.

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
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import polars as pl
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

AgeRules = Mapping[int, tuple[float | None, float | None]]


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

    age_universe: AgeRules = field(default_factory=lambda: dict(DEFAULT_AGE_UNIVERSE))

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
# Helpers
# --------------------------------------------------------------------------

# Names of working columns.  They never appear in a returned frame.
_CONTROL, _RAKE, _FLAG = "__control", "__rake", "__flag"
_ROW, _GROUP, _X, _ELIG, _TIE = "__row", "__group", "__x", "__eligible", "__tiebreaker"
_INT, _MANT, _TOTAL, _POOL, _STUFFED, _OUT = (
    "__integer", "__mantissa", "__total", "__pool", "__stuffed", "__out",
)


def _value(col: str) -> pl.Expr:
    """A value column as Float64, with NaN turned into null.

    Polars orders NaN above every number (NaN > 0 is true), so a NaN left in
    place would be coarsened and raked as if it were a large positive value.
    """
    return pl.col(col).cast(pl.Float64).fill_nan(None)


def sas_round(x: pl.Expr, unit: float = 1.0) -> pl.Expr:
    """Round to the nearest multiple of `unit`, halves away from zero.

    Mirrors SAS ROUND(x, unit).  Polars' Expr.round() offers other tie
    rules; this spells the SAS one out, including its "fuzz": a value that
    is only approximately halfway (floating-point noise below the half)
    still rounds up.  The tolerance, min(1e-6, 1e-12 * magnitude), is the
    one the SAS documentation uses to illustrate the behaviour; SAS does not
    publish the exact rule.
    """
    q = x.abs() / unit
    fuzz = (q * 1e-12).clip(upper_bound=1e-6)
    return x.sign() * (q + 0.5 + fuzz).floor() * unit


def coarsen(x: pl.Expr, base: float = 5.0, min_positive: float = 1.0) -> pl.Expr:
    """Round positive values to the nearest `base`; never let one round to zero.

    SAS:
        if blend_pop > 0 then do;
            blend_pop = round(blend_pop, 5);
            if blend_pop = 0 then blend_pop = 1;
        end;

    Zeros, negatives and nulls are returned unchanged.

        df.with_columns(coarsen(pl.col("blend_pop")))
    """
    rounded = sas_round(x, base)
    # A when/then/otherwise takes its output name from the first `then`, so
    # that branch is `rounded` (named after x) rather than the literal.
    never_zero = pl.when(rounded > 0).then(rounded).otherwise(min_positive)
    return (
        pl.when(x > 0)  # null > 0 is null, which `when` treats as false
        .then(never_zero)
        .otherwise(x)
    )


def _require_columns(df: pl.DataFrame, cols: Sequence[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{what}: missing column(s) {missing}")


def _require_complete_keys(df: pl.DataFrame, keys: Sequence[str], what: str) -> None:
    n_bad = df.select(pl.any_horizontal(pl.col(keys).is_null()).sum()).item()
    if n_bad:
        raise ValueError(
            f"{what}: {n_bad} row(s) have a missing value in a group key "
            f"{list(keys)}. SAS would silently drop these rows from every total."
        )


_PACKABLE_INTEGERS = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32)


def sort_by_keys(df: pl.DataFrame, keys: Sequence[str]) -> pl.DataFrame:
    """Stable sort by several key columns.

    Gives exactly what df.sort(keys, maintain_order=True) gives, several
    times faster on a large frame.  Polars sorts one integer column much
    faster than several columns, so the keys are packed into one integer
    whose order is the order of the key combination: each key becomes a
    digit (its offset from the column minimum for integers, its rank among
    the distinct values otherwise) of a mixed-radix number.

    Falls back to the plain sort if a key has nulls or the packed number
    would not fit in 64 bits.
    """
    keys = list(keys)
    if not keys or df.height < 2:
        return df
    if df.select(pl.any_horizontal(pl.col(keys).is_null()).any()).item():
        return df.sort(keys, maintain_order=True)

    work, digits, span = df, [], 1
    for k in keys:
        if df.schema[k] in _PACKABLE_INTEGERS:
            low, high = df[k].min(), df[k].max()
            digit, size = pl.col(k).cast(pl.Int64) - low, high - low + 1
        else:
            levels = df.select(k).unique().sort(k).with_row_index(f"__digit_{k}")
            work = work.join(levels, on=k, how="left", maintain_order="left")
            digit, size = pl.col(f"__digit_{k}"), levels.height
        digits.append((digit, size))
        span *= size
    if span >= 2**63:
        return df.sort(keys, maintain_order=True)

    packed = pl.lit(0, dtype=pl.UInt64)
    for digit, size in digits:
        packed = packed * size + digit.cast(pl.UInt64)
    order = work.select(packed.alias("__packed")).with_row_index("__row")
    order = order.sort("__packed", maintain_order=True)
    return df[order["__row"]]


# --------------------------------------------------------------------------
# %rakeit
# --------------------------------------------------------------------------


def rakeit(
    df: pl.DataFrame,
    value_col: str,
    group_cols: Sequence[str],
    control: pl.DataFrame | None = None,
    control_col: str | None = None,
    out_col: str | None = None,
    factor_col: str | None = None,
) -> pl.DataFrame:
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

    Returns `df` with `out_col` set; same rows, same order.
    """
    keys = list(group_cols)
    control = df if control is None else control
    control_col = control_col or value_col
    out_col = out_col or value_col
    _require_columns(df, [*keys, value_col], "rakeit input")
    _require_columns(control, [*keys, control_col], "rakeit control")
    _require_complete_keys(df, keys, "rakeit input")
    _require_complete_keys(control, keys, "rakeit control")

    x = _value(value_col)

    # proc summary data=&CDSN nway; class &CGRP; var &CVAR; output sum=control;
    ctrl = (
        control.group_by(keys)
        .agg(_value(control_col).sum().alias(_CONTROL))
        .cast({k: df.schema[k] for k in keys})  # so the join keys line up
    )
    # proc summary data=&UDSN nway; ... output sum=uncontrolled n=records;
    # (sum and count both skip missing values)
    unc = df.group_by(keys).agg(x.sum().alias("uncontrolled"), x.count().alias("records"))
    _warn_unmatched_controls(unc, keys, ctrl)

    # data rake; merge uncontrolled control; by &CGRP;   -- one row per group
    uncontrolled, records, c = pl.col("uncontrolled"), pl.col("records"), pl.col(_CONTROL)
    rake = (
        unc.join(ctrl, on=keys, how="left")
        .with_columns(c.fill_null(0.0))  # if control = . then control = 0
        .select(
            *keys,
            pl.when(uncontrolled > 0).then(c / uncontrolled)  # rake = control / uncontrolled
            .when(records > 0).then(c / records)              # else rake = dist
            .otherwise(1.0)
            .alias(_RAKE),
            (uncontrolled <= 0).alias(_FLAG),
        )
    )

    # data &ODSN; merge &UDSN rake; by &CGRP;
    #   if flag = 0 then &OVAR = &UVAR * rake; else &OVAR = rake;
    out = pl.when(pl.col(_FLAG)).then(pl.col(_RAKE)).otherwise(x * pl.col(_RAKE))
    new_cols = [out.alias(out_col)]
    if factor_col:
        new_cols.append(pl.col(_RAKE).alias(factor_col))
    return (
        df.join(rake, on=keys, how="left", maintain_order="left")
        .with_columns(new_cols)
        .drop(_RAKE, _FLAG)
    )


def _warn_unmatched_controls(groups: pl.DataFrame, keys: list[str], ctrl: pl.DataFrame) -> None:
    """Warn about control groups that have no rows to receive the control."""
    unmatched = ctrl.join(groups, on=keys, how="anti")
    if unmatched.height:
        log.warning(
            "rakeit: %d control group(s) by %s have no rows to rake; "
            "their control total of %.6g is not allocated.",
            unmatched.height, keys, unmatched[_CONTROL].sum(),
        )


# --------------------------------------------------------------------------
# %roundit
# --------------------------------------------------------------------------


def age_universe_expr(
    columns: Collection[str],
    age_col: str = "age",
    gqt_col: str = "gqt",
    rules: AgeRules | None = None,
) -> pl.Expr:
    """Boolean expression: true where a row may receive an added person.

    `columns` is the frame's column names.  The rules apply only when BOTH
    `age_col` and `gqt_col` are among them; otherwise every row is eligible
    (as in the SAS macro).

    A missing age counts as below every threshold, as it does in SAS, so it
    is out of universe for any GQ type that has a lower bound.
    """
    if age_col not in columns or gqt_col not in columns:
        return pl.lit(True)
    rules = DEFAULT_AGE_UNIVERSE if rules is None else rules
    age = _value(age_col)
    gqt = pl.col(gqt_col).cast(pl.Float64)
    out_of_universe = pl.lit(False)
    for code, (out_at_or_below, out_at_or_above) in rules.items():
        in_type = gqt == code
        if out_at_or_below is not None:  # SAS: missing <= anything is true
            out_of_universe |= in_type & (age.is_null() | (age <= out_at_or_below))
        if out_at_or_above is not None:  # SAS: missing >= anything is false
            out_of_universe |= in_type & (age >= out_at_or_above)
    # A comparison with a null gqt or age is null; SAS treats it as false.
    return ~out_of_universe.fill_null(False)


def age_universe_flag(
    df: pl.DataFrame,
    age_col: str = "age",
    gqt_col: str = "gqt",
    rules: AgeRules | None = None,
) -> pl.Series:
    """age_universe_expr() evaluated on `df`, as a Boolean Series."""
    expr = age_universe_expr(df.columns, age_col, gqt_col, rules)
    return df.select(expr.alias("age_uni")).to_series()


def roundit(
    df: pl.DataFrame,
    value_col: str,
    group_cols: Sequence[str],
    out_col: str | None = None,
    *,
    seed: int = 42,
    age_col: str = "age",
    gqt_col: str = "gqt",
    age_universe: AgeRules | None = None,
    tiebreaker_col: str | None = None,
    return_groups: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame]:
    """Greatest-mantissa rounding within groups.

    SAS parameters:  IDSN=df  IVAR=value_col  LIST=group_cols  OVAR=out_col

    Within each group:
      1. integer = floor(value), mantissa = value - integer
      2. stuffpool = ROUND(sum(value), 1) - sum(integer)
      3. order rows by mantissa desc, integer desc, random tiebreaker desc
      4. the first `stuffpool` ELIGIBLE rows in that order get integer + 1;
         every other row gets integer

    Eligibility comes from age_universe_expr(); ineligible rows are skipped
    without using up the pool.  If a group has fewer eligible rows than its
    pool, it ends short of its total (a warning is logged; SAS is silent).

    `tiebreaker_col` names a column of uniform draws to use instead of
    generating them, e.g. the tiebreaker exported from a SAS run.

    Returns `df` with `out_col` set; same rows, same order.  With
    return_groups=True also returns one row per group with its pool, number
    of eligible rows, number stuffed and shortfall.
    """
    keys = list(group_cols)
    out_col = out_col or value_col
    _require_columns(df, [*keys, value_col], "roundit input")
    _require_complete_keys(df, keys, "roundit input")
    if tiebreaker_col is not None:
        _require_columns(df, [tiebreaker_col], "roundit input")

    # One integer per group, numbered in sorted key order.  Grouping and
    # sorting on it is much cheaper than on the key columns themselves.
    group_ids = df.select(keys).unique().sort(keys).with_row_index(_GROUP)

    # A narrow working frame, so the sorts below do not carry the caller's
    # other columns along.
    cols = [_value(value_col).alias(_X),
            age_universe_expr(df.columns, age_col, gqt_col, age_universe).alias(_ELIG)]
    if tiebreaker_col is not None:
        cols.append(pl.col(tiebreaker_col).cast(pl.Float64).alias(_TIE))
    work = (
        df.select(*keys, *cols)
        .with_row_index(_ROW)
        .join(group_ids, on=keys, how="left", maintain_order="left")
        .drop(keys)
        # proc sort data = rounding; by &LIST;   (a stable sort)
        .sort(_GROUP, maintain_order=True)
    )

    # SAS draws the tiebreaker row by row in that sorted order, so draw i
    # belongs to the i-th row here.
    if tiebreaker_col is None:
        draws = np.random.Generator(np.random.PCG64(seed)).random(work.height)
        work = work.with_columns(pl.Series(_TIE, draws, dtype=pl.Float64))

    work = (
        work.with_columns(pl.col(_X).floor().alias(_INT))
        .with_columns(
            (pl.col(_X) - pl.col(_INT)).alias(_MANT),
            # proc summary ... sum=total integer;  stuffpool = round(total, 1) - integer
            pl.col(_X).sum().over(_GROUP).alias(_TOTAL),
        )
        .with_columns(
            (sas_round(pl.col(_TOTAL)) - pl.col(_INT).sum().over(_GROUP)).alias(_POOL)
        )
        # proc sort; by &LIST descending mantissa descending integer descending tiebreaker;
        # (a missing value sorts last, as it does in a SAS descending sort)
        .sort(
            [_GROUP, _MANT, _INT, _TIE],
            descending=[False, True, True, True],
            nulls_last=True,
            maintain_order=True,
        )
        # Eligible rows ahead of each row within its group.  This equals the
        # SAS `counter` for as long as the pool has not run out, so
        #   if age_uni = 1 and counter < stuffpool then OVAR = integer + 1;
        # becomes "eligible, with fewer than stuffpool eligible rows ahead".
        .with_columns(
            (
                pl.col(_ELIG)
                & (
                    pl.col(_ELIG).cast(pl.Int64).cum_sum().over(_GROUP)
                    - pl.col(_ELIG).cast(pl.Int64)
                    < pl.col(_POOL)
                )
            ).alias(_STUFFED)
        )
        .with_columns((pl.col(_INT) + pl.col(_STUFFED).cast(pl.Float64)).alias(_OUT))
    )

    groups = (
        work.group_by(_GROUP)
        .agg(
            pl.col(_TOTAL).first().alias("total"),
            pl.col(_INT).sum().alias("integer_sum"),
            pl.col(_POOL).first().alias("stuffpool"),
            pl.col(_ELIG).sum().cast(pl.Int64).alias("eligible_rows"),
            pl.col(_STUFFED).sum().cast(pl.Int64).alias("stuffed"),
        )
        .with_columns(
            (pl.col("stuffpool") - pl.col("stuffed")).clip(lower_bound=0).alias("shortfall")
        )
        .join(group_ids, on=_GROUP, how="left")
        .sort(_GROUP)
        .select(*keys, pl.exclude(_GROUP, *keys))
    )
    short = groups.filter(pl.col("shortfall") > 0)
    if short.height:
        log.warning(
            "roundit: %d group(s) by %s had fewer eligible rows than their stuffing "
            "pool; %d person(s) could not be placed and those groups end short.",
            short.height, keys, int(short["shortfall"].sum()),
        )

    # Back to the caller's row order, then attach the result to the caller's frame.
    rounded = work.select(_ROW, _OUT).sort(_ROW)[_OUT]
    result = df.with_columns(rounded.alias(out_col))
    return (result, groups) if return_groups else result


# --------------------------------------------------------------------------
# Main program (prog03_dao_roundmethod8.sas)
# --------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Outputs of run_pipeline()."""

    detail: pl.DataFrame            # final file (SAS: base_round6_cr)
    county_gqt: pl.DataFrame        # stage 1: original, coarsened, raked, control
    state_gqt_check: pl.DataFrame   # do the outputs hit the state x GQT controls?
    stage1_groups: pl.DataFrame     # roundit diagnostics, one row per state x GQT
    stage2_groups: pl.DataFrame     # roundit diagnostics, one row per county x GQT
    stats: pl.DataFrame             # PROC MEANS equivalents on the differences
    compare: dict                   # PROC COMPARE equivalent, county x GQT level


def run_pipeline(df: pl.DataFrame, cfg: Config | None = None) -> PipelineResult:
    """Run both stages and the review checks on a detail file."""
    cfg = cfg or Config()
    v = cfg.value_col
    detail_keys = list(cfg.detail_keys)
    state_keys = list(cfg.state_gqt_keys)
    county_keys = list(cfg.county_gqt_keys)

    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"input: expected a polars.DataFrame, got {type(df).__name__}. "
            "polars.from_pandas(df) converts a pandas frame."
        )
    _require_columns(df, [*detail_keys, v], "input")
    _require_complete_keys(df, detail_keys, "input")
    if not df.schema[v].is_numeric():
        raise TypeError(f"input: column {v!r} must be numeric")
    coarsened = coarsen(pl.col(v), cfg.rounding_base, cfg.min_positive)
    roundit_opts = {
        "seed": cfg.seed, "age_col": cfg.age_col, "gqt_col": cfg.gqt_col,
        "age_universe": cfg.age_universe, "return_groups": True,
    }

    # ---- Preparation -------------------------------------------------------
    # proc sort data=rounded_input; by nation state county age sex race31 hisp gqt;
    detail = sort_by_keys(df.select(*detail_keys, _value(v)), detail_keys)
    n_missing = detail[v].null_count()
    if n_missing and not cfg.allow_missing_values:
        raise ValueError(
            f"input: {n_missing} missing value(s) in {v!r}. Set "
            "allow_missing_values=True to carry them through as SAS does."
        )
    log.info("input: %d rows, total %s = %.4f", detail.height, v, detail[v].sum())

    # rounded_input_gqt: unrounded county x GQT totals
    county = detail.group_by(county_keys).agg(pl.col(v).sum()).sort(county_keys)
    # rounded_input_stategqt: unrounded state x GQT totals (summed from the county totals)
    state = county.group_by(state_keys).agg(pl.col(v).sum()).sort(state_keys)

    # ---- Stage 1: county x GQT controls -------------------------------------
    county_r = county.with_columns(coarsened)
    county_rc = rakeit(county_r, v, state_keys, control=state)
    # No age column here, so the age-universe rules are off.
    county_rcr, stage1_groups = roundit(county_rc, v, state_keys, **roundit_opts)
    log.info("stage 1: %d county x GQT controls", county_rcr.height)

    # ---- Stage 2: detail cells ----------------------------------------------
    detail_r = detail.with_columns(coarsened)
    detail_c = rakeit(detail_r, v, county_keys, control=county_rcr)
    detail_cr, stage2_groups = roundit(detail_c, v, county_keys, **roundit_opts)
    # Keep the nine columns.  Rows are still sorted by the eight keys.  The
    # values are whole numbers, so store them as integers (null if missing).
    final = detail_cr.select(*detail_keys, pl.col(v).cast(pl.Int64))
    log.info("stage 2: %d detail rows, total %s = %d", final.height, v, final[v].sum())

    # ---- Review checks ------------------------------------------------------
    county_out = county.select(*county_keys, pl.col(v).alias("original"))
    steps = {
        "coarsened": county_r,
        "raked": county_rc,
        "control": county_rcr,
        "final": final.group_by(county_keys).agg(pl.col(v).sum()),
    }
    for name, frame in steps.items():
        county_out = county_out.join(
            frame.select(*county_keys, pl.col(v).alias(name)),
            on=county_keys, how="left", maintain_order="left",
        )
    county_out = county_out.with_columns(
        (pl.col("final") - pl.col("original")).alias("diff")
    ).with_columns(pl.col("diff").abs().alias("absdiff"))

    state_check = (
        state.select(*state_keys, pl.col(v).alias("original"))
        .with_columns(sas_round(pl.col("original")).alias("target"))
        .join(
            county_out.group_by(state_keys).agg(
                pl.col("raked").sum().alias("raked_county_sum"),
                pl.col("control").sum().alias("control_sum"),
                pl.col("final").sum().alias("final_sum"),
            ),
            on=state_keys, how="left", maintain_order="left",
        )
        .with_columns((pl.col("final_sum") - pl.col("target")).alias("final_minus_target"))
    )

    # `final` and `detail` hold the same cells in the same order.
    detail_diff = (final[v] - detail[v]).to_numpy()
    stats = pl.concat([
        _difference_stats("county_gqt", county_out["diff"].to_numpy(),
                          county_out[cfg.gqt_col].cast(pl.Float64).to_numpy()),
        _difference_stats("detail", detail_diff,
                          final[cfg.gqt_col].cast(pl.Float64).to_numpy()),
    ])

    compare = {
        "level": "county_gqt",
        "rows_compared": county_out.height,
        "rows_unequal": int((county_out["diff"] != 0).sum()),
        "max_abs_diff": float(county_out["absdiff"].max() or 0.0),
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


_STAT_NAMES = ["sum", "n", "mean", "std", "min", "max", "mode", "median", "p5", "p95"]


def _describe(x: np.ndarray) -> dict:
    """PROC MEANS statistics: sum N mean std min max mode median p5 p95.

    Done in numpy because two of them have no Polars built-in: the SAS
    default percentile definition and the PROC MEANS mode.
    """
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {**dict.fromkeys(_STAT_NAMES, float("nan")), "n": 0}
    values, counts = np.unique(x, return_counts=True)
    # Lowest of the most frequent values; missing when no value repeats.
    # (Assumed to match PROC MEANS; its documentation does not spell this out.)
    mode = values[counts.argmax()] if counts.max() > 1 else float("nan")
    # SAS default percentile definition (QNTLDEF=5): empirical distribution
    # function with averaging = numpy's "averaged_inverted_cdf".
    p5, p95 = np.percentile(x, [5, 95], method="averaged_inverted_cdf")
    stats = {
        "sum": x.sum(),
        "n": int(x.size),
        "mean": x.mean(),
        "std": x.std(ddof=1) if x.size > 1 else float("nan"),
        "min": x.min(),
        "max": x.max(),
        "mode": mode,
        "median": np.median(x),
        "p5": p5,
        "p95": p95,
    }
    return {k: (val if k == "n" else float(val)) for k, val in stats.items()}


def _difference_stats(level: str, diff: np.ndarray, gqt: np.ndarray) -> pl.DataFrame:
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
    schema = {"level": pl.String, "universe": pl.String, "variable": pl.String,
              **{name: pl.Int64 if name == "n" else pl.Float64 for name in _STAT_NAMES}}
    return pl.DataFrame(rows, schema=schema)


# --------------------------------------------------------------------------
# File I/O and command line
# --------------------------------------------------------------------------


def read_table(path: str | Path) -> pl.DataFrame:
    """Read .parquet, .csv or .sas7bdat by file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pl.read_parquet(path)
    if suffix == ".csv":
        # Infer types from the whole file: the default looks at the first 100
        # rows only, and would call a column of 0, 0, ..., 1.5 an integer.
        return pl.read_csv(path, infer_schema_length=None)
    if suffix == ".sas7bdat":
        try:
            import pyreadstat
        except ImportError as exc:
            raise ImportError(
                "Reading .sas7bdat needs pyreadstat (pip install pyreadstat)"
            ) from exc
        df, _meta = pyreadstat.read_sas7bdat(str(path), output_format="polars")
        # SAS names are case-insensitive; the config uses lower case.
        return df.rename(str.lower)
    raise ValueError(f"Unsupported input type: {path.suffix!r} (use .parquet, .csv or .sas7bdat)")


def write_table(df: pl.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.write_parquet(path)
    else:
        df.write_csv(path)


def write_outputs(result: PipelineResult, out_dir: Path, fmt: str = "parquet") -> None:
    """Write the rounded file, the stage 1 controls and the review tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(result.detail, out_dir / f"rounded_detail.{fmt}")
    write_table(result.county_gqt, out_dir / f"county_gqt_controls.{fmt}")
    result.state_gqt_check.write_csv(out_dir / "state_gqt_check.csv")
    result.stats.write_csv(out_dir / "difference_stats.csv")
    # The two stages have different group keys, hence the diagonal concat.
    short = pl.concat(
        [
            result.stage1_groups.with_columns(pl.lit(1).alias("stage")),
            result.stage2_groups.with_columns(pl.lit(2).alias("stage")),
        ],
        how="diagonal",
    )
    short.filter(pl.col("shortfall") > 0).write_csv(out_dir / "shortfall_groups.csv")


def summarize(result: PipelineResult) -> str:
    """Short text summary of the review checks."""
    sc = result.state_gqt_check
    off = int((sc["final_minus_target"] != 0).sum())
    with pl.Config(
        tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4,
        tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True,
        tbl_formatting="ASCII_MARKDOWN",
    ):
        stats_table = str(result.stats)
    lines = [
        f"Detail rows: {result.detail.height:,}",
        f"County x GQT groups: {result.county_gqt.height:,} "
        f"({result.compare['rows_unequal']:,} differ from the original; "
        f"max abs diff {result.compare['max_abs_diff']:.4f})",
        f"State x GQT groups: {sc.height:,} ({off} do not match their rounded original total)",
        f"Stage 1 shortfall: {int(result.stage1_groups['shortfall'].sum())} person(s)",
        f"Stage 2 shortfall: {int(result.stage2_groups['shortfall'].sum())} person(s)",
        "",
        stats_table,
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
