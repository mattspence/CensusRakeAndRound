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

Extension beyond the SAS program: Config.control_levels turns the two
stages into a chain of any depth, for example

    state x GQT -> county -> tract -> block -> block x householder -> detail

Each level's totals are coarsened, raked to the integer controls of the level
above and integerized, exactly as stage 1 does for counties.  With a
relationship column configured, a derived 0/1 `householder` column can be
used as a level (which controls householders per block), and relationship
rules join the age rules in deciding which cells may receive an added
person.  With the default Config none of this is active and the result is
the SAS program's.

Usage
-----
    python dao_rake_round.py input.parquet --out-dir run1
    python dao_rake_round.py input.sas7bdat --out-dir run1 --config config.toml

or from Python:

    import polars as pl
    from dao_rake_round import Config, run_pipeline
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
class RelationshipRule:
    """Where persons with one of `codes` can exist.

    A row whose relationship is in `codes` may receive an added person only
    if its GQ type is in `gqt` (when given) and its age is within
    [min_age, max_age] (when given).  A missing age fails a `min_age` test,
    as a missing age does in the SAS age rules.
    """

    codes: tuple[int, ...]
    gqt: tuple[int, ...] | None = None
    min_age: float | None = None
    max_age: float | None = None


# 2020 Census relationship codes (RELSHIP): 20 householder, 21-24 spouses and
# unmarried partners, 25-36 other household members, 37 institutional GQ
# person, 38 noninstitutional GQ person.  GQ types 1-4 are taken to be
# institutional and 5-7 noninstitutional.
# ASSUMPTION TO CONFIRM: the minimum age of 15 for householders, spouses and
# partners.  Add further age rules (parents, in-laws, ...) in the config.
DEFAULT_RELATIONSHIP_UNIVERSE: tuple[RelationshipRule, ...] = (
    RelationshipRule(codes=(20, 21, 22, 23, 24), gqt=(0,), min_age=15),
    RelationshipRule(codes=tuple(range(25, 37)), gqt=(0,)),
    RelationshipRule(codes=(37,), gqt=(1, 2, 3, 4)),
    RelationshipRule(codes=(38,), gqt=(5, 6, 7)),
)


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

    # ---- Extension beyond the SAS program ----------------------------------
    # The chain of control levels, coarsest first; each entry is the key
    # columns of one level and must contain the keys of the one before it.
    # The first level is the unrounded control.  None means the SAS
    # program's two levels: (state_gqt_keys, county_gqt_keys).
    control_levels: tuple[tuple[str, ...], ...] | None = None

    # Relationship to householder.  None means there is no such column.
    relationship_col: str | None = None
    # Relationship codes that mean "householder".
    householder_codes: tuple[int, ...] = (20,)
    # Name of a derived 0/1 column (1 = householder) that control_levels may
    # use as a key.  It is not written to the output.
    householder_col: str = "householder"
    # Where each relationship can exist.  None means the 2020 Census defaults.
    relationship_universe: tuple[RelationshipRule, ...] | None = None

    @property
    def levels(self) -> tuple[tuple[str, ...], ...]:
        """The control chain in use."""
        return self.control_levels or (self.state_gqt_keys, self.county_gqt_keys)

    @property
    def relationship_rules(self) -> tuple[RelationshipRule, ...]:
        if self.relationship_universe is None:
            return DEFAULT_RELATIONSHIP_UNIVERSE
        return self.relationship_universe

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load a Config from a TOML file.  Unknown keys are an error."""
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        for name in ("detail_keys", "state_gqt_keys", "county_gqt_keys", "householder_codes"):
            if name in raw:
                raw[name] = tuple(raw[name])
        if "control_levels" in raw:
            raw["control_levels"] = tuple(tuple(level) for level in raw["control_levels"])
        if "age_universe" in raw:
            # TOML keys are strings; each value is a table such as
            # { out_at_or_below = 15, out_at_or_above = 66 }.
            raw["age_universe"] = {
                int(gqt): (rule.get("out_at_or_below"), rule.get("out_at_or_above"))
                for gqt, rule in raw["age_universe"].items()
            }
        if "relationship_universe" in raw:
            # An array of tables such as { codes = [20, 21], gqt = [0], min_age = 15 }.
            rules = []
            for rule in raw["relationship_universe"]:
                extra = set(rule) - {"codes", "gqt", "min_age", "max_age"}
                if extra or "codes" not in rule:
                    raise ValueError(f"Bad relationship_universe entry in {path}: {rule}")
                rules.append(RelationshipRule(
                    codes=tuple(rule["codes"]),
                    gqt=tuple(rule["gqt"]) if "gqt" in rule else None,
                    min_age=rule.get("min_age"),
                    max_age=rule.get("max_age"),
                ))
            raw["relationship_universe"] = tuple(rules)
        return cls(**raw)


def level_names(levels: Sequence[Sequence[str]]) -> list[str]:
    """A short name per control level, from the keys it adds.

    The first level is named by its keys after the first one, every other
    level by the keys after the prefix it shares with the level above:
    (nation, state, gqt), (nation, state, county, gqt) -> state_gqt, county_gqt.
    """
    names = []
    for i, keys in enumerate(levels):
        if i == 0:
            new = list(keys[1:]) or list(keys)
        else:
            shared = 0
            for a, b in zip(levels[i - 1], keys, strict=False):
                if a != b:
                    break
                shared += 1
            new = list(keys[shared:])
        names.append("_".join(new))
    return names


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
    digit of a mixed-radix number.  The digit is the key's offset from the
    column minimum for integers with a compact range, and its rank among the
    distinct values otherwise (text, floats, sparse codes such as tracts).

    Keys that do not fit together in 64 bits are packed into two or more
    integers, which are then sorted together.  Falls back to the plain sort
    only if a key has nulls.
    """
    keys = list(keys)
    if not keys or df.height < 2:
        return df
    if df.select(pl.any_horizontal(pl.col(keys).is_null()).any()).item():
        return df.sort(keys, maintain_order=True)

    work, digits = df, []
    for k in keys:
        low = high = None
        if df.schema[k] in _PACKABLE_INTEGERS:
            low, high = df[k].min(), df[k].max()
        if low is not None and high - low < df.height:
            digit, size = pl.col(k).cast(pl.Int64) - low, high - low + 1
        else:
            levels = df.select(k).unique().sort(k).with_row_index(f"__digit_{k}")
            work = work.join(levels, on=k, how="left", maintain_order="left")
            digit, size = pl.col(f"__digit_{k}"), levels.height
        digits.append((digit.cast(pl.UInt64), size))  # every size is <= df.height

    # Pack the digits, in key order, into as few 63-bit integers as possible.
    packed, current, span = [], None, 1
    for digit, size in digits:
        if current is not None and span * size < 2**63:
            current, span = current * size + digit, span * size
        else:
            if current is not None:
                packed.append(current)
            current, span = digit, size
    packed.append(current)

    names = [f"__packed_{i}" for i in range(len(packed))]
    order = work.select(p.alias(n) for p, n in zip(packed, names, strict=True))
    order = order.with_row_index("__row").sort(names, maintain_order=True)
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


def relationship_universe_expr(
    columns: Collection[str],
    relationship_col: str | None,
    rules: Sequence[RelationshipRule] | None = None,
    age_col: str = "age",
    gqt_col: str = "gqt",
) -> pl.Expr:
    """Boolean expression: true where the relationship rules allow an added person.

    Not part of the SAS macro.  Every row is eligible when the frame has no
    relationship column.  A rule's GQ-type test is skipped when the frame has
    no GQ-type column, and its age tests when it has no age column.  A
    relationship code that no rule mentions is unrestricted.
    """
    if relationship_col is None or relationship_col not in columns:
        return pl.lit(True)
    rules = DEFAULT_RELATIONSHIP_UNIVERSE if rules is None else rules
    relationship = pl.col(relationship_col).cast(pl.Float64)
    out_of_universe = pl.lit(False)
    for rule in rules:
        in_rule = relationship.is_in([float(code) for code in rule.codes])
        if rule.gqt is not None and gqt_col in columns:
            gqt = pl.col(gqt_col).cast(pl.Float64)
            out_of_universe |= in_rule & ~gqt.is_in([float(code) for code in rule.gqt])
        if age_col in columns:
            age = _value(age_col)
            if rule.min_age is not None:  # a missing age is below every minimum
                out_of_universe |= in_rule & (age.is_null() | (age < rule.min_age))
            if rule.max_age is not None:
                out_of_universe |= in_rule & (age > rule.max_age)
    return ~out_of_universe.fill_null(False)


def eligibility_expr(columns: Collection[str], cfg: Config) -> pl.Expr:
    """The age rules and the relationship rules together, for a frame's columns."""
    return age_universe_expr(columns, cfg.age_col, cfg.gqt_col, cfg.age_universe) & (
        relationship_universe_expr(
            columns, cfg.relationship_col, cfg.relationship_rules, cfg.age_col, cfg.gqt_col
        )
    )


def eligibility_expr_flag(df: pl.DataFrame, cfg: Config) -> pl.Series:
    """eligibility_expr() evaluated on `df`, as a Boolean Series."""
    return df.select(eligibility_expr(df.columns, cfg).alias("eligible")).to_series()


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
    eligible: pl.Expr | None = None,
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

    Eligibility comes from age_universe_expr(), or from `eligible` (a Boolean
    expression on `df`) when that is given.  Ineligible rows are skipped
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
    if eligible is None:
        eligible = age_universe_expr(df.columns, age_col, gqt_col, age_universe)
    cols = [_value(value_col).alias(_X), eligible.alias(_ELIG)]
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
    level_names: list[str]          # one name per control level, coarsest first
    controls: dict[str, pl.DataFrame]  # per level below the first: original,
    #                                    coarsened, raked, control, final, diff
    state_gqt_check: pl.DataFrame   # do the outputs hit the first level's totals?
    groups: dict[str, pl.DataFrame]  # roundit diagnostics per stage; last is "detail"
    level_summary: pl.DataFrame     # one line per level and one for the detail
    stats: pl.DataFrame             # PROC MEANS equivalents on the differences
    compare: dict                   # PROC COMPARE equivalent, second level
    householder_check: dict | None = None  # only with a householder level

    # The SAS program's two stages, by their old names.
    @property
    def county_gqt(self) -> pl.DataFrame:
        return self.controls[self.level_names[1]]

    @property
    def stage1_groups(self) -> pl.DataFrame:
        return self.groups[self.level_names[1]]

    @property
    def stage2_groups(self) -> pl.DataFrame:
        return self.groups["detail"]


def _check_config(cfg: Config, levels: list[list[str]], names: list[str]) -> None:
    """Fail early, with a clear message, on a control chain that cannot work."""
    if len(levels) < 2:
        raise ValueError("control_levels needs at least two levels")
    allowed = set(cfg.detail_keys) | {cfg.householder_col}
    for i, keys in enumerate(levels):
        unknown = [k for k in keys if k not in allowed]
        if unknown:
            raise ValueError(f"control level {names[i]}: {unknown} are not detail keys")
        if i and not set(levels[i - 1]) <= set(keys):
            raise ValueError(
                f"control level {names[i]} must contain every key of the level above it "
                f"({names[i - 1]}): {levels[i - 1]}"
            )
        if i and len(set(keys)) == len(set(levels[i - 1])):
            raise ValueError(f"control level {i + 1} must add at least one key to the level above")
    if len(set(names)) != len(names) or "detail" in names:
        raise ValueError(f"control_levels give ambiguous level names: {names}")
    uses_flag = any(cfg.householder_col in keys for keys in levels)
    if uses_flag and cfg.householder_col not in cfg.detail_keys and cfg.relationship_col is None:
        raise ValueError(
            f"control_levels use {cfg.householder_col!r}; set relationship_col so it can be derived"
        )
    if cfg.relationship_col is not None and cfg.relationship_col not in cfg.detail_keys:
        raise ValueError(f"relationship_col {cfg.relationship_col!r} must be one of detail_keys")


def run_pipeline(df: pl.DataFrame, cfg: Config | None = None) -> PipelineResult:
    """Run every stage of the control chain and the review checks on a detail file."""
    cfg = cfg or Config()
    v = cfg.value_col
    detail_keys = list(cfg.detail_keys)
    levels = [list(keys) for keys in cfg.levels]
    names = level_names(levels)

    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"input: expected a polars.DataFrame, got {type(df).__name__}. "
            "polars.from_pandas(df) converts a pandas frame."
        )
    _check_config(cfg, levels, names)
    _require_columns(df, [*detail_keys, v], "input")
    _require_complete_keys(df, detail_keys, "input")
    if not df.schema[v].is_numeric():
        raise TypeError(f"input: column {v!r} must be numeric")
    coarsened = coarsen(pl.col(v), cfg.rounding_base, cfg.min_positive)

    def integerize(frame: pl.DataFrame, group: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
        # The rules only bite where the frame has the columns they test, so
        # they are off for the totals tables, as in the SAS macro.
        return roundit(frame, v, group, seed=cfg.seed, return_groups=True,
                       eligible=eligibility_expr(frame.columns, cfg))

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

    flag = cfg.householder_col
    if flag not in detail_keys and any(flag in keys for keys in levels):
        codes = [float(code) for code in cfg.householder_codes]
        is_householder = pl.col(cfg.relationship_col).cast(pl.Float64).is_in(codes)
        detail = detail.with_columns(is_householder.cast(pl.Int8).alias(flag))

    # Unrounded totals at every level, each summed from the level below it
    # (SAS: rounded_input_gqt from the detail, rounded_input_stategqt from that).
    totals: list[pl.DataFrame] = [detail] * len(levels)
    source = detail
    for i in reversed(range(len(levels))):
        source = sort_by_keys(source.group_by(levels[i]).agg(pl.col(v).sum()), levels[i])
        totals[i] = source

    # ---- One stage per level below the first --------------------------------
    # Coarsen the level's totals, rake them to the controls of the level
    # above (within that level's groups), integerize.  The first level's
    # controls are its unrounded totals.  SAS stage 1 is this for counties.
    control = totals[0]
    steps: dict[str, dict[str, pl.DataFrame]] = {}
    groups: dict[str, pl.DataFrame] = {}
    for i in range(1, len(levels)):
        above = levels[i - 1]
        level_r = totals[i].with_columns(coarsened)
        level_rc = rakeit(level_r, v, above, control=control)
        level_rcr, groups[names[i]] = integerize(level_rc, above)
        steps[names[i]] = {"coarsened": level_r, "raked": level_rc, "control": level_rcr}
        control = level_rcr
        log.info("level %s: %d controls", names[i], level_rcr.height)

    # ---- Last stage: detail cells (SAS stage 2) -----------------------------
    detail_r = detail.with_columns(coarsened)
    detail_c = rakeit(detail_r, v, levels[-1], control=control)
    detail_cr, groups["detail"] = integerize(detail_c, levels[-1])
    # The values are whole numbers, so store them as integers (null if
    # missing).  Rows are still sorted by the detail keys.
    detail_cr = detail_cr.with_columns(pl.col(v).cast(pl.Int64))
    final = detail_cr.select(*detail_keys, v)
    log.info("detail: %d rows, total %s = %d", final.height, v, final[v].sum())

    # ---- Review checks ------------------------------------------------------
    # The final file summed to every level, each from the level below it.
    final_sums: list[pl.DataFrame] = [detail_cr] * len(levels)
    source = detail_cr
    for i in reversed(range(len(levels))):
        source = source.group_by(levels[i]).agg(pl.col(v).sum())
        final_sums[i] = source

    controls: dict[str, pl.DataFrame] = {}
    for i in range(1, len(levels)):
        table = totals[i].select(*levels[i], pl.col(v).alias("original"))
        for column, frame in {**steps[names[i]], "final": final_sums[i]}.items():
            table = table.join(
                frame.select(*levels[i], pl.col(v).alias(column)),
                on=levels[i], how="left", maintain_order="left",
            )
        controls[names[i]] = table.with_columns(
            (pl.col("final") - pl.col("original")).alias("diff")
        ).with_columns(pl.col("diff").abs().alias("absdiff"))

    first = controls[names[1]]
    top_check = (
        totals[0].select(*levels[0], pl.col(v).alias("original"))
        .with_columns(sas_round(pl.col("original")).alias("target"))
        .join(
            first.group_by(levels[0]).agg(
                pl.col("raked").sum().alias("raked_county_sum"),
                pl.col("control").sum().alias("control_sum"),
            ),
            on=levels[0], how="left", maintain_order="left",
        )
        .join(
            final_sums[0].select(*levels[0], pl.col(v).alias("final_sum")),
            on=levels[0], how="left", maintain_order="left",
        )
        .with_columns((pl.col("final_sum") - pl.col("target")).alias("final_minus_target"))
    )

    # `final` and `detail` hold the same cells in the same order.
    detail_table = final.select(
        *([cfg.gqt_col] if cfg.gqt_col in detail_keys else []),
        detail[v].alias("original"), pl.col(v).alias("final"),
    ).with_columns((pl.col("final") - pl.col("original")).alias("diff"))
    tables = {**controls, "detail": detail_table}
    stats = pl.concat(
        _difference_stats(name, table["diff"].to_numpy(), _gqt_values(table, cfg.gqt_col))
        for name, table in tables.items()
    )
    level_summary = pl.DataFrame([
        {
            "level": name,
            "rows": table.height,
            "differ": int((table["diff"] != 0).sum()),
            "max_abs_diff": float(table["diff"].abs().max() or 0.0),
            "positive_to_zero": int(((table["original"] > 0) & (table["final"] == 0)).sum()),
            "shortfall": int(groups[name]["shortfall"].sum()),
        }
        for name, table in tables.items()
    ])

    compare = {
        "level": names[1],
        "rows_compared": first.height,
        "rows_unequal": int((first["diff"] != 0).sum()),
        "max_abs_diff": float(first["absdiff"].max() or 0.0),
    }

    return PipelineResult(
        detail=final,
        level_names=names,
        controls=controls,
        state_gqt_check=top_check,
        groups=groups,
        level_summary=level_summary,
        stats=stats,
        compare=compare,
        householder_check=_householder_check(detail, detail_cr, cfg, levels),
    )


def _gqt_values(table: pl.DataFrame, gqt_col: str) -> np.ndarray | None:
    if gqt_col not in table.columns:
        return None
    return table[gqt_col].cast(pl.Float64).to_numpy()


def _householder_check(
    before: pl.DataFrame, after: pl.DataFrame, cfg: Config, levels: list[list[str]]
) -> dict | None:
    """Count places left with a household population but no householder.

    A "place" is a group of the level just above the householder level
    (a block, in the block-level chain), counted over household population
    only (GQ type 0).  The method can produce such places: it controls how
    many householders a block has, not which households they head.
    """
    flag, v = cfg.householder_col, cfg.value_col
    with_flag = [keys for keys in levels if flag in keys]
    if not with_flag:
        return None
    place = [k for k in with_flag[0] if k != flag]

    def count(frame: pl.DataFrame) -> tuple[int, int, int]:
        if cfg.gqt_col in frame.columns:
            frame = frame.filter(pl.col(cfg.gqt_col).cast(pl.Float64) == 0)
        by_place = frame.group_by(place).agg(
            pl.col(v).sum().alias("persons"),
            (pl.col(v) * pl.col(flag)).sum().alias("householders"),
        ).filter(pl.col("persons") > 0)
        without = by_place.filter(pl.col("householders") == 0).height
        return by_place.height, without, int(round(by_place["householders"].sum()))

    places_0, without_0, householders_0 = count(before)
    places_1, without_1, householders_1 = count(after)
    return {
        "place_keys": place,
        "places_with_household_population": {"original": places_0, "final": places_1},
        "places_without_a_householder": {"original": without_0, "final": without_1},
        "householders": {"original": householders_0, "final": householders_1},
    }


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


def _difference_stats(level: str, diff: np.ndarray, gqt: np.ndarray | None) -> pl.DataFrame:
    """Stats on diff and |diff| for all rows, households (gqt = 0) and GQ (gqt ne 0).

    Without a GQ-type column only the "all" lines are produced.
    """
    universes = {"all": np.ones(len(diff), dtype=bool)}
    if gqt is not None:
        universes |= {"household": gqt == 0, "group_quarters": gqt != 0}
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
    """Write the rounded file, each level's controls and the review tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(result.detail, out_dir / f"rounded_detail.{fmt}")
    for name, table in result.controls.items():
        write_table(table, out_dir / f"{name}_controls.{fmt}")
    result.state_gqt_check.write_csv(out_dir / f"{result.level_names[0]}_check.csv")
    result.level_summary.write_csv(out_dir / "level_summary.csv")
    result.stats.write_csv(out_dir / "difference_stats.csv")
    # The stages have different group keys, hence the diagonal concat.
    short = pl.concat(
        [
            table.with_columns(pl.lit(i).alias("stage"), pl.lit(name).alias("level"))
            for i, (name, table) in enumerate(result.groups.items(), start=1)
        ],
        how="diagonal",
    )
    short.filter(pl.col("shortfall") > 0).write_csv(out_dir / "shortfall_groups.csv")


def _as_text(table: pl.DataFrame) -> str:
    with pl.Config(
        tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4,
        tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True,
        tbl_formatting="ASCII_MARKDOWN",
    ):
        return str(table)


def summarize(result: PipelineResult) -> str:
    """Short text summary of the review checks."""
    top = result.state_gqt_check
    off = int((top["final_minus_target"] != 0).sum())
    lines = [
        f"Control chain: {' -> '.join(result.level_names)} -> detail",
        f"{result.level_names[0]} groups: {top.height:,} "
        f"({off} do not match their rounded original total)",
        "",
        "Per level: rows, how many differ from the original, largest difference,",
        "positive originals that ended at zero, persons that could not be placed.",
        _as_text(result.level_summary),
    ]
    check = result.householder_check
    if check:
        places = check["places_with_household_population"]
        without = check["places_without_a_householder"]
        lines += [
            "",
            f"Householders: {check['householders']['original']:,} originally, "
            f"{check['householders']['final']:,} in the output.",
            f"Places ({', '.join(check['place_keys'])}) with a household population: "
            f"{places['original']:,} originally, {places['final']:,} in the output.",
            f"Of those, with no householder: {without['original']:,} originally, "
            f"{without['final']:,} in the output.",
        ]
    lines += ["", _as_text(result.stats)]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rake-and-round disclosure avoidance (port of prog03_dao_roundmethod8.sas)."
    )
    p.add_argument("input", type=Path, help="detail file: .parquet, .csv or .sas7bdat")
    p.add_argument("--out-dir", type=Path, required=True, help="folder for outputs")
    p.add_argument("--config", type=Path, help="TOML file overriding Config defaults")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                   help="format of the large outputs (default: parquet)")
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
