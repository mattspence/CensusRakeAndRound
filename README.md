# CensusRakeAndRound
Raking and Rounding as Disclosure Avoidance

`dao_rake_round.py` is a Python (Polars) port of the SAS rake-and-round
programs in [`sas/`](sas). The method is described in
[Rake-and-Round Disclosure Avoidance Specification.pdf](Rake-and-Round%20Disclosure%20Avoidance%20Specification.pdf).

## What it does

The input is a population file with one row per detail cell and a count.
The program:

1. **Coarsens** every positive count to the nearest 5 (a positive count
   that would round to 0 becomes 1).
2. **Rakes** the coarsened counts so they sum to control totals.
3. **Integerizes** the raked values (greatest-mantissa rounding within each
   control group), so the output is whole numbers.

This is done in stages, coarsest geography first. By default there are two,
as in the SAS program:

| Stage | Cells | Raked to |
|---|---|---|
| 1 | county x GQ type totals | unrounded state x GQ type totals |
| 2 | detail cells | the stage 1 integer controls |

The output sums to the rounded state x GQ type totals of the original file.
When a person has to be added to a cell during integerization, only cells
inside the GQ type's age universe can receive one (for example, ages 15+
for GQ type 1).

## Requirements

- Python 3.11 or later
- `polars` and `numpy`
- `pyreadstat`, only to read `.sas7bdat` input
- `duckdb`, only for `ppmf_to_input.py`
- `pytest`, only to run the tests

```bash
pip install polars numpy
```

## Input file

A `.parquet`, `.csv` or `.sas7bdat` file with one row per detail cell.
With the default settings it needs these columns:

| Column | Meaning |
|---|---|
| `nation`, `state`, `county` | geography |
| `age`, `sex`, `race31`, `hisp` | demographic detail |
| `gqt` | GQ type: 0 = household, 1-7 = major group quarters type |
| `blend_pop` | the count to be rounded (may be fractional) |

Column names can be changed in the config file. Columns not named in the
config are ignored. SAS column names are converted to lower case.

The program stops with an error if a key column has a missing value, or if
`blend_pop` has one (set `allow_missing_values = true` to carry missing
counts through as SAS does).

## Running it

```bash
python dao_rake_round.py input.parquet --out-dir run1
```

With a config file:

```bash
python dao_rake_round.py input.parquet --out-dir run1 --config config.toml
```

| Option | Meaning |
|---|---|
| `input` | the detail file |
| `--out-dir DIR` | folder for the outputs (required; created if needed) |
| `--config FILE` | TOML file of settings |
| `--format parquet\|csv` | format of the large outputs (default `parquet`) |
| `--base N` | rounding base; overrides the config (default 5) |
| `--seed N` | tiebreaker seed; overrides the config (default 42) |

The same seed and the same input give the same output.

## Outputs

Written to `--out-dir`. Level names come from the control chain; the names
below are for the default two stages.

| File | Contents |
|---|---|
| `rounded_detail.parquet` | **the rounded file**: the detail keys and the integer count, sorted by the keys |
| `county_gqt_controls.parquet` | per county x GQ type: `original`, `coarsened`, `raked`, `control`, `final`, `diff`, `absdiff` (one such file per control level below the first) |
| `state_gqt_check.csv` | per state x GQ type: original total, rounded target, and what the output sums to; `final_minus_target` should be 0 |
| `level_summary.csv` | per level: rows, how many differ from the original, largest difference, positive cells that ended at zero, persons that could not be placed |
| `difference_stats.csv` | sum, mean, std, min, max, mode, median, p5, p95 of the differences, for all cells, households and group quarters |
| `shortfall_groups.csv` | groups that ended short of their control because too few cells were eligible to receive a person (empty in a clean run) |
| `report.txt` | the text summary that is also printed to the screen |

`worked_example.txt` and `worked_example_block.txt` show these tables for a
small synthetic file, with one group traced through each step.

## Configuration

Every setting is optional. [`config.example.toml`](config.example.toml)
lists the SAS-equivalent settings with their defaults, which reproduce
`prog03_dao_roundmethod8.sas`. An unknown key is an error.

| Key | Default | Meaning |
|---|---|---|
| `value_col` | `"blend_pop"` | the count column |
| `detail_keys` | `nation, state, county, age, sex, race31, hisp, gqt` | the columns that identify a cell; also the sort order |
| `state_gqt_keys` | `nation, state, gqt` | first control level |
| `county_gqt_keys` | `nation, state, county, gqt` | second control level |
| `age_col`, `gqt_col` | `"age"`, `"gqt"` | columns the age-universe rules test |
| `rounding_base` | `5.0` | coarsen to the nearest multiple of this |
| `min_positive` | `1.0` | value given to a positive count that would round to 0 |
| `seed` | `42` | seed for the random tiebreaker in integerization |
| `allow_missing_values` | `false` | `true` carries missing counts through as SAS does |
| `[age_universe]` | see the example file | per GQ type, the ages that cannot receive an added person |
| `control_levels` | not set | a longer control chain; replaces the two `*_keys` levels |
| `relationship_col` | not set | relationship-to-householder column |
| `householder_codes` | `[20]` | relationship codes that mean householder |
| `householder_col` | `"householder"` | name of the derived 0/1 householder column |
| `[[relationship_universe]]` | 2020 Census defaults | where each relationship code can exist (GQ types, ages) |

### Block-level runs

`control_levels` extends the two stages to a chain of any depth. Each level
is coarsened, raked to the integer controls of the level above, and
integerized; the detail cells are then raked to the last level. Each level
must contain every key of the level above it and add at least one.

[`config.block.toml`](config.block.toml) sets up

```
state x GQT -> county -> tract -> block -> block x householder -> detail
```

The last level controls the number of householders per block. It uses
`householder`, a 0/1 column derived from `relationship_col`. With a
relationship column, the relationship rules join the age rules in deciding
which cells may receive an added person, and the report counts blocks left
with a household population but no householder.

## Running on the PPMF

`ppmf_to_input.py` counts the 2020 Census PPMF person file into cells.
Race there is `cenrace` (63 categories) rather than `race31`, so use
[`config.ppmf.toml`](config.ppmf.toml).

```bash
python ppmf_to_input.py 2020_ppmf_person.csv --out ppmf_cells.parquet
```

```bash
python dao_rake_round.py ppmf_cells.parquet --out-dir run1 --config config.ppmf.toml
```

For a block-level run:

```bash
python ppmf_to_input.py 2020_ppmf_person.csv --out cells.parquet --level block
```

```bash
python dao_rake_round.py cells.parquet --out-dir run_blocks --config config.block.toml
```

`--states 10 44` limits `ppmf_to_input.py` to those state FIPS codes.

Without the PPMF, `make_synthetic_ppmf.py` writes a synthetic person file
in the same layout (about 1.1 million made-up records, no Census data):

```bash
python make_synthetic_ppmf.py synthetic_ppmf_person.csv
```

## Using it from Python

```python
import polars as pl
from dao_rake_round import Config, run_pipeline, write_outputs, summarize

df = pl.read_parquet("input.parquet")
cfg = Config()                              # or Config.from_toml("config.toml")
result = run_pipeline(df, cfg)

result.detail            # the rounded file
result.controls          # dict: level name -> controls table
result.state_gqt_check   # do the outputs hit the top-level totals?
result.level_summary     # one line per level
result.stats             # difference statistics
result.groups            # per stage: pool, eligible rows, shortfall per group
print(summarize(result))
```

`run_pipeline` takes a Polars DataFrame; convert a pandas frame with
`pl.from_pandas(df)`. `Config` is frozen, so change a setting with
`dataclasses.replace(cfg, rounding_base=10)`.

The two SAS macros are also available on their own:

- `rakeit(df, value_col, group_cols, control=...)` is `%rakeit`: scale a
  column within groups to control totals.
- `roundit(df, value_col, group_cols, ...)` is `%roundit`: integerize a
  column within groups, keeping each group's rounded total.

## Differences from the SAS program

Group totals match SAS. Individual cells can differ because the random
tiebreaker stream is numpy's, not SAS's. For an exact comparison, export the
SAS tiebreaker column and pass it to `roundit(tiebreaker_col=...)`. Rows
with a missing key raise an error rather than being silently dropped. The
full list is in the docstring at the top of `dao_rake_round.py`.

## Other files

| File | Contents |
|---|---|
| `dao_rake_round_polars.py` | the earlier Polars port, without `control_levels` and relationship rules |
| `dao_rake_round_pandas.py` | the original pandas port |
| `test_dao_rake_round.py` | tests: `python -m pytest test_dao_rake_round.py` |
| `sas/` | the SAS programs the port follows |
