"""Turn the 2020 Census PPMF person file into input for dao_rake_round.py.

The PPMF person file (2020_ppmf_person.csv) has one row per person.  The
rake-and-round pipeline wants one row per cell with a count.  This script
counts persons by

    nation, state, county, age, sex, cenrace, hisp, gqt

and writes the result as Parquet (or CSV).

PPMF field        ->  pipeline column
-----------------     ---------------------------------------------------
(constant 1)      ->  nation
TABBLKST          ->  state     2-character FIPS, kept as text
TABBLKCOU         ->  county    3-character FIPS, kept as text
QAGE (0-115)      ->  age       top-coded at --max-age (default 100)
QSEX              ->  sex       1 male, 2 female
CENRACE (01-63)   ->  cenrace   all 63 categories, as an integer
CENHISP           ->  hisp      1 not Hispanic, 2 Hispanic
GQTYPE_PL (0-7)   ->  gqt       0 = housing unit, 1-7 = major GQ type
count of persons  ->  blend_pop

Two things differ from the file the SAS program reads:

* Race is CENRACE's 63 categories, not the 31-category race31.  Race is
  never used as a rake or rounding group, so this only changes the number
  of cells.  Use config.ppmf.toml, which names the column `cenrace`.
* blend_pop is a whole-number count here, not a fractional estimate.

Usage
-----
    python ppmf_to_input.py 2020_ppmf_person.csv --out ppmf_cells.parquet
    python ppmf_to_input.py 2020_ppmf_person.csv --out de_ri.parquet --states 10 44

Then:
    python dao_rake_round.py ppmf_cells.parquet --out-dir run1 --config config.ppmf.toml

--fill-zeros adds a zero row for every age x sex x race x Hispanic-origin
combination in every county x GQ type, like a full master file.  That is
203,616 rows per county at the defaults, so use it with --states.  Zero
cells only matter when the age-universe rules push an added person past
the cells that have people in them.

Block level
-----------
--level block also keeps the tract, the block and the relationship:

TABTRACTCE        ->  tract     6-character tract code, kept as text
TABBLK            ->  block     4-character block number, kept as text
RELSHIP (20-38)   ->  relship   relationship to householder, as an integer

    python ppmf_to_input.py 2020_ppmf_person.csv --out de_blocks.parquet --level block --states 10
    python dao_rake_round.py de_blocks.parquet --out-dir run_blocks --config config.block.toml

Only populated cells are written (--fill-zeros is not available: the full
rectangle at block level is far too large).  Expect close to one row per
person, so run one state at a time.

Requires duckdb (pip install duckdb).  DuckDB streams the CSV, so the
national file does not have to fit in memory.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import duckdb

REQUIRED = ["TABBLKST", "TABBLKCOU", "RTYPE", "QAGE", "QSEX", "CENRACE", "CENHISP", "GQTYPE_PL"]
N_RACES = 63
N_GQT = 8  # 0 = housing unit, 1-7 = major GQ types
BLOCK_REQUIRED = ["TABTRACTCE", "TABBLK", "RELSHIP"]
COUNTY_KEYS = ["nation", "state", "county", "age", "sex", "cenrace", "hisp", "gqt"]
BLOCK_KEYS = ["nation", "state", "county", "tract", "block",
              "age", "sex", "cenrace", "hisp", "gqt", "relship"]


def build_cells(
    con: duckdb.DuckDBPyConnection,
    person_csv: Path,
    states: Sequence[str] | None = None,
    max_age: int = 100,
    fill_zeros: bool = False,
    level: str = "county",
) -> None:
    """Create the table `cells` in `con` from the PPMF person CSV."""
    block_level = level == "block"
    if block_level and fill_zeros:
        raise SystemExit("--fill-zeros is not available with --level block.")
    src = str(person_csv).replace("'", "''")
    con.execute(
        "CREATE OR REPLACE VIEW person AS "
        f"SELECT * FROM read_csv('{src}', header = true, all_varchar = true)"
    )
    have = {row[0].upper() for row in con.execute("DESCRIBE person").fetchall()}
    missing = [c for c in REQUIRED + (BLOCK_REQUIRED if block_level else []) if c not in have]
    if missing:
        raise SystemExit(
            f"{person_csv}: expected PPMF person-file columns not found: {missing}.\n"
            f"Columns present: {sorted(have)}"
        )

    where = ""
    if states:
        codes = ", ".join(f"'{int(s):02d}'" for s in states)
        where = f"WHERE TABBLKST IN ({codes})"

    block_columns = "TABTRACTCE AS tract, TABBLK AS block," if block_level else ""
    relship_column = "CAST(RELSHIP AS INTEGER) AS relship," if block_level else ""
    con.execute(f"""
        CREATE OR REPLACE TABLE counted AS
        SELECT 1                                   AS nation,
               TABBLKST                            AS state,
               TABBLKCOU                           AS county,
               {block_columns}
               LEAST(CAST(QAGE AS INTEGER), {int(max_age)}) AS age,
               CAST(QSEX AS INTEGER)               AS sex,
               CAST(CENRACE AS INTEGER)            AS cenrace,
               CAST(CENHISP AS INTEGER)            AS hisp,
               CAST(GQTYPE_PL AS INTEGER)          AS gqt,
               {relship_column}
               CAST(COUNT(*) AS DOUBLE)            AS blend_pop,
               COUNT(*) FILTER ((RTYPE = '3') <> (CAST(GQTYPE_PL AS INTEGER) = 0)) AS rtype_mismatch
        FROM person
        {where}
        GROUP BY ALL
    """)

    n_bad = con.execute("SELECT COALESCE(SUM(rtype_mismatch), 0) FROM counted").fetchone()[0]
    if n_bad:
        print(f"WARNING: {int(n_bad)} person(s) where RTYPE and GQTYPE_PL disagree "
              "about housing unit vs group quarters.", file=sys.stderr)

    if not fill_zeros:
        con.execute(
            "CREATE OR REPLACE TABLE cells AS SELECT * EXCLUDE (rtype_mismatch) FROM counted"
        )
    else:
        con.execute(f"""
            CREATE OR REPLACE TABLE cells AS
            SELECT c.nation, c.state, c.county, a.age, s.sex, r.cenrace, h.hisp, g.gqt,
                   COALESCE(k.blend_pop, 0.0) AS blend_pop
            FROM (SELECT DISTINCT nation, state, county FROM counted) c
            CROSS JOIN (SELECT UNNEST(RANGE(0, {int(max_age)} + 1)) AS age) a
            CROSS JOIN (SELECT UNNEST([1, 2]) AS sex) s
            CROSS JOIN (SELECT UNNEST(RANGE(1, {N_RACES} + 1)) AS cenrace) r
            CROSS JOIN (SELECT UNNEST([1, 2]) AS hisp) h
            CROSS JOIN (SELECT UNNEST(RANGE(0, {N_GQT})) AS gqt) g
            LEFT JOIN counted k
              ON  k.state = c.state AND k.county = c.county AND k.age = a.age AND k.sex = s.sex
              AND k.cenrace = r.cenrace AND k.hisp = h.hisp AND k.gqt = g.gqt
        """)
        lost = con.execute(
            "SELECT (SELECT SUM(blend_pop) FROM counted) - (SELECT SUM(blend_pop) FROM cells)"
        ).fetchone()[0]
        if lost:
            raise SystemExit(f"{lost:.0f} person(s) have codes outside the expected ranges "
                             "(sex 1-2, cenrace 1-63, hisp 1-2, gqt 0-7).")


def summarize(con: duckdb.DuckDBPyConnection) -> str:
    rows, persons, positive = con.execute(
        "SELECT COUNT(*), SUM(blend_pop), COUNT(*) FILTER (blend_pop > 0) FROM cells"
    ).fetchone()
    n_state, n_county = con.execute(
        "SELECT COUNT(DISTINCT state), COUNT(DISTINCT (state, county)) FROM cells"
    ).fetchone()
    by_gqt = con.execute(
        "SELECT gqt, CAST(SUM(blend_pop) AS BIGINT) FROM cells GROUP BY gqt ORDER BY gqt"
    ).fetchall()
    geography = f"states {n_state}   counties {n_county}"
    if "block" in {row[0] for row in con.execute("DESCRIBE cells").fetchall()}:
        n_tract, n_block = con.execute(
            "SELECT COUNT(DISTINCT (state, county, tract)), "
            "COUNT(DISTINCT (state, county, tract, block)) FROM cells"
        ).fetchone()
        geography += f"   tracts {n_tract:,}   blocks {n_block:,}"
    lines = [
        f"persons {int(persons):,}   {geography}",
        f"cells   {rows:,} ({positive:,} with at least one person)",
        "persons by gqt: " + "  ".join(f"{g}: {n:,}" for g, n in by_gqt),
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Count the PPMF person file into rake-and-round cells."
    )
    p.add_argument("person_csv", type=Path, help="2020_ppmf_person.csv (or an extract of it)")
    p.add_argument("--out", type=Path, required=True, help="output file (.parquet or .csv)")
    p.add_argument("--states", nargs="+",
                   help="state FIPS codes to keep, e.g. 10 44 (default: all)")
    p.add_argument("--max-age", type=int, default=100,
                   help="top-code age at this value (default 100)")
    p.add_argument("--fill-zeros", action="store_true",
                   help="add zero rows so every county x GQ type has every detail cell")
    p.add_argument("--level", choices=["county", "block"], default="county",
                   help="block: also keep tract, block and relationship (default: county)")
    args = p.parse_args(argv)

    con = duckdb.connect()
    build_cells(con, args.person_csv, args.states, args.max_age, args.fill_zeros, args.level)
    order = "ORDER BY " + ", ".join(BLOCK_KEYS if args.level == "block" else COUNTY_KEYS)
    fmt = "(FORMAT PARQUET)" if args.out.suffix == ".parquet" else "(FORMAT CSV, HEADER)"
    con.execute(f"COPY (SELECT * FROM cells {order}) TO '{args.out}' {fmt}")
    print(summarize(con))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
