"""Write a SYNTHETIC person file in the 2020 PPMF person-file layout.

Nothing here is Census data.  The file exists so the chain

    person file -> ppmf_to_input.py -> dao_rake_round.py

can be run end to end when the real PPMF is not at hand.  State codes 91
and 92 are not real FIPS codes, on purpose.

The shapes are loosely realistic so the run exercises the procedure:
five counties from 6,000 to 600,000 people, about 2.5% of people in group
quarters with type-appropriate ages, a long-tailed 63-category race
distribution, and a handful of people at out-of-universe ages for their
GQ type.

For block-level runs every person also has a tract (TABTRACTCE), a block
(TABBLK) and a relationship to householder (RELSHIP).  Tracts hold about
4,000 people and blocks are very uneven in size.  Each GQ type sits in a
few "facility" blocks per county.  Household members get a relationship
code with roughly one householder per 2.5 people; a block whose household
residents are all under 15 still gets a householder, which makes a few
out-of-universe householders.

    python make_synthetic_ppmf.py synthetic_ppmf_person.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SEED = 2020

# (state, county, persons, GQ share, weights for GQ types 1-7)
COUNTIES = [
    ("91", "001", 600_000, 0.030, [10, 1, 20, 3, 50, 2, 14]),   # city with a university
    ("91", "003", 150_000, 0.020, [15, 1, 40, 4, 10, 15, 15]),  # suburb with a base
    ("91", "005", 40_000, 0.080, [75, 2, 12, 2, 0, 0, 9]),      # rural, large prison
    ("92", "001", 300_000, 0.022, [22, 1, 30, 4, 25, 3, 15]),
    ("92", "003", 6_000, 0.015, [0, 0, 70, 0, 0, 0, 30]),       # very small
]

# GQ type -> (youngest, oldest, mean, sd, share male) for the usual residents
GQ_AGES = {
    1: (18, 85, 36, 11, 0.91),   # adult correctional
    2: (10, 20, 16, 2, 0.80),    # juvenile
    3: (40, 105, 82, 11, 0.33),  # nursing
    4: (18, 95, 50, 18, 0.55),   # other institutional
    5: (17, 30, 20, 1.6, 0.46),  # college housing
    6: (18, 50, 24, 5, 0.83),    # military
    7: (16, 95, 42, 17, 0.58),   # other noninstitutional
}
GQTYPE_CODE = {0: "000", 1: "101", 2: "201", 3: "301", 4: "401", 5: "501", 6: "601", 7: "701"}
STRAY_SHARE = 0.002  # GQ residents given an arbitrary age 0-100

PERSONS_PER_TRACT = 4000
BLOCKS_PER_TRACT = 60
ADULTS_PER_HOUSEHOLDER = 1.9
# RELSHIP codes and shares for household members who are not the householder
ADULT_RELSHIP = ([21, 22, 23, 24, 25, 28, 29, 31, 32, 33, 34, 36],
                 [0.45, 0.08, 0.01, 0.01, 0.15, 0.04, 0.05, 0.02, 0.02, 0.05, 0.07, 0.05])
CHILD_RELSHIP = ([25, 26, 27, 30, 33, 35, 36], [0.80, 0.03, 0.05, 0.08, 0.02, 0.01, 0.01])


def assign_blocks(rng: np.random.Generator, gqt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A tract number and a block number for every person in one county."""
    n = len(gqt)
    n_tracts = max(1, round(n / PERSONS_PER_TRACT))
    n_blocks = n_tracts * BLOCKS_PER_TRACT
    weights = rng.lognormal(0.0, 1.3, n_blocks)          # a few big blocks, many small
    block = rng.choice(n_blocks, n, p=weights / weights.sum())
    for t in range(1, 8):                                # GQ: a few facility blocks per type
        members = np.flatnonzero(gqt == t)
        if len(members):
            n_facilities = min(n_blocks, max(1, round(len(members) / 400)))
            facilities = rng.choice(n_blocks, n_facilities, replace=False)
            share = rng.dirichlet(np.full(n_facilities, 0.8))
            block[members] = rng.choice(facilities, len(members), p=share)
    return block // BLOCKS_PER_TRACT, block % BLOCKS_PER_TRACT


def assign_relship(rng: np.random.Generator, gqt: np.ndarray, age: np.ndarray,
                   block_id: np.ndarray) -> np.ndarray:
    """A RELSHIP code for every person in one county."""
    relship = np.where(gqt == 0, 0, np.where(gqt <= 4, 37, 38))
    hh = np.flatnonzero(gqt == 0)
    adult = age[hh] >= 18
    relship[hh[adult]] = rng.choice(ADULT_RELSHIP[0], adult.sum(), p=ADULT_RELSHIP[1])
    relship[hh[~adult]] = rng.choice(CHILD_RELSHIP[0], (~adult).sum(), p=CHILD_RELSHIP[1])
    # Householders: in each block the oldest residents, one per
    # ADULTS_PER_HOUSEHOLDER adults and at least one, in random order among
    # equals.  A block with no adults therefore gets a householder under 18.
    order = hh[np.lexsort((rng.random(len(hh)), -age[hh], block_id[hh]))]
    blocks, start, size = np.unique(block_id[order], return_index=True, return_counts=True)
    adults = np.bincount(np.searchsorted(blocks, block_id[hh[adult]]), minlength=len(blocks))
    n_householders = np.maximum(1, np.rint(adults / ADULTS_PER_HOUSEHOLDER)).astype(int)
    rank = np.arange(len(order)) - np.repeat(start, size)
    relship[order[rank < np.repeat(n_householders, size)]] = 20
    return relship


def race_probs(rng: np.random.Generator) -> np.ndarray:
    """63 CENRACE categories: six single races, then 57 combinations with a long tail."""
    p = np.zeros(63)
    p[:6] = [0.60, 0.13, 0.011, 0.06, 0.002, 0.085]
    tail = np.sort(rng.dirichlet(np.full(57, 0.25)))[::-1]
    p[6:] = tail * (1 - p[:6].sum())
    return p


def household_ages(rng: np.random.Generator, n: int) -> np.ndarray:
    weights = np.where(np.arange(111) < 60, 1.0, np.exp(-(np.arange(111) - 60) / 13.0))
    return rng.choice(111, n, p=weights / weights.sum())


def main(path: str) -> None:
    rng = np.random.default_rng(SEED)
    # A separate stream, so the columns generated before this was added are unchanged.
    geo_rng = np.random.default_rng(SEED + 1)
    base_race = race_probs(rng)
    frames = []
    for state, county, n, gq_share, type_weights in COUNTIES:
        # county-specific tilt of the race distribution
        race_p = base_race * rng.lognormal(0, 0.5, 63)
        race_p /= race_p.sum()

        in_gq = rng.random(n) < gq_share
        w = np.array(type_weights, dtype=float)
        gqt = np.zeros(n, dtype=int)
        gqt[in_gq] = rng.choice(np.arange(1, 8), in_gq.sum(), p=w / w.sum())

        age = household_ages(rng, n)
        male = rng.random(n) < 0.49
        for t, (lo, hi, mean, sd, p_male) in GQ_AGES.items():
            m = gqt == t
            age[m] = np.clip(np.rint(rng.normal(mean, sd, m.sum())), lo, hi)
            male[m] = rng.random(m.sum()) < p_male
        stray = in_gq & (rng.random(n) < STRAY_SHARE)
        age[stray] = rng.integers(0, 101, stray.sum())

        cenrace = rng.choice(np.arange(1, 64), n, p=race_p)
        p_hisp = np.where(cenrace == 6, 0.90, np.where(cenrace == 1, 0.12, 0.06))
        hisp = np.where(rng.random(n) < p_hisp, 2, 1)

        tract, block = assign_blocks(geo_rng, gqt)
        relship = assign_relship(geo_rng, gqt, age, tract * BLOCKS_PER_TRACT + block)

        frames.append(pd.DataFrame({
            "VINTAGE": "SYNTHETIC",
            "TABBLKST": state,
            "TABBLKCOU": county,
            "TABTRACTCE": [f"{(t + 1) * 100:06d}" for t in tract],
            "TABBLK": [f"{1000 + (b % 4) * 1000 + b // 4:04d}" for b in block],
            "RTYPE": np.where(gqt == 0, "3", "5"),
            "GQTYPE": pd.Series(gqt).map(GQTYPE_CODE).to_numpy(),
            "RELSHIP": relship,
            "QSEX": np.where(male, 1, 2),
            "QAGE": age,
            "CENHISP": hisp,
            "CENRACE": [f"{c:02d}" for c in cenrace],
            "GQTYPE_PL": gqt,
        }))

    person = pd.concat(frames, ignore_index=True)
    person.insert(5, "EPNUM", np.arange(1, len(person) + 1))
    person.to_csv(path, index=False)
    print(f"wrote {len(person):,} synthetic person records to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic_ppmf_person.csv")
