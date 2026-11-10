import json
from pathlib import Path

from histoseg.data.split import SPLITS, split_patients

FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}
PATIENTS = [f"TCGA-XX-{i:04d}" for i in range(151)]


def test_committed_splits_have_no_shared_patients():
    splits = json.loads(Path("configs/splits.json").read_text())
    seen = [p for name in SPLITS for p in splits[name]]
    assert len(seen) == len(set(seen)), "a patient appears in more than one split"
    assert splits["counts"] == {name: len(splits[name]) for name in SPLITS}


def test_split_is_disjoint_and_complete():
    splits = split_patients(PATIENTS, FRACTIONS, seed=42)
    sets = [set(splits[name]) for name in SPLITS]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(PATIENTS)


def test_split_sizes_follow_fractions():
    splits = split_patients(PATIENTS, FRACTIONS, seed=42)
    assert [len(splits[name]) for name in SPLITS] == [106, 23, 22]


def test_same_seed_gives_identical_split_regardless_of_input_order():
    a = split_patients(PATIENTS, FRACTIONS, seed=42)
    b = split_patients(list(reversed(PATIENTS)) + PATIENTS[:5], FRACTIONS, seed=42)
    assert a == b


def test_different_seed_gives_different_split():
    assert split_patients(PATIENTS, FRACTIONS, seed=1) != split_patients(
        PATIENTS, FRACTIONS, seed=2
    )
