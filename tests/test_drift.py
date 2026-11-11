import numpy as np

from histoseg.config import load_config
from histoseg.serving.drift import StainDriftMonitor
from serving_helpers import REFERENCE_STAIN

DRIFT = load_config("configs/serve.yaml")["drift"]
REF_MEAN = np.array([REFERENCE_STAIN["h_mean"], REFERENCE_STAIN["e_mean"]])
REF_STD = np.array([REFERENCE_STAIN["h_std"], REFERENCE_STAIN["e_std"]])


def fill(monitor: StainDriftMonitor, features: np.ndarray) -> float:
    score = 0.0
    for f in features:
        score = monitor.update(f)
    return score


def test_score_is_zero_before_any_request():
    assert StainDriftMonitor(REFERENCE_STAIN).score() == 0.0


def test_reference_like_inputs_score_near_zero():
    # Inputs spread like the training data, centred on the reference mean.
    rng = np.random.default_rng(0)
    features = REF_MEAN + rng.normal(size=(500, 2)) * REF_STD
    monitor = StainDriftMonitor(REFERENCE_STAIN, window=DRIFT["window"])
    assert fill(monitor, features) < 0.5 < DRIFT["threshold"]


def test_shifted_inputs_score_above_threshold():
    rng = np.random.default_rng(0)
    features = REF_MEAN + rng.normal(size=(500, 2)) * REF_STD
    features[:, 0] += 2 * REF_STD[0]  # two reference std more haematoxylin
    monitor = StainDriftMonitor(REFERENCE_STAIN, window=DRIFT["window"])
    assert fill(monitor, features) > DRIFT["threshold"]


def test_score_is_in_reference_std_units_and_uses_the_worse_stain():
    monitor = StainDriftMonitor(REFERENCE_STAIN, window=10)
    shifted = REF_MEAN + np.array([0.5, -3.0]) * REF_STD
    assert np.isclose(fill(monitor, np.tile(shifted, (10, 1))), 3.0)


def test_window_forgets_old_requests():
    monitor = StainDriftMonitor(REFERENCE_STAIN, window=5)
    fill(monitor, np.tile(REF_MEAN + 5 * REF_STD, (5, 1)))
    assert np.isclose(fill(monitor, np.tile(REF_MEAN, (5, 1))), 0.0)


def test_images_without_tissue_are_ignored():
    monitor = StainDriftMonitor(REFERENCE_STAIN, window=5)
    fill(monitor, np.tile(REF_MEAN, (5, 1)))
    assert np.isclose(monitor.update(np.array([np.nan, np.nan])), 0.0)
    assert monitor.size() == 5
