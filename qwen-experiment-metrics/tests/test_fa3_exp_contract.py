"""Fail closed on numeric regressions, drift and incomplete FA3 comparisons."""
import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[2] / "kernels/probes/fa3-exp.py"))


def test_numeric_contract_requires_relative_and_absolute_accuracy():
    delta = {"relative_l2": .003, "max_scaled": .02}
    control = {"relative_l2": .025, "max_scaled": .04}
    candidate = {"relative_l2": .0251, "max_scaled": .0401}
    assert MODULE["accept_errors"](delta, control, candidate)
    assert not MODULE["accept_errors"](delta, control, {**candidate, "relative_l2": .03})
    assert not MODULE["accept_errors"]({**delta, "relative_l2": .02}, control, candidate)
    assert not MODULE["accept_errors"](delta, control, {**candidate, "max_scaled": .06})
    for bad in (float("nan"), float("inf")):
        assert not MODULE["accept_errors"](delta, control, {**candidate, "relative_l2": bad})


def timing():
    return [{"rows": n, "ratio": .95, "saving_us": 20., "control_drift": .01, "paired_ratio_p90": .97} for n in (3986, 4027, 4096)]


@pytest.mark.parametrize(("field", "value"), [("ratio", .98), ("ratio", float("nan")), ("saving_us", 4.), ("control_drift", -.06), ("paired_ratio_p90", 1.01)])
def test_performance_contract_needs_every_shape_and_stable_pairs(field, value):
    rows = timing()
    assert MODULE["eligible"](rows)
    rows[-1][field] = value
    assert not MODULE["eligible"](rows)


def test_incomplete_or_duplicate_shapes_do_not_qualify():
    rows = timing()
    assert not MODULE["eligible"](rows[:-1])
    assert not MODULE["eligible"]([rows[0], rows[0], rows[2]])
