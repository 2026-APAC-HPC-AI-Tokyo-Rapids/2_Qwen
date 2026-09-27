import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("serving_goodput", Path(__file__).parents[1]/"calibrate_serving_goodput.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ServingGoodputTests(unittest.TestCase):
    def dataset(self):
        return {"calibration": {"run_id": "a", "predicted_goodput": 1200, "measured_goodput": 1000, "parser_valid": True}, "holdout": [{"run_id": "b", "predicted_goodput": 2400, "measured_goodput": 1900, "parser_valid": True}]}

    def test_factor_depends_only_on_calibration(self):
        data = self.dataset()
        first = module.evaluate(data)
        data["holdout"][0]["measured_goodput"] = 1500
        self.assertEqual(first["scale"], module.evaluate(data)["scale"])
        self.assertEqual(first["holdout"][0]["calibrated_goodput"], 2000)

    def test_reused_run_is_rejected(self):
        data = self.dataset()
        data["holdout"][0]["run_id"] = "a"
        with self.assertRaisesRegex(ValueError, "disjoint"):
            module.evaluate(data)

    def test_ineligible_run_is_not_a_calibration_point(self):
        data = self.dataset()
        data["calibration"]["parser_valid"] = False
        with self.assertRaisesRegex(ValueError, "validated"):
            module.evaluate(data)
