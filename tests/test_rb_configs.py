import unittest
from pathlib import Path

import yaml

from scripts.gen_configs import make


class RBConfigTests(unittest.TestCase):
    def test_generator_uses_frozen_rb_learning_rates(self):
        self.assertEqual(make("rayleigh_benard", "jepa")["optim"]["lr"], 5e-5)
        self.assertEqual(make("rayleigh_benard", "mae")["optim"]["lr"], 1e-4)

    def test_checked_in_rb_configs_match_frozen_learning_rates(self):
        root = Path(__file__).parents[1] / "configs"
        jepa = yaml.safe_load((root / "rayleigh_benard_jepa.yaml").read_text())
        mae = yaml.safe_load((root / "rayleigh_benard_mae.yaml").read_text())
        self.assertEqual(jepa["optim"]["lr"], 5e-5)
        self.assertEqual(mae["optim"]["lr"], 1e-4)

    def test_rb_lr_fix_does_not_change_other_generated_configs(self):
        root = Path(__file__).parents[1] / "configs"
        for dataset in ("active_matter", "shear_flow"):
            for objective in ("jepa", "mae"):
                checked_in = yaml.safe_load((root / f"{dataset}_{objective}.yaml").read_text())
                self.assertEqual(make(dataset, objective)["optim"]["lr"], checked_in["optim"]["lr"])


if __name__ == "__main__":
    unittest.main()
