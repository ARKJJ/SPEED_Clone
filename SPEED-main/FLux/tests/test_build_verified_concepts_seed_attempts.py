import importlib.util
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "Flux1" / "build_verified_concepts.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_verified_concepts", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildVerifiedConceptsSeedAttemptsTest(unittest.TestCase):
    def test_each_task_considers_thirty_candidate_seeds_per_template(self):
        module = load_module()
        specs = (
            module.task_spec("retain"),
            module.task_spec("erase", 10),
            module.task_spec("erase", 50),
            module.task_spec("erase", 100),
        )
        self.assertEqual([spec.max_attempts for spec in specs], [30, 30, 30, 30])


if __name__ == "__main__":
    unittest.main()
