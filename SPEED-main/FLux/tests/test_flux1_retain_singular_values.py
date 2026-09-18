import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "Flux1" / "analyze_retain_singular_values.py"


def load_module():
    spec = importlib.util.spec_from_file_location("flux1_retain_singular_values", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Flux1RetainSingularValuesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_summary_counts_exact_zero_and_strict_thresholds(self):
        summary = self.module.summarize_singular_values(
            [0.0, 1e-8, 1e-4, 0.2], [0.0, 1e-4, 1e-3]
        )

        self.assertEqual(summary["exact_zero_count"], 1)
        self.assertEqual(summary["count_below_0"], 0)
        self.assertEqual(summary["count_below_0.0001"], 2)
        self.assertEqual(summary["count_below_0.001"], 3)
        self.assertEqual(summary["numerical_rank_exact"], 3)

    def test_csv_writer_emits_one_row_per_module_threshold(self):
        rows = [
            {
                "module": "transformer_blocks.0.ff_context.net.2",
                "layer": 0,
                "matrix_dim": 4,
                "sample_columns": 20,
                "concept_count": 2,
                "threshold": 1e-4,
                "count_below_threshold": 2,
                "null_space_dim": 2,
                "exact_zero_count": 1,
                "numerical_rank_exact": 3,
                "min_singular_value": 0.0,
                "max_singular_value": 0.2,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "singular_values.csv"
            self.module.write_summary_csv(rows, output)
            with output.open(newline="") as handle:
                written = list(csv.DictReader(handle))

        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["module"], rows[0]["module"])
        self.assertEqual(written[0]["count_below_threshold"], "2")
        self.assertEqual(written[0]["null_space_dim"], "2")

    def test_retain_loader_selects_retain_rows_by_default(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False) as handle:
            handle.write("id,type,concept\n")
            handle.write("1,erase,Erase Person\n")
            handle.write("2,retain,Keep Person\n")
            csv_path = Path(handle.name)
        try:
            texts = self.module._load_retain_texts(csv_path, "concept", [])
        finally:
            csv_path.unlink()

        self.assertEqual(texts, ["Keep Person"])

    def test_retain_token_indices_select_all_positions(self):
        class Tokenizer:
            def __call__(self, *args, **kwargs):
                raise AssertionError("all-position mode must not tokenize each prompt")

        class Pipeline:
            tokenizer_2 = Tokenizer()

        indices = self.module._retain_token_indices(
            Pipeline(), ["Keep Person", ""], max_sequence_length=512
        )

        self.assertEqual(indices["Keep Person"], list(range(512)))
        self.assertEqual(indices[""], list(range(512)))

    def test_analysis_normalizes_retain_covariance_by_concept_count(self):
        text = SOURCE.read_text()
        self.assertIn("concept_count[name] += 1", text)
        self.assertIn("covariance = second_moment[name] / count", text)
        self.assertNotIn("covariance = second_moment[name] / sample_columns[name]", text)


if __name__ == "__main__":
    unittest.main()
