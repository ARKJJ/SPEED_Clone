import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "Flux1" / "mlp.py"


class Flux1MlpTokenSelectionStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text()
        cls.tree = ast.parse(cls.text, filename=str(SOURCE))

    def test_target_and_anchor_use_all_fixed_sequence_positions(self):
        self.assertIn(
            "full_token_indices = list(range(max_sequence_length))",
            self.text,
        )
        self.assertIn(
            "target_token_indices = {concept: full_token_indices for concept in target_concepts}",
            self.text,
        )
        self.assertIn(
            "anchor_token_indices = {concept: full_token_indices for concept in anchor_concepts}",
            self.text,
        )

    def test_retain_selection_uses_all_fixed_sequence_positions(self):
        self.assertIn(
            "retain_token_indices = {",
            self.text,
        )
        self.assertIn(
            "concept: full_token_indices for concept in retain_texts",
            self.text,
        )

    def test_retain_second_moment_normalizes_by_retain_concept_count(self):
        self.assertIn(
            "retain_count_by_module[module_name] += 1",
            self.text,
        )
        self.assertNotIn(
            "retain_count_by_module[module_name] += retain_inputs.shape[1]",
            self.text,
        )

    def test_target_and_anchor_are_not_repeated_for_column_alignment(self):
        self.assertNotIn("target_count = target_inputs.shape[1] // anchor_inputs.shape[1]", self.text)
        self.assertNotIn("anchor_inputs = anchor_inputs.repeat_interleave(target_count, dim=1)", self.text)

    def test_target_and_cross_grams_normalize_by_target_token_count(self):
        self.assertIn(
            "target_matrix = target_inputs @ target_inputs.T / target_inputs.shape[1]",
            self.text,
        )
        self.assertIn(
            "cross_matrix = anchor_inputs @ target_inputs.T / target_inputs.shape[1]",
            self.text,
        )

    def test_mlp_layers_are_obtained_directly_from_transformer_blocks(self):
        self.assertIn("pipeline.transformer.transformer_blocks", self.text)
        self.assertIn("block.ff_context.net[2]", self.text)
        self.assertNotIn("for name, module in pipeline.transformer.named_modules()", self.text)


if __name__ == "__main__":
    unittest.main()
