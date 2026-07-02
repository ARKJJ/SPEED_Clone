import ast
import unittest
from pathlib import Path


CE_FLUX_PATH = Path(__file__).resolve().parents[1] / "CE_Flux.py"


def _source():
    return CE_FLUX_PATH.read_text()


def _tree():
    return ast.parse(_source())


def _function_names():
    return {node.name for node in ast.walk(_tree()) if isinstance(node, ast.FunctionDef)}


class CEFluxStaticTests(unittest.TestCase):
    def test_attention_selection_helpers_are_consolidated(self):
        names = _function_names()

        self.assertIn("_selected_attention_suffixes", names)
        self.assertNotIn("_attention_suffixes", names)
        self.assertNotIn("_attention_suffix", names)
        self.assertNotIn("_final_modules_by_suffix", names)
        self.assertNotIn("_remaining_modules_with_suffix", names)


    def test_trace_prompt_keeps_hook_cleanup_finally_block(self):
        trace_prompt = next(
            node for node in ast.walk(_tree())
            if isinstance(node, ast.FunctionDef) and node.name == "_trace_prompt"
        )

        try_nodes = [node for node in ast.walk(trace_prompt) if isinstance(node, ast.Try)]

        self.assertTrue(any(node.finalbody for node in try_nodes))
        self.assertIn("handle.remove()", _source())


    def test_layer_update_does_not_silently_skip_missing_traces(self):
        source = _source()

        self.assertNotIn("Warning: no final anchor trace", source)
        self.assertNotIn("Warning: no edit trace", source)
        self.assertNotIn("max(_remaining_modules_with_suffix", source)

    def test_anchor_targets_are_averaged_and_expanded(self):
        names = _function_names()
        source = _source()

        self.assertIn("_mean_outputs", names)
        self.assertIn(".mean(dim=1, keepdim=True)", source)
        self.assertIn("anchor_final_means", source)
        self.assertIn("expand(-1, final_current.shape[1])", source)
        self.assertNotIn("anchor_outputs.shape[1] != final_current.shape[1]", source)

    def test_trace_compaction_keeps_token_samples_for_mean_target_expansion(self):
        source = _source()

        self.assertNotIn("_compact_trace_records", _function_names())
        self.assertIn("reshape(-1, record[\"inputs\"][0].shape[-1]).T", source)
        self.assertIn("reshape(-1, record[\"outputs\"][0].shape[-1]).T", source)


if __name__ == "__main__":
    unittest.main()
