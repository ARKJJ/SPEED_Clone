import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "Flux1" / "sample2.py"


class Flux1Sample2PairingStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text()
        cls.tree = ast.parse(cls.text, filename=str(SOURCE))

    def test_paired_sampling_uses_the_sample_style_serial_path(self):
        self.assertNotIn("ThreadPoolExecutor", self.text)
        self.assertNotIn("torch.cuda.Stream", self.text)
        self.assertIn("pipe_edit = copy.deepcopy(pipe)", self.text)

    def test_paired_batch_generates_original_then_edit(self):
        functions = {
            node.name: node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("generate_paired_batch", functions)
        calls = [
            node
            for node in ast.walk(functions["generate_paired_batch"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "flux_generate"
        ]
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            ast.unparse(calls[0].keywords[-1].value),
            "shared_latents",
        )
        self.assertEqual(
            ast.unparse(calls[1].keywords[-1].value),
            "shared_latents",
        )


if __name__ == "__main__":
    unittest.main()
