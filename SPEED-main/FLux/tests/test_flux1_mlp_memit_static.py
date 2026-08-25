import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "Flux1" / "mlp_memit.py"


class Flux1MlpMemitStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text()
        cls.tree = ast.parse(cls.text, filename=str(SOURCE))

    def test_flux1_specific_contract_and_simplified_structure(self):
        imported_names = {
            alias.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom) and node.module == "diffusers"
            for alias in node.names
        }
        function_names = {
            node.name for node in self.tree.body if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("DiffusionPipeline", imported_names)
        self.assertIn("pipeline.tokenizer_2", self.text)
        self.assertIn('FLUX1_MLP_SUFFIX = ".ff_context.net.2"', self.text)
        self.assertIn("args.residual_scale", self.text)
        self.assertNotIn("_parse_concepts", function_names)
        self.assertNotIn("_load_retain_texts", function_names)
        self.assertFalse(any(isinstance(node, ast.Try) for node in ast.walk(self.tree)))


if __name__ == "__main__":
    unittest.main()
