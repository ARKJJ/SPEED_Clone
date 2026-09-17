import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "Flux2" / "mlp_adversarial.py"
SAMPLE2 = Path(__file__).resolve().parents[1] / "Flux2" / "sample2.py"
SCRIPT = Path(__file__).resolve().parents[1] / "Flux2" / "scripts" / "eval_nudity_adversarial.sh"
FLUX2_EDITORS = [
    Path(__file__).resolve().parents[1] / "Flux2" / "mlp.py",
    Path(__file__).resolve().parents[1] / "Flux2" / "mlp_memit.py",
    Path(__file__).resolve().parents[1] / "Flux2" / "attn.py",
    Path(__file__).resolve().parents[1] / "Flux2" / "attn_memit.py",
]


class Flux2MlpAdversarialStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text()
        cls.sample2_text = SAMPLE2.read_text()
        cls.tree = ast.parse(cls.text, filename=str(SOURCE))
        cls.script_text = SCRIPT.read_text()

    def test_uses_all_matching_dual_block_text_mlp_modules(self):
        self.assertIn('FLUX2_MLP_SUFFIX = ".ff_context.linear_out"', self.text)
        self.assertIn('re.match(r"transformer_blocks\\.(\\d+)\\.", name)', self.text)
        self.assertNotIn("layer_start", self.text)
        self.assertNotIn("layer_end", self.text)

    def test_builds_adversarial_target_from_base_and_current_weights(self):
        self.assertIn("def _adversarial_inputs(", self.text)
        self.assertIn("base_weight", self.text)
        self.assertIn("current_weight", self.text)
        self.assertIn("torch.linalg.solve", self.text)

    def test_releases_the_base_pipeline_before_tracing_the_current_model(self):
        self.assertIn("base_weights = {", self.text)
        self.assertIn("del base_pipeline", self.text)
        self.assertIn("torch.cuda.empty_cache()", self.text)

    def test_reedits_adversarial_target_toward_empty_anchor(self):
        self.assertIn("adversarial_inputs @ adversarial_inputs.T", self.text)
        self.assertIn("empty_inputs @ adversarial_inputs.T", self.text)
        self.assertIn("_closed_form_update(", self.text)

    def test_retraces_nudity_and_empty_for_each_layer_but_retain_is_fixed(self):
        self.assertIn("for _layer_index, layer_modules in sorted(grouped_modules.items()):", self.text)
        self.assertIn("[target_concept]", self.text)
        self.assertIn("[\"\"]", self.text)
        self.assertIn("retain_inputs_by_module", self.text)
        self.assertIn("base_retain_inputs_by_module", self.text)
        loop_start = self.text.index("for _layer_index, layer_modules in sorted(grouped_modules.items()):")
        loop_body = self.text[loop_start:]
        self.assertNotIn("_trace_concepts(pipeline, retain_", loop_body)
        self.assertIn("_collect_retain_inputs(", self.text)
        self.assertIn("base_pipeline,", self.text)

    def test_evaluation_builds_ordinary_mlp_checkpoint_before_adversarial_edit(self):
        self.assertIn("--edited_ckpt", self.text)
        self.assertIn("mlp.py", self.script_text)
        self.assertIn("--anchor_concepts", self.script_text)
        self.assertIn('""', self.script_text)
        self.assertIn("mlp_adversarial.py", self.script_text)

    def test_evaluation_only_writes_mlp_and_adversarial_checkpoints(self):
        self.assertIn("MLP_CKPT", self.script_text)
        self.assertIn("ROBUST_CKPT", self.script_text)
        self.assertIn("Generated checkpoints:", self.script_text)
        self.assertNotIn("run_sampling", self.script_text)
        self.assertNotIn("sample2.py", self.script_text)
        self.assertNotIn("i2p_cal.py", self.script_text)
        self.assertNotIn("SAVE_ROOT_ORIGINAL", self.script_text)
        self.assertNotIn("SAVE_ROOT_MLP", self.script_text)
        self.assertNotIn("SAVE_ROOT_ADVERSARIAL", self.script_text)
        self.assertNotIn("--i2p_path", self.script_text)
        self.assertNotIn("combine", self.script_text)

    def test_sample2_supports_speed_i2p_rows_and_no_combine_output(self):
        self.assertIn('content in ["nudity", "i2p", "coco", "erase", "retain"]', self.sample2_text)
        self.assertIn('data["prompt"]', self.sample2_text)
        self.assertIn('data["sd_seed"]', self.sample2_text)
        self.assertNotIn("guidance_scale", self.sample2_text)
        self.assertNotIn("sd_guidance_scale", self.sample2_text)
        self.assertIn('args.i2p_path', self.sample2_text)
        self.assertNotIn("combine_images_horizontally", self.sample2_text)
        self.assertNotIn('os.path.join(save_path, "combine")', self.sample2_text)

    def test_flux2_mlp_selects_all_512_tokens_for_nudity_target_and_empty_anchor(self):
        mlp_text = (Path(__file__).resolve().parents[1] / "Flux2" / "mlp.py").read_text()
        self.assertIn("full_token_indices = list(range(max_sequence_length))", mlp_text)
        self.assertIn("def _uses_full_nudity_tokens(concept):", mlp_text)
        self.assertIn("concept: full_token_indices if _uses_full_nudity_tokens(concept)", mlp_text)
        self.assertIn("use_full_empty_anchor = any(", mlp_text)
        self.assertIn("concept: full_token_indices if concept == \"\" and use_full_empty_anchor", mlp_text)
        self.assertIn("retain_token_indices = {", mlp_text)
        self.assertIn("concept: full_token_indices if concept == \"\" and use_full_empty_anchor", mlp_text)
        self.assertIn("else concept_token_indices[concept]", mlp_text)

        adv_text = SOURCE.read_text()
        self.assertIn("full_token_indices = list(range(max_sequence_length))", adv_text)
        self.assertIn("target_token_indices = {target_concept: full_token_indices}", adv_text)
        self.assertIn('empty_token_indices = {"": full_token_indices}', adv_text)
        self.assertIn("retain_token_indices = {text: full_token_indices", adv_text)
        self.assertNotIn("list(range(1, max_sequence_length))", adv_text)

    def test_flux2_klein_scripts_do_not_configure_guidance_scale(self):
        scripts = list(SCRIPT.parent.glob("*.sh"))
        for script in scripts:
            text = script.read_text()
            self.assertNotIn("GUIDANCE_SCALE", text, script.name)
            self.assertNotIn("--guidance_scale", text, script.name)

    def test_flux2_edit_traces_do_not_configure_guidance_scale(self):
        for path in (
            SOURCE,
            SOURCE.parent / "mlp.py",
        ):
            self.assertNotIn("guidance_scale", path.read_text(), str(path))


if __name__ == "__main__":
    unittest.main()
