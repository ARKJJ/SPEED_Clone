# Flux1 MLP-MEMIT Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce redundant structure in `Flux1/mlp_memit.py` so it follows `Flux2/mlp_memit.py` more closely without removing Flux1-specific model paths or `residual_scale`.

**Architecture:** Keep the shared MEMIT pipeline unchanged: collect anchor, retain, and target traces; use the final MLP output residual; distribute it with `remaining_counts`; solve the retain-projected update; save sparse weights. Simplify only source organization, inline the retain/concept parsing used once, and use Flux2-style compact tracing and update code. Flux1 continues to use `DiffusionPipeline`, `tokenizer_2`, `.ff_context.net.2`, sequence length `256`, and configurable trace guidance.

**Tech Stack:** Python, `ast`/`unittest` static test, PyTorch, Diffusers, pandas, safetensors.

---

### Task 1: Add the structural regression test

**Files:**
- Create: `SPEED-main/FLux/tests/test_flux1_mlp_memit_static.py`
- Test: `SPEED-main/FLux/tests/test_flux1_mlp_memit_static.py`

- [x] **Step 1: Write the failing test**

Create a dependency-free `unittest` that parses `Flux1/mlp_memit.py` with `ast`:

```python
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
```

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
python3 SPEED-main/FLux/tests/test_flux1_mlp_memit_static.py
```

Expected result before the refactor: failure because the current file still contains `_parse_concepts`, `_load_retain_texts`, and the `try/finally` tracing block.

### Task 2: Compress Flux1 tracing and parsing structure

**Files:**
- Modify: `SPEED-main/FLux/Flux1/mlp_memit.py:1-166,318-381`

- [x] **Step 1: Match Flux2 import and tracing style**

Keep only imports needed by the final file. Preserve `DiffusionPipeline`, pandas, PyTorch, and safetensors. Remove the Diffusers logging import and explicit progress configuration. Rewrite `_trace_concepts()` in the Flux2 compact style while retaining Flux1-specific `trace_guidance_scale`, output capture, and `tokenizer_2` behavior. Do not add `try/finally`; retain the direct handle removal after the pipeline call.

- [x] **Step 2: Keep the shared closed-form update and simplify its formatting**

Preserve covariance, SVD, null-space projector, `system`, and `torch.linalg.solve`. Only compress formatting and variable declarations; do not alter operands, transpose placement, dtype/device conversion, or threshold semantics.

- [x] **Step 3: Inline one-time retain loading and concept parsing**

Remove `_load_retain_texts()` and `_parse_concepts()`. In `__main__`, parse target concepts directly, read `args.retain_path` directly when present, load the requested heads, and apply the existing target-concept filter. Keep the current `residual_scale`, `trace_guidance_scale`, `trace_resolution`, and `max_sequence_length` CLI arguments.

- [x] **Step 4: Simplify `edit_model()` without changing its data flow**

Use Flux2-style compact module filtering, token-index dictionary construction, anchor/retain tracing, retain concatenation, per-module target tracing, residual construction, closed-form solve, and sparse checkpoint assembly. Preserve Flux1’s `.ff_context.net.2` selector and `remaining_counts` distribution. Do not modify `Flux2/mlp_memit.py`.

### Task 3: Verify the refactor

**Files:**
- Test: `SPEED-main/FLux/tests/test_flux1_mlp_memit_static.py`
- Verify: `SPEED-main/FLux/Flux1/mlp_memit.py`

- [x] **Step 1: Run the static test and verify GREEN**

Run:

```bash
python3 SPEED-main/FLux/tests/test_flux1_mlp_memit_static.py
```

Expected result: all structural assertions pass.

- [x] **Step 2: Compile the edited Python files**

Run:

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m py_compile SPEED-main/FLux/Flux1/mlp_memit.py SPEED-main/FLux/tests/test_flux1_mlp_memit_static.py
```

Expected result: exit code `0`. This checks syntax only; it does not load Diffusers or run a GPU trace.

- [x] **Step 3: Check the patch and scope**

Run:

```bash
git diff --check
git status --short SPEED-main/FLux/Flux1/mlp_memit.py SPEED-main/FLux/Flux2/mlp_memit.py SPEED-main/FLux/tests/test_flux1_mlp_memit_static.py
```

Expected result: no whitespace errors, Flux2 remains unchanged, and only the Flux1 editor plus its static test are modified by implementation.
