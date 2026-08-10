# FLUX Multi-Concept Step Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `CE_Flux.py` aggregate FLUX traces per concept across timesteps before combining multiple concepts, instead of concatenating every concept-step column directly.

**Architecture:** Keep the existing trace collection and closed-form update intact. Add a small aggregation pass inside `edit_model()` that collapses the timestep axis for target, anchor, and retain traces, then average target/anchor concept representations across concepts while leaving retain samples as a pooled covariance set.

**Tech Stack:** Python, PyTorch, Diffusers, safetensors

---

### Task 1: Collapse timestep columns before solving

**Files:**
- Modify: `SPEED-main/FLux/CE_Flux.py`

- [ ] **Step 1: Reproduce the current multi-concept layout mentally from the trace shape**

The stored trace for one concept is built from `input_steps[:, batch_index, :, :].reshape(-1, input_steps.shape[-1]).T`, so the timestep axis is already flattened into the column axis.

- [ ] **Step 2: Add a timestep-mean reshape in `edit_model()`**

Use the stored column count and `args.trace_num_steps` to recover the trace tensor and average over timesteps before any target/anchor concatenation.

```python
target_inputs = target_inputs.T.reshape(args.trace_num_steps, -1, target_inputs.shape[0]).mean(dim=0).T
anchor_inputs = anchor_inputs.T.reshape(args.trace_num_steps, -1, anchor_inputs.shape[0]).mean(dim=0).T
```

- [ ] **Step 3: Replace target/anchor concatenation with concept averaging**

After timestep averaging, stack the per-concept tensors and average them so a multi-concept edit uses one shared target tensor and one shared anchor tensor.

```python
target_inputs = torch.stack(target_inputs, dim=0).mean(dim=0).to(module.weight.device, torch.float32)
anchor_inputs = torch.stack(anchor_inputs, dim=0).mean(dim=0).to(module.weight.device, torch.float32)
```

- [ ] **Step 4: Keep retain statistics pooled at the concept level**

Collapse retain traces over timesteps before appending them to `retain_inputs_by_module`, then keep the existing covariance and null-space projector solve.

```python
retain_inputs_by_module[module_name].append(
    retain_traces[concept][module_name]["inputs"].T.reshape(args.trace_num_steps, -1, retain_traces[concept][module_name]["inputs"].shape[0]).mean(dim=0).T
)
```

- [ ] **Step 5: Run a syntax check**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m py_compile SPEED-main/FLux/CE_Flux.py
```

Expected: no syntax errors.

- [ ] **Step 6: Review the edited weight logging**

Confirm the existing `||delta||` print and final status print still fire once per edited module and once per run.

- [ ] **Step 7: Commit**

```bash
git add SPEED-main/FLux/CE_Flux.py docs/superpowers/plans/2026-08-10-flux-multi-concept-step-aggregation.md
git commit -m "feat: aggregate FLUX multi-concept traces by timestep"
```
