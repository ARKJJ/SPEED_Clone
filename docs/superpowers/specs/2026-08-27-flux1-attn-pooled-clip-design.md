# Flux1 Attention Editor With Optional Pooled CLIP Update

## Scope

Modify only `SPEED-main/FLux/Flux1/attn.py`. Preserve its existing text-side
attention Q/K/V edit path. Add an opt-in global conditioning edit for
`time_text_embed.text_embedder.linear_1`.

## Interface

Add these CLI options:

- `--edit_clip_global`: disabled by default; when enabled, include the pooled
  CLIP projection in the sparse checkpoint.
- `--clip_update_lambda`: optional ridge coefficient for the pooled CLIP solve;
  it defaults to `--update_lambda` when unspecified.

## Data Flow

For each target, anchor, and retain prompt, obtain the pooled CLIP embedding
from `pipeline.encode_prompt`. Treat each pooled embedding as one column.

The existing attention update continues to trace token-sequence module inputs
during diffusion. The pooled CLIP update is independent: it must not enter an
attention hook or be concatenated with T5 token inputs.

For the selected global linear module, collect target, anchor, and retain
pooled vectors. Build the same retain-projected closed-form update used by the
attention path, using its own target and retain matrices. Add the resulting
weight to that module and save it together with sparse attention weights.

## Validation And Errors

- Resolve the global module by exact suffix
  `time_text_embed.text_embedder.linear_1` under the transformer.
- Require exactly one matching weighted module.
- Require the module input width to equal the pooled embedding width.
- Raise a clear error if `encode_prompt` does not expose a rank-two pooled
  embedding compatible with the module.
- Do not change the existing target-token, anchor-token, or attention-retain
  selection semantics.

## Tests

Add static regression coverage that asserts:

- the global-edit flag defaults to disabled;
- the exact global module suffix is used;
- pooled embeddings are collected through `encode_prompt` rather than an
  attention hook;
- the global update is independently added to `edit_dict` only when enabled;
- width validation is present.

Static tests establish source wiring only. A GPU run must separately verify
nonzero checkpoint differences, successful sparse-weight loading, and paired
target/retain samples.
