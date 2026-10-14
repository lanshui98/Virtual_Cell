# Reconstruction provenance

Reconstructed on 2026-10-14 from retained local Codex session records dated 2026-09-04 and 2026-09-06, plus the 2026-10-05 deletion inventory. Private conversations, absolute workstation paths, credentials and raw tool outputs are not bundled.

## Recovered components

- `train_lowrank_effect_basis.py`: LowRank architecture, uncentered training-only SVD, four-part objective and AdamW settings.
- `train_lowrank_effect_full.py`: rank 16, seed 2, 15 optimizer updates, random minibatches of up to 32 rows.
- `output_extension.py`: context-control Pearson/cosine nearest-neighbor effect transfer, 16 neighbors and correlation cutoff 0.05.
- `generate_lowrank_abc_effects.py`: target prediction, exact panel alignment, 1,024 sampled controls/context, imputation and mean-count reconstruction.
- `expand_lowrank_abc_to_vcc_h5ad.py`: library-size-stratified sampling, sparse-support log1p effect application, Poisson counts and streaming CSR output.

## New or corrected code

- Portable installed package and explicit paths replace workstation-specific imports and hardcoded paths.
- `prepare` is a new portable aggregation implementation based on the recorded batch-matching definition. It is not the recovered historical data preprocessing script. It requires a prealigned panel and weights per-batch effects by perturbation cell counts.
- New target-level splits replace the deleted historical split JSON. Target identity is normalized to uppercase for splits; aliases still require explicit preprocessing.
- Validation always calls `eval()`. The recovered five-seed reproduction script omitted this and evaluated with dropout enabled; its best-over-step statistics are not directly reproducible here.
- “Epoch” is renamed “step” to match the original optimizer loop; the default remains 15 updates.
- Historical nullable-string and empty HDF5 attribute failures are addressed with standard AnnData string-array metadata.
- Backed sparse controls are read in blocks; sampled indices are sorted, dense controls are accepted and empty sampling strata are avoided.
- Additional checks reject missing target embeddings, misaligned axes, nonfinite effects, invalid ranks and noninteger controls. No silent zero-vector target substitution.
- Rates are clipped before exponentiation to avoid overflow. Positive effects are still applied only on observed sparse support, preserving the historical limitation.
- New synthetic demo and tests exercise preparation through H5AD readback. They establish software behavior only.

## Historical result status

The final recorded entry was `7jLs37MwSfUkywemE1ul`, model `esm2-lowrank-r16-controlpoisson-v1`, last observed in `scoring`. Later notes incorrectly inferred “not submitted” from missing local logs; the explicit accepted-entry/upload record is stronger evidence. No official final score was recovered.

Original data, checkpoints and split artifacts are unavailable. This is an executable reconstruction of the method, not a byte-for-byte recovery or a verified recreation of historical benchmark results.
