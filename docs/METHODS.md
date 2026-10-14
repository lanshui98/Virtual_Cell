# Method

For a given context, target and batch, compute a pseudobulk effect from raw counts:

`delta_batch = log1p(mean(target_counts)) - log1p(mean(non_targeting_counts))`.

Only batches containing sufficient perturbation and matching control cells enter aggregation. Effects for a target across valid batches are weighted by perturbation cell counts. Controls from a different batch are never substituted.

During validation, hold out all rows of selected targets across every context. Compute the uncentered SVD exclusively on retained training rows. Fit an ESM2-to-module-coordinate MLP with a target-level amplitude gate. The normalization for local NMAE proxy is `mean(abs(pred - true)) / mean(abs(true))`; zero effect has value 1 when true effects are nonzero. This does not reproduce the competition's Wilcoxon DE metric, differential-gene selection or official scoring transformations.

For each official context, estimate a gene coexpression graph from sampled control log1p counts. Standardized gene vectors yield Pearson correlation via cosine distance. Transfer predicted effects from positive known-gene neighbors to missing output genes; leave effects at zero without reliable neighbors. Exact official gene order is enforced.

Sample each context's controls across ten library-size strata, with replacement if a stratum is too small. Apply the target's predicted effect to each sampled cell's own nonzero counts, then sample Poisson rates. Stream groups to a sparse H5AD with `obs.target_gene`, `obs.context` and official gene symbols in `var_names`.

## Scope of extrapolation

An unseen perturbation target can be evaluated if its ESM2 vector exists. This is not evidence that its biological response is accurate. Unknown output genes are imputed, not learned through an output-gene language-model decoder. New context adaptation uses the supplied controls; the network itself is target-only. Perturbation-induced new expression, state transitions, variance changes and zero inflation are not explicitly learned. No ZINB, diffusion model, State backbone or MAP-KG branch is used in the final rank-16 implementation.

## Experiment practice

Use fixed target splits, compare against the zero-effect baseline, report held-out errors separately from full-data fitting and assess multiple seeds. Basis rank and step count are validation choices. Neither synthetic demo outputs nor historical best-over-step proxy values establish official challenge performance.
