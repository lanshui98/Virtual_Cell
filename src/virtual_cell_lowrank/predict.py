import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch
from .data import embeddings, check_counts
from .imputer import ControlCoexpressionEffectImputer
from .model import LowRank


def predict(a):
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {out}')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    emb = embeddings(a.embeddings)
    model = LowRank(ck['dim'], ck['rank'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    data = Path(a.controls)
    genes = pd.read_csv(data / 'gene_names.csv').iloc[:, 0].astype(str).to_numpy()
    targets = pd.read_csv(data / 'pert_counts.csv').iloc[:, 0].astype(str).to_numpy()
    genes, targets = np.asarray(genes, dtype=str), np.asarray(targets, dtype=str)
    for names in (genes, targets):
        if len(set(np.char.upper(names))) != len(names):
            raise ValueError('Panel gene and target identifiers must be unique ignoring case')
    missing = [t for t in targets if t.upper() not in emb]
    if missing:
        raise ValueError(f'Missing target embeddings: {missing}')
    lookup = {g.upper(): i for i, g in enumerate(genes)}
    try:
        idx = np.array([lookup[g.upper()] for g in ck['genes']])
    except KeyError as e:
        raise ValueError(f'Model gene missing from official panel: {e}') from e
    with torch.no_grad():
        direct = model(torch.tensor(np.stack([emb[t.upper()] for t in targets])), ck['basis']).numpy()
    if not np.isfinite(direct).all():
        raise ValueError('Model produced nonfinite effects')
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    manifest = {'checkpoint': str(a.checkpoint), 'contexts': {}, 'target_count': len(targets)}
    for c in 'ABC':
        controls = ad.read_h5ad(data / f'context_{c}.h5ad', backed='r')
        try:
            if not np.array_equal(controls.var_names.astype(str), genes):
                raise ValueError(f'Context {c} does not follow official gene order')
            if not controls.n_obs or a.control_sample < 2:
                raise ValueError('Need nonempty controls and control-sample >= 2')
            ii = np.sort(rng.choice(controls.n_obs, min(a.control_sample, controls.n_obs), replace=False))
            sample = controls.X[ii]
            sample = sample.toarray() if hasattr(sample, 'toarray') else np.asarray(sample)
            imp = ControlCoexpressionEffectImputer.fit(np.log1p(sample), np.isin(np.arange(len(genes)), idx))
            full = imp.impute_effects(direct[:, np.argsort(idx)])
            total = np.zeros(len(genes), dtype=np.float64)
            for start in range(0, controls.n_obs, 2048):
                block = controls.X[start:start + 2048]
                check_counts(block)
                total += np.asarray(block.sum(axis=0)).ravel()
            base = (total / controls.n_obs).astype(np.float32)
            mean = np.expm1(np.clip(np.log1p(base)[None, :] + full, 0, 20)).astype(np.float32)
            np.savez_compressed(out / f'context_{c}_lowrank_effects.npz', targets=targets, genes=genes,
                                log1p_effect=full, control_mean_counts=base, predicted_mean_counts=mean)
            manifest['contexts'][c] = {'control_cells': controls.n_obs, 'effect_abs_mean': float(np.abs(full).mean())}
        finally:
            controls.file.close()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
