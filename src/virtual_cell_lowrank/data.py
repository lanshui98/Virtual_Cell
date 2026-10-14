"""Batch-matched raw-count aggregation. No downloaded data is bundled."""
import json
from pathlib import Path
import anndata as ad
import numpy as np
import torch


def embeddings(path):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            table = dict(zip(z['genes'].astype(str), z['vectors']))
    else:
        # Accept the historical torch dictionary format; use trusted local files.
        table = torch.load(path, map_location='cpu', weights_only=True)
    result = {}
    for gene, vector in table.items():
        key = str(gene).upper()
        vector = np.asarray(vector, dtype=np.float32)
        if key in result or vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f'Invalid or duplicate embedding: {gene}')
        result[key] = vector
    if not result or len({len(x) for x in result.values()}) != 1:
        raise ValueError('Embeddings must have a common, nonzero vector dimension')
    return result


def check_counts(x):
    values = x.data if hasattr(x, 'tocsr') else np.asarray(x)
    if not np.isfinite(values).all() or (values < 0).any() or not np.array_equal(values, np.rint(values)):
        raise ValueError('Expected finite nonnegative integer raw counts in X')


def prepare(manifest, output):
    """Aggregate within each batch, then cell-count-weight across valid batches."""
    manifest = Path(manifest)
    specs = json.loads(manifest.read_text())
    genes, rows, targets, contexts, weights = None, [], [], [], []
    for spec in specs:
        path = Path(spec['path'])
        if not path.is_absolute():
            path = manifest.parent / path
        a = ad.read_h5ad(path, backed='r')
        try:
            names = a.var[spec['gene_key']].astype(str).to_numpy() if spec.get('gene_key') else a.var_names.astype(str).to_numpy()
            names = np.asarray(names, dtype=str)
            if len(set(np.char.upper(names))) != len(names):
                raise ValueError('Gene symbols must be unique')
            if genes is None:
                genes = names
            if not np.array_equal(genes, names):
                raise ValueError('All sources must use the same gene axis; explicitly align panels first')
            obs = a.obs
            target = obs[spec['target_key']].astype(str).to_numpy()
            batch = obs[spec['batch_key']].astype(str).to_numpy()
            control = spec.get('control_label', 'non-targeting')
            min_cells = int(spec.get('min_cells', 5))
            accumulated = {}
            for b in np.unique(batch):
                ci = np.flatnonzero((batch == b) & (target == control))
                if len(ci) < min_cells:
                    continue
                cx = a.X[ci]
                check_counts(cx)
                cm = np.asarray(cx.mean(axis=0)).ravel()
                for t in np.unique(target[batch == b]):
                    if t == control:
                        continue
                    ti = np.flatnonzero((batch == b) & (target == t))
                    if len(ti) < min_cells:
                        continue
                    tx = a.X[ti]
                    check_counts(tx)
                    delta = np.log1p(np.asarray(tx.mean(axis=0)).ravel()) - np.log1p(cm)
                    accumulated.setdefault(t, []).append((delta, len(ti)))
            for t, pairs in sorted(accumulated.items()):
                w = np.array([n for _, n in pairs])
                rows.append(np.average(np.stack([d for d, _ in pairs]), axis=0, weights=w))
                targets.append(t)
                contexts.append(spec['context'])
                weights.append(w.sum())
        finally:
            a.file.close()
    if not rows:
        raise ValueError('No batch-matched perturbation/control groups passed min_cells')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, genes=genes, targets=np.array(targets), contexts=np.array(contexts),
                        delta=np.asarray(rows, dtype=np.float32), n_cells=np.array(weights))
    output.with_suffix('.json').write_text(json.dumps({'effect_space': 'log1p_mean_raw_count_delta',
        'aggregation': 'within-batch effects, weighted by perturbation cell count', 'n_rows': len(rows), 'sources': specs}, indent=2))
