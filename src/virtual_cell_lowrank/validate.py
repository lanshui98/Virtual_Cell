"""Local structural/count validation, not the official VCC evaluator."""
import json
import anndata as ad
import numpy as np
from .data import check_counts


def validate(path, cells_per_target=400):
    a = ad.read_h5ad(path, backed='r')
    try:
        if not a.var_names.is_unique or not a.obs_names.is_unique:
            raise ValueError('Gene/cell identifiers must be unique')
        if set(a.obs['context'].astype(str)) != set('ABC'):
            raise ValueError('Expected exactly contexts A, B, C')
        counts = a.obs.groupby(['context', 'target_gene'], observed=True).size()
        targets = set(a.obs['target_gene'].astype(str))
        if len(counts) != 3 * len(targets) or not (counts == cells_per_target).all():
            raise ValueError('Every target must have exactly cells-per-target cells in every context')
        for s in range(0, a.n_obs, 2048):
            check_counts(a.X[s:s + 2048])
        result = {'cells': a.n_obs, 'genes': a.n_vars, 'targets': len(targets),
                  'cells_per_target': cells_per_target, 'local_validation': 'passed'}
    finally:
        a.file.close()
    print(json.dumps(result, indent=2))
    return result
