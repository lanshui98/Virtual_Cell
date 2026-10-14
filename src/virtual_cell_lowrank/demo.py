import json
from pathlib import Path
from types import SimpleNamespace
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from .data import prepare
from .train import train
from .predict import predict
from .expand import main as expand
from .validate import validate


def demo(output):
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Demo output must be empty')
    data = root / 'data'
    data.mkdir(parents=True)
    rng = np.random.default_rng(42)
    genes = np.array([f'G{i}' for i in range(8)])
    targets = genes[:5]
    pd.DataFrame({'gene': genes}).to_csv(data / 'gene_names.csv', index=False)
    pd.DataFrame({'target': targets, 'count': 400}).to_csv(data / 'pert_counts.csv', index=False)
    np.savez(data / 'embeddings.npz', genes=genes[:7], vectors=rng.normal(size=(7, 12)).astype('float32'))
    sources = []
    for c in 'ABC':
        base = rng.poisson(3, (30, 8)).astype('float32')
        ad.AnnData(sparse.csr_matrix(base), var=pd.DataFrame(index=genes)).write_h5ad(data / f'context_{c}.h5ad')
        matrix, labels = [], []
        for batch in ['b1', 'b2']:
            for target in ['non-targeting', *targets]:
                x = rng.poisson(3 if target == 'non-targeting' else 4, (8, 8)).astype('float32')
                matrix.append(x)
                labels.extend([(target, batch)] * 8)
        obs = pd.DataFrame(labels, columns=['target', 'batch'])
        obs.index = obs.index.astype(str)
        path = data / f'train_{c}.h5ad'
        ad.AnnData(sparse.csr_matrix(np.vstack(matrix)), obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(path)
        sources.append({'path': path.name, 'context': c, 'target_key': 'target', 'batch_key': 'batch'})
    (data / 'sources.json').write_text(json.dumps(sources))
    prepare(data / 'sources.json', data / 'effects.npz')
    train(SimpleNamespace(effects=data / 'effects.npz', embeddings=data / 'embeddings.npz', out=root / 'model',
        rank=2, steps=3, batch_size=8, seed=2, device='cpu', validation_fraction=0.2, train_all=False))
    predict(SimpleNamespace(checkpoint=root / 'model/model.pt', embeddings=data / 'embeddings.npz',
        controls=data, out=root / 'effects', control_sample=20, seed=0))
    expand(SimpleNamespace(effects_dir=root / 'effects', controls_dir=data, output=root / 'prediction.h5ad',
        cells_per_target=4, library_strata=10, seed=20260907))
    validate(root / 'prediction.h5ad', 4)
