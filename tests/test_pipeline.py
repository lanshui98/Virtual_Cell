import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from virtual_cell_lowrank.demo import demo
from virtual_cell_lowrank.imputer import ControlCoexpressionEffectImputer
from virtual_cell_lowrank.train import train
from virtual_cell_lowrank.validate import validate


def test_synthetic_pipeline_roundtrip(tmp_path):
    demo(tmp_path)
    assert validate(tmp_path / 'prediction.h5ad', 4)['cells'] == 60
    split = json.loads((tmp_path / 'model/split.json').read_text())
    with np.load(tmp_path / 'data/effects.npz') as raw:
        t = raw['targets']
        assert set(t[split['train_indices']]).isdisjoint(t[split['validation_indices']])
        expected = torch.tensor(raw['delta'][split['train_indices'], :7])
    checkpoint = torch.load(tmp_path / 'model/model.pt', weights_only=True)
    _, _, vh = torch.linalg.svd(expected, full_matrices=False)
    # Compare the projection, which is invariant to SVD sign choices.
    b = checkpoint['basis']
    torch.testing.assert_close(b.T @ b, vh[:2].T @ vh[:2])
    with pytest.raises(ValueError, match='cells-per-target'):
        validate(tmp_path / 'prediction.h5ad', 5)


def test_imputation_panel_order_and_null_fallback():
    x = np.arange(20, dtype='float32')
    controls = np.stack([x, x, np.ones(20)], axis=1)
    imputer = ControlCoexpressionEffectImputer.fit(controls, np.array([False, True, False]), neighbours=1)
    out = imputer.impute_effects(np.array([[0.4]], dtype='float32'))
    np.testing.assert_allclose(out, [[0.4, 0.4, 0]], atol=1e-5)


def test_invalid_rank_rejected(tmp_path):
    np.savez(tmp_path / 'effects.npz', genes=np.array(['G']), targets=np.array(['A', 'B']), delta=np.ones((2, 1)))
    np.savez(tmp_path / 'embeddings.npz', genes=np.array(['G', 'A', 'B']), vectors=np.ones((3, 2)))
    with pytest.raises(ValueError, match='rank exceeds'):
        train(SimpleNamespace(effects=tmp_path / 'effects.npz', embeddings=tmp_path / 'embeddings.npz',
            out=tmp_path / 'model', rank=16, steps=1, batch_size=32, seed=2, device='cpu',
            validation_fraction=0.5, train_all=False))


def test_prepare_does_not_mix_batch_controls(tmp_path):
    import anndata as ad
    import pandas as pd
    from scipy import sparse
    from virtual_cell_lowrank.data import prepare
    # Batch b has treatment only: its huge effect must not enter the label.
    obs = pd.DataFrame({'target': ['non-targeting', 'A', 'A'], 'batch': ['a', 'a', 'b']}, index=['0', '1', '2'])
    a = ad.AnnData(sparse.csr_matrix([[10.], [20.], [1000.]]), obs=obs, var=pd.DataFrame(index=['G']))
    a.write_h5ad(tmp_path / 'counts.h5ad')
    spec = [{'path': 'counts.h5ad', 'context': 'test', 'target_key': 'target', 'batch_key': 'batch', 'min_cells': 1}]
    (tmp_path / 'manifest.json').write_text(json.dumps(spec))
    prepare(tmp_path / 'manifest.json', tmp_path / 'effects.npz')
    with np.load(tmp_path / 'effects.npz') as z:
        np.testing.assert_allclose(z['delta'], [[np.log1p(20) - np.log1p(10)]])
        assert z['n_cells'].tolist() == [1]
