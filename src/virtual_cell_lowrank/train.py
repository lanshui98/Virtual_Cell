import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from .data import embeddings
from .model import LowRank, effect_loss, nmae_proxy


def train(a):
    if a.steps <= 0 or a.batch_size <= 0 or a.rank <= 0:
        raise ValueError('steps, batch-size and rank must be positive')
    if not a.train_all and not 0 < a.validation_fraction < 1:
        raise ValueError('validation-fraction must be between zero and one')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty run directory: {out}')
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    emb = embeddings(a.embeddings)
    with np.load(a.effects, allow_pickle=False) as raw:
        targets, genes = raw['targets'].astype(str), raw['genes'].astype(str)
        y = np.asarray(raw['delta'], dtype=np.float32)
    if y.shape != (len(targets), len(genes)) or not np.isfinite(y).all():
        raise ValueError('Invalid effect matrix dimensions or values')
    tok = np.array([t.upper() in emb for t in targets])
    gok = np.array([g.upper() in emb for g in genes])
    original_rows = np.flatnonzero(tok)
    missing_targets = sorted(set(targets[~tok]))
    targets, genes, y = targets[tok], genes[gok], y[tok][:, gok]
    if not len(targets) or not len(genes):
        raise ValueError('No embedding-covered targets or genes')
    upper = np.char.upper(targets)
    unique = np.unique(upper)
    if a.train_all:
        tr, va = np.arange(len(targets)), np.array([], dtype=int)
    else:
        if len(unique) < 2:
            raise ValueError('Need at least two distinct targets for held-out validation')
        n = min(len(unique) - 1, max(1, int(round(len(unique) * a.validation_fraction))))
        held = rng.choice(unique, n, replace=False)
        tr, va = np.flatnonzero(~np.isin(upper, held)), np.flatnonzero(np.isin(upper, held))
    device = torch.device(a.device)
    truth = torch.tensor(y, device=device)
    x = torch.tensor(np.stack([emb[t.upper()] for t in targets]), device=device)
    if a.rank > min(len(tr), len(genes)):
        raise ValueError('rank exceeds the training matrix dimensions')
    # Historical SVD was uncentered; retain it. Only training labels enter basis.
    _, _, vh = torch.linalg.svd(truth[tr], full_matrices=False)
    basis = vh[:a.rank]
    model = LowRank(x.shape[1], a.rank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'split.json').write_text(json.dumps({'train_indices': original_rows[tr].tolist(),
        'validation_indices': original_rows[va].tolist(), 'validation_targets': sorted(set(targets[va]))}, indent=2))
    best, best_step = float('inf'), None
    with (out / 'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'train_loss', 'validation_nmae_proxy'])
        writer.writeheader()
        for step in range(1, a.steps + 1):
            model.train()
            rows = rng.choice(tr, min(a.batch_size, len(tr)), replace=False)
            pred = model(x[rows], basis)
            loss = effect_loss(pred, truth[rows], torch.sigmoid(model.gate(x[rows])))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.eval()
            score = None
            if len(va):
                with torch.no_grad():
                    score = nmae_proxy(model(x[va], basis), truth[va])
            writer.writerow({'step': step, 'train_loss': float(loss.detach()), 'validation_nmae_proxy': score})
            if score is None or score < best:
                best_step = step
                if score is not None:
                    best = score
                torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'basis': basis.cpu(), 'genes': genes.tolist(), 'rank': a.rank, 'dim': x.shape[1],
                    'step': step, 'seed': a.seed, 'effect_space': 'log1p_mean_raw_count_delta',
                    'validation_nmae_proxy': score}, out / 'model.pt')
    summary = {'rank': a.rank, 'steps': a.steps, 'batch_size': a.batch_size, 'seed': a.seed,
        'train_all': a.train_all, 'n_rows': len(targets), 'n_targets': len(unique), 'n_genes': len(genes),
        'best_step': best_step, 'best_validation_nmae_proxy': best if len(va) else None,
        'missing_embedding_targets': missing_targets, 'missing_embedding_gene_count': int((~gok).sum()),
        'torch_version': torch.__version__, 'numpy_version': np.__version__, 'device': str(device)}
    (out / 'manifest.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
