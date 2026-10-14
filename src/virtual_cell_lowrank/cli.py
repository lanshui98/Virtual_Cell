import argparse


def main():
    p = argparse.ArgumentParser(description='ESM2 low-rank effect pipeline (historical reconstruction)')
    commands = p.add_subparsers(dest='command', required=True)
    q = commands.add_parser('prepare', help='Build batch-matched effect labels from raw-count H5ADs')
    q.add_argument('--manifest', required=True)
    q.add_argument('--output', required=True)
    q = commands.add_parser('train', help='Fit a training-only SVD basis and target-conditioned predictor')
    q.add_argument('--effects', required=True)
    q.add_argument('--embeddings', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--rank', type=int, default=16)
    q.add_argument('--steps', type=int, default=15, help='Optimizer updates, NOT full dataset epochs')
    q.add_argument('--batch-size', type=int, default=32)
    q.add_argument('--seed', type=int, default=2)
    q.add_argument('--device', default='cpu')
    q.add_argument('--validation-fraction', type=float, default=0.2)
    q.add_argument('--train-all', action='store_true')
    q = commands.add_parser('predict', help='Generate full-panel A/B/C mean effects')
    for key in ['checkpoint', 'embeddings', 'controls', 'out']:
        q.add_argument('--' + key, required=True)
    q.add_argument('--control-sample', type=int, default=1024)
    q.add_argument('--seed', type=int, default=0)
    q = commands.add_parser('expand', help='Stream matched-control Poisson counts to H5AD')
    from pathlib import Path
    for key in ['effects-dir', 'controls-dir', 'output']:
        q.add_argument('--' + key, type=Path, required=True)
    q.add_argument('--cells-per-target', type=int, default=400)
    q.add_argument('--library-strata', type=int, default=10)
    q.add_argument('--seed', type=int, default=20260907)
    q = commands.add_parser('validate', help='Check local H5AD structure and raw counts')
    q.add_argument('--h5ad', required=True)
    q.add_argument('--cells-per-target', type=int, default=400)
    q = commands.add_parser('demo', help='Run synthetic end-to-end example; no downloads needed')
    q.add_argument('--out', required=True)
    a = p.parse_args()
    if a.command == 'prepare':
        from .data import prepare
        prepare(a.manifest, a.output)
    elif a.command == 'train':
        from .train import train
        train(a)
    elif a.command == 'predict':
        from .predict import predict
        predict(a)
    elif a.command == 'expand':
        from .expand import main as expand
        expand(a)
    elif a.command == 'validate':
        from .validate import validate
        validate(a.h5ad, a.cells_per_target)
    else:
        from .demo import demo
        demo(a.out)


if __name__ == '__main__':
    main()
