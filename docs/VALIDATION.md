# Local validation

Validated on 2026-10-14 using the existing Python environment, CPU execution.

- Editable package installation: passed.
- Four tests: passed (synthetic end-to-end H5AD roundtrip, training-only SVD and target isolation, coexpression/null fallback, invalid rank, and batch-control isolation).
- Installed CLI synthetic demo: passed; 60 cells, 8 genes, 5 targets across A/B/C.
- No real-data training, official validator or official scoring was performed.

Observed package versions (a tested environment, not a dependency lock):

| Package | Version |
|---|---|
| Python | 3.10.20 |
| NumPy | 2.2.6 |
| PyTorch | 2.5.1+cu124 |
| pandas | 2.3.3 |
| AnnData | 0.11.4 |
| SciPy | 1.15.3 |
| h5py | 3.16.0 |
| scikit-learn | 1.7.2 |
