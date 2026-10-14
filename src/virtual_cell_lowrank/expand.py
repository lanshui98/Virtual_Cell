#!/usr/bin/env python3
"""Create a VCC-shaped raw-count submission from low-rank A/B/C effects.

For every context and target, this samples 400 real matched control cells across
library-size strata and applies the predicted log1p effect to each cell's own
counts. Poisson sampling supplies count-level variation while preserving the
control cell's sparse support and library-size heterogeneity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
from scipy import sparse


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--effects-dir", type=Path, required=True)
    p.add_argument("--controls-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cells-per-target", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--library-strata", type=int, default=10)
    return p.parse_args()


def nullable_strings(parent: h5py.Group, name: str, values: np.ndarray) -> None:
    dataset = parent.create_dataset(name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype("utf-8"))
    dataset.attrs.update({"encoding-type": "string-array", "encoding-version": "0.2.0"})


def write_categorical(parent: h5py.Group, name: str, codes: np.ndarray, categories: np.ndarray) -> None:
    group = parent.create_group(name)
    group.attrs.update({"encoding-type": "categorical", "encoding-version": "0.2.0", "ordered": False})
    dataset = group.create_dataset("codes", data=codes.astype(np.int32, copy=False))
    dataset.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
    nullable_strings(group, "categories", categories)


def stratified_control_indices(library_sizes: np.ndarray, n: int, strata: int, rng: np.random.Generator) -> np.ndarray:
    """Sample controls across library-size quantiles, preserving their mix."""
    order = np.argsort(library_sizes, kind="stable")
    if not len(order) or strata <= 0:
        raise ValueError("Need nonempty controls and positive strata")
    chunks = np.array_split(order, min(strata, len(order)))
    base, remainder = divmod(n, len(chunks))
    selected = []
    for i, chunk in enumerate(chunks):
        take = base + (i < remainder)
        selected.append(rng.choice(chunk, size=take, replace=take > len(chunk)))
    return np.sort(np.concatenate(selected))


def main(a=None) -> None:
    a = args() if a is None else a
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output}")
    if a.cells_per_target <= 0:
        raise ValueError("--cells-per-target must be positive")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    loaded = [np.load(a.effects_dir / f"context_{c}_lowrank_effects.npz", allow_pickle=False) for c in "ABC"]
    targets = loaded[0]["targets"].astype(str)
    genes = loaded[0]["genes"].astype(str)
    if any(not np.array_equal(z["targets"].astype(str), targets) or not np.array_equal(z["genes"].astype(str), genes) for z in loaded[1:]):
        raise ValueError("Context effect target/gene axes disagree")

    n_rows = 3 * len(targets) * a.cells_per_target
    rng = np.random.default_rng(a.seed)
    with h5py.File(a.output, "w") as out:
        out.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
        x = out.create_group("X")
        x.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": np.array([n_rows, len(genes)])})
        data = x.create_dataset("data", (0,), maxshape=(None,), dtype="f4", chunks=True, compression="gzip", compression_opts=1)
        indices = x.create_dataset("indices", (0,), maxshape=(None,), dtype="i4", chunks=True, compression="gzip", compression_opts=1)
        indptr = x.create_dataset("indptr", (n_rows + 1,), dtype="i8")
        indptr[0] = 0
        row, nnz = 0, 0
        summary: dict[str, object] = {"seed": a.seed, "cells_per_target": a.cells_per_target, "library_strata": a.library_strata, "contexts": {}}

        for context, effect_file in zip("ABC", loaded):
            controls = ad.read_h5ad(a.controls_dir / f"context_{context}.h5ad", backed="r")
            if controls.n_vars != len(genes) or not np.array_equal(controls.var_names.astype(str), genes):
                raise ValueError(f"Context {context} controls do not match effect-gene axis")
            library_sizes = np.concatenate([np.asarray(controls.X[s:s + 2048].sum(axis=1)).ravel() for s in range(0, controls.n_obs, 2048)])
            effects = effect_file["log1p_effect"].astype(np.float32, copy=False)
            if effects.shape != (len(targets), len(genes)) or not np.isfinite(effects).all():
                raise ValueError("Invalid effect shape or nonfinite effects")
            before_nnz, before_row = nnz, row
            for target_i, effect in enumerate(effects):
                control_rows = stratified_control_indices(library_sizes, a.cells_per_target, a.library_strata, rng)
                base = controls.X[control_rows]
                base = base.to_memory() if hasattr(base, "to_memory") else base
                base = sparse.csr_matrix(base, dtype=np.float32)
                base.eliminate_zeros()
                if not np.isfinite(base.data).all() or (base.data < 0).any() or not np.array_equal(base.data, np.rint(base.data)):
                    raise ValueError("Controls must contain raw integer counts")
                # delta was learned in log1p-count space.  Applying it only to
                # observed control entries avoids inventing dense expression from
                # a small average effect, and Poisson draws restore count noise.
                scaled = base.copy()
                scaled.data = np.expm1(np.clip(np.log1p(scaled.data) + effect[scaled.indices], 0, np.log1p(1_000_000.0)))
                scaled.data = np.clip(scaled.data, 0.0, 1_000_000.0)
                scaled.data = rng.poisson(scaled.data).astype(np.float32)
                scaled.eliminate_zeros()
                count = scaled.nnz
                data.resize((nnz + count,))
                indices.resize((nnz + count,))
                data[nnz:nnz + count] = scaled.data
                indices[nnz:nnz + count] = scaled.indices
                indptr[row + 1:row + a.cells_per_target + 1] = nnz + scaled.indptr[1:]
                nnz += count
                row += a.cells_per_target
                if (target_i + 1) % 25 == 0 or target_i + 1 == len(targets):
                    print(f"{context}: {target_i + 1}/{len(targets)} targets; {row:,}/{n_rows:,} cells", flush=True)
            summary["contexts"][context] = {
                "control_cells": int(controls.n_obs),
                "library_size_mean": float(library_sizes.mean()),
                "output_cells": row - before_row,
                "output_nnz": nnz - before_nnz,
            }
            controls.file.close()

        obs = out.create_group("obs")
        obs.attrs.update({"encoding-type": "dataframe", "encoding-version": "0.2.0", "_index": "_index", "column-order": np.asarray(["target_gene", "context"], dtype=h5py.string_dtype("utf-8"))})
        nullable_strings(obs, "_index", np.arange(n_rows).astype(str))
        write_categorical(obs, "target_gene", np.tile(np.repeat(np.arange(len(targets)), a.cells_per_target), 3), targets)
        write_categorical(obs, "context", np.repeat(np.arange(3), len(targets) * a.cells_per_target), np.asarray(["A", "B", "C"]))
        var = out.create_group("var")
        var.attrs.update({"encoding-type": "dataframe", "encoding-version": "0.2.0", "_index": "_index", "column-order": np.asarray([], dtype=h5py.string_dtype("utf-8"))})
        nullable_strings(var, "_index", genes)
        for key in ("layers", "obsm", "obsp", "uns", "varm", "varp"):
            group = out.create_group(key)
            group.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
        out.attrs["expanded_from"] = str(a.effects_dir)
        out.attrs["expansion_method"] = "matched_control_log1p_effect_poisson_sparse_v1"
        out.attrs["stored_nnz"] = nnz
    (a.output.with_suffix(".expansion_manifest.json")).write_text(json.dumps(summary, indent=2))
    print(f"wrote {a.output} with {n_rows:,} cells and {nnz:,} nonzeros", flush=True)


if __name__ == "__main__":
    main()
