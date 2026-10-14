"""Merge ESM2-decoded effects with control-coexpression inference.

Genes with an ESM2 vector are decoded directly.  For panel genes without one,
the imputer uses only their context-matched control expression to select
co-expressed ESM2-covered genes, then transfers the *predicted effect* (not
the baseline count) from those neighbours.  Counts are always reconstructed
from each gene's own matched-control expression downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class ControlCoexpressionEffectImputer:
    known_indices: np.ndarray
    unknown_indices: np.ndarray
    neighbour_indices: np.ndarray
    neighbour_weights: np.ndarray

    @classmethod
    def fit(
        cls,
        control_log_expression: np.ndarray,
        esm2_known: np.ndarray,
        neighbours: int = 16,
        min_correlation: float = 0.05,
    ) -> "ControlCoexpressionEffectImputer":
        """Fit per-context gene neighbours from control cells only.

        ``control_log_expression`` is [sampled control cells, panel genes].
        Sampling controls is permitted because this estimates a baseline
        coexpression graph; perturbation outcome cells never enter this fit.
        """
        x=np.asarray(control_log_expression,dtype=np.float32)
        esm2_known=np.asarray(esm2_known,dtype=bool)
        if x.ndim != 2 or x.shape[1] != len(esm2_known) or not np.isfinite(x).all() or neighbours <= 0:
            raise ValueError("Invalid control matrix, known-gene mask, or neighbour count")
        known=np.flatnonzero(esm2_known); unknown=np.flatnonzero(~esm2_known)
        if not len(known): raise ValueError('No ESM2-covered panel genes')
        if not len(unknown): return cls(known,unknown,np.empty((0,0),np.int64),np.empty((0,0),np.float32))
        # Standardize gene vectors across control cells: cosine becomes Pearson r.
        g=x.T; g=(g-g.mean(1,keepdims=True))/np.maximum(g.std(1,keepdims=True),1e-6)
        g/=np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-6)
        k=min(neighbours,len(known)); nn=NearestNeighbors(n_neighbors=k,metric='cosine').fit(g[known])
        distances, local=nn.kneighbors(g[unknown]); corr=np.clip(1.0-distances,0.0,None)
        corr[corr<min_correlation]=0.0
        # A row with no reliable neighbour yields a conservative null effect.
        weights=corr/np.maximum(corr.sum(1,keepdims=True),1e-8)
        return cls(known,unknown,local.astype(np.int64),weights.astype(np.float32))

    def impute_effects(self, direct_effects: np.ndarray) -> np.ndarray:
        """Return [perturbations, all panel genes] effects in panel order."""
        direct=np.asarray(direct_effects,dtype=np.float32)
        if direct.shape[-1] != len(self.known_indices): raise ValueError('direct_effects must be ESM2-covered genes only')
        out=np.zeros((direct.shape[0],len(self.known_indices)+len(self.unknown_indices)),dtype=np.float32)
        out[:,self.known_indices]=direct
        if len(self.unknown_indices):
            out[:,self.unknown_indices]=(direct[:,self.neighbour_indices]*self.neighbour_weights[None,:,:]).sum(-1)
        return out
