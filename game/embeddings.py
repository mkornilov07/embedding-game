"""Utilities for postprocessing and storing embedding vectors."""

import numpy as np


def apply_abtt(model):
    """Apply All-but-the-Top postprocessing to a Gensim KeyedVectors model."""
    vectors = model.vectors.astype(np.float32)
    centered = vectors - vectors.mean(axis=0)
    n_components = max(1, vectors.shape[1] // 100)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    top_components = components[:n_components]
    projection = (centered @ top_components.T) @ top_components
    model.vectors = (centered - projection).astype(vectors.dtype)
    if hasattr(model, "norms"):
        model.norms = None
    return model
