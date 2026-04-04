"""Chuẩn hóa hàng/cột ma trận thưa phục vụ cosine (tiện ích dùng chung)."""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as spla


def row_l2_normalize_csr(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Chia mỗi hàng cho norm L2 (hàng 0-norm giữ nguyên 0)."""
    norms = spla.norm(matrix, axis=1)
    norms = np.asarray(norms).ravel().astype(np.float64)
    norms[norms == 0.0] = 1.0
    inv = sp.diags(1.0 / norms, shape=(matrix.shape[0], matrix.shape[0]))
    return inv @ matrix


def col_l2_normalize_csr(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Chia mỗi cột cho norm L2."""
    norms = spla.norm(matrix, axis=0)
    norms = np.asarray(norms).ravel().astype(np.float64)
    norms[norms == 0.0] = 1.0
    inv = sp.diags(1.0 / norms, shape=(matrix.shape[1], matrix.shape[1]))
    return matrix @ inv
