"""Độ tương đồng cosine trên vector hàng (user) hoặc cột (item) — thay thế được (OCP)."""

from __future__ import annotations

from scipy import sparse as sp

from .sparse_normalization import col_l2_normalize_csr, row_l2_normalize_csr


class SparseCosineSimilarity:
    """
    Cosine similarity qua tích vô hướng trên vector đã chuẩn hóa L2.

    - axis=0: ma trận (n_users, n_users) giữa các user.
    - axis=1: ma trận (n_items, n_items) giữa các item.
    """

    def pairwise_similarity(self, matrix: sp.csr_matrix, axis: int) -> sp.csr_matrix:
        if axis == 0:
            ru = row_l2_normalize_csr(matrix)
            return ru @ ru.transpose()
        if axis == 1:
            ci = col_l2_normalize_csr(matrix)
            return ci.transpose() @ ci
        raise ValueError("axis phải là 0 (user) hoặc 1 (item)")
