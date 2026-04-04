"""Lọc cộng tác dựa user: dự đoán qua tổng trọng số hành vi của láng giềng gần."""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp

from ..domain.protocols import UserItemSimilarity
from .cosine_similarity import SparseCosineSimilarity


class UserBasedCollaborativeFiltering:
    """
    User-user CF trên ma trận implicit nhị phân.

    Điểm dự đoán cho (u, i): tổng theo v của sim(u,v) * R[v,i], chuẩn hóa theo tổng |sim| của top-K láng giềng (bỏ chính u).
    """

    def __init__(
        self,
        neighbor_k: int = 40,
        similarity: UserItemSimilarity | None = None,
    ) -> None:
        self._neighbor_k = neighbor_k
        self._similarity = similarity or SparseCosineSimilarity()
        self._r: sp.csr_matrix | None = None
        self._sim_users: sp.csr_matrix | None = None

    def fit(self, matrix: sp.csr_matrix) -> None:
        self._r = matrix.tocsr()
        self._sim_users = self._similarity.pairwise_similarity(self._r, axis=0).tocsr()
        self._sim_users.setdiag(0)
        self._sim_users.eliminate_zeros()

    def recommend(
        self,
        user_index: int,
        k: int,
        exclude_item_indices: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if self._r is None or self._sim_users is None:
            raise RuntimeError("Gọi fit() trước khi recommend().")

        exclude = exclude_item_indices or set()
        row = self._sim_users.getrow(user_index)
        if row.nnz == 0:
            return self._fallback_popular(k, exclude)

        idx = row.indices
        val = row.data
        order = np.argsort(-val)[: self._neighbor_k]
        neigh_idx = idx[order]
        neigh_w = val[order].astype(np.float64)
        denom = np.sum(np.abs(neigh_w))
        if denom <= 0:
            return self._fallback_popular(k, exclude)

        acc = np.zeros(self._r.shape[1], dtype=np.float64)
        for v, w in zip(neigh_idx, neigh_w):
            acc += w * self._r.getrow(v).toarray().ravel()
        acc /= denom
        for j in exclude:
            if 0 <= j < acc.shape[0]:
                acc[j] = -np.inf

        top = np.argsort(-acc)[:k]
        return [(int(j), float(acc[j])) for j in top if acc[j] > -np.inf]

    def _fallback_popular(self, k: int, exclude: set[int]) -> list[tuple[int, float]]:
        assert self._r is not None
        pop = np.asarray(self._r.sum(axis=0)).ravel()
        for j in exclude:
            if 0 <= j < pop.shape[0]:
                pop[j] = -1.0
        top = np.argsort(-pop)[:k]
        return [(int(j), float(pop[j])) for j in top if pop[j] >= 0]
