"""Lọc cộng tác dựa item: profile người dùng là tổng có trọng số của vector item tương đồng."""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp

from ..domain.protocols import UserItemSimilarity
from .cosine_similarity import SparseCosineSimilarity


class ItemBasedCollaborativeFiltering:
    """
    Item-item CF: với user u, vector điểm = tổng_j ( R[u,j] * top-K cột sim(j, *) ).

    Chuẩn hóa theo tổng trọng số tương đồng đóng góp cho từng cột (trung bình có trọng số).
    """

    def __init__(
        self,
        neighbor_k: int = 40,
        similarity: UserItemSimilarity | None = None,
    ) -> None:
        self._neighbor_k = neighbor_k
        self._similarity = similarity or SparseCosineSimilarity()
        self._r: sp.csr_matrix | None = None
        self._sim_items: sp.csr_matrix | None = None

    def fit(self, matrix: sp.csr_matrix) -> None:
        self._r = matrix.tocsr()
        self._sim_items = self._similarity.pairwise_similarity(self._r, axis=1).tocsr()
        self._sim_items.setdiag(0)
        self._sim_items.eliminate_zeros()

    def recommend(
        self,
        user_index: int,
        k: int,
        exclude_item_indices: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if self._r is None or self._sim_items is None:
            raise RuntimeError("Gọi fit() trước khi recommend().")

        exclude = exclude_item_indices or set()
        ru = self._r.getrow(user_index)
        if ru.nnz == 0:
            return self._fallback_popular(k, exclude)

        liked = ru.indices
        liked_w = ru.data.astype(np.float64)
        acc = np.zeros(self._r.shape[1], dtype=np.float64)
        wsum = np.zeros(self._r.shape[1], dtype=np.float64)

        for j, wj in zip(liked, liked_w):
            row = self._sim_items.getrow(j)
            if row.nnz == 0:
                continue
            order = np.argsort(-row.data)[: self._neighbor_k]
            for t in order:
                jj = row.indices[t]
                s = row.data[t]
                acc[jj] += float(wj) * float(s)
                wsum[jj] += abs(float(s))

        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)

        for j in exclude:
            if 0 <= j < scores.shape[0]:
                scores[j] = -np.inf

        top = np.argsort(-scores)[:k]
        out = [(int(j), float(scores[j])) for j in top if scores[j] > -np.inf and scores[j] > 0]
        if not out:
            return self._fallback_popular(k, exclude)
        return out

    def _fallback_popular(self, k: int, exclude: set[int]) -> list[tuple[int, float]]:
        assert self._r is not None
        pop = np.asarray(self._r.sum(axis=0)).ravel()
        for j in exclude:
            if 0 <= j < pop.shape[0]:
                pop[j] = -1.0
        top = np.argsort(-pop)[:k]
        return [(int(j), float(pop[j])) for j in top if pop[j] >= 0]
