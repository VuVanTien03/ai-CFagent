"""
Matrix Factorization với học BPR (Rendle et al.) cho phản hồi ngầm.

Điểm dự đoán: <u_emb, i_emb>. Huấn luyện bằng SGD trên bộ ba (u, i, j)
với i là item tích cực, j là mẫu âm (người dùng chưa tương tác).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp


class BPRMatrixFactorization:
    def __init__(
        self,
        dim: int = 32,
        learning_rate: float = 0.05,
        reg: float = 0.01,
        n_epochs: int = 50,
        neg_samples_per_pos: int = 1,
        rng_seed: int = 42,
    ) -> None:
        self._dim = dim
        self._lr = learning_rate
        self._reg = reg
        self._n_epochs = n_epochs
        self._neg_per_pos = neg_samples_per_pos
        self._rng = np.random.default_rng(rng_seed)
        self._u: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._r: sp.csr_matrix | None = None
        self._user_pos: list[np.ndarray] | None = None
        self._user_item_set: list[set[int]] | None = None

    def fit(self, matrix: sp.csr_matrix) -> None:
        self._r = matrix.tocsr()
        n_users, n_items = self._r.shape
        scale = 1.0 / np.sqrt(self._dim)
        self._u = self._rng.normal(0.0, scale, (n_users, self._dim)).astype(np.float64)
        self._v = self._rng.normal(0.0, scale, (n_items, self._dim)).astype(np.float64)

        self._user_pos = []
        self._user_item_set = []
        for u in range(n_users):
            row = self._r.getrow(u)
            idx = row.indices.astype(np.int64)
            self._user_pos.append(idx)
            self._user_item_set.append(set(idx.tolist()))

        positives: list[tuple[int, int]] = []
        for u in range(n_users):
            for i in self._user_pos[u]:
                positives.append((u, int(i)))

        for _ in range(self._n_epochs):
            self._rng.shuffle(positives)
            assert self._user_item_set is not None
            for u, i in positives:
                pos_s = self._user_item_set[u]
                for _ in range(self._neg_per_pos):
                    j = int(self._rng.integers(0, n_items))
                    while j in pos_s:
                        j = int(self._rng.integers(0, n_items))
                    self._sgd_step(u, i, j)

    def _sgd_step(self, u: int, i: int, j: int) -> None:
        assert self._u is not None and self._v is not None
        x = float(np.dot(self._u[u], self._v[i] - self._v[j]))
        sig = 1.0 / (1.0 + np.exp(-x))
        g = (1.0 - sig)  # gradient của -log σ(x)

        u_vec = self._u[u].copy()
        vi = self._v[i].copy()
        vj = self._v[j].copy()

        self._u[u] += self._lr * (g * (vi - vj) - self._reg * self._u[u])
        self._v[i] += self._lr * (g * u_vec - self._reg * self._v[i])
        self._v[j] += self._lr * (-g * u_vec - self._reg * self._v[j])

    def recommend(
        self,
        user_index: int,
        k: int,
        exclude_item_indices: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if self._u is None or self._v is None or self._r is None or self._user_pos is None:
            raise RuntimeError("Gọi fit() trước khi recommend().")

        exclude = exclude_item_indices or set()
        scores = self._v @ self._u[user_index]

        if self._user_pos[user_index].size == 0:
            return self._fallback_popular(k, exclude)

        for j in exclude:
            if 0 <= j < scores.shape[0]:
                scores[j] = -np.inf

        top = np.argsort(-scores)[:k]
        out = [(int(j), float(scores[j])) for j in top if scores[j] > -np.inf]
        if not out:
            return self._fallback_popular(k, exclude)
        return out

    def _fallback_popular(self, k: int, exclude: set[int]) -> list[tuple[int, float]]:
        assert self._r is not None
        pop = np.asarray(self._r.sum(axis=0)).ravel().astype(np.float64)
        for j in exclude:
            if 0 <= j < pop.shape[0]:
                pop[j] = -1.0
        top = np.argsort(-pop)[:k]
        return [(int(j), float(pop[j])) for j in top if pop[j] >= 0]
