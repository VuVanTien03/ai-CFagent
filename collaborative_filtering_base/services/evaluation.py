"""Đánh giá Hit@K và NDCG@K trên file .inter (giữ nguyên ánh xạ chỉ số từ tập huấn luyện)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..domain.entities import IndexMaps
from ..domain.protocols import CollaborativeRecommender


def _rank_of_target(rec: list[tuple[int, float]], target_item_idx: int) -> int | None:
    """Thứ hạng 1-based của target trong danh sách gợi ý; None nếu không có trong list."""
    for pos, (j, _) in enumerate(rec, start=1):
        if j == target_item_idx:
            return pos
    return None


def _ndcg_at_k_binary_single_relevant(rank_1based: int | None, k: int) -> float:
    """NDCG@k với một mục liên quan nhị phân; IDCG@k = 1 (mục tiêu ở hạng 1)."""
    if rank_1based is None or rank_1based > k:
        return 0.0
    dcg = 1.0 / math.log2(rank_1based + 1)
    idcg = 1.0 / math.log2(2)
    return dcg / idcg


@dataclass(frozen=True)
class NDCGAtKResult:
    """Trung bình NDCG@1 … NDCG@max_k trên các dòng test hợp lệ."""

    evaluated: int
    max_k: int
    mean_ndcg_by_k: tuple[float, ...]

    def mean_ndcg_at(self, k: int) -> float:
        if k < 1 or k > self.max_k:
            raise ValueError(f"k phải trong [1, {self.max_k}]")
        return self.mean_ndcg_by_k[k - 1]


@dataclass(frozen=True)
class HitRateResult:
    """Kết quả đánh giá đơn giản trên tập test."""

    evaluated: int
    hits: int
    k: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.evaluated if self.evaluated else 0.0


@dataclass(frozen=True)
class HitAtKResult:
    """Hit@1 … Hit@max_k: số lần trúng trong top-k / tổng số dòng test hợp lệ."""

    evaluated: int
    max_k: int
    hits_by_k: tuple[int, ...]

    def hits_at(self, k: int) -> int:
        if k < 1 or k > self.max_k:
            raise ValueError(f"k phải trong [1, {self.max_k}]")
        return self.hits_by_k[k - 1]

    def hit_rate_at(self, k: int) -> float:
        return self.hits_at(k) / self.evaluated if self.evaluated else 0.0


class SequentialInterEvaluator:
    """
    Mỗi dòng test: (user, lịch sử, mục tiêu). Loại các item trong lịch sử khỏi gợi ý, kiểm tra mục tiêu trong top-K.

    Bỏ qua dòng nếu user hoặc item mục tiêu không có trong IndexMaps của train (cold start).
    """

    def __init__(self, field_separator: str = "\t", seq_separator: str = " ") -> None:
        self._field_separator = field_separator
        self._seq_separator = seq_separator

    def _iter_sequential_test_cases(
        self,
        inter_path: Path,
        maps: IndexMaps,
    ) -> Iterator[tuple[int, set[int], int]]:
        """Mỗi phần tử: (user_index, exclude_item_indices, target_item_index)."""
        lines = inter_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split(self._field_separator)
            if len(parts) < 3:
                continue
            uid, list_raw, target = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if uid not in maps.user_to_idx or target not in maps.item_to_idx:
                continue
            u = maps.user_index(uid)
            exclude: set[int] = set()
            for tok in list_raw.split(self._seq_separator):
                tok = tok.strip()
                if tok and tok in maps.item_to_idx:
                    exclude.add(maps.item_index(tok))
            yield u, exclude, maps.item_index(target)

    def hit_at_k(
        self,
        inter_path: Path,
        maps: IndexMaps,
        recommender: CollaborativeRecommender,
        k: int = 10,
    ) -> HitRateResult:
        hits = 0
        evaluated = 0
        for u, exclude, target_idx in self._iter_sequential_test_cases(inter_path, maps):
            evaluated += 1
            rec = recommender.recommend(u, k=k, exclude_item_indices=exclude)
            top_items = {j for j, _ in rec}
            if target_idx in top_items:
                hits += 1

        return HitRateResult(evaluated=evaluated, hits=hits, k=k)

    def hit_at_1_to_k(
        self,
        inter_path: Path,
        maps: IndexMaps,
        recommender: CollaborativeRecommender,
        max_k: int = 10,
    ) -> HitAtKResult:
        """
        Hit@1 … Hit@max_k: mục tiêu nằm trong top-k (theo thứ hạng từ một lần recommend top-max_k).
        """
        if max_k < 1:
            raise ValueError("max_k phải >= 1")

        hits = [0] * max_k
        evaluated = 0
        for u, exclude, target_idx in self._iter_sequential_test_cases(inter_path, maps):
            evaluated += 1
            rec = recommender.recommend(u, k=max_k, exclude_item_indices=exclude)
            rank = _rank_of_target(rec, target_idx)
            for ki in range(1, max_k + 1):
                if rank is not None and rank <= ki:
                    hits[ki - 1] += 1

        return HitAtKResult(evaluated=evaluated, max_k=max_k, hits_by_k=tuple(hits))

    def ndcg_at_1_to_k(
        self,
        inter_path: Path,
        maps: IndexMaps,
        recommender: CollaborativeRecommender,
        max_k: int = 10,
    ) -> NDCGAtKResult:
        """
        Trung bình NDCG@1 … NDCG@max_k (một mục tiêu liên quan / dòng; IDCG chuẩn hoá theo hạng 1).

        Gợi ý top-max_k dùng chung cho mọi k để thứ hạng mục tiêu nhất quán.
        """
        if max_k < 1:
            raise ValueError("max_k phải >= 1")

        sums = [0.0] * max_k
        evaluated = 0
        for u, exclude, target_idx in self._iter_sequential_test_cases(inter_path, maps):
            evaluated += 1
            rec = recommender.recommend(u, k=max_k, exclude_item_indices=exclude)
            rank = _rank_of_target(rec, target_idx)
            for ki in range(1, max_k + 1):
                sums[ki - 1] += _ndcg_at_k_binary_single_relevant(rank, ki)

        if evaluated == 0:
            mean = tuple(0.0 for _ in range(max_k))
        else:
            inv = 1.0 / evaluated
            mean = tuple(s * inv for s in sums)

        return NDCGAtKResult(evaluated=evaluated, max_k=max_k, mean_ndcg_by_k=mean)
