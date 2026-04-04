"""Lớp façade ứng dụng — phụ thuộc vào CollaborativeRecommender (DIP), không phụ thuộc lớp cụ thể."""

from __future__ import annotations

from dataclasses import dataclass

from ..data.dataset_sources import CdsInterDatasetSource
from ..domain.protocols import CollaborativeRecommender
from .evaluation import HitRateResult, SequentialInterEvaluator


@dataclass
class RecommenderApplication:
    """
    Dịch vụ gom: tải train từ nguồn CDs (dense/sparse), fit mô hình, (tuỳ chọn) đánh giá test.

    Người gọi inject recommender để dễ đổi user-based / item-based mà không sửa lớp này.
    """

    source: CdsInterDatasetSource
    recommender: CollaborativeRecommender
    evaluator: SequentialInterEvaluator | None = None

    def __post_init__(self) -> None:
        if self.evaluator is None:
            self.evaluator = SequentialInterEvaluator()

    def fit_from_train(self) -> None:
        bundle = self.source.load_train()
        self.recommender.fit(bundle.matrix)

    def evaluate_test_hit_at_k(self, k: int = 10) -> HitRateResult:
        bundle = self.source.load_train()
        test_path = self.source.paths.test_inter
        return self.evaluator.hit_at_k(test_path, bundle.maps, self.recommender, k=k)
