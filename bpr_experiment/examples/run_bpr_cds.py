"""
Huấn luyện BPR-MF trên CDs-100-user-dense và CDs-100-user-sparse; đánh giá Hit@1..10 và NDCG@1..10.

Chạy từ thư mục gốc repo:

    python -m bpr_experiment.examples.run_bpr_cds
"""

from __future__ import annotations

from collaborative_filtering_base.services.evaluation import SequentialInterEvaluator

from bpr_experiment.config.dataset_paths import CDsDatasetPaths
from bpr_experiment.data.cds_source import CdsInterDatasetSource
from bpr_experiment.models.bpr_mf import BPRMatrixFactorization


def _run_label(label: str, source: CdsInterDatasetSource) -> None:
    bundle = source.load_train()
    print(f"\n=== {label} ===")
    print(f"Train matrix: {bundle.matrix.shape[0]} users × {bundle.matrix.shape[1]} items, nnz={bundle.matrix.nnz}")

    model = BPRMatrixFactorization(
        dim=32,
        learning_rate=0.05,
        reg=0.02,
        n_epochs=80,
        neg_samples_per_pos=1,
        rng_seed=42,
    )
    model.fit(bundle.matrix)

    evaluator = SequentialInterEvaluator()
    test_path = source.paths.test_inter
    hit_multi = evaluator.hit_at_1_to_k(test_path, bundle.maps, model, max_k=10)
    hit_parts = [
        f"Hit@{k}={hit_multi.hit_rate_at(k):.4f} ({hit_multi.hits_at(k)}/{hit_multi.evaluated})"
        for k in range(1, hit_multi.max_k + 1)
    ]
    print(f"BPR-MF | {' | '.join(hit_parts)}")

    ndcg = evaluator.ndcg_at_1_to_k(test_path, bundle.maps, model, max_k=10)
    parts = [f"NDCG@{k}={ndcg.mean_ndcg_at(k):.4f}" for k in range(1, ndcg.max_k + 1)]
    print(f"BPR-MF | {' | '.join(parts)}")


def main() -> None:
    _run_label("CDs-100-user-dense", CdsInterDatasetSource(CDsDatasetPaths.cds_dense()))
    _run_label("CDs-100-user-sparse", CdsInterDatasetSource(CDsDatasetPaths.cds_sparse()))


if __name__ == "__main__":
    main()
