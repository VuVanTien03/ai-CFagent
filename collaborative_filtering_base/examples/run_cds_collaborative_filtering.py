"""
Ví dụ chạy nhanh: so sánh Hit@K giữa CDs dense và sparse (cùng mã, khác nguồn dữ liệu).

Chạy từ thư mục AgentCF với PYTHONPATH gồm thư mục hiện tại:
    python -m collaborative_filtering_base.examples.run_cds_collaborative_filtering
"""

from __future__ import annotations

from collaborative_filtering_base.config.dataset_paths import CDsDatasetPaths
from collaborative_filtering_base.data.dataset_sources import CdsInterDatasetSource
from collaborative_filtering_base.models.item_based_cf import ItemBasedCollaborativeFiltering
from collaborative_filtering_base.models.user_based_cf import UserBasedCollaborativeFiltering
from collaborative_filtering_base.services.recommender_app import RecommenderApplication


def _run_label(label: str, source: CdsInterDatasetSource, k: int = 10) -> None:
    print(f"=== {label} | User-based CF ===")
    app_u = RecommenderApplication(source=source, recommender=UserBasedCollaborativeFiltering(neighbor_k=30))
    app_u.fit_from_train()
    r_u = app_u.evaluate_test_hit_at_k(k=k)
    print(f"Hit@{k}: {r_u.hits}/{r_u.evaluated} = {r_u.hit_rate:.4f}")

    print(f"=== {label} | Item-based CF ===")
    app_i = RecommenderApplication(source=source, recommender=ItemBasedCollaborativeFiltering(neighbor_k=30))
    app_i.fit_from_train()
    r_i = app_i.evaluate_test_hit_at_k(k=k)
    print(f"Hit@{k}: {r_i.hits}/{r_i.evaluated} = {r_i.hit_rate:.4f}\n")


def main() -> None:
    dense = CdsInterDatasetSource(CDsDatasetPaths.cds_dense())
    sparse = CdsInterDatasetSource(CDsDatasetPaths.cds_sparse())
    _run_label("CDs-100-user-dense", dense)
    _run_label("CDs-100-user-sparse", sparse)


if __name__ == "__main__":
    main()
