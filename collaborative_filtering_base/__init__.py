"""
Lọc cộng tác dựa bộ nhớ (user-based / item-based) cho dữ liệu RecBole `.inter`.

Cấu trúc thư mục (hướng SOLID):
- `config/`: đường dẫn bộ CDs dense & sparse (SRP).
- `domain/`: thực thể + `Protocol` (ISP/DIP), không phụ thuộc SciPy cụ thể ở tầng hợp đồng.
- `data/`: đọc file & dựng ma trận thưa (tách I/O khỏi thuật toán).
- `models/`: thuật toán CF & cosine (OCP: thay `UserItemSimilarity`).
- `services/`: façade ứng dụng + đánh giá Hit@K (DIP: inject `CollaborativeRecommender`).
- `examples/`: script chạy thử trên `agentcf/dataset/CDs-100-user-{dense,sparse}`.

Chạy ví dụ (từ thư mục `AgentCF`):
`python -m collaborative_filtering_base.examples.run_cds_collaborative_filtering`
"""
# Gói lọc cộng tác (memory-based) cơ sở — API công khai gọn cho người dùng thư viện.
from .config.dataset_paths import CDsDatasetPaths
from .models.item_based_cf import ItemBasedCollaborativeFiltering
from .models.user_based_cf import UserBasedCollaborativeFiltering
from .services.recommender_app import RecommenderApplication

__all__ = [
    "CDsDatasetPaths",
    "UserBasedCollaborativeFiltering",
    "ItemBasedCollaborativeFiltering",
    "RecommenderApplication",
]
