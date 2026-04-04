"""Giao thức (Protocol) — tách interface theo ISP, phụ thuộc ngược vào abstraction (DIP)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from scipy import sparse as sp


@runtime_checkable
class InteractionMatrixBuilder(Protocol):
    """Xây ma trận tương tác user–item từ luồng (user, item) đã mã hóa chỉ số."""

    def build(self, rows: list[int], cols: list[int], n_users: int, n_items: int) -> sp.csr_matrix:
        """Trả về ma trận thưa CSR kích thước (n_users, n_items)."""
        ...


@runtime_checkable
class UserItemSimilarity(Protocol):
    """Chiến lược tính độ tương đồng giữa các vector hàng (user) hoặc cột (item)."""

    def pairwise_similarity(self, matrix: sp.csr_matrix, axis: int) -> sp.csr_matrix:
        """
        axis=0: độ tương đồng giữa các user (theo hàng).
        axis=1: độ tương đồng giữa các item (theo cột).
        """
        ...


@runtime_checkable
class CollaborativeRecommender(Protocol):
    """Hợp đồng cho mô hình lọc cộng tác dựa bộ nhớ (không huấn luyện gradient)."""

    def fit(self, matrix: sp.csr_matrix) -> None:
        """Học cấu trúc lân cận / chuẩn hóa từ ma trận tương tác."""
        ...

    def recommend(
        self,
        user_index: int,
        k: int,
        exclude_item_indices: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Trả về danh sách (chỉ số item, điểm dự đoán) giảm dần theo điểm."""
        ...
