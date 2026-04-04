"""Thực thể miền: ánh xạ ID gốc ↔ chỉ số nội bộ và bản ghi ma trận."""

from __future__ import annotations

from dataclasses import dataclass

from scipy import sparse as sp


@dataclass(frozen=True)
class IndexMaps:
    """Ánh xạ hai chiều giữa token (chuỗi ID RecBole) và chỉ số ma trận."""

    user_to_idx: dict[str, int]
    item_to_idx: dict[str, int]

    @property
    def n_users(self) -> int:
        return len(self.user_to_idx)

    @property
    def n_items(self) -> int:
        return len(self.item_to_idx)

    def user_index(self, user_id: str) -> int:
        return self.user_to_idx[user_id]

    def item_index(self, item_id: str) -> int:
        return self.item_to_idx[item_id]

    def idx_to_item(self) -> list[str]:
        """Danh sách item_id theo thứ tự cột ma trận."""
        pairs = sorted(((j, tid) for tid, j in self.item_to_idx.items()), key=lambda x: x[0])
        return [tid for _, tid in pairs]


@dataclass
class UserItemInteractionBundle:
    """Gói dữ liệu sau khi đọc .inter: ma trận thưa + ánh xạ chỉ số."""

    matrix: sp.csr_matrix
    maps: IndexMaps
