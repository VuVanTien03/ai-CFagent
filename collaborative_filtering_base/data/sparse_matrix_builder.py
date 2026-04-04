"""Dựng ma trận thưa từ danh sách (row, col) — có thể thay thế để mở rộng (OCP)."""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp


class CsrInteractionMatrixBuilder:
    """Gom các cặp (user_idx, item_idx) thành CSR nhị phân (1 nếu có tương tác)."""

    def build(self, rows: list[int], cols: list[int], n_users: int, n_items: int) -> sp.csr_matrix:
        if n_users == 0 or n_items == 0:
            return sp.csr_matrix((n_users, n_items), dtype=np.float32)
        data = np.ones(len(rows), dtype=np.float32)
        r = np.asarray(rows, dtype=np.int32)
        c = np.asarray(cols, dtype=np.int32)
        mat = sp.coo_matrix((data, (r, c)), shape=(n_users, n_items), dtype=np.float32)
        # Gộp trùng (coo): dùng sum_duplicates; chuyển CSR sẽ cộng dồn phần tử trùng chỉ mục.
        mat.sum_duplicates()
        return mat.tocsr()
