"""Đọc định dạng atomic .inter của RecBole (user_id, item_id_list, item_id)."""

from __future__ import annotations

from pathlib import Path

from ..domain.entities import IndexMaps, UserItemInteractionBundle
from .sparse_matrix_builder import CsrInteractionMatrixBuilder


class RecBoleInterReader:
    """
    Trách nhiệm duy nhất: parse một file .inter và trích các cặp (user, item).

    Mỗi dòng: lịch sử item_id_list (cách nhau bởi seq_separator) và item mục tiêu;
    coi mọi item trong list và item mục tiêu đều là tương tác đã quan sát (implicit).
    """

    def __init__(
        self,
        field_separator: str = "\t",
        seq_separator: str = " ",
        matrix_builder: CsrInteractionMatrixBuilder | None = None,
    ) -> None:
        self._field_separator = field_separator
        self._seq_separator = seq_separator
        self._matrix_builder = matrix_builder or CsrInteractionMatrixBuilder()

    def read_bundle(self, inter_path: Path) -> UserItemInteractionBundle:
        """Đọc file và trả về ma trận CSR cùng ánh xạ ID."""
        user_to_idx: dict[str, int] = {}
        item_to_idx: dict[str, int] = {}
        pairs_rows: list[int] = []
        pairs_cols: list[int] = []

        def _user(u: str) -> int:
            if u not in user_to_idx:
                user_to_idx[u] = len(user_to_idx)
            return user_to_idx[u]

        def _item(i: str) -> int:
            if i not in item_to_idx:
                item_to_idx[i] = len(item_to_idx)
            return item_to_idx[i]

        lines = inter_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            maps = IndexMaps(user_to_idx=user_to_idx, item_to_idx=item_to_idx)
            empty = self._matrix_builder.build([], [], 0, 0)
            return UserItemInteractionBundle(matrix=empty, maps=maps)

        data_lines = [ln for ln in lines[1:] if ln.strip()]
        for line in data_lines:
            parts = line.split(self._field_separator)
            if len(parts) < 3:
                continue
            uid, list_raw, target = parts[0].strip(), parts[1].strip(), parts[2].strip()
            ui = _user(uid)
            items_tokens = [t for t in list_raw.split(self._seq_separator) if t]
            for it in items_tokens:
                pairs_rows.append(ui)
                pairs_cols.append(_item(it))
            pairs_rows.append(ui)
            pairs_cols.append(_item(target))

        maps = IndexMaps(user_to_idx=user_to_idx, item_to_idx=item_to_idx)
        mat = self._matrix_builder.build(pairs_rows, pairs_cols, maps.n_users, maps.n_items)
        return UserItemInteractionBundle(matrix=mat, maps=maps)
