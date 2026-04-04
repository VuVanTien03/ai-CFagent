"""Nguồn dữ liệu cụ thể cho CDs dense / sparse — cùng parser, khác đường dẫn (SRP)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config.dataset_paths import CDsDatasetPaths
from .inter_reader import RecBoleInterReader
from ..domain.entities import UserItemInteractionBundle


@dataclass
class CdsInterDatasetSource:
    """
    Đóng gói đường dẫn bộ CDs (dense hoặc sparse) và thao tác đọc train bundle.

    Liskov: hoán đổi nguồn dense/sparse không đổi mã phía trên nếu chỉ cần UserItemInteractionBundle.
    """

    paths: CDsDatasetPaths
    reader: RecBoleInterReader | None = None

    def __post_init__(self) -> None:
        if self.reader is None:
            self.reader = RecBoleInterReader()

    def load_train(self) -> UserItemInteractionBundle:
        return self.reader.read_bundle(self.paths.train_inter)

    def inter_path(self, split: str) -> Path:
        if split == "train":
            return self.paths.train_inter
        if split == "valid":
            return self.paths.valid_inter
        if split == "test":
            return self.paths.test_inter
        raise ValueError("split phải là train | valid | test")
