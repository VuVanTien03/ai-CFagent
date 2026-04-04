"""Nạp train bundle từ file .inter — tái dùng parser RecBoleInterReader của collaborative_filtering_base."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from collaborative_filtering_base.data.inter_reader import RecBoleInterReader
from collaborative_filtering_base.domain.entities import UserItemInteractionBundle

from ..config.dataset_paths import CDsDatasetPaths


@dataclass
class CdsInterDatasetSource:
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
