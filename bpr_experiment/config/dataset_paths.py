"""Đường dẫn tới bộ CDs trong AgentCF/agentcf/dataset (layout chuẩn của repo này)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    # bpr_experiment/config/dataset_paths.py → repo root
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CDsDatasetPaths:
    """Train/valid/test .inter cho CDs-100-user-dense hoặc sparse."""

    dataset_dir: Path
    prefix: str

    @property
    def train_inter(self) -> Path:
        return self.dataset_dir / f"{self.prefix}.train.inter"

    @property
    def valid_inter(self) -> Path:
        return self.dataset_dir / f"{self.prefix}.valid.inter"

    @property
    def test_inter(self) -> Path:
        return self.dataset_dir / f"{self.prefix}.test.inter"

    @classmethod
    def cds_dense(cls) -> CDsDatasetPaths:
        root = _repo_root() / "AgentCF" / "agentcf" / "dataset" / "CDs-100-user-dense"
        return cls(dataset_dir=root, prefix="CDs-100-user-dense")

    @classmethod
    def cds_sparse(cls) -> CDsDatasetPaths:
        root = _repo_root() / "AgentCF" / "agentcf" / "dataset" / "CDs-100-user-sparse"
        return cls(dataset_dir=root, prefix="CDs-100-user-sparse")
