"""Đường dẫn mặc định tới bộ CDs dạng dense và sparse (cùng cây AgentCF/agentcf/dataset)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _agentcf_dataset_dir() -> Path:
    # collaborative_filtering_base/config/ → AgentCF/agentcf/dataset
    agentcf_pkg = Path(__file__).resolve().parents[2] / "agentcf"
    return agentcf_pkg / "dataset"


@dataclass(frozen=True)
class CDsDatasetPaths:
    """Tập đường dẫn tới các file .inter train/valid/test cho CDs-100-user."""

    dataset_dir: Path
    prefix: str  # ví dụ: "CDs-100-user-dense" hoặc "CDs-100-user-sparse"

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
        root = _agentcf_dataset_dir() / "CDs-100-user-dense"
        return cls(dataset_dir=root, prefix="CDs-100-user-dense")

    @classmethod
    def cds_sparse(cls) -> CDsDatasetPaths:
        root = _agentcf_dataset_dir() / "CDs-100-user-sparse"
        return cls(dataset_dir=root, prefix="CDs-100-user-sparse")
