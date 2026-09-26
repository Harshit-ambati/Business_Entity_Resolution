"""Configuration for candidate retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CandidateRetrievalConfig:
    """Configuration for candidate retrieval and blocking."""

    source1_path: Path | str | None = None
    source2_path: Path | str | None = None
    source3_path: Path | str | None = None
    truth_path: Path | str | None = None
    output_path: Path | str | None = None
    work_dir: Path | str = Path("artifacts/blocking")

    # Blocking route switches (Phase 1 baseline: exact normalized name only)
    exact_name: bool = True
    name_token: bool = False
    address: bool = False
    combined: bool = False
    fuzzy_name: bool = False

    # Capacity and safety cutoffs
    max_candidates_per_entity: int = 32
    max_block_size: int = 500  # Safeguard for high-frequency normalized names

    # Split label
    split: str = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        return {
            "source1_path": str(self.source1_path) if self.source1_path else None,
            "source2_path": str(self.source2_path) if self.source2_path else None,
            "source3_path": str(self.source3_path) if self.source3_path else None,
            "truth_path": str(self.truth_path) if self.truth_path else None,
            "output_path": str(self.output_path) if self.output_path else None,
            "work_dir": str(self.work_dir),
            "exact_name": self.exact_name,
            "name_token": self.name_token,
            "address": self.address,
            "combined": self.combined,
            "fuzzy_name": self.fuzzy_name,
            "max_candidates_per_entity": self.max_candidates_per_entity,
            "max_block_size": self.max_block_size,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateRetrievalConfig:
        """Create a CandidateRetrievalConfig from a dictionary."""
        return cls(
            source1_path=Path(data["source1_path"]) if data.get("source1_path") else None,
            source2_path=Path(data["source2_path"]) if data.get("source2_path") else None,
            source3_path=Path(data["source3_path"]) if data.get("source3_path") else None,
            truth_path=Path(data["truth_path"]) if data.get("truth_path") else None,
            output_path=Path(data["output_path"]) if data.get("output_path") else None,
            work_dir=Path(data.get("work_dir", "artifacts/blocking")),
            exact_name=data.get("exact_name", True),
            name_token=data.get("name_token", False),
            address=data.get("address", False),
            combined=data.get("combined", False),
            fuzzy_name=data.get("fuzzy_name", False),
            max_candidates_per_entity=data.get("max_candidates_per_entity", 32),
            max_block_size=data.get("max_block_size", 500),
            split=data.get("split", "unspecified"),
        )
