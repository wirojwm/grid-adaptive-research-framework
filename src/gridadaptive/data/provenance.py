"""Dataset provenance tracking.

Raw and processed market-data files are git-ignored (too large, don't belong in
history); this module is what git tracks instead. A `DatasetProvenance` record ties
a derived dataset's content hash (`dataset_id`, the same hash `load_ohlcv_csv`
computes) back to where it came from and what was done to it. Together with
`gridadaptive.experiment.record.ExperimentRecord` (whose `dataset_id` field cites
the same hash), these two records form the complete manifest chain — raw source ->
derived dataset -> experiment run — with no separate "manifest" file needed.

A record may also cover **raw acquisition only**, before any derived dataset exists
(`derived_data_hash=""`) — e.g. a large raw file has been obtained and hashed but no
parser/aggregator has been built yet. `dataset_id` for such a record is the raw file's
own hash. `raw_data_hash` is always required; `derived_data_hash` is empty until a
derived dataset actually exists, at which point a *new*, separate
`DatasetProvenance` record is created for it (citing this record's `raw_data_hash`
for lineage) — never edited into this one, per the append-only rule below.

Records are immutable once saved: a `ProvenanceStore` refuses to overwrite an
existing `dataset_id`, mirroring `ExperimentLedger`'s append-only discipline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetProvenance:
    dataset_id: str
    source_name: str
    source_identifier: str
    retrieval_params: dict
    retrieval_timestamp: str  # ISO8601 UTC
    requested_date_range: dict
    actual_date_range: dict
    row_count: int
    raw_data_hash: str
    derived_data_hash: str = ""  # empty = raw acquired, derived dataset not built yet
    cleaning_steps: list[str] = field(default_factory=list)
    source_bar_convention: str = "close"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if not self.source_name:
            raise ValueError("source_name must be non-empty")
        if not self.raw_data_hash:
            raise ValueError("raw_data_hash must be non-empty")
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative")
        if self.source_bar_convention not in ("open", "close"):
            raise ValueError("source_bar_convention must be 'open' or 'close'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProvenanceStoreError(RuntimeError):
    pass


class ProvenanceStore:
    """One immutable JSON file per dataset_id under `directory`."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _path(self, dataset_id: str) -> Path:
        return self.directory / f"{dataset_id}.json"

    def save(self, provenance: DatasetProvenance) -> Path:
        path = self._path(provenance.dataset_id)
        if path.exists():
            raise ProvenanceStoreError(
                f"provenance record for dataset_id {provenance.dataset_id!r} already "
                f"exists at {path}; provenance records are immutable"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(provenance.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    def load(self, dataset_id: str) -> dict | None:
        path = self._path(dataset_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
