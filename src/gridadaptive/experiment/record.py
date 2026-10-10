"""Experiment record schema and append-only ledger.

Experiment records are the unit of reproducibility for this project. They are
never overwritten in place; a correction is filed as a new record whose
`supersedes` field points at the record it corrects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    strategy_version: str
    git_commit: str
    dataset_id: str
    symbol: str
    timeframe: str
    date_range: dict  # {"start": ..., "end": ..., "split": "train"|"validation"|"test"}
    strategy_params: dict
    execution_assumptions: dict
    starting_capital: float
    output_metrics: dict
    random_seed: int | None = None
    supersedes: str | None = None
    test_set_touch_count: int = 0
    git_dirty: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentLedgerError(RuntimeError):
    pass


class ExperimentLedger:
    """Append-only JSONL store, one ExperimentRecord per line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: ExperimentRecord) -> None:
        if self.get(record.experiment_id) is not None:
            raise ExperimentLedgerError(
                f"experiment_id {record.experiment_id!r} already exists; experiment "
                "records are append-only. File a new record with 'supersedes' set "
                "instead of overwriting this one."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), default=str) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def get(self, experiment_id: str) -> dict | None:
        for rec in self.read_all():
            if rec["experiment_id"] == experiment_id:
                return rec
        return None
