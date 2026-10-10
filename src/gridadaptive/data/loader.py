"""CSV OHLCV loader implementing the dataset contract.

Dataset versioning uses a content hash (sha256) of the raw file bytes as
`dataset_id`, so any change to the source file — including silent in-place
mutation — produces a different id and invalidates any experiment record that
cites the old one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from gridadaptive.data.schema import validate_ohlcv


_DEFAULT_HASH_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB - bounded memory regardless of file size


def hash_file(path: str | Path, chunk_size: int = _DEFAULT_HASH_CHUNK_SIZE) -> str:
    """SHA-256 of a file's bytes, streamed in bounded chunks rather than reading the
    whole file into memory at once - safe for multi-GB raw files (e.g. a broker tick
    export) as well as the small derived CSVs this was originally written for.
    `chunk_size` only affects how much memory a single read uses; the resulting
    digest is identical regardless of chunk size."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class OHLCVDataset:
    bars: pd.DataFrame
    dataset_id: str
    source_path: str
    timeframe: str
    bar_convention: str
    gaps: list[dict] = field(default_factory=list)


def load_ohlcv_csv(
    path: str | Path, timeframe: str, bar_convention: str = "close"
) -> OHLCVDataset:
    path = Path(path)
    dataset_id = hash_file(path)

    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column")

    # utc=False preserves naive timestamps as naive, so the schema validator's
    # tz-aware check can reject them rather than the loader silently assuming UTC.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.set_index("timestamp").sort_index()

    report = validate_ohlcv(df, timeframe=timeframe, bar_convention=bar_convention)

    return OHLCVDataset(
        bars=df,
        dataset_id=dataset_id,
        source_path=str(path),
        timeframe=timeframe,
        bar_convention=bar_convention,
        gaps=report.gaps,
    )
