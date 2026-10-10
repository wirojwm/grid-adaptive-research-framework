import hashlib
from pathlib import Path

import pytest

from gridadaptive.data.loader import hash_file, load_ohlcv_csv
from gridadaptive.data.schema import OHLCVValidationError

FIXTURE = Path(__file__).parent / "fixtures" / "sample_ohlcv.csv"


def test_load_ohlcv_csv_basic_shape():
    ds = load_ohlcv_csv(FIXTURE, timeframe="1h", bar_convention="close")
    assert len(ds.bars) == 6
    assert ds.bars.index.tz is not None
    assert ds.timeframe == "1h"
    assert ds.bar_convention == "close"


def test_loader_flags_the_fixture_gap():
    ds = load_ohlcv_csv(FIXTURE, timeframe="1h", bar_convention="close")
    assert len(ds.gaps) == 1


def test_dataset_id_is_a_stable_content_hash():
    ds1 = load_ohlcv_csv(FIXTURE, timeframe="1h", bar_convention="close")
    ds2 = load_ohlcv_csv(FIXTURE, timeframe="1h", bar_convention="close")
    assert ds1.dataset_id == ds2.dataset_id
    assert len(ds1.dataset_id) == 64  # sha256 hex digest


def test_dataset_id_changes_if_file_content_changes(tmp_path):
    p1 = tmp_path / "a.csv"
    p1.write_text(FIXTURE.read_text())
    p2 = tmp_path / "b.csv"
    p2.write_text(FIXTURE.read_text() + "\n2024-01-01T07:00:00Z,1.1,1.11,1.09,1.1,10\n")

    ds1 = load_ohlcv_csv(p1, timeframe="1h", bar_convention="close")
    ds2 = load_ohlcv_csv(p2, timeframe="1h", bar_convention="close")
    assert ds1.dataset_id != ds2.dataset_id


def test_naive_timestamps_rejected_not_silently_assumed_utc(tmp_path):
    p = tmp_path / "naive.csv"
    p.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01 00:00:00,1.1,1.11,1.09,1.1,100\n"
        "2024-01-01 01:00:00,1.1,1.11,1.09,1.1,100\n"
    )
    with pytest.raises(OHLCVValidationError):
        load_ohlcv_csv(p, timeframe="1h", bar_convention="close")


def test_missing_timestamp_column_rejected(tmp_path):
    p = tmp_path / "no_ts.csv"
    p.write_text("open,high,low,close\n1.1,1.11,1.09,1.1\n")
    with pytest.raises(ValueError):
        load_ohlcv_csv(p, timeframe="1h", bar_convention="close")


def test_hash_file_matches_direct_sha256_for_small_file(tmp_path):
    p = tmp_path / "small.bin"
    content = b"grid adaptive test content"
    p.write_bytes(content)
    assert hash_file(p) == hashlib.sha256(content).hexdigest()


def test_hash_file_matches_direct_sha256_across_multiple_chunks(tmp_path):
    p = tmp_path / "multi_chunk.bin"
    content = bytes(range(256)) * 1000  # 256,000 bytes
    p.write_bytes(content)
    # a tiny chunk_size forces many read iterations - result must still match exactly
    assert hash_file(p, chunk_size=17) == hashlib.sha256(content).hexdigest()


def test_hash_file_default_chunk_size_matches_small_chunk_size(tmp_path):
    p = tmp_path / "consistency.bin"
    content = b"x" * 500_000
    p.write_bytes(content)
    assert hash_file(p) == hash_file(p, chunk_size=4096)


def test_hash_file_handles_empty_file(tmp_path):
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert hash_file(p) == hashlib.sha256(b"").hexdigest()


def test_hash_file_never_reads_whole_file_at_once(tmp_path, monkeypatch):
    p = tmp_path / "large_ish.bin"
    p.write_bytes(b"y" * 1_000_000)

    def _forbidden_read_bytes(self):
        raise AssertionError("hash_file must stream in bounded chunks, not Path.read_bytes()")

    monkeypatch.setattr(Path, "read_bytes", _forbidden_read_bytes)
    # must succeed without ever calling the now-forbidden whole-file read
    assert hash_file(p, chunk_size=65536) == hashlib.sha256(b"y" * 1_000_000).hexdigest()
