import pytest

from gridadaptive.data.provenance import DatasetProvenance, ProvenanceStore, ProvenanceStoreError


def _provenance(dataset_id="abc123"):
    return DatasetProvenance(
        dataset_id=dataset_id,
        source_name="HistData.com",
        source_identifier="example_m5_export",
        retrieval_params={"symbol": "EURUSD", "timeframe": "M5"},
        retrieval_timestamp="2026-08-14T12:00:00Z",
        requested_date_range={"start": "2015-01-01", "end": "2020-12-31"},
        actual_date_range={"start": "2015-01-02", "end": "2020-12-31"},
        row_count=500_000,
        raw_data_hash="rawhash123",
        derived_data_hash=dataset_id,
        cleaning_steps=["shifted timestamps +5min (open->close)", "dropped 3 exact-duplicate rows"],
        source_bar_convention="close",
    )


def test_save_and_load_round_trip(tmp_path):
    store = ProvenanceStore(tmp_path)
    store.save(_provenance())
    loaded = store.load("abc123")
    assert loaded["source_name"] == "HistData.com"
    assert loaded["row_count"] == 500_000
    assert loaded["cleaning_steps"] == [
        "shifted timestamps +5min (open->close)",
        "dropped 3 exact-duplicate rows",
    ]


def test_save_is_immutable_and_rejects_overwrite(tmp_path):
    store = ProvenanceStore(tmp_path)
    store.save(_provenance())
    with pytest.raises(ProvenanceStoreError):
        store.save(_provenance())


def test_load_returns_none_for_unknown_dataset_id(tmp_path):
    store = ProvenanceStore(tmp_path)
    assert store.load("nope") is None


def test_empty_dataset_id_rejected():
    with pytest.raises(ValueError):
        DatasetProvenance(
            dataset_id="",
            source_name="HistData.com",
            source_identifier="x",
            retrieval_params={},
            retrieval_timestamp="2026-08-14T12:00:00Z",
            requested_date_range={},
            actual_date_range={},
            row_count=0,
            raw_data_hash="x",
            derived_data_hash="x",
            cleaning_steps=[],
            source_bar_convention="close",
        )


def test_negative_row_count_rejected():
    with pytest.raises(ValueError):
        prov = _provenance()
        DatasetProvenance(**{**prov.to_dict(), "row_count": -1})


def test_derived_data_hash_may_be_empty_for_raw_only_record():
    # a raw file may be acquired and hashed before any derived dataset exists yet
    # (e.g. before a parser/aggregator is built) - this must be representable
    # without fabricating a placeholder derived hash.
    prov = DatasetProvenance(
        dataset_id="rawhash_only_123",
        source_name="example_broker_demo",
        source_identifier="example tick export",
        retrieval_params={},
        retrieval_timestamp="2026-08-15T00:00:00Z",
        requested_date_range={},
        actual_date_range={},
        row_count=0,
        raw_data_hash="rawhash_only_123",
        derived_data_hash="",
        notes="raw acquisition only, no derived dataset yet",
    )
    assert prov.derived_data_hash == ""


def test_raw_data_hash_still_required_even_for_raw_only_record():
    with pytest.raises(ValueError):
        DatasetProvenance(
            dataset_id="x",
            source_name="x",
            source_identifier="x",
            retrieval_params={},
            retrieval_timestamp="2026-08-15T00:00:00Z",
            requested_date_range={},
            actual_date_range={},
            row_count=0,
            raw_data_hash="",
            derived_data_hash="",
        )


def test_raw_only_record_round_trips_through_store(tmp_path):
    store = ProvenanceStore(tmp_path)
    prov = DatasetProvenance(
        dataset_id="rawhash_only_456",
        source_name="example_broker_demo",
        source_identifier="example tick export",
        retrieval_params={},
        retrieval_timestamp="2026-08-15T00:00:00Z",
        requested_date_range={},
        actual_date_range={},
        row_count=0,
        raw_data_hash="rawhash_only_456",
        derived_data_hash="",
    )
    store.save(prov)
    loaded = store.load("rawhash_only_456")
    assert loaded["derived_data_hash"] == ""
