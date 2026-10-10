import pytest

from gridadaptive.experiment.record import (
    ExperimentLedger,
    ExperimentLedgerError,
    ExperimentRecord,
)


def _record(experiment_id="exp-0001", supersedes=None):
    return ExperimentRecord(
        experiment_id=experiment_id,
        strategy_version="0.0.0-example",
        git_commit="deadbeef",
        dataset_id="abc123",
        symbol="EURUSD",
        timeframe="1h",
        date_range={"start": "2024-01-01", "end": "2024-01-02", "split": "train"},
        strategy_params={},
        execution_assumptions={"spread": 0.0002, "commission_per_unit": 0.00001},
        starting_capital=10000.0,
        output_metrics={"total_return": 0.0},
        supersedes=supersedes,
    )


def test_append_and_read_round_trip(tmp_path):
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    ledger.append(_record())
    rows = ledger.read_all()
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "exp-0001"
    assert rows[0]["symbol"] == "EURUSD"
    assert rows[0]["date_range"]["split"] == "train"


def test_duplicate_experiment_id_is_rejected(tmp_path):
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    ledger.append(_record())
    with pytest.raises(ExperimentLedgerError):
        ledger.append(_record())


def test_correction_uses_supersedes_not_overwrite(tmp_path):
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    ledger.append(_record("exp-0001"))
    ledger.append(_record("exp-0002", supersedes="exp-0001"))
    rows = ledger.read_all()
    assert len(rows) == 2
    assert rows[1]["supersedes"] == "exp-0001"


def test_get_returns_none_for_unknown_id(tmp_path):
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    assert ledger.get("nope") is None


def test_ledger_file_is_plain_jsonl_one_record_per_line(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = ExperimentLedger(path)
    ledger.append(_record("exp-0001"))
    ledger.append(_record("exp-0002"))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
