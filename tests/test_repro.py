import pytest

from gridadaptive.experiment.repro import capture_git_commit


def test_capture_git_commit_on_current_repo_or_raises_before_first_commit():
    try:
        commit, dirty = capture_git_commit(".")
    except RuntimeError:
        pytest.skip("repo has no commits yet — expected before the first commit exists")
    else:
        assert len(commit) == 40
        assert isinstance(dirty, bool)


def test_capture_git_commit_outside_a_repo_raises(tmp_path):
    with pytest.raises(RuntimeError):
        capture_git_commit(tmp_path)
