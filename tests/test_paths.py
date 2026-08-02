from common.paths import PROJECT_ROOT, data_dir, project_root


def test_project_root_points_at_repo():
    assert project_root() == PROJECT_ROOT
    assert (PROJECT_ROOT / "injected-bug-debugging-benchmark-design.md").exists()


def test_data_dir_respects_env_override(tmp_path, monkeypatch):
    target = tmp_path / "mounted" / "data"
    monkeypatch.setenv("IBB_DATA_ROOT", str(target))
    assert data_dir() == target
    assert target.exists()