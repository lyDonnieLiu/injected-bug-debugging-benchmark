import json
from pathlib import Path

import pytest

from credibility_report.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    assert "credibility" in capsys.readouterr().out.lower()


def test_cli_report_smoke(tmp_path):
    json_out = tmp_path / "report.json"
    rc = main(
        [
            "--config",
            str(DEFAULT_CONFIG),
            "--model",
            "gpt2",
            "--task",
            "ioi",
            "--json-out",
            str(json_out),
        ]
    )
    assert rc == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["model"] == "gpt2"
    assert payload["task"] == "ioi"
    assert payload["grade"] == "D"
    assert len(payload["checks"]) == 6