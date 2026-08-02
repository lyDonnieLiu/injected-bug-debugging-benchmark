from pathlib import Path

from common.config import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads():
    cfg = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / "default.yaml")
    assert cfg.device == "auto"
    assert cfg.seed == 42


def test_cloud_config_sets_cuda_device():
    cfg = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / "cloud_gpu.yaml")
    assert cfg.device.startswith("cuda")


def test_phase_config_keeps_unknown_keys():
    cfg = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / "phase_a_toy.yaml")
    assert cfg.phase == "A"  # extra key preserved
    assert cfg.device == "auto"