import pytest

from common import device as device_mod


def test_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.delenv(device_mod.DEVICE_ENV_VAR, raising=False)
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: False)
    assert device_mod.resolve_device() == "cpu"


def test_env_var_wins_over_preference(monkeypatch):
    monkeypatch.setenv(device_mod.DEVICE_ENV_VAR, "cpu")
    assert device_mod.resolve_device(preference="cuda") == "cpu"


def test_requested_cuda_without_gpu_raises(monkeypatch):
    monkeypatch.delenv(device_mod.DEVICE_ENV_VAR, raising=False)
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        device_mod.resolve_device("cuda")


def test_preference_used_when_env_unset(monkeypatch):
    monkeypatch.delenv(device_mod.DEVICE_ENV_VAR, raising=False)
    assert device_mod.resolve_device("cpu") == "cpu"