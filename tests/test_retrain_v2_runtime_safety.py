from types import SimpleNamespace

import pytest

from open_r1.utils.callbacks import RuntimeSafetyLoggingCallback


def _callback(tmp_path):
    return RuntimeSafetyLoggingCallback(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(),
    )


def test_runtime_safety_logs_finite_metrics(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    callback = _callback(tmp_path)
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)

    callback.on_log(
        None,
        state,
        None,
        logs={
            "loss": 0.5,
            "reward": 0.4,
            "grad_norm": 0.1,
            "completions/clipped_ratio": 0.25,
        },
    )

    assert (tmp_path / "runtime_safety.jsonl").is_file()


def test_runtime_safety_only_logs_high_reserved_memory(tmp_path, monkeypatch):
    import json

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 20 * 1024**3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 23 * 1024**3)
    callback = _callback(tmp_path)
    state = SimpleNamespace(global_step=56, is_world_process_zero=True)

    callback.on_log(None, state, None, logs={"reward": 0.2, "grad_norm": 0.1})

    record = json.loads((tmp_path / "runtime_safety.jsonl").read_text())
    assert record["peak_allocated_gib"] == 20.0
    assert record["peak_reserved_gib"] == 23.0


def test_runtime_safety_rejects_nonfinite_metric(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    callback = _callback(tmp_path)
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)

    with pytest.raises(RuntimeError, match="non-finite"):
        callback.on_log(None, state, None, logs={"reward": float("nan")})


def test_runtime_safety_rejects_rolling_zero_gradient_rate(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    callback = _callback(tmp_path)
    state = SimpleNamespace(global_step=0, is_world_process_zero=True)

    for step in range(1, 20):
        state.global_step = step
        callback.on_log(None, state, None, logs={"grad_norm": 0.0})
    state.global_step = 20
    with pytest.raises(RuntimeError, match="rolling-20"):
        callback.on_log(None, state, None, logs={"grad_norm": 0.0})
