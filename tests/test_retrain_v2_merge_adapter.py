from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jobs.retrain_v2.merge_adapter import _merge_model


class _FakeTensor:
    def __init__(self, *, is_meta: bool = False, elements: int = 1) -> None:
        self.is_meta = is_meta
        self._elements = elements

    def numel(self) -> int:
        return self._elements


class _FakeModel:
    def __init__(
        self,
        *,
        parameters: dict[str, _FakeTensor] | None = None,
        buffers: dict[str, _FakeTensor] | None = None,
        modules: list[Any] | None = None,
    ) -> None:
        self._parameters = parameters or {"model.weight": _FakeTensor(elements=7)}
        self._buffers = buffers or {"model.cache": _FakeTensor()}
        self._modules = modules or [self]

    def named_parameters(self):
        return iter(self._parameters.items())

    def named_buffers(self):
        return iter(self._buffers.items())

    def parameters(self):
        return iter(self._parameters.values())

    def modules(self):
        return iter(self._modules)

    def state_dict(self) -> dict[str, _FakeTensor]:
        return dict(self._parameters)

    def save_pretrained(self, output: Path, **kwargs: Any) -> None:
        assert kwargs == {"safe_serialization": True, "max_shard_size": "5GB"}
        (output / "model.safetensors").write_bytes(b"merged")


class _FakeTokenizer:
    def save_pretrained(self, output: Path) -> None:
        (output / "tokenizer.json").write_text("{}\n", encoding="utf-8")


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pre_meta_kind: str | None = None,
    post_meta_kind: str | None = None,
) -> dict[str, Any]:
    import peft
    import peft.tuners.lora as lora_module
    import torch
    import transformers

    calls: dict[str, Any] = {"merge_called": False}

    class FakeLoraLayer:
        pass

    class FakePeftModel(_FakeModel):
        def merge_and_unload(self) -> _FakeModel:
            calls["merge_called"] = True
            parameters = {"model.weight": _FakeTensor(elements=7)}
            buffers = {"model.cache": _FakeTensor()}
            if post_meta_kind == "parameter":
                parameters["merged.meta"] = _FakeTensor(is_meta=True)
            elif post_meta_kind == "buffer":
                buffers["merged.meta"] = _FakeTensor(is_meta=True)
            return _FakeModel(parameters=parameters, buffers=buffers)

    def load_base(path: str, **kwargs: Any) -> _FakeModel:
        calls["base_path"] = path
        calls["base_kwargs"] = kwargs
        return _FakeModel()

    def load_tokenizer(path: str, **kwargs: Any) -> _FakeTokenizer:
        calls["tokenizer_path"] = path
        calls["tokenizer_kwargs"] = kwargs
        return _FakeTokenizer()

    def load_peft(base_model: _FakeModel, path: str, **kwargs: Any) -> FakePeftModel:
        calls["peft_base_model"] = base_model
        calls["peft_path"] = path
        calls["peft_kwargs"] = kwargs
        # Model the regression: with CUDA visible, an omitted map would allow
        # PEFT's default `auto` dispatch to choose GPUs and CPU offload.
        if torch.cuda.is_available() and kwargs.get("device_map") != {"": "cpu"}:
            raise AssertionError("GPU-visible merge was not pinned to CPU")
        parameters = {"model.weight": _FakeTensor(elements=7)}
        buffers = {"model.cache": _FakeTensor()}
        if pre_meta_kind == "parameter":
            parameters["adapter.meta"] = _FakeTensor(is_meta=True)
        elif pre_meta_kind == "buffer":
            buffers["adapter.meta"] = _FakeTensor(is_meta=True)
        model = FakePeftModel(parameters=parameters, buffers=buffers)
        model._modules = [model, FakeLoraLayer()]
        return model

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(lora_module, "LoraLayer", FakeLoraLayer)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(load_base),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(load_tokenizer),
    )
    monkeypatch.setattr(
        peft.PeftModel,
        "from_pretrained",
        staticmethod(load_peft),
    )
    return calls


def test_merge_model_pins_peft_to_cpu_when_gpus_are_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_runtime(monkeypatch)
    output = tmp_path / "merged"

    evidence = _merge_model(tmp_path / "base", tmp_path / "adapter", output)

    assert calls["peft_kwargs"] == {
        "device_map": {"": "cpu"},
        "local_files_only": True,
    }
    assert calls["merge_called"] is True
    assert evidence["lora_modules_before"] == 1
    assert evidence["residual_lora_modules_after"] == 0
    assert evidence["residual_lora_state_keys_after"] == 0
    assert evidence["merged_parameter_tensors"] == 1
    assert evidence["merged_parameter_count"] == 7
    assert (output / "model.safetensors").is_file()
    assert (output / "tokenizer.json").is_file()


@pytest.mark.parametrize("meta_kind", ["parameter", "buffer"])
def test_merge_model_rejects_pre_merge_meta_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    meta_kind: str,
) -> None:
    calls = _install_fake_runtime(monkeypatch, pre_meta_kind=meta_kind)
    output = tmp_path / "merged"

    with pytest.raises(
        RuntimeError,
        match=rf"pre-merge adapter load.*{meta_kind}:adapter\.meta",
    ):
        _merge_model(tmp_path / "base", tmp_path / "adapter", output)

    assert calls["merge_called"] is False
    assert not output.exists()


@pytest.mark.parametrize("meta_kind", ["parameter", "buffer"])
def test_merge_model_rejects_post_merge_meta_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    meta_kind: str,
) -> None:
    calls = _install_fake_runtime(monkeypatch, post_meta_kind=meta_kind)
    output = tmp_path / "merged"

    with pytest.raises(
        RuntimeError,
        match=rf"post-merge unload.*{meta_kind}:merged\.meta",
    ):
        _merge_model(tmp_path / "base", tmp_path / "adapter", output)

    assert calls["merge_called"] is True
    assert not output.exists()
