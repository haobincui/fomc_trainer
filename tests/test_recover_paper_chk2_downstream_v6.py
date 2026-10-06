from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jobs.generation import generate_paper_chk2_chk1_analysis_rewrite as v5
from jobs.generation import generate_paper_chk2_downstream_v6 as v6
from jobs.generation import recover_paper_chk2_downstream_v6 as recovery


def _fixtures():
    path = Path(__file__).with_name("test_generate_paper_chk2_chk1_analysis_rewrite.py")
    spec = importlib.util.spec_from_file_location("recovery_v5_fixtures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_pass() -> dict[str, Any]:
    return {
        "complete": True,
        "machine_pass": True,
        "reasons": [],
        "result": {"primary": {}},
        "provider": {},
        "contract_repair_used": False,
    }


def _run_terminal(tmp_path: Path, backend: Any):
    fixtures = _fixtures()
    row = fixtures._row()
    reference = fixtures._reference_bank()
    terminal = v6._process_downstream_terminal(
        row,
        output=tmp_path,
        tokenizer=fixtures.FakeTokenizer(),
        reference_bank=reference,
        official_reference_bank_sha256="a" * 64,
        backend=backend,
        identity=v5.ProviderIdentityRegistry(),
        environment={v5.API_KEY_ENV: "not-persisted"},
        config=v5.ProviderConfig(),
        code_sha256="b" * 64,
        source_audit=_source_pass(),
    )
    return fixtures, row, reference, terminal


class InvalidStyleRepairBackend:
    def __init__(self, fixtures: Any) -> None:
        self.fixtures = fixtures
        self.delegate = fixtures.StyleRepairBackend()

    def generate(self, *, role, user_prompt, **kwargs):
        if role != v5.ROLE_REWRITE_STYLE_REPAIR:
            return self.delegate.generate(role=role, user_prompt=user_prompt, **kwargs)
        self.delegate.requests.append(
            {
                "role": role,
                "user_prompt": user_prompt,
                "payload": v5._payload_from_prompt(user_prompt),
            }
        )
        return self.fixtures._response(
            "invalid-style-repair",
            {"answer": self.fixtures.STYLE_REWRITE.replace("2 percent", "3 percent")},
            reasoning="Attempted style repair while preserving the source.",
        )


def _compat_load(tmp_path: Path, fixtures: Any, row: Any, reference: Any):
    return recovery.load_terminal_compat(
        tmp_path,
        row,
        code_sha256="b" * 64,
        official_reference_bank_sha256="a" * 64,
        identity=v5.ProviderIdentityRegistry(),
        reference_bank=reference,
        config=v5.ProviderConfig(),
        environment={v5.API_KEY_ENV: "not-persisted"},
        source_audit=_source_pass(),
        tokenizer=fixtures.FakeTokenizer(),
    )


def test_style_deterministic_reject_uses_narrow_compatibility(tmp_path: Path) -> None:
    fixtures = _fixtures()
    fixtures, row, reference, terminal = _run_terminal(
        tmp_path, InvalidStyleRepairBackend(fixtures)
    )
    assert recovery._compatibility_kind(terminal) == "style_repair_deterministic_reject"
    assert terminal["terminal_status"] == v5.TERMINAL_STYLE_FIDELITY_REJECT
    with pytest.raises(
        v5.SyntheticRewriteError, match="selected rewrite candidate drift"
    ):
        v6._load_downstream_terminal(
            tmp_path,
            row,
            code_sha256="b" * 64,
            official_reference_bank_sha256="a" * 64,
            identity=v5.ProviderIdentityRegistry(),
            reference_bank=reference,
            config=v5.ProviderConfig(),
            environment={v5.API_KEY_ENV: "not-persisted"},
            source_audit=_source_pass(),
            tokenizer=fixtures.FakeTokenizer(),
        )
    assert _compat_load(tmp_path, fixtures, row, reference) == terminal


def test_a_triggered_fidelity_deterministic_reject_is_compatible(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures()
    fixtures, row, reference, terminal = _run_terminal(
        tmp_path, fixtures.FidelityRepairFailureBackend()
    )
    assert terminal["rejection_stage"] == "fidelity_repair_deterministic_gate"
    assert (
        recovery._compatibility_kind(terminal) == "fidelity_repair_deterministic_reject"
    )
    assert _compat_load(tmp_path, fixtures, row, reference) == terminal


def test_generation_triggered_fidelity_repair_never_enters_compatibility() -> None:
    record = {
        "terminal_status": v5.TERMINAL_GENERATION_REJECT,
        "rejection_stage": "fidelity_repair_deterministic_gate",
        "generation": {
            "selected_attempt": "fidelity_repair",
            "fidelity_repair_used": True,
            "style_repair_used": False,
            "deterministic_validation": {"machine_pass": False},
        },
        "repair_history": [
            {
                "repair_type": "fidelity",
                "trigger_stage": "rewrite_primary_deterministic_gate",
            }
        ],
    }
    assert recovery._compatibility_kind(record) is None


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("repair_history", 0, "attempt"), "wrong-attempt"),
        (("repair_history", 0, "reason_codes"), ["tampered"]),
    ],
)
def test_style_compatibility_rejects_history_tamper(
    tmp_path: Path, path: tuple[Any, ...], value: Any
) -> None:
    fixtures = _fixtures()
    fixtures, row, reference, _terminal = _run_terminal(
        tmp_path, InvalidStyleRepairBackend(fixtures)
    )
    terminal_path = v5._terminal_path(tmp_path, row)
    payload = json.loads(terminal_path.read_text())
    target: Any = payload["record"]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    payload["record_sha256"] = v5.sha256_text(v5.canonical_json(payload["record"]))
    terminal_path.write_text(v5.canonical_json(payload) + "\n")
    with pytest.raises(recovery.RecoveryError, match="compatibility|style"):
        _compat_load(tmp_path, fixtures, row, reference)


def test_compatibility_uses_prebound_provider_identity(tmp_path: Path) -> None:
    fixtures = _fixtures()
    fixtures, row, reference, _terminal = _run_terminal(
        tmp_path, InvalidStyleRepairBackend(fixtures)
    )
    identity = v5.ProviderIdentityRegistry()
    identity.bind(
        "preexisting",
        fixtures._response(
            "different-provider",
            {"answer": fixtures.REWRITE},
            fingerprint="fp-different",
        ),
    )
    with pytest.raises(v5.ModelDriftError):
        recovery.load_terminal_compat(
            tmp_path,
            row,
            code_sha256="b" * 64,
            official_reference_bank_sha256="a" * 64,
            identity=identity,
            reference_bank=reference,
            config=v5.ProviderConfig(),
            environment={},
            source_audit=_source_pass(),
            tokenizer=fixtures.FakeTokenizer(),
        )


def _fake_state(*, terminals: dict[str, Any], missing: tuple[str, ...]):
    return recovery.RecoveryState(
        acquisition_root=Path("/unused"),
        terminals=terminals,
        missing_ids=missing,
        compatibility_ids=(),
        cache_counts={"terminal": len(terminals)},
        provider_identities={},
        implementation_sha256="a" * 64,
        run_binding_sha256="b" * 64,
        reference_sha256="c" * 64,
        prompt_contract_sha256="d" * 64,
    )


def _minimal_partial_tree(root: Path) -> None:
    for name in recovery.FIXED_ACQUISITION_FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("\n" if name.endswith(".jsonl") else "{}\n")
    cache = root / "cache/terminal"
    cache.mkdir(parents=True)
    (cache / "one.json").write_text("{}\n")


def test_seal_verifies_staging_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "original"
    output = tmp_path / "recovered"
    _minimal_partial_tree(original)
    handoff_root = tmp_path / "handoff"
    handoff_root.mkdir()
    (handoff_root / "handoff_manifest.json").write_text("{}\n")
    handoff = SimpleNamespace(
        root=handoff_root,
        manifest={"manifest_sha256": "e" * 64},
    )
    state = _fake_state(terminals={}, missing=())
    monkeypatch.setattr(recovery, "inspect_acquisition", lambda *a, **k: state)
    monkeypatch.setattr(
        recovery, "_validate_baseline_compatibility_scope", lambda s: None
    )

    def reject_staging(*args, **kwargs):
        del args, kwargs
        raise recovery.RecoveryError("copied snapshot failed verification")

    monkeypatch.setattr(recovery, "verify_partial_handoff", reject_staging)
    with pytest.raises(recovery.RecoveryError, match="copied snapshot"):
        recovery.seal_partial_acquisition(
            original, output, handoff=handoff, tokenizer=object()
        )
    assert not output.exists()


def test_attempt_ledger_is_append_only_and_partitioned(tmp_path: Path) -> None:
    manifest = {
        "manifest_sha256": "f" * 64,
        "baseline_missing_ids": ["a", "b"],
    }
    recovery._record_attempt(
        tmp_path,
        manifest,
        kind="provider_attempt",
        requested_ids=["a", "b"],
        completed_ids=["a"],
        failures=[{"sample_id": "b", "error_type": "ProviderRequestError"}],
    )
    recovery._record_attempt(
        tmp_path,
        manifest,
        kind="provider_attempt",
        requested_ids=["b"],
        completed_ids=["b"],
        failures=[],
    )
    attempts, completed = recovery._load_attempts(tmp_path, manifest)
    assert len(attempts) == 2
    assert completed == {"a", "b"}
    first = tmp_path / "attempts/attempt-0001.json"
    first_bytes = first.read_bytes()
    recovery._load_attempts(tmp_path, manifest)
    assert first.read_bytes() == first_bytes


def test_sealed_failure_ledger_remains_valid_after_all_rows_recover() -> None:
    failures = [
        {"sample_id": "a", "error_type": "ProviderRequestError"},
        {"sample_id": "b", "error_type": "ProviderRequestError"},
    ]
    recovery._validate_historical_failure_ledger(
        failures,
        row_ids=("a", "b", "already-terminal"),
        missing_ids=("b",),
        expected_failure_ids=("a", "b"),
    )
    recovery._validate_historical_failure_ledger(
        failures,
        row_ids=("a", "b", "already-terminal"),
        missing_ids=(),
        expected_failure_ids=("a", "b"),
    )
    with pytest.raises(recovery.RecoveryError, match="failure/missing partition"):
        recovery._validate_historical_failure_ledger(
            failures,
            row_ids=("a", "b", "already-terminal"),
            missing_ids=("already-terminal",),
            expected_failure_ids=("a", "b"),
        )


@pytest.mark.parametrize("drift", ("modify", "remove", "add-cache"))
def test_original_v6_inventory_drift_is_rejected(tmp_path: Path, drift: str) -> None:
    original = tmp_path / "original"
    _minimal_partial_tree(original)
    files = recovery._partial_file_paths(original)
    inventory = []
    for path in files:
        relative = path.relative_to(original)
        inventory.append(
            {
                "path": (Path("acquisition") / relative).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": recovery.sha256_file(path),
            }
        )
    manifest = {
        "original_v6_root": recovery._display_path(original),
        "inventory": inventory,
    }
    recovery.verify_original_v6_unchanged(original, manifest)
    if drift == "modify":
        (original / recovery.FIXED_ACQUISITION_FILES[0]).write_text("tampered\n")
    elif drift == "remove":
        (original / recovery.FIXED_ACQUISITION_FILES[0]).unlink()
    else:
        (original / "cache/terminal/extra.json").write_text("{}\n")
    with pytest.raises(
        recovery.RecoveryError, match="protocol artifact drift|inventory set drift"
    ):
        recovery.verify_original_v6_unchanged(original, manifest)


def test_resume_calls_worker_only_for_current_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _fixtures()
    row_done = fixtures._row("train", 1)
    row_missing = fixtures._row("train", 2)
    handoff = SimpleNamespace(
        prepared={"train": (row_done, row_missing), "validation": (), "test": ()},
        source_results={
            row_done.sample_id: _source_pass(),
            row_missing.sample_id: _source_pass(),
        },
    )
    manifest = {
        "manifest_sha256": "f" * 64,
        "baseline_terminal_ids": [row_done.sample_id],
        "baseline_missing_ids": [row_missing.sample_id],
    }
    initial = _fake_state(
        terminals={row_done.sample_id: {"terminal_status": v5.TERMINAL_SOURCE_REJECT}},
        missing=(row_missing.sample_id,),
    )
    final = _fake_state(
        terminals={
            row_done.sample_id: {"terminal_status": v5.TERMINAL_SOURCE_REJECT},
            row_missing.sample_id: {"terminal_status": v5.TERMINAL_SOURCE_REJECT},
        },
        missing=(),
    )
    states = iter((initial, final))
    identities: list[Any] = []

    def inspect(*args, **kwargs):
        del args
        identities.append(kwargs.get("identity"))
        return next(states)

    monkeypatch.setattr(recovery, "verify_partial_handoff", lambda *a, **k: manifest)
    monkeypatch.setattr(recovery, "verify_original_v6_unchanged", lambda *a, **k: None)
    monkeypatch.setattr(recovery, "inspect_acquisition", inspect)
    monkeypatch.setattr(
        recovery.official_v2,
        "deserialize_official_reference_bank",
        lambda data: {},
    )
    calls: list[str] = []

    def process(row, **kwargs):
        del kwargs
        calls.append(row.sample_id)
        return {"terminal_status": v5.TERMINAL_SOURCE_REJECT}

    monkeypatch.setattr(v6, "_process_downstream_terminal", process)
    monkeypatch.setattr(
        recovery, "_finalize_recovery", lambda *a, **k: {"status": "complete"}
    )
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir()
    (acquisition / "official_pre_action_reference_bank.jsonl").write_text("x\n")
    result = recovery.resume_recovery(
        tmp_path,
        original_root=tmp_path / "original",
        handoff=handoff,
        tokenizer=fixtures.FakeTokenizer(),
        backend=object(),
        environment={},
        concurrency=128,
        resume=True,
    )
    assert result == {"status": "complete"}
    assert calls == [row_missing.sample_id]
    assert identities[0] is not None


def test_resume_complete_state_without_receipt_runs_finalizer_without_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _fixtures()
    row = fixtures._row("train", 1)
    terminal = {"terminal_status": v5.TERMINAL_SOURCE_REJECT}
    handoff = SimpleNamespace(
        prepared={"train": (row,), "validation": (), "test": ()},
        source_results={row.sample_id: _source_pass()},
    )
    manifest = {
        "manifest_sha256": "f" * 64,
        "baseline_terminal_ids": [row.sample_id],
        "baseline_missing_ids": [],
    }
    state = _fake_state(terminals={row.sample_id: terminal}, missing=())
    monkeypatch.setattr(recovery, "verify_partial_handoff", lambda *a, **k: manifest)
    monkeypatch.setattr(recovery, "verify_original_v6_unchanged", lambda *a, **k: None)
    monkeypatch.setattr(recovery, "inspect_acquisition", lambda *a, **k: state)
    finalized: list[recovery.RecoveryState] = []

    def finalize(*args, **kwargs):
        del args
        finalized.append(kwargs["state"])
        return {"status": "complete"}

    monkeypatch.setattr(recovery, "_finalize_recovery", finalize)

    class NoProviderCalls:
        def generate(self, **kwargs):
            del kwargs
            raise AssertionError("provider must not be called for a complete state")

    result = recovery.resume_recovery(
        tmp_path,
        original_root=tmp_path / "original",
        handoff=handoff,
        tokenizer=fixtures.FakeTokenizer(),
        backend=NoProviderCalls(),
        environment={},
        concurrency=128,
        resume=True,
    )
    assert result == {"status": "complete"}
    assert finalized == [state]


def test_resume_requires_exact_128_and_explicit_resume() -> None:
    with pytest.raises(recovery.RecoveryError, match="concurrency=128"):
        recovery.resume_recovery(
            Path("unused"),
            original_root=Path("unused-original"),
            handoff=object(),
            tokenizer=object(),
            backend=object(),
            environment={},
            concurrency=16,
            resume=True,
        )
    with pytest.raises(recovery.RecoveryError, match="explicit --resume"):
        recovery.resume_recovery(
            Path("unused"),
            original_root=Path("unused-original"),
            handoff=object(),
            tokenizer=object(),
            backend=object(),
            environment={},
            concurrency=128,
            resume=False,
        )
