"""Fail-closed local model runner for clean chk1 teacher targets.

The runner deliberately has no HTTP client and no process-control operations.
It accepts only the two repository-local model trees used by the retrain-v2
contract, performs a read-only ``nvidia-smi`` preflight, generates two strict
Qwen JSON candidates, verifies each deterministically, and asks chk0 to judge
each surviving candidate independently.  A single Qwen repair is allowed only
when neither original candidate passes both gates.

Large model imports are lazy.  Unit tests and planning use an injected backend
or ``dry_run=True`` and therefore do not initialize CUDA or load weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlparse

from open_r1.provenance import fingerprint_artifact_path, validate_sha256

from .contracts import (
    CANDIDATE_SCHEMA_VERSION,
    CRITIC_SCHEMA_VERSION,
    CRITIC_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    canonical_json,
)
from .verifier import (
    CandidateContent,
    VerificationResult,
    critic_accepts,
    select_candidate,
    validate_critic,
    verify_candidate,
)


QWEN_MODEL_RELATIVE_PATH = Path("models/Qwen3.5-9B")
CHK0_MODEL_RELATIVE_PATH = Path("models/DeepSeek-R1-Distill-Llama-8B")
QWEN_SEEDS = (42, 43)
CACHE_SCHEMA_VERSION = "chk1-local-teacher-cache-v3"

_GPU_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total",
    "--format=csv,noheader,nounits",
)
_COMPUTE_QUERY = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
    "--format=csv,noheader,nounits",
)
_CREDENTIAL_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "FOMC_REPORT_API_KEY",
    "OPENAI_API_KEY",
)
_ENDPOINT_ENV_KEYS = (
    "DEEPSEEK_BASE_URL",
    "FOMC_REPORT_BASE_URL",
    "OPENAI_BASE_URL",
)
_DEEPSEEK_REMOTE_HOSTS = (
    "api.deepseek.com",
    "deepseek.com",
    "deepseek.ai",
)
_KNOWN_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|end|>",
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
    "<｜end▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
)
_GENERIC_SPECIAL_TOKEN_RE = re.compile(r"<\|[^<>\n]{1,80}\|>|<｜[^<>\n]{1,80}｜>")
_GEMMA_CHANNEL_RE = re.compile(
    r"^\s*<\|channel\>thought\n(?P<reasoning>.*?)(?:\n)?<channel\|>(?P<answer>.*)\s*$",
    flags=re.DOTALL,
)
_QWEN_CHANNEL_RE = re.compile(
    r"(?:<\|channel\|>\s*(?:analysis|thinking)\s*<\|message\|>)"
    r"(?P<reasoning>.*?)"
    r"(?:<\|channel\|>\s*final\s*<\|message\|>)"
    r"(?P<answer>.*)",
    flags=re.DOTALL | re.IGNORECASE,
)


class LocalModelSafetyError(RuntimeError):
    """Base error for a fail-closed local generation preflight."""


class GpuSafetyError(LocalModelSafetyError):
    """Raised when the two-GPU host is not exclusively available."""


class NetworkSafetyError(LocalModelSafetyError):
    """Raised when credentials or a remote endpoint could enable API access."""


class LocalModelPathError(LocalModelSafetyError):
    """Raised when a configured model is not the pinned repository-local tree."""


class ModelOutputContractError(LocalModelSafetyError):
    """Raised when teacher or critic output does not satisfy its strict contract."""


class CriticRejectedError(ModelOutputContractError):
    """Raised when neither original candidate nor the single repair passes."""

    def __init__(
        self,
        message: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.payload = dict(payload) if payload is not None else None


class CacheIntegrityError(LocalModelSafetyError):
    """Raised when an idempotent cache entry is malformed or conflicts."""


@dataclass(frozen=True)
class GpuDevice:
    index: int
    uuid: str
    name: str
    memory_total_mib: int


@dataclass(frozen=True)
class ComputeProcess:
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mib: int | None


@dataclass(frozen=True)
class GpuSafetyReport:
    safe: bool
    devices: tuple[GpuDevice, ...]
    compute_processes: tuple[ComputeProcess, ...]
    external_processes: tuple[ComputeProcess, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "devices": [asdict(item) for item in self.devices],
            "compute_processes": [asdict(item) for item in self.compute_processes],
            "external_processes": [asdict(item) for item in self.external_processes],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class QwenTeacherConfig:
    model_path: Path
    seeds: tuple[int, int] = QWEN_SEEDS
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"
    thinking: bool = True
    do_sample: bool = True
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 1536
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if self.seeds != QWEN_SEEDS:
            raise ValueError(f"Qwen seeds must be exactly {QWEN_SEEDS}")
        if not self.load_in_4bit or self.quant_type != "nf4" or not self.double_quant:
            raise ValueError("Qwen must use 4-bit NF4 with double quantization")
        if not self.thinking:
            raise ValueError("Qwen thinking mode must be enabled")
        if self.temperature != 0.2 or self.top_p != 0.9:
            raise ValueError("Qwen sampling must use temperature=0.2 and top_p=0.9")
        if self.max_new_tokens != 1536:
            raise ValueError("Qwen max_new_tokens must be 1536")

    def contract(self, *, repo_root: Path) -> dict[str, Any]:
        return {
            "model_path": _repository_relative(self.model_path, repo_root),
            "local_files_only": True,
            "seeds": list(self.seeds),
            "quantization": {
                "load_in_4bit": self.load_in_4bit,
                "bnb_4bit_quant_type": self.quant_type,
                "bnb_4bit_use_double_quant": self.double_quant,
                "bnb_4bit_compute_dtype": self.compute_dtype,
            },
            "chat_template_kwargs": {"enable_thinking": self.thinking},
            "generation": {
                "do_sample": self.do_sample,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
            },
            "device": self.device,
        }


@dataclass(frozen=True)
class Chk0CriticConfig:
    model_path: Path
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"
    seed: int = 42
    do_sample: bool = False
    max_new_tokens: int = 768
    device: str = "cuda:1"

    def contract(self, *, repo_root: Path) -> dict[str, Any]:
        return {
            "model_path": _repository_relative(self.model_path, repo_root),
            "local_files_only": True,
            "seed": self.seed,
            "quantization": {
                "load_in_4bit": self.load_in_4bit,
                "bnb_4bit_quant_type": self.quant_type,
                "bnb_4bit_use_double_quant": self.double_quant,
                "bnb_4bit_compute_dtype": self.compute_dtype,
            },
            "generation": {
                "do_sample": self.do_sample,
                "max_new_tokens": self.max_new_tokens,
            },
            "response_schema": CHK0_CRITIC_JSON_SCHEMA,
            "device": self.device,
        }


@dataclass(frozen=True)
class LocalGenerationRequest:
    role: str
    model_path: Path
    messages: tuple[dict[str, str], ...]
    seed: int
    device: str
    load_in_4bit: bool
    quant_type: str
    double_quant: bool
    compute_dtype: str
    generation_kwargs: Mapping[str, Any]
    chat_template_kwargs: Mapping[str, Any]
    response_schema: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CriticVerdict:
    grounded: bool
    unsupported_claims: tuple[str, ...]
    style_score: int
    reasoning_consistency: bool

    @property
    def accepted(self) -> bool:
        """Apply the one canonical chk1 critic acceptance policy."""

        return critic_accepts(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "unsupported_claims": list(self.unsupported_claims),
            "style_score": self.style_score,
            "reasoning_consistency": self.reasoning_consistency,
        }


class GenerationBackend(Protocol):
    """Minimal injectable interface used by real and mock local backends."""

    def generate(self, request: LocalGenerationRequest) -> str: ...


CHK0_CRITIC_JSON_SCHEMA: dict[str, Any] = {
    "name": "chk1_local_groundedness_critic",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "grounded": {"type": "boolean"},
            "unsupported_claims": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 20,
            },
            "style_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "reasoning_consistency": {"type": "boolean"},
        },
        "required": [
            "grounded",
            "unsupported_claims",
            "style_score",
            "reasoning_consistency",
        ],
        "additionalProperties": False,
    },
}

TEACHER_CANDIDATE_JSON_SCHEMA: dict[str, Any] = {
    "name": "chk1_local_teacher_candidate",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "minLength": 1},
            "final_analysis": {"type": "string", "minLength": 1},
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "uniqueItems": True,
            },
        },
        "required": ["reasoning", "final_analysis", "evidence_ids"],
        "additionalProperties": False,
    },
}


def _parse_csv_line(line: str, expected_columns: int) -> list[str]:
    columns = [item.strip() for item in line.split(",", maxsplit=expected_columns - 1)]
    if len(columns) != expected_columns:
        raise ValueError(
            f"nvidia-smi row has {len(columns)} columns; expected {expected_columns}: {line!r}"
        )
    return columns


def parse_gpu_query(output: str) -> tuple[GpuDevice, ...]:
    """Parse the read-only physical-GPU query, failing on ambiguous rows."""

    devices: list[GpuDevice] = []
    for line in str(output or "").splitlines():
        if not line.strip():
            continue
        index, uuid, name, memory = _parse_csv_line(line, 4)
        devices.append(
            GpuDevice(
                index=int(index),
                uuid=uuid,
                name=name,
                memory_total_mib=int(memory),
            )
        )
    if len({item.index for item in devices}) != len(devices):
        raise ValueError("nvidia-smi returned duplicate GPU indices")
    if len({item.uuid for item in devices}) != len(devices):
        raise ValueError("nvidia-smi returned duplicate GPU UUIDs")
    return tuple(sorted(devices, key=lambda item: item.index))


def parse_compute_query(output: str) -> tuple[ComputeProcess, ...]:
    """Parse active compute processes without inspecting or controlling them."""

    processes: list[ComputeProcess] = []
    normalized = str(output or "").strip()
    if not normalized or normalized.lower().startswith("no running processes"):
        return ()
    for line in normalized.splitlines():
        gpu_uuid, pid, process_name, memory = _parse_csv_line(line, 4)
        memory_value = None if memory in {"", "N/A", "[N/A]"} else int(memory)
        processes.append(
            ComputeProcess(
                gpu_uuid=gpu_uuid,
                pid=int(pid),
                process_name=process_name,
                used_memory_mib=memory_value,
            )
        )
    return tuple(processes)


def _run_readonly_command(argv: Sequence[str]) -> str:
    completed = subprocess.run(
        list(argv),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout


def inspect_gpu_safety(
    *,
    command_runner: Callable[[Sequence[str]], str] = _run_readonly_command,
    allowed_pids: Sequence[int] = (),
    expected_gpu_count: int = 2,
    expected_gpu_name: str | None = "NVIDIA A30",
) -> GpuSafetyReport:
    """Inspect GPU exclusivity using only read-only ``nvidia-smi`` queries.

    Any command, parse, inventory, or ownership ambiguity is represented as an
    unsafe report.  The function never sends a signal and never terminates a
    process, including processes it classifies as external.
    """

    errors: list[str] = []
    devices: tuple[GpuDevice, ...] = ()
    processes: tuple[ComputeProcess, ...] = ()
    try:
        devices = parse_gpu_query(command_runner(_GPU_QUERY))
    except Exception as exc:  # noqa: BLE001 - fail closed on any probe failure
        errors.append(f"gpu inventory unavailable: {type(exc).__name__}: {exc}")
    try:
        processes = parse_compute_query(command_runner(_COMPUTE_QUERY))
    except Exception as exc:  # noqa: BLE001 - fail closed on any probe failure
        errors.append(
            f"compute process inventory unavailable: {type(exc).__name__}: {exc}"
        )

    if len(devices) != expected_gpu_count:
        errors.append(
            f"expected exactly {expected_gpu_count} GPUs, found {len(devices)}"
        )
    if expected_gpu_name is not None:
        wrong_names = [
            item.name for item in devices if expected_gpu_name not in item.name
        ]
        if wrong_names:
            errors.append(f"expected {expected_gpu_name!r} GPUs, found {wrong_names!r}")
    known_uuids = {item.uuid for item in devices}
    unknown_process_gpus = sorted(
        {item.gpu_uuid for item in processes if item.gpu_uuid not in known_uuids}
    )
    if unknown_process_gpus:
        errors.append(
            f"compute processes reference unknown GPU UUIDs: {unknown_process_gpus}"
        )

    allowed = {int(pid) for pid in allowed_pids}
    external = tuple(item for item in processes if item.pid not in allowed)
    if external:
        errors.append(
            "external GPU compute processes are present: "
            + ", ".join(f"pid={item.pid}:{item.process_name}" for item in external)
        )
    return GpuSafetyReport(
        safe=not errors,
        devices=devices,
        compute_processes=processes,
        external_processes=external,
        errors=tuple(errors),
    )


def assert_gpu_safe(**kwargs: Any) -> GpuSafetyReport:
    report = inspect_gpu_safety(**kwargs)
    if not report.safe:
        raise GpuSafetyError("; ".join(report.errors))
    return report


def _is_loopback_endpoint(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def assert_no_deepseek_api_access(
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Reject credentials and non-loopback endpoints before local generation."""

    env = os.environ if environment is None else environment
    present_keys = [
        key for key in _CREDENTIAL_ENV_KEYS if str(env.get(key, "")).strip()
    ]
    if api_key or present_keys:
        raise NetworkSafetyError(
            "API credentials are forbidden for chk1 local generation: "
            + ", ".join(present_keys or ["explicit api_key"])
        )

    endpoints = [value for value in [endpoint] if value]
    endpoints.extend(
        str(env[key]).strip()
        for key in _ENDPOINT_ENV_KEYS
        if str(env.get(key, "")).strip()
    )
    for value in endpoints:
        parsed = urlparse(value)
        hostname = (parsed.hostname or "").lower()
        deepseek_remote = any(
            hostname == host or hostname.endswith(f".{host}")
            for host in _DEEPSEEK_REMOTE_HOSTS
        )
        if deepseek_remote or not _is_loopback_endpoint(value):
            raise NetworkSafetyError(
                f"remote model/API endpoint is forbidden for chk1: {value!r}"
            )


def offline_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a sanitized child environment without mutating the caller."""

    result = dict(os.environ if environment is None else environment)
    for key in (*_CREDENTIAL_ENV_KEYS, *_ENDPOINT_ENV_KEYS):
        result.pop(key, None)
    result.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    return result


def _repository_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise LocalModelPathError(f"model path escapes repository: {path}") from exc


def resolve_local_model_path(
    *,
    repo_root: str | Path,
    model_path: str | Path,
    expected_relative_path: str | Path,
) -> Path:
    """Resolve one pinned model path and reject hub IDs, URLs, and symlink escapes."""

    root = Path(repo_root).resolve()
    raw = str(model_path)
    if "://" in raw or raw.startswith(("hf://", "hub://")):
        raise LocalModelPathError(f"remote model identifiers are forbidden: {raw!r}")
    candidate = Path(model_path)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    expected = (root / Path(expected_relative_path)).resolve()
    _repository_relative(resolved, root)
    _repository_relative(expected, root)
    if resolved != expected:
        raise LocalModelPathError(
            f"model path must be {Path(expected_relative_path).as_posix()!r}, got {raw!r}"
        )
    if not resolved.is_dir():
        raise LocalModelPathError(f"local model directory does not exist: {resolved}")
    return resolved


def build_local_configs(
    *,
    repo_root: str | Path,
    qwen_model_path: str | Path = QWEN_MODEL_RELATIVE_PATH,
    chk0_model_path: str | Path = CHK0_MODEL_RELATIVE_PATH,
) -> tuple[QwenTeacherConfig, Chk0CriticConfig]:
    root = Path(repo_root).resolve()
    qwen = resolve_local_model_path(
        repo_root=root,
        model_path=qwen_model_path,
        expected_relative_path=QWEN_MODEL_RELATIVE_PATH,
    )
    chk0 = resolve_local_model_path(
        repo_root=root,
        model_path=chk0_model_path,
        expected_relative_path=CHK0_MODEL_RELATIVE_PATH,
    )
    return QwenTeacherConfig(model_path=qwen), Chk0CriticConfig(model_path=chk0)


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _local_artifact_provenance_sha256(
    *,
    qwen_config: QwenTeacherConfig,
    critic_config: Chk0CriticConfig,
) -> str:
    """Fallback provenance for direct API callers.

    The batch pipeline supplies its more specific, precomputed provenance.
    Direct callers still hash both complete local trees so cache reuse cannot
    cross a weight, tokenizer, config, or repository-code change.
    """

    payload = {
        "qwen_artifact_sha256": fingerprint_artifact_path(qwen_config.model_path)[
            "sha256"
        ],
        "critic_artifact_sha256": fingerprint_artifact_path(critic_config.model_path)[
            "sha256"
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _resolve_generation_provenance_sha256(
    value: str | None,
    *,
    qwen_config: QwenTeacherConfig,
    critic_config: Chk0CriticConfig,
) -> str:
    if value is None:
        return _local_artifact_provenance_sha256(
            qwen_config=qwen_config,
            critic_config=critic_config,
        )
    return validate_sha256(value, label="generation_provenance_sha256")


def build_cache_key(
    *,
    prompt: str,
    qwen_config: QwenTeacherConfig,
    critic_config: Chk0CriticConfig,
    repo_root: str | Path,
    fact_card: Mapping[str, Any] | None = None,
    same_sample_minutes: str = "",
    generation_provenance_sha256: str | None = None,
) -> str:
    """Build a key from every input that can affect generation or acceptance."""

    root = Path(repo_root).resolve()
    provenance_sha256 = _resolve_generation_provenance_sha256(
        generation_provenance_sha256,
        qwen_config=qwen_config,
        critic_config=critic_config,
    )
    fact_card_json = canonical_json(dict(fact_card)) if fact_card is not None else None
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "critic_schema_version": CRITIC_SCHEMA_VERSION,
        "generation_provenance_sha256": provenance_sha256,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "fact_card_sha256": (
            hashlib.sha256(fact_card_json.encode("utf-8")).hexdigest()
            if fact_card_json is not None
            else None
        ),
        "same_sample_minutes_sha256": hashlib.sha256(
            same_sample_minutes.encode("utf-8")
        ).hexdigest(),
        "system_prompt_sha256": {
            "generator": hashlib.sha256(
                GENERATOR_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "critic": hashlib.sha256(CRITIC_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "repair": hashlib.sha256(REPAIR_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        },
        "qwen": qwen_config.contract(repo_root=root),
        "critic": critic_config.contract(repo_root=root),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def clean_qwen_special_tokens(text: str) -> str:
    cleaned = str(text or "").replace("\r\n", "\n")
    for token in _KNOWN_SPECIAL_TOKENS:
        cleaned = cleaned.replace(token, "")
    cleaned = re.sub(r"^\s*assistant\s*\n", "", cleaned, flags=re.IGNORECASE)
    cleaned = _GENERIC_SPECIAL_TOKEN_RE.sub("", cleaned)
    return cleaned.strip()


def qwen_to_deepseek_response(text: str) -> str:
    """Convert a thinking Qwen completion to DeepSeek's completion-only form.

    DeepSeek's tokenizer already pre-fills the opening ``<think>`` token, so the
    stored completion is ``reasoning\n</think>\nanswer`` with exactly one closing
    boundary and no Qwen control tokens.
    """

    raw = str(text or "").replace("\r\n", "\n").strip()
    for token in ("<|im_start|>assistant", "<|im_start|>", "<|im_end|>"):
        raw = raw.replace(token, "")

    reasoning = ""
    answer = ""
    channel_match = _QWEN_CHANNEL_RE.search(raw)
    gemma_match = _GEMMA_CHANNEL_RE.match(raw)
    think_matches = list(
        re.finditer(r"<think>\s*(.*?)\s*</think>", raw, flags=re.DOTALL | re.IGNORECASE)
    )
    if channel_match:
        reasoning = channel_match.group("reasoning")
        answer = channel_match.group("answer")
    elif gemma_match:
        reasoning = gemma_match.group("reasoning")
        answer = gemma_match.group("answer")
    elif len(think_matches) == 1:
        match = think_matches[0]
        reasoning = match.group(1)
        answer = raw[match.end() :]
    elif not think_matches and raw.count("</think>") == 1:
        reasoning, answer = raw.split("</think>", 1)
    else:
        raise ModelOutputContractError(
            "Qwen output must contain exactly one explicit thinking boundary"
        )

    reasoning = clean_qwen_special_tokens(reasoning)
    answer = clean_qwen_special_tokens(answer)
    if not reasoning:
        raise ModelOutputContractError("Qwen output has empty reasoning")
    if not answer:
        raise ModelOutputContractError("Qwen output has empty final answer")
    if "<think>" in reasoning.lower() or "</think>" in reasoning.lower():
        raise ModelOutputContractError(
            "Qwen reasoning contains a nested thinking boundary"
        )
    if "<think>" in answer.lower() or "</think>" in answer.lower():
        raise ModelOutputContractError("Qwen final answer contains a thinking boundary")
    converted = f"{reasoning}\n</think>\n{answer}"
    if converted.count("</think>") != 1 or _GENERIC_SPECIAL_TOKEN_RE.search(converted):
        raise ModelOutputContractError("Qwen special-token cleanup was incomplete")
    return converted


def _critic_json_text(text: str) -> str:
    raw = str(text or "").strip()
    for token in ("<|im_start|>assistant", "<|im_start|>", "<|im_end|>"):
        raw = raw.replace(token, "")
    if "</think>" in raw:
        if raw.count("</think>") != 1:
            raise ModelOutputContractError(
                "critic output has multiple thinking boundaries"
            )
        raw = raw.split("</think>", 1)[1]
    raw = clean_qwen_special_tokens(raw)
    if raw.startswith("```") or raw.endswith("```"):
        raise ModelOutputContractError("critic JSON must not use Markdown fences")
    return raw.strip()


def _teacher_json_text(text: str) -> str:
    """Extract the JSON answer without sanitizing candidate field contents."""

    raw = str(text or "").replace("\r\n", "\n").strip()
    prefixes = ("<|im_start|>assistant", "<|im_start|>")
    for prefix in prefixes:
        if raw.startswith(prefix):
            raw = raw[len(prefix) :].lstrip()
            break
    suffixes = (
        "<|im_end|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<|eot_id|>",
        "<|end|>",
        "<｜end▁of▁sentence｜>",
    )
    stripped_suffix = True
    while stripped_suffix:
        stripped_suffix = False
        for suffix in suffixes:
            if raw.endswith(suffix):
                raw = raw[: -len(suffix)].rstrip()
                stripped_suffix = True
                break
    if "</think>" in raw:
        if raw.count("</think>") != 1:
            raise ModelOutputContractError(
                "teacher output has multiple thinking boundaries"
            )
        raw = raw.split("</think>", 1)[1].lstrip()
    if raw.startswith("```") or raw.endswith("```"):
        raise ModelOutputContractError("teacher JSON must not use Markdown fences")
    return raw.strip()


def parse_teacher_candidate(text: str) -> dict[str, Any]:
    """Parse the teacher's strict three-key JSON after any native thought."""

    raw = _teacher_json_text(text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelOutputContractError(f"teacher returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelOutputContractError("teacher response must be one JSON object")
    expected = {"reasoning", "final_analysis", "evidence_ids"}
    if set(payload) != expected:
        raise ModelOutputContractError(
            f"teacher keys must be exactly {sorted(expected)}, got {sorted(payload)}"
        )
    reasoning = payload["reasoning"]
    final = payload["final_analysis"]
    evidence_ids = payload["evidence_ids"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ModelOutputContractError("teacher reasoning must be non-empty text")
    if not isinstance(final, str) or not final.strip():
        raise ModelOutputContractError("teacher final_analysis must be non-empty text")
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or any(not isinstance(item, str) or not item.strip() for item in evidence_ids)
    ):
        raise ModelOutputContractError(
            "teacher evidence_ids must be a non-empty string list"
        )
    cleaned_ids = [item.strip() for item in evidence_ids]
    if len(cleaned_ids) != len(set(cleaned_ids)):
        raise ModelOutputContractError("teacher evidence_ids must be unique")
    for field_text in (reasoning, final):
        if (
            _GENERIC_SPECIAL_TOKEN_RE.search(field_text)
            or "<think>" in field_text.casefold()
            or "</think>" in field_text.casefold()
        ):
            raise ModelOutputContractError("teacher JSON contains model-control tokens")
    return {
        "reasoning": reasoning.strip(),
        "final_analysis": final.strip(),
        "evidence_ids": cleaned_ids,
    }


def candidate_to_deepseek_response(candidate: Mapping[str, Any]) -> str:
    parsed = parse_teacher_candidate(canonical_json(dict(candidate)))
    return f"{parsed['reasoning']}\n</think>\n{parsed['final_analysis']}"


def parse_critic_verdict(text: str) -> CriticVerdict:
    """Decode one independent exact chk0 groundedness verdict."""

    raw = _critic_json_text(text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelOutputContractError(f"critic returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelOutputContractError("critic response must be one JSON object")
    try:
        validated = validate_critic(payload)
    except ValueError as exc:
        raise ModelOutputContractError(str(exc)) from exc
    return CriticVerdict(
        grounded=validated["grounded"],
        unsupported_claims=tuple(validated["unsupported_claims"]),
        style_score=validated["style_score"],
        reasoning_consistency=validated["reasoning_consistency"],
    )


def _teacher_request(
    config: QwenTeacherConfig, *, prompt: str, seed: int
) -> LocalGenerationRequest:
    return LocalGenerationRequest(
        role="qwen_teacher",
        model_path=config.model_path,
        messages=(
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ),
        seed=seed,
        device=config.device,
        load_in_4bit=config.load_in_4bit,
        quant_type=config.quant_type,
        double_quant=config.double_quant,
        compute_dtype=config.compute_dtype,
        generation_kwargs={
            "do_sample": config.do_sample,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
        },
        chat_template_kwargs={"enable_thinking": config.thinking},
        response_schema=TEACHER_CANDIDATE_JSON_SCHEMA,
    )


def _critic_request(
    config: Chk0CriticConfig,
    *,
    prompt: str,
    candidate: Mapping[str, Any],
    candidate_id: int | str,
) -> LocalGenerationRequest:
    quoted_payload = _canonical_json(
        {
            "evidence_prompt": prompt,
            "candidate_seed": candidate_id,
            "candidate": dict(candidate),
            "required_json_schema": CHK0_CRITIC_JSON_SCHEMA["schema"],
        }
    )
    return LocalGenerationRequest(
        role="chk0_critic",
        model_path=config.model_path,
        messages=(
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": quoted_payload},
        ),
        seed=config.seed,
        device=config.device,
        load_in_4bit=config.load_in_4bit,
        quant_type=config.quant_type,
        double_quant=config.double_quant,
        compute_dtype=config.compute_dtype,
        generation_kwargs={
            "do_sample": config.do_sample,
            "max_new_tokens": config.max_new_tokens,
        },
        chat_template_kwargs={},
        response_schema=CHK0_CRITIC_JSON_SCHEMA,
    )


def _repair_request(
    config: QwenTeacherConfig,
    *,
    prompt: str,
    rejected_candidates: Mapping[str, Any],
    error_codes: Sequence[str],
) -> LocalGenerationRequest:
    repair_payload = _canonical_json(
        {
            "original_evidence_prompt": prompt,
            "rejected_candidates": {
                str(candidate_id): candidate
                for candidate_id, candidate in sorted(rejected_candidates.items())
            },
            "verifier_error_codes": sorted(set(str(code) for code in error_codes)),
            "required_json_schema": TEACHER_CANDIDATE_JSON_SCHEMA["schema"],
        }
    )
    return LocalGenerationRequest(
        role="qwen_repair",
        model_path=config.model_path,
        messages=(
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": repair_payload},
        ),
        seed=42,
        device=config.device,
        load_in_4bit=config.load_in_4bit,
        quant_type=config.quant_type,
        double_quant=config.double_quant,
        compute_dtype=config.compute_dtype,
        generation_kwargs={
            "do_sample": False,
            "max_new_tokens": config.max_new_tokens,
        },
        chat_template_kwargs={"enable_thinking": config.thinking},
        response_schema=TEACHER_CANDIDATE_JSON_SCHEMA,
    )


def build_generation_plan(
    *,
    prompt: str,
    repo_root: str | Path,
    fact_card: Mapping[str, Any] | None = None,
    same_sample_minutes: str = "",
    qwen_model_path: str | Path = QWEN_MODEL_RELATIVE_PATH,
    chk0_model_path: str | Path = CHK0_MODEL_RELATIVE_PATH,
    generation_provenance_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    qwen, critic = build_local_configs(
        repo_root=root,
        qwen_model_path=qwen_model_path,
        chk0_model_path=chk0_model_path,
    )
    provenance_sha256 = _resolve_generation_provenance_sha256(
        generation_provenance_sha256,
        qwen_config=qwen,
        critic_config=critic,
    )
    key = build_cache_key(
        prompt=prompt,
        qwen_config=qwen,
        critic_config=critic,
        repo_root=root,
        fact_card=fact_card,
        same_sample_minutes=same_sample_minutes,
        generation_provenance_sha256=provenance_sha256,
    )
    fact_card_json = canonical_json(dict(fact_card)) if fact_card is not None else None
    return {
        "status": "dry_run",
        "cache_key": key,
        "generation_provenance_sha256": provenance_sha256,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "qwen": qwen.contract(repo_root=root),
        "critic": critic.contract(repo_root=root),
        "verification": {
            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "critic_schema_version": CRITIC_SCHEMA_VERSION,
            "fact_card_sha256": (
                hashlib.sha256(fact_card_json.encode("utf-8")).hexdigest()
                if fact_card_json is not None
                else None
            ),
            "same_sample_minutes_sha256": hashlib.sha256(
                same_sample_minutes.encode("utf-8")
            ).hexdigest(),
            "critic_acceptance": {
                "grounded": True,
                "unsupported_claims": [],
                "minimum_style_score": 4,
                "reasoning_consistency": True,
            },
            "maximum_repairs": 1,
        },
        "network": "local_files_only",
        "gpu_preflight_required_for_execution": True,
    }


def _cache_path(cache_dir: str | Path, cache_key: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", cache_key):
        raise CacheIntegrityError("cache key must be a lowercase SHA-256 digest")
    return Path(cache_dir) / cache_key[:2] / f"{cache_key}.json"


def load_cache_entry(
    cache_dir: str | Path,
    cache_key: str,
    *,
    expected_generation_provenance_sha256: str | None = None,
) -> dict[str, Any] | None:
    path = _cache_path(cache_dir, cache_key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheIntegrityError(f"invalid cache entry {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CacheIntegrityError(f"cache entry is not an object: {path}")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheIntegrityError(f"cache schema mismatch: {path}")
    if payload.get("cache_key") != cache_key:
        raise CacheIntegrityError(f"cache key mismatch: {path}")
    try:
        cached_provenance = validate_sha256(
            payload.get("generation_provenance_sha256"),
            label="cached generation_provenance_sha256",
        )
    except ValueError as exc:
        raise CacheIntegrityError(f"cache provenance is invalid: {path}") from exc
    if expected_generation_provenance_sha256 is not None and (
        cached_provenance
        != validate_sha256(
            expected_generation_provenance_sha256,
            label="expected generation_provenance_sha256",
        )
    ):
        raise CacheIntegrityError(
            f"cached generation provenance does not match execution: {path}"
        )
    if payload.get("selected_response"):
        selected = str(payload["selected_response"])
        if selected.count("</think>") != 1:
            raise CacheIntegrityError(
                f"cached response has invalid DeepSeek format: {path}"
            )
    return payload


def store_cache_entry(cache_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    cache_key = str(payload.get("cache_key", ""))
    try:
        validate_sha256(
            payload.get("generation_provenance_sha256"),
            label="cache generation_provenance_sha256",
        )
    except ValueError as exc:
        raise CacheIntegrityError("cache payload has invalid provenance") from exc
    path = _cache_path(cache_dir, cache_key)
    serialized = _canonical_json(dict(payload)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = path.read_text(encoding="utf-8")
        if existing != serialized:
            raise CacheIntegrityError(f"immutable cache collision at {path}")
        return path
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


class MockGenerationBackend:
    """Deterministic backend for tests and offline orchestration rehearsals."""

    def __init__(self, outputs: Mapping[tuple[str, int | str] | str, str]):
        self.outputs = dict(outputs)
        self.requests: list[LocalGenerationRequest] = []

    def generate(self, request: LocalGenerationRequest) -> str:
        self.requests.append(request)
        key: tuple[str, int | str] | str
        if request.role == "chk0_critic":
            try:
                critic_payload = json.loads(request.messages[-1]["content"])
                candidate_id = critic_payload["candidate_seed"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ModelOutputContractError(
                    "mock critic request is missing candidate_seed"
                ) from exc
            candidate_key = (request.role, candidate_id)
            key = candidate_key if candidate_key in self.outputs else "critic"
        else:
            key = (request.role, request.seed)
        if key not in self.outputs:
            raise ModelOutputContractError(f"mock output is missing for {key!r}")
        return self.outputs[key]


class TransformersLocalBackend:
    """Lazy, local-files-only Transformers backend.

    Constructing the object is cheap; CUDA and model weights are touched only by
    ``generate``.  This class intentionally exposes no URL, token, or hub ID.
    """

    def __init__(self) -> None:
        self._loaded: dict[tuple[str, str], tuple[Any, Any]] = {}

    @staticmethod
    def _dtype(torch_module: Any, name: str) -> Any:
        if name != "bfloat16":
            raise LocalModelSafetyError(f"unsupported local compute dtype: {name}")
        return torch_module.bfloat16

    def _load(self, request: LocalGenerationRequest) -> tuple[Any, Any]:
        key = (str(request.model_path), request.device)
        if key in self._loaded:
            return self._loaded[key]
        import torch  # lazy: no CUDA initialization during import/dry-run tests
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        quantization = BitsAndBytesConfig(
            load_in_4bit=request.load_in_4bit,
            bnb_4bit_quant_type=request.quant_type,
            bnb_4bit_use_double_quant=request.double_quant,
            bnb_4bit_compute_dtype=self._dtype(torch, request.compute_dtype),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(request.model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        config = AutoConfig.from_pretrained(
            str(request.model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        architectures = set(getattr(config, "architectures", ()) or ())
        if "Qwen3_5ForConditionalGeneration" in architectures:
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError as exc:  # pragma: no cover - version contract
                raise LocalModelSafetyError(
                    "Installed Transformers lacks AutoModelForImageTextToText "
                    "required by local Qwen3.5-9B"
                ) from exc
            model_class = AutoModelForImageTextToText
        else:
            model_class = AutoModelForCausalLM
        model = model_class.from_pretrained(
            str(request.model_path),
            local_files_only=True,
            trust_remote_code=True,
            quantization_config=quantization,
            torch_dtype=self._dtype(torch, request.compute_dtype),
            device_map={"": request.device},
        )
        model.eval()
        self._loaded[key] = (tokenizer, model)
        return tokenizer, model

    def generate(self, request: LocalGenerationRequest) -> str:
        import torch

        tokenizer, model = self._load(request)
        rendered = tokenizer.apply_chat_template(
            list(request.messages),
            tokenize=False,
            add_generation_prompt=True,
            **dict(request.chat_template_kwargs),
        )
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to(request.device) for key, value in encoded.items()}
        generator = torch.Generator(device=request.device).manual_seed(request.seed)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                **dict(request.generation_kwargs),
                generator=generator,
            )
        prompt_tokens = encoded["input_ids"].shape[-1]
        return tokenizer.decode(
            generated[0, prompt_tokens:],
            skip_special_tokens=False,
        )


def _candidate_payload(candidate: CandidateContent | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "reasoning": candidate.reasoning,
        "final_analysis": candidate.final_analysis,
        "evidence_ids": list(candidate.evidence_ids),
    }


def _contract_failure(
    verification: VerificationResult,
    *,
    message: str,
) -> VerificationResult:
    details = dict(verification.details)
    details["output_contract_error"] = message
    return VerificationResult(
        passed=False,
        error_codes=tuple(
            dict.fromkeys((*verification.error_codes, "candidate_output_contract"))
        ),
        details=details,
        candidate=None,
    )


def _verify_teacher_output(
    raw: str,
    *,
    fact_card: Mapping[str, Any],
    same_sample_minutes: str,
) -> VerificationResult:
    """Normalize only outer model framing, then run the deterministic verifier."""

    try:
        candidate_json = _teacher_json_text(raw)
    except ModelOutputContractError as exc:
        verification = verify_candidate(
            raw,
            fact_card=fact_card,
            same_sample_minutes=same_sample_minutes,
        )
        return _contract_failure(verification, message=str(exc))

    verification = verify_candidate(
        candidate_json,
        fact_card=fact_card,
        same_sample_minutes=same_sample_minutes,
    )
    if not verification.passed:
        return verification
    try:
        # ``verify_candidate`` owns the deterministic semantic checks; this
        # parser adds the local generation contract's broader control-token
        # hygiene before the output can reach chk0.
        parse_teacher_candidate(candidate_json)
    except ModelOutputContractError as exc:
        return _contract_failure(verification, message=str(exc))
    return verification


def _critic_rejection_codes(
    verdict: CriticVerdict | None,
    *,
    contract_error: str | None,
) -> list[str]:
    if contract_error is not None:
        return ["critic_output_contract"]
    if verdict is None:
        return ["critic_missing"]
    codes: list[str] = []
    if not verdict.grounded:
        codes.append("critic_not_grounded")
    if verdict.unsupported_claims:
        codes.append("critic_unsupported_claims")
    if verdict.style_score < 4:
        codes.append("critic_style_score_below_4")
    if not verdict.reasoning_consistency:
        codes.append("critic_reasoning_inconsistent")
    return codes


def run_local_chk1_generation(
    *,
    prompt: str,
    repo_root: str | Path,
    fact_card: Mapping[str, Any] | None = None,
    same_sample_minutes: str = "",
    cache_dir: str | Path | None = None,
    qwen_model_path: str | Path = QWEN_MODEL_RELATIVE_PATH,
    chk0_model_path: str | Path = CHK0_MODEL_RELATIVE_PATH,
    backend: GenerationBackend | None = None,
    dry_run: bool = False,
    mock: bool = False,
    environment: Mapping[str, str] | None = None,
    gpu_command_runner: Callable[[Sequence[str]], str] = _run_readonly_command,
    allowed_pids: Sequence[int] = (),
    generation_provenance_sha256: str | None = None,
) -> dict[str, Any]:
    """Generate one chk1 target locally, or return its side-effect-free plan."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(fact_card, Mapping):
        raise ValueError("fact_card must be provided as a mapping")
    if not isinstance(same_sample_minutes, str):
        raise ValueError("same_sample_minutes must be a string")
    # Snapshot the nested mapping once so the cache key, dry-run plan, and
    # deterministic verifier all observe byte-for-byte identical evidence.
    frozen_fact_card = json.loads(canonical_json(dict(fact_card)))
    assert_no_deepseek_api_access(environment=environment)
    root = Path(repo_root).resolve()
    qwen, critic = build_local_configs(
        repo_root=root,
        qwen_model_path=qwen_model_path,
        chk0_model_path=chk0_model_path,
    )
    provenance_sha256 = _resolve_generation_provenance_sha256(
        generation_provenance_sha256,
        qwen_config=qwen,
        critic_config=critic,
    )
    cache_key = build_cache_key(
        prompt=prompt,
        qwen_config=qwen,
        critic_config=critic,
        repo_root=root,
        fact_card=frozen_fact_card,
        same_sample_minutes=same_sample_minutes,
        generation_provenance_sha256=provenance_sha256,
    )
    plan = build_generation_plan(
        prompt=prompt,
        repo_root=root,
        fact_card=frozen_fact_card,
        same_sample_minutes=same_sample_minutes,
        qwen_model_path=qwen_model_path,
        chk0_model_path=chk0_model_path,
        generation_provenance_sha256=provenance_sha256,
    )
    if dry_run:
        return plan

    if cache_dir is not None:
        cached = load_cache_entry(
            cache_dir,
            cache_key,
            expected_generation_provenance_sha256=provenance_sha256,
        )
        if cached is not None:
            result = dict(cached)
            result["cache_hit"] = True
            if result.get("status") == "accepted" and result.get("selected_response"):
                verifier_payload = result.get("verifier")
                critic_payload = result.get("critic")
                selected_candidate = result.get("selected_candidate")
                try:
                    cached_critic = validate_critic(critic_payload)
                    expected_response = candidate_to_deepseek_response(
                        selected_candidate
                    )
                except (TypeError, ValueError, ModelOutputContractError) as exc:
                    raise CacheIntegrityError(
                        "cached accepted result has invalid candidate or critic data"
                    ) from exc
                if (
                    not isinstance(verifier_payload, Mapping)
                    or verifier_payload.get("passed") is not True
                    or verifier_payload.get("error_codes") not in ([], ())
                    or not critic_accepts(cached_critic)
                    or expected_response != result["selected_response"]
                ):
                    raise CacheIntegrityError(
                        "cached accepted result no longer satisfies chk1 gates"
                    )
                return result
            if result.get("status") == "rejected":
                if result.get("selected_response") is not None:
                    raise CacheIntegrityError(
                        "cached rejected result unexpectedly contains a selected response"
                    )
                raise CriticRejectedError(
                    "cached chk1 candidate set was rejected after its single repair",
                    payload=result,
                )
            raise CacheIntegrityError(
                "cache entry is neither an accepted target nor a quarantined rejection"
            )

    if not mock:
        assert_gpu_safe(
            command_runner=gpu_command_runner,
            allowed_pids=allowed_pids,
            expected_gpu_count=2,
            expected_gpu_name="NVIDIA A30",
        )
    if backend is None:
        if mock:
            raise ValueError("mock=True requires an injected backend")
        backend = TransformersLocalBackend()

    raw_candidates: dict[str, str] = {}
    candidates: dict[str, dict[str, Any] | None] = {}
    verifications: dict[str, VerificationResult] = {}
    for seed in QWEN_SEEDS:
        request = _teacher_request(qwen, prompt=prompt, seed=seed)
        raw = backend.generate(request)
        seed_key = str(seed)
        raw_candidates[seed_key] = raw
        verification = _verify_teacher_output(
            raw,
            fact_card=frozen_fact_card,
            same_sample_minutes=same_sample_minutes,
        )
        verifications[seed_key] = verification
        candidates[seed_key] = _candidate_payload(verification.candidate)

    raw_critics: dict[str, str] = {}
    critic_verdicts: dict[str, CriticVerdict | None] = {
        str(seed): None for seed in QWEN_SEEDS
    }
    critic_contract_errors: dict[str, str | None] = {
        str(seed): None for seed in QWEN_SEEDS
    }
    for seed in QWEN_SEEDS:
        seed_key = str(seed)
        verification = verifications[seed_key]
        candidate = candidates[seed_key]
        if not verification.passed or candidate is None:
            continue
        request = _critic_request(
            critic,
            prompt=prompt,
            candidate=candidate,
            candidate_id=seed,
        )
        raw_critic = backend.generate(request)
        raw_critics[seed_key] = raw_critic
        try:
            critic_verdicts[seed_key] = parse_critic_verdict(raw_critic)
        except ModelOutputContractError as exc:
            critic_contract_errors[seed_key] = str(exc)

    accepted_seeds = [
        seed
        for seed in QWEN_SEEDS
        if (
            critic_verdicts[str(seed)] is not None
            and critic_verdicts[str(seed)].accepted
        )
    ]
    selected_seed: int | None = None
    if len(accepted_seeds) == 1:
        selected_seed = accepted_seeds[0]
    elif len(accepted_seeds) == 2:
        ranked_inputs = [
            (
                verifications[str(seed)],
                critic_verdicts[str(seed)].to_dict(),
            )
            for seed in accepted_seeds
            if critic_verdicts[str(seed)] is not None
        ]
        selected_index = select_candidate(ranked_inputs)
        if selected_index is not None:
            selected_seed = accepted_seeds[selected_index]

    repair: dict[str, Any] | None = None
    selected_from: str | None = (
        str(selected_seed) if selected_seed is not None else None
    )
    selected_verification = (
        verifications[str(selected_seed)] if selected_seed is not None else None
    )
    selected_verdict = (
        critic_verdicts[str(selected_seed)] if selected_seed is not None else None
    )
    selected_candidate = (
        candidates[str(selected_seed)] if selected_seed is not None else None
    )

    if selected_seed is None:
        rejected_candidates: dict[str, Any] = {}
        repair_error_codes: list[str] = []
        for seed in QWEN_SEEDS:
            seed_key = str(seed)
            verification = verifications[seed_key]
            verdict = critic_verdicts[seed_key]
            contract_error = critic_contract_errors[seed_key]
            critic_codes = (
                _critic_rejection_codes(verdict, contract_error=contract_error)
                if verification.passed
                else []
            )
            repair_error_codes.extend(verification.error_codes)
            repair_error_codes.extend(critic_codes)
            rejected_candidates[seed_key] = {
                "candidate": candidates[seed_key],
                "verifier_error_codes": list(verification.error_codes),
                "critic": verdict.to_dict() if verdict is not None else None,
                "critic_contract_error": contract_error,
                "critic_rejection_codes": critic_codes,
            }
        if not repair_error_codes:
            repair_error_codes.append("candidate_selection_failed")

        repair_request = _repair_request(
            qwen,
            prompt=prompt,
            rejected_candidates=rejected_candidates,
            error_codes=repair_error_codes,
        )
        raw_repair = backend.generate(repair_request)
        repair_verification = _verify_teacher_output(
            raw_repair,
            fact_card=frozen_fact_card,
            same_sample_minutes=same_sample_minutes,
        )
        repair_candidate = _candidate_payload(repair_verification.candidate)
        raw_repair_critic: str | None = None
        repair_verdict: CriticVerdict | None = None
        repair_critic_contract_error: str | None = None
        if repair_verification.passed and repair_candidate is not None:
            repair_critic_request = _critic_request(
                critic,
                prompt=prompt,
                candidate=repair_candidate,
                candidate_id="repair",
            )
            raw_repair_critic = backend.generate(repair_critic_request)
            try:
                repair_verdict = parse_critic_verdict(raw_repair_critic)
            except ModelOutputContractError as exc:
                repair_critic_contract_error = str(exc)
        repair = {
            "raw_candidate": raw_repair,
            "candidate": repair_candidate,
            "verification": repair_verification.to_dict(),
            "raw_critic": raw_repair_critic,
            "critic": repair_verdict.to_dict() if repair_verdict is not None else None,
            "critic_contract_error": repair_critic_contract_error,
            "critic_rejection_codes": (
                _critic_rejection_codes(
                    repair_verdict,
                    contract_error=repair_critic_contract_error,
                )
                if repair_verification.passed
                else []
            ),
        }
        if repair_verdict is not None and repair_verdict.accepted:
            selected_from = "repair"
            selected_verification = repair_verification
            selected_verdict = repair_verdict
            selected_candidate = repair_candidate

    selected_response = (
        selected_verification.candidate.response
        if selected_verification is not None
        and selected_verification.candidate is not None
        and selected_verdict is not None
        and selected_verdict.accepted
        else None
    )
    status = "accepted" if selected_response is not None else "rejected"
    selected_raw_critic = (
        repair.get("raw_critic")
        if selected_from == "repair" and repair is not None
        else raw_critics.get(selected_from or "")
    )
    rejection_error_codes: list[str] = []
    if status == "rejected" and repair is not None:
        rejection_error_codes.extend(repair["verification"]["error_codes"])
        rejection_error_codes.extend(repair["critic_rejection_codes"])
    payload: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "generation_provenance_sha256": provenance_sha256,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "contracts": {"qwen": plan["qwen"], "critic": plan["critic"]},
        "raw_candidates": raw_candidates,
        "candidates": candidates,
        "verifications": {
            seed: verification.to_dict() for seed, verification in verifications.items()
        },
        "raw_critics": raw_critics,
        "critics": {
            seed: verdict.to_dict() if verdict is not None else None
            for seed, verdict in critic_verdicts.items()
        },
        "critic_contract_errors": critic_contract_errors,
        "repair": repair,
        "repair_attempted": repair is not None,
        "selected_from": selected_from,
        "selected_seed": selected_seed if selected_from != "repair" else None,
        "selected_candidate": selected_candidate,
        "verifier": (
            selected_verification.to_dict()
            if selected_verification is not None
            else None
        ),
        "critic": (
            selected_verdict.to_dict() if selected_verdict is not None else None
        ),
        "raw_critic": selected_raw_critic,
        "rejection_error_codes": list(dict.fromkeys(rejection_error_codes)),
        "selected_response": selected_response,
        "status": status,
        "cache_hit": False,
    }
    cache_payload = dict(payload)
    cache_payload.pop("cache_hit")
    if cache_dir is not None:
        store_cache_entry(cache_dir, cache_payload)
    if status == "rejected":
        raise CriticRejectedError(
            "chk0 rejected both Qwen candidates and the single repair",
            payload=payload,
        )
    return payload


# Explicit aliases keep call sites readable without weakening any invariant.
preflight_gpu_safety = assert_gpu_safe
generate_chk1_target = run_local_chk1_generation


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CANDIDATE_SCHEMA_VERSION",
    "CHK0_CRITIC_JSON_SCHEMA",
    "CHK0_MODEL_RELATIVE_PATH",
    "CRITIC_SCHEMA_VERSION",
    "QWEN_MODEL_RELATIVE_PATH",
    "QWEN_SEEDS",
    "TEACHER_CANDIDATE_JSON_SCHEMA",
    "CacheIntegrityError",
    "Chk0CriticConfig",
    "ComputeProcess",
    "CriticRejectedError",
    "CriticVerdict",
    "GenerationBackend",
    "GpuDevice",
    "GpuSafetyError",
    "GpuSafetyReport",
    "LocalGenerationRequest",
    "LocalModelPathError",
    "LocalModelSafetyError",
    "MockGenerationBackend",
    "ModelOutputContractError",
    "NetworkSafetyError",
    "QwenTeacherConfig",
    "TransformersLocalBackend",
    "assert_gpu_safe",
    "assert_no_deepseek_api_access",
    "build_cache_key",
    "build_generation_plan",
    "build_local_configs",
    "candidate_to_deepseek_response",
    "clean_qwen_special_tokens",
    "generate_chk1_target",
    "inspect_gpu_safety",
    "load_cache_entry",
    "offline_environment",
    "parse_compute_query",
    "parse_critic_verdict",
    "parse_gpu_query",
    "parse_teacher_candidate",
    "preflight_gpu_safety",
    "qwen_to_deepseek_response",
    "resolve_local_model_path",
    "run_local_chk1_generation",
    "store_cache_entry",
]
