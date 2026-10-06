from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.generation import publish_paper_chk2_chk1_analysis_rewrite as publisher
from jobs.generation import paper_chk2_official_reference_v2 as official_reference
from jobs.generation.publish_paper_chk2_chk1_analysis_rewrite import (
    BOUNDARY,
    DATASET_ROLE,
    STYLE_DIMENSIONS,
    TERMINAL_SCHEMA_VERSION,
    PublicationError,
    publish_release,
    render_user_prompt,
    sha256_file,
    sha256_text,
    verify_release,
)


SPLITS = ("train", "validation", "test")
SOURCE_SPLITS = {"train": "train", "validation": "eval", "test": "test"}
LINEAGE = dict(publisher.REQUIRED_LINEAGE)


class FakeDeepSeekTokenizer:
    bos_token = "<BOS>"
    eos_token = "<EOS>"
    bos_token_id = 1
    eos_token_id = 2

    def _encode(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        if text.startswith(self.bos_token):
            text = text[len(self.bos_token) :]
        has_eos = text.endswith(self.eos_token)
        if has_eos:
            text = text[: -len(self.eos_token)]
        ids.extend(10 + ord(character) for character in text)
        if has_eos:
            ids.append(self.eos_token_id)
        return ids

    def __call__(self, *, text: str):
        return {"input_ids": self._encode(text)}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        truncation: bool = False,
        return_dict: bool = False,
    ):
        assert add_generation_prompt is True
        assert truncation is False
        rendered = self.bos_token
        for message in messages:
            rendered += f"[{message['role']}]\n{message['content']}\n"
        rendered += "[assistant]\n<think>\n"
        return self._encode(rendered) if tokenize else rendered


def test_publisher_tokenizer_loader_uses_the_exact_runtime_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = object()
    calls: list[tuple[Path, dict[str, object]]] = []

    class AutoTokenizerStub:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: object) -> object:
            calls.append((path, kwargs))
            return loaded

    verified: list[object] = []
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=AutoTokenizerStub),
    )
    monkeypatch.setattr(
        publisher.generator,
        "_verify_tokenizer_runtime_contract",
        lambda tokenizer: verified.append(tokenizer),
    )

    assert publisher._load_tokenizer(tmp_path) is loaded
    assert calls == [(tmp_path, publisher.generator.TOKENIZER_LOADER_KWARGS)]
    assert calls[0][1]["fix_mistral_regex"] is False
    assert calls[0][1]["fix_mistral_regex"] is not True
    assert verified == [loaded]


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _file_descriptor(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    result: dict[str, object] = {
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".jsonl":
        result["rows"] = len(path.read_text(encoding="utf-8").splitlines())
    return result


def _id_digest(values: list[str]) -> str:
    return sha256_text("".join(f"{value}\n" for value in sorted(values)))


def _source_id_digest(values: list[str]) -> str:
    return sha256_text(
        json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _provider(role: str, payload: dict) -> dict:
    fields = list(publisher.generator._role_fields(role))
    assert set(payload) == set(fields)
    projection = {
        "role": role,
        "allowed_fields": fields,
        "field_value_sha256": {
            field: sha256_text(
                json.dumps(
                    payload[field],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            for field in fields
        },
        "canonical_payload_sha256": sha256_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "official_text_policy": (
            "corresponding_meeting_pre_action_analysis_body"
            if role.startswith("validator_b")
            else "forbidden"
        ),
        "api_key_env": "DEEPSEEK_API_KEY",
        "plaintext_credential_persisted": False,
    }
    return {
        "response_id": "response-id",
        "returned_model": "deepseek-v4-flash",
        "system_fingerprint": "fixed-fingerprint",
        "finish_reason": "stop",
        "created": None,
        "usage": {},
        "raw_reasoning_sha256": sha256_text(""),
        "raw_content_sha256": sha256_text("{}"),
        "request_projection": projection,
    }


def _source_report(analysis: str, provided: str, *, passed: bool, adjudication=False):
    claims_key = "reviewed_claims" if adjudication else "claims"
    issues_key = "confirmed_blocking_issues" if adjudication else "blocking_issues"
    claim = {
        "analysis_span": analysis,
        "evidence_span": provided if passed else None,
        "verdict": "supported" if passed else "unsupported",
        "issue_code": None if passed else "FACTUAL_UNSUPPORTED",
    }
    issue = {
        "analysis_span": analysis,
        "evidence_span": None,
        "issue_code": "FACTUAL_UNSUPPORTED",
    }
    return {
        claims_key: [claim],
        issues_key: [] if passed else [issue],
        "overall_pass": not passed,
        "machine_pass": passed,
    }


def _validator_a_result(analysis: str, reasoning: str, answer: str, *, passed=True):
    reasons = [] if passed else ["material source claim omitted"]
    return {
        "source_claims": [
            {
                "source_span": analysis,
                "reasoning_evidence": reasoning if passed else None,
                "rewrite_evidence": answer if passed else None,
                "rewrite_verdict": "entailed" if passed else "omitted",
                "reasoning_verdict": "covered" if passed else "not_covered",
            }
        ],
        "rewrite_claims": [
            {
                "rewrite_span": answer,
                "source_evidence": analysis,
                "verdict": "supported",
            }
        ],
        "reasoning_issues": [],
        "bidirectional_entailment": passed,
        "reasoning_compatible": passed,
        "issues": reasons,
        "overall_pass": passed,
        "reasoning_compatible_for_gate": passed,
        "overall_pass_for_gate": passed,
        "reported_issues": reasons,
        "ignored_operational_issues": [],
        "ignored_lexical_overlap_issues": [],
        "ignored_operational_reasoning_issues": [],
        "complete_native_cot_preserved": True,
        "reasoning_meta_used_for_rejection": False,
        "reported_bidirectional_entailment": passed,
        "reported_reasoning_compatible": passed,
        "reported_overall_pass": passed,
        "local_verdict_recomputed": True,
        "contract_reasons": [],
        "machine_pass": passed,
        "contract_repair_used": False,
        "contract_repair": None,
    }


def _unrun() -> dict:
    return {
        "complete": False,
        "machine_pass": False,
        "reasons": [],
        "result": None,
        "provider": {},
    }


def _source_fixture(candidate_root: Path, generation_root: Path):
    source_rows: dict[str, list[dict]] = {}
    repair_rows: list[dict] = []
    all_ids: list[str] = []
    for split_number, split in enumerate(SPLITS):
        source_split = SOURCE_SPLITS[split]
        candidates: list[dict] = []
        generations: list[dict] = []
        source_rows[split] = []
        for index in range(2):
            sample_id = f"chk1-{split}-{index}"
            meeting = f"2026-0{split_number + 1}-{index + 1:02d}"
            prompt = f"Original chk1 prompt for {sample_id}."
            provided = json.dumps(
                {"evidence": [{"metric": "activity", "value": str(index + 1)}]},
                separators=(",", ":"),
            )
            analysis = (
                f"Economic activity in {split} row {index} remained steady, "
                f"with the observed index at {index + 1}."
            )
            response = f"Original chk1 reasoning {index}.{BOUNDARY}{analysis}"
            candidate = {
                "prompt": prompt,
                "provided_data": provided,
                "response": response,
            }
            generation = {
                "sample_id": sample_id,
                "split": source_split,
                "meeting_date": meeting,
                "atomic_topic": f"Topic {index}",
                "section_style_id": f"style-{index % 2}",
                "prompt_sha256": sha256_text(prompt),
                "provided_data_sha256": sha256_text(provided),
                "final_analysis_sha256": sha256_text(analysis),
            }
            repair = {
                "sample_id": sample_id,
                "split": source_split,
                "source_line_number": index + 1,
                "generation_manifest_line_number": index + 1,
                "prompt_sha256": sha256_text(prompt),
                "provided_data_sha256": sha256_text(provided),
                "candidate_response_sha256": sha256_text(response),
                "source_response_sha256": (
                    "f" * 64
                    if split == "train" and index == 0
                    else sha256_text(response)
                ),
                "candidate_answer_sha256": sha256_text(analysis),
                "source_answer_sha256": sha256_text(analysis),
            }
            candidates.append(candidate)
            generations.append(generation)
            repair_rows.append(repair)
            source_rows[split].append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "source_split": source_split,
                    "source_index": index + 1,
                    "meeting_date": meeting,
                    "atomic_topic": generation["atomic_topic"],
                    "section_style_id": generation["section_style_id"],
                    "provided_data": provided,
                    "analysis": analysis,
                }
            )
            all_ids.append(sample_id)
        _jsonl(candidate_root / f"analysis_sft/{source_split}.jsonl", candidates)
        _jsonl(generation_root / f"{source_split}.jsonl", generations)
    _jsonl(candidate_root / "audits/repair_manifest.jsonl", repair_rows)
    source_paths = {
        "candidate/audits/repair_manifest.jsonl": candidate_root
        / "audits/repair_manifest.jsonl",
        **{
            f"candidate/analysis_sft/{source_split}.jsonl": candidate_root
            / f"analysis_sft/{source_split}.jsonl"
            for source_split in SOURCE_SPLITS.values()
        },
        **{
            f"generation/manifests/{source_split}.jsonl": generation_root
            / f"{source_split}.jsonl"
            for source_split in SOURCE_SPLITS.values()
        },
    }
    pins = {key: sha256_file(path) for key, path in source_paths.items()}
    bindings = {
        key: {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": len(path.read_text(encoding="utf-8").splitlines()),
        }
        for key, path in source_paths.items()
    }
    return source_rows, pins, bindings, _source_id_digest(all_ids)


def _official_reference_fixture(root: Path, source_rows: dict[str, list[dict]]):
    raw_root = root / "raw"
    roster_rows: list[dict] = []
    fields = (
        "line_id",
        "section_name",
        "raw_text",
        "label",
        "label_type",
        "explanation",
        "reason",
        "response",
    )
    for split in SPLITS:
        for source in source_rows[split]:
            meeting = source["meeting_date"]
            raw_path = raw_root / f"{meeting}.csv"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "line_id": "1",
                        "section_name": "Staff Review of the Economic Situation",
                        "raw_text": (
                            "Participants observed that economic activity remained "
                            "steady over the period while incoming information "
                            "continued to indicate balanced conditions."
                        ),
                        "label": "core",
                        "label_type": "core",
                        "explanation": "",
                        "reason": "",
                        "response": "",
                    }
                )
                writer.writerow(
                    {
                        "line_id": "2",
                        "section_name": "Committee Policy Action",
                        "raw_text": "The Committee discussed its policy decision.",
                        "label": "core",
                        "label_type": "core",
                        "explanation": "",
                        "reason": "",
                        "response": "",
                    }
                )
            roster_rows.append(
                {
                    "schema_version": official_reference.EXPECTED_ROSTER_SCHEMA_VERSION,
                    "meeting_end_date": meeting,
                    "meeting_id": f"meeting-{meeting}",
                    "source_split": split,
                    "raw_minutes_csv": str(raw_path),
                    "raw_minutes_csv_sha256": sha256_file(raw_path),
                }
            )
    roster_path = root / "official_meeting_roster.jsonl"
    _jsonl(roster_path, sorted(roster_rows, key=lambda row: row["meeting_end_date"]))
    roster_sha = sha256_file(roster_path)
    source_projection = [
        {"meeting_date": row["meeting_date"], "split": split}
        for split in SPLITS
        for row in source_rows[split]
    ]
    bank = official_reference.build_official_reference_bank(
        source_projection,
        roster_path=roster_path,
        expected_roster_sha256=roster_sha,
        expected_meeting_counts={split: 2 for split in SPLITS},
        exception_boundaries={},
        expected_boundary_method_counts={
            official_reference.EXACT_SECTION_BOUNDARY_METHOD: 6
        },
    )
    return bank, roster_path, roster_sha


def _validator_b_result(reference, answer: str, *, score: int = 8) -> dict:
    dimensions = {}
    paragraph = reference.paragraphs[0]
    for name in STYLE_DIMENSIONS:
        dimensions[name] = {
            "score": score,
            "candidate_evidence": ["remained steady"],
            "official_evidence": [
                {
                    "paragraph_id": paragraph.paragraph_id,
                    "exact_span": "economic activity remained steady",
                    "style_feature_code": "INSTITUTIONAL_REGISTER",
                }
            ],
            "issue_codes": [],
            "action_codes": [],
        }
    passed = score >= 7
    return {
        "comparison_status": "comparable",
        "passage_matches": [
            {
                "candidate_span": answer,
                "official_paragraph_id": paragraph.paragraph_id,
                "official_span": "economic activity remained steady",
                "match_type": "SAME_TOPIC",
            }
        ],
        "dimensions": dimensions,
        "critical_style_errors": [],
        "overall_pass": passed,
        "machine_pass": passed,
        "mean_score": float(score),
        "min_score": score,
        "contract_reasons": [],
        "contract_repair_used": False,
        "contract_repair": None,
    }


def _terminal(source: dict, reference_bank, *, passed: bool) -> dict:
    analysis = source["analysis"]
    if passed:
        row_number = source["source_index"] - 1
        index_value = source["source_index"]
        reasoning = (
            "Preserve the source claims, direction, quantity, and uncertainty "
            "faithfully while selecting a neutral institutional sentence structure."
        )
        answer = (
            f"Economic activity in {source['split']} row {row_number} remained "
            f"steady over the period, while the observed index remained at "
            f"{index_value} amid otherwise unchanged conditions."
        )
        prompt = render_user_prompt(analysis)
        response = reasoning + BOUNDARY + answer
        token_replay = publisher._token_replay(
            tokenizer=FakeDeepSeekTokenizer(),
            system_prompt=publisher.generator.STUDENT_SYSTEM_PROMPT,
            prompt=prompt,
            response=response,
            sample_id=source["sample_id"],
        )
        deterministic_diagnostics = {
            "reasoning_words": len(publisher._WORD_RE.findall(reasoning)),
            "rewritten_minutes_words": len(publisher._WORD_RE.findall(answer)),
            "prompt_tokens": token_replay["prompt_tokens"],
            "completion_tokens": token_replay["completion_tokens"],
            "total_tokens": token_replay["total_tokens"],
            "total_token_limit": publisher.MAX_TOTAL_TOKENS,
            "truncation": False,
            "source_numbers": dict(sorted(publisher._numeric_values(analysis).items())),
            "rewrite_numbers": dict(sorted(publisher._numeric_values(answer).items())),
            "source_dates": sorted(publisher._date_values(analysis)),
            "rewrite_dates": sorted(publisher._date_values(answer)),
            "source_attributions": sorted(publisher._attribution_categories(analysis)),
            "rewrite_attributions": sorted(publisher._attribution_categories(answer)),
            "raw_reasoning_preserved": True,
            "complete_native_cot_preserved": True,
            "reasoning_sanitization": "none",
            "reasoning_meta_used_for_rejection": False,
            "near_copy_used_for_rejection": False,
            "exact_full_source_copy_used_for_rejection": True,
            "tokenizer_replay": {
                "prompt_tokens": token_replay["prompt_tokens"],
                "completion_tokens": token_replay["completion_tokens"],
                "total_tokens": token_replay["total_tokens"],
                "single_bos": True,
                "single_eos": True,
                "completion_only_prompt_masked": True,
                "completion_mask_covers_reasoning_boundary_answer_eos": True,
                "no_truncation": True,
            },
        }
        source_result = _source_report(analysis, source["provided_data"], passed=True)
        source_audit = {
            "complete": True,
            "machine_pass": True,
            "reasons": [],
            "contract_repair_used": False,
            "result": {
                "primary": source_result,
                "adjudication": None,
                "contract_repair": None,
            },
            "provider": {
                publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY: _provider(
                    publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY,
                    {
                        "source_analysis": analysis,
                        "provided_data": source["provided_data"],
                    },
                )
            },
        }
        a_result = _validator_a_result(analysis, reasoning, answer)
        validator_a = {
            "complete": True,
            "machine_pass": True,
            "reasons": [],
            "result": a_result,
            "provider": {
                publisher.generator.ROLE_VALIDATOR_A_PRIMARY: _provider(
                    publisher.generator.ROLE_VALIDATOR_A_PRIMARY,
                    {
                        "source_analysis": analysis,
                        "teacher_response_analysis": reasoning,
                        "rewritten_minutes": answer,
                    },
                )
            },
        }
        reference = reference_bank.reference_for_meeting(source["meeting_date"])
        b_result = _validator_b_result(reference, answer)
        official_payload = {
            "meeting_date": reference.meeting_date,
            "paragraphs": [
                {
                    "paragraph_id": paragraph.paragraph_id,
                    "section_name": paragraph.section_name,
                    "text": paragraph.text,
                }
                for paragraph in reference.paragraphs
            ],
        }
        validator_b = {
            "complete": True,
            "machine_pass": True,
            "mean_score": 8.0,
            "min_score": 8,
            "reasons": [],
            "result": b_result,
            "provider": {
                publisher.generator.ROLE_VALIDATOR_B_PRIMARY: _provider(
                    publisher.generator.ROLE_VALIDATOR_B_PRIMARY,
                    {
                        "rewritten_minutes": answer,
                        "corresponding_official_minutes_pre_action": (official_payload),
                    },
                )
            },
        }
    else:
        reasoning = answer = prompt = response = ""
        primary = _source_report(analysis, source["provided_data"], passed=False)
        adjudication = _source_report(
            analysis, source["provided_data"], passed=False, adjudication=True
        )
        source_audit = {
            "complete": True,
            "machine_pass": False,
            "reasons": ["source_claim_not_supported"],
            "contract_repair_used": False,
            "result": {
                "primary": primary,
                "adjudication": adjudication,
                "contract_repair": None,
            },
            "provider": {
                publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY: _provider(
                    publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY,
                    {
                        "source_analysis": analysis,
                        "provided_data": source["provided_data"],
                    },
                ),
                publisher.generator.ROLE_SOURCE_AUDIT_ADJUDICATION: _provider(
                    publisher.generator.ROLE_SOURCE_AUDIT_ADJUDICATION,
                    {
                        "source_analysis": analysis,
                        "provided_data": source["provided_data"],
                        "primary_findings": primary,
                    },
                ),
            },
        }
        validator_a = {}
        validator_b = {}
    return {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "sample_id": source["sample_id"],
        "split": source["split"],
        "source_split": source["source_split"],
        "source_index": source["source_index"],
        "meeting_date": source["meeting_date"],
        "atomic_topic": source["atomic_topic"],
        "section_style_id": source["section_style_id"],
        "terminal_status": "PASS" if passed else "SOURCE_QUALITY_REJECT",
        "training_pass": passed,
        "rejection_stage": None if passed else "source_admission",
        "rejection_reasons": [] if passed else ["source_claim_not_supported"],
        "provided_data_sha256": sha256_text(source["provided_data"]),
        "source_analysis": analysis,
        "source_analysis_sha256": sha256_text(analysis),
        "source_audit": source_audit,
        "teacher_response_analysis": reasoning,
        "rewritten_minutes": answer,
        "student_prompt": prompt,
        "sft_response": response,
        "teacher_response_analysis_sha256": sha256_text(reasoning)
        if reasoning
        else None,
        "rewritten_minutes_sha256": sha256_text(answer) if answer else None,
        "prompt_sha256": sha256_text(prompt) if prompt else None,
        "response_sha256": sha256_text(response) if response else None,
        "generation": (
            {
                "selected_attempt": "primary" if passed else None,
                "fidelity_repair_used": False,
                "style_repair_used": False,
                "deterministic_validation": {
                    "machine_pass": passed,
                    "reasons": [],
                    "diagnostics": deterministic_diagnostics,
                },
                "provider": {
                    publisher.generator.ROLE_REWRITE_PRIMARY: _provider(
                        publisher.generator.ROLE_REWRITE_PRIMARY,
                        {"analysis": analysis},
                    )
                }
                if passed
                else {},
            }
            if passed
            else {}
        ),
        "validator_a": validator_a,
        "validator_b": validator_b,
        "repair_history": [],
        "lineage": dict(LINEAGE),
    }


def _prompt_contract() -> dict:
    implementation = publisher.generator._implementation_contract()
    return publisher.generator._prompt_contract(
        code_sha256=implementation["composite_sha256"],
        config=publisher.generator.ProviderConfig(),
    )


def _materialize_provider_cache(
    root: Path,
    *,
    source: dict,
    role: str,
    system_prompt: str,
    user_prompt: str,
    raw_content: dict,
    raw_reasoning: str,
    provider_container: dict,
    reference_bank_sha256: str,
) -> None:
    projection = publisher.generator._request_projection(role, user_prompt)
    response = publisher.generator.ProviderResponse(
        raw_reasoning=raw_reasoning,
        raw_content=json.dumps(
            raw_content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        response_id=f"{role}-{source['sample_id']}",
        returned_model="deepseek-v4-flash",
        system_fingerprint="fixed-fingerprint",
        finish_reason="stop",
        created=None,
        usage={},
    )
    binding = {
        "schema_version": publisher.generator.CACHE_SCHEMA_VERSION,
        "role": role,
        "sample_id": source["sample_id"],
        "split": source["split"],
        "source_analysis_sha256": sha256_text(source["analysis"]),
        "provided_data_sha256": sha256_text(source["provided_data"]),
        "official_reference_bank_sha256": (
            reference_bank_sha256 if role.startswith("validator_b") else None
        ),
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "provider_contract_sha256": (
            publisher.generator.ProviderConfig().contract_sha256
        ),
        "code_sha256": publisher.generator._implementation_contract()[
            "composite_sha256"
        ],
        "request_projection_sha256": sha256_text(
            json.dumps(
                projection,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    }
    cache = {
        "binding": binding,
        "binding_sha256": sha256_text(
            json.dumps(
                binding,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "provider_response": response.as_dict(),
        "request_projection": projection,
        "raw_reasoning_sha256": sha256_text(response.raw_reasoning),
        "raw_content_sha256": sha256_text(response.raw_content),
    }
    provider_container[role] = publisher.generator._provider_record(response, cache)
    _json(
        root / "cache" / role / f"{sha256_text(source['sample_id'])}.json",
        cache,
    )


def _materialize_terminal_provider_caches(
    root: Path,
    *,
    source: dict,
    terminal: dict,
    reference_bank,
    reference_bank_sha256: str,
) -> None:
    row = SimpleNamespace(
        sample_id=source["sample_id"],
        split=source["split"],
        source_analysis=source["analysis"],
        source_analysis_sha256=sha256_text(source["analysis"]),
        provided_data=source["provided_data"],
        provided_data_sha256=sha256_text(source["provided_data"]),
        meeting_date=source["meeting_date"],
    )
    source_result = terminal["source_audit"]["result"]
    primary_raw = {
        key: value
        for key, value in source_result["primary"].items()
        if key != "machine_pass"
    }
    _materialize_provider_cache(
        root,
        source=source,
        role=publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY,
        system_prompt=publisher.generator.SOURCE_AUDIT_SYSTEM_PROMPT,
        user_prompt=publisher.generator._source_audit_user_prompt(row),
        raw_content=primary_raw,
        raw_reasoning="Source-only primary audit.",
        provider_container=terminal["source_audit"]["provider"],
        reference_bank_sha256=reference_bank_sha256,
    )
    adjudication = source_result["adjudication"]
    if adjudication is not None:
        adjudication_raw = {
            key: value for key, value in adjudication.items() if key != "machine_pass"
        }
        _materialize_provider_cache(
            root,
            source=source,
            role=publisher.generator.ROLE_SOURCE_AUDIT_ADJUDICATION,
            system_prompt=publisher.generator.SOURCE_ADJUDICATION_SYSTEM_PROMPT,
            user_prompt=publisher.generator._source_adjudication_user_prompt(
                row, source_result["primary"]
            ),
            raw_content=adjudication_raw,
            raw_reasoning="Source-only adjudication.",
            provider_container=terminal["source_audit"]["provider"],
            reference_bank_sha256=reference_bank_sha256,
        )
    if terminal["generation"] == {}:
        return
    candidate = SimpleNamespace(
        teacher_response_analysis=terminal["teacher_response_analysis"],
        rewritten_minutes=terminal["rewritten_minutes"],
    )
    _materialize_provider_cache(
        root,
        source=source,
        role=publisher.generator.ROLE_REWRITE_PRIMARY,
        system_prompt=publisher.generator.REWRITE_SYSTEM_PROMPT,
        user_prompt=publisher.generator._teacher_user_prompt(row),
        raw_content={"answer": terminal["rewritten_minutes"]},
        raw_reasoning=terminal["teacher_response_analysis"],
        provider_container=terminal["generation"]["provider"],
        reference_bank_sha256=reference_bank_sha256,
    )
    a_result = terminal["validator_a"]["result"]
    a_raw = {
        key: a_result[key]
        for key in (
            "source_claims",
            "rewrite_claims",
            "reasoning_issues",
            "bidirectional_entailment",
            "reasoning_compatible",
            "issues",
            "overall_pass",
        )
    }
    _materialize_provider_cache(
        root,
        source=source,
        role=publisher.generator.ROLE_VALIDATOR_A_PRIMARY,
        system_prompt=publisher.generator.VALIDATOR_A_SYSTEM_PROMPT,
        user_prompt=publisher.generator._validator_a_user_prompt(row, candidate),
        raw_content=a_raw,
        raw_reasoning="Fidelity verification.",
        provider_container=terminal["validator_a"]["provider"],
        reference_bank_sha256=reference_bank_sha256,
    )
    b_result = terminal["validator_b"]["result"]
    b_raw = {
        key: b_result[key]
        for key in (
            "comparison_status",
            "passage_matches",
            "dimensions",
            "critical_style_errors",
            "overall_pass",
        )
    }
    _materialize_provider_cache(
        root,
        source=source,
        role=publisher.generator.ROLE_VALIDATOR_B_PRIMARY,
        system_prompt=publisher.generator.VALIDATOR_B_SYSTEM_PROMPT,
        user_prompt=publisher.generator._validator_b_user_prompt(
            row, candidate, reference_bank
        ),
        raw_content=b_raw,
        raw_reasoning="Official-reference style verification.",
        provider_container=terminal["validator_b"]["provider"],
        reference_bank_sha256=reference_bank_sha256,
    )


def _materialize_terminal_cache(
    root: Path,
    *,
    source: dict,
    terminal: dict,
    reference_bank_sha256: str,
) -> None:
    row = SimpleNamespace(
        sample_id=source["sample_id"],
        source_analysis_sha256=sha256_text(source["analysis"]),
        provided_data_sha256=sha256_text(source["provided_data"]),
    )
    code_sha256 = publisher.generator._implementation_contract()["composite_sha256"]
    binding = publisher.generator._terminal_binding(
        row, code_sha256, reference_bank_sha256
    )
    _json(
        root / "cache" / "terminal" / f"{sha256_text(source['sample_id'])}.json",
        {
            "binding": binding,
            "record": terminal,
            "record_sha256": sha256_text(
                json.dumps(
                    terminal,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        },
    )


def _acquisition_fixture(
    root: Path,
    source_rows: dict[str, list[dict]],
    source_bindings: dict[str, dict],
    reference_bank,
    roster_path: Path,
    roster_sha: str,
) -> None:
    reference_path = root / "official_pre_action_reference_bank.jsonl"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_bytes(
        official_reference.serialize_official_reference_bank(reference_bank)
    )
    _json(root / "prompt_contract.json", _prompt_contract())
    artifacts: dict[str, object] = {}
    rejection_audit: list[dict] = []
    validator_b_audit: list[dict] = []
    for split in SPLITS:
        terminal = [
            _terminal(source_rows[split][0], reference_bank, passed=False),
            _terminal(source_rows[split][1], reference_bank, passed=True),
        ]
        for source, terminal_row in zip(source_rows[split], terminal):
            _materialize_terminal_provider_caches(
                root,
                source=source,
                terminal=terminal_row,
                reference_bank=reference_bank,
                reference_bank_sha256=sha256_file(reference_path),
            )
            _materialize_terminal_cache(
                root,
                source=source,
                terminal=terminal_row,
                reference_bank_sha256=sha256_file(reference_path),
            )
        _jsonl(root / f"terminal/{split}.jsonl", terminal)
        rejection_audit.append(terminal[0])
        validator_b_audit.append(
            {
                "sample_id": terminal[1]["sample_id"],
                "split": split,
                "terminal_status": "PASS",
                "validator_b": terminal[1]["validator_b"],
            }
        )
        _jsonl(
            root / f"sft_candidate/{split}.jsonl",
            [
                {
                    "prompt": terminal[1]["student_prompt"],
                    "response": terminal[1]["sft_response"],
                }
            ],
        )
        _jsonl(
            root / f"manifests/{split}.jsonl",
            [
                {
                    "sample_id": terminal[1]["sample_id"],
                    "source_analysis_sha256": terminal[1]["source_analysis_sha256"],
                    "prompt_sha256": terminal[1]["prompt_sha256"],
                    "response_sha256": terminal[1]["response_sha256"],
                }
            ],
        )
        artifacts[split] = {
            name: _file_descriptor(root, f"{directory}/{split}.jsonl")
            for name, directory in (
                ("terminal", "terminal"),
                ("sft_candidate", "sft_candidate"),
                ("manifest", "manifests"),
            )
        }
    _jsonl(root / "audits/rejections.jsonl", rejection_audit)
    _jsonl(root / "audits/validator_b.jsonl", validator_b_audit)
    _jsonl(root / "audits/repair_history.jsonl", [])
    artifacts["audits"] = {
        "rejections": _file_descriptor(root, "audits/rejections.jsonl"),
        "validator_b": _file_descriptor(root, "audits/validator_b.jsonl"),
        "repair_history": _file_descriptor(root, "audits/repair_history.jsonl"),
    }
    artifacts["official_reference_bank"] = _file_descriptor(
        root, "official_pre_action_reference_bank.jsonl"
    )
    source_provider_identity = {
        "returned_model": "deepseek-v4-flash",
        "system_fingerprint": "fixed-fingerprint",
        "roles_observed": sorted(
            {
                publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY,
                publisher.generator.ROLE_SOURCE_AUDIT_ADJUDICATION,
            }
        ),
    }
    full_provider_identity = {
        "returned_model": "deepseek-v4-flash",
        "system_fingerprint": "fixed-fingerprint",
        "roles_observed": sorted(
            {
                *source_provider_identity["roles_observed"],
                publisher.generator.ROLE_REWRITE_PRIMARY,
                publisher.generator.ROLE_VALIDATOR_A_PRIMARY,
                publisher.generator.ROLE_VALIDATOR_B_PRIMARY,
            }
        ),
    }
    repair_provenance = {
        "candidate_response_changed_rows": 1,
        "candidate_final_answer_changed_rows": 0,
        "source_rows": 6,
    }
    _json(
        root / "preparation_summary.json",
        {
            "schema_version": publisher.generator.PREPARED_SCHEMA_VERSION,
            "source": {
                "split_counts": {split: 2 for split in SPLITS},
                "meeting_counts": {split: 2 for split in SPLITS},
                "total_rows": 6,
                "sample_id_sha256": _source_id_digest(
                    [row["sample_id"] for split in SPLITS for row in source_rows[split]]
                ),
                "artifacts": source_bindings,
                "upstream_chk1_candidate_repair_provenance": repair_provenance,
            },
            "meeting_split_isolation": True,
            "lineage": dict(LINEAGE),
        },
    )
    prepare_manifest = {
        "schema_version": publisher.generator.PREPARED_SCHEMA_VERSION,
        "source_artifacts": source_bindings,
        "official_reference_bank_sha256": sha256_file(reference_path),
        "official_roster_path": str(roster_path),
        "official_roster_sha256": roster_sha,
        "invariants": dict(LINEAGE),
    }
    prepare_manifest["prepare_manifest_sha256"] = sha256_text(
        json.dumps(
            prepare_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    _json(root / "prepare_manifest.json", prepare_manifest)
    _json(
        root / "source_admission_receipt.json",
        {
            "schema_version": publisher.generator.SOURCE_AUDIT_SCHEMA_VERSION,
            "status": "source_admission_complete",
            "authorized_scope": "paper_chk2_dataset_construction_only",
            "source": json.loads((root / "preparation_summary.json").read_text())[
                "source"
            ],
            "status_counts": {
                split: {"PASS": 1, "SOURCE_QUALITY_REJECT": 1} for split in SPLITS
            },
            "artifacts": {},
            "provider_identities": source_provider_identity,
            "lineage": dict(LINEAGE),
        },
    )
    _json(
        root / "final_summary.json",
        {
            "schema_version": publisher.generator.SUMMARY_SCHEMA_VERSION,
            "quality_status": "passed",
            "unresolved_failure_count": 0,
            "total_source_rows": 6,
            "terminal_classified": 6,
            "status_counts": {
                split: {"PASS": 1, "SOURCE_QUALITY_REJECT": 1} for split in SPLITS
            },
            "training_ready_candidate": True,
            "evaluation_eligible": False,
            "lineage": dict(LINEAGE),
            "preparation_summary_sha256": sha256_file(
                root / "preparation_summary.json"
            ),
            "source_admission_receipt_sha256": sha256_file(
                root / "source_admission_receipt.json"
            ),
            "prompt_contract_sha256": sha256_file(root / "prompt_contract.json"),
            "official_reference_bank_sha256": sha256_file(reference_path),
            "artifacts": artifacts,
            "provider_identities": full_provider_identity,
        },
    )


def _refresh_acquisition(root: Path) -> None:
    final = json.loads((root / "final_summary.json").read_text(encoding="utf-8"))
    for split in SPLITS:
        for name, directory in (
            ("terminal", "terminal"),
            ("sft_candidate", "sft_candidate"),
            ("manifest", "manifests"),
        ):
            final["artifacts"][split][name] = _file_descriptor(
                root, f"{directory}/{split}.jsonl"
            )
    for name in ("rejections", "validator_b", "repair_history"):
        final["artifacts"]["audits"][name] = _file_descriptor(
            root, f"audits/{name}.jsonl"
        )
    final["preparation_summary_sha256"] = sha256_file(root / "preparation_summary.json")
    final["source_admission_receipt_sha256"] = sha256_file(
        root / "source_admission_receipt.json"
    )
    final["prompt_contract_sha256"] = sha256_file(root / "prompt_contract.json")
    final["official_reference_bank_sha256"] = sha256_file(
        root / "official_pre_action_reference_bank.jsonl"
    )
    _json(root / "final_summary.json", final)


def _add_final_provider_roles(root: Path, *roles: str) -> None:
    final_path = root / "final_summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    observed = set(final["provider_identities"]["roles_observed"])
    observed.update(roles)
    final["provider_identities"]["roles_observed"] = sorted(observed)
    _json(final_path, final)


def _enable_validator_contract_repairs(
    *,
    acquisition: Path,
    source: dict,
    terminal: dict,
    reference_bank,
) -> None:
    """Replace both primary validator reports with malformed+repaired calls."""

    row = SimpleNamespace(
        sample_id=source["sample_id"],
        split=source["split"],
        source_analysis=source["analysis"],
        source_analysis_sha256=sha256_text(source["analysis"]),
        provided_data=source["provided_data"],
        provided_data_sha256=sha256_text(source["provided_data"]),
        meeting_date=source["meeting_date"],
    )
    candidate = SimpleNamespace(
        teacher_response_analysis=terminal["teacher_response_analysis"],
        rewritten_minutes=terminal["rewritten_minutes"],
    )
    reference_sha = sha256_file(
        acquisition / "official_pre_action_reference_bank.jsonl"
    )

    a_result = terminal["validator_a"]["result"]
    a_replacement = {
        key: a_result[key]
        for key in (
            "source_claims",
            "rewrite_claims",
            "reasoning_issues",
            "bidirectional_entailment",
            "reasoning_compatible",
            "issues",
            "overall_pass",
        )
    }
    a_invalid = {"answer": "This nested answer wrapper is not a Validator-A report."}
    a_invalid_text = json.dumps(
        a_invalid, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    probe = publisher.generator.ProviderResponse(
        raw_reasoning="",
        raw_content=a_invalid_text,
        response_id="contract-probe",
        returned_model="deepseek-v4-flash",
        system_fingerprint="fixed-fingerprint",
        finish_reason="stop",
        created=None,
        usage={},
    )
    a_probe_result, _, _ = publisher.generator._validate_validator_a(
        row, candidate, probe
    )
    a_codes = publisher.generator._controlled_validator_contract_codes(
        "validator_a", a_probe_result["contract_reasons"]
    )
    _materialize_provider_cache(
        acquisition,
        source=source,
        role=publisher.generator.ROLE_VALIDATOR_A_PRIMARY,
        system_prompt=publisher.generator.VALIDATOR_A_SYSTEM_PROMPT,
        user_prompt=publisher.generator._validator_a_user_prompt(row, candidate),
        raw_content=a_invalid,
        raw_reasoning="Malformed primary Validator-A transport.",
        provider_container=terminal["validator_a"]["provider"],
        reference_bank_sha256=reference_sha,
    )
    _materialize_provider_cache(
        acquisition,
        source=source,
        role=publisher.generator.ROLE_VALIDATOR_A_PRIMARY_CONTRACT_REPAIR,
        system_prompt=publisher.generator.VALIDATOR_A_CONTRACT_REPAIR_SYSTEM_PROMPT,
        user_prompt=publisher.generator._validator_a_contract_repair_user_prompt(
            row, candidate, a_codes
        ),
        raw_content=a_replacement,
        raw_reasoning="Repaired Validator-A report contract.",
        provider_container=terminal["validator_a"]["provider"],
        reference_bank_sha256=reference_sha,
    )
    a_replacement_text = json.dumps(
        a_replacement, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    a_result["contract_repair_used"] = True
    a_result["contract_repair"] = {
        "target_validator_role": publisher.generator.ROLE_VALIDATOR_A_PRIMARY,
        "contract_repair_role": (
            publisher.generator.ROLE_VALIDATOR_A_PRIMARY_CONTRACT_REPAIR
        ),
        "trigger_contract_error_codes": a_codes,
        "invalid_report_sha256": sha256_text(a_invalid_text),
        "replacement_report_sha256": sha256_text(a_replacement_text),
    }

    b_result = terminal["validator_b"]["result"]
    b_replacement = {
        key: b_result[key]
        for key in (
            "comparison_status",
            "passage_matches",
            "dimensions",
            "critical_style_errors",
            "overall_pass",
        )
    }
    b_invalid = {"answer": "This nested answer wrapper is not a Validator-B report."}
    b_invalid_text = json.dumps(
        b_invalid, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    b_probe = publisher.generator.ProviderResponse(
        raw_reasoning="",
        raw_content=b_invalid_text,
        response_id="contract-probe-b",
        returned_model="deepseek-v4-flash",
        system_fingerprint="fixed-fingerprint",
        finish_reason="stop",
        created=None,
        usage={},
    )
    b_probe_result, _, _ = publisher.generator._validate_validator_b(
        row, candidate, b_probe, reference_bank
    )
    b_codes = publisher.generator._controlled_validator_contract_codes(
        "validator_b", b_probe_result["contract_reasons"]
    )
    _materialize_provider_cache(
        acquisition,
        source=source,
        role=publisher.generator.ROLE_VALIDATOR_B_PRIMARY,
        system_prompt=publisher.generator.VALIDATOR_B_SYSTEM_PROMPT,
        user_prompt=publisher.generator._validator_b_user_prompt(
            row, candidate, reference_bank
        ),
        raw_content=b_invalid,
        raw_reasoning="Malformed primary Validator-B transport.",
        provider_container=terminal["validator_b"]["provider"],
        reference_bank_sha256=reference_sha,
    )
    _materialize_provider_cache(
        acquisition,
        source=source,
        role=publisher.generator.ROLE_VALIDATOR_B_PRIMARY_CONTRACT_REPAIR,
        system_prompt=publisher.generator.VALIDATOR_B_CONTRACT_REPAIR_SYSTEM_PROMPT,
        user_prompt=publisher.generator._validator_b_contract_repair_user_prompt(
            row, candidate, reference_bank, b_codes
        ),
        raw_content=b_replacement,
        raw_reasoning="Repaired Validator-B report contract.",
        provider_container=terminal["validator_b"]["provider"],
        reference_bank_sha256=reference_sha,
    )
    b_replacement_text = json.dumps(
        b_replacement, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    b_result["contract_repair_used"] = True
    b_result["contract_repair"] = {
        "target_validator_role": publisher.generator.ROLE_VALIDATOR_B_PRIMARY,
        "contract_repair_role": (
            publisher.generator.ROLE_VALIDATOR_B_PRIMARY_CONTRACT_REPAIR
        ),
        "trigger_contract_error_codes": b_codes,
        "invalid_report_sha256": sha256_text(b_invalid_text),
        "replacement_report_sha256": sha256_text(b_replacement_text),
    }
    validator_b_audit_path = acquisition / "audits/validator_b.jsonl"
    validator_b_audits = [
        json.loads(line) for line in validator_b_audit_path.read_text().splitlines()
    ]
    matched = False
    for audit in validator_b_audits:
        if audit.get("sample_id") == source["sample_id"]:
            audit["validator_b"] = terminal["validator_b"]
            matched = True
    assert matched
    _jsonl(validator_b_audit_path, validator_b_audits)


def _fixture(tmp_path: Path):
    source = tmp_path / "source"
    generation = tmp_path / "generation_manifests"
    acquisition = tmp_path / "acquisition"
    release = tmp_path / "release"
    source_rows, pins, bindings, id_digest = _source_fixture(source, generation)
    reference_bank, roster_path, roster_sha = _official_reference_fixture(
        tmp_path / "official_reference", source_rows
    )
    _acquisition_fixture(
        acquisition,
        source_rows,
        bindings,
        reference_bank,
        roster_path,
        roster_sha,
    )
    tokenizer_path = tmp_path / "cp200-tokenizer"
    tokenizer_path.mkdir(parents=True)
    kwargs = {
        "source_root": source,
        "generation_manifest_root": generation,
        "acquisition_root": acquisition,
        "release_root": release,
        "tokenizer": FakeDeepSeekTokenizer(),
        "tokenizer_path": tokenizer_path,
        "expected_tokenizer_path": tokenizer_path,
        "expected_split_counts": {split: 2 for split in SPLITS},
        "expected_meeting_counts": {split: 2 for split in SPLITS},
        "expected_sample_id_sha256": id_digest,
        "expected_source_file_sha256": pins,
        "official_roster_path": roster_path,
        "expected_official_roster_sha256": roster_sha,
        "expected_boundary_method_counts": {
            official_reference.EXACT_SECTION_BOUNDARY_METHOD: 6
        },
        "official_reference_exception_boundaries": {},
    }
    return source_rows, acquisition, release, kwargs


def test_publishes_exact_chk1_pass_only_release_with_style_gate(tmp_path: Path) -> None:
    _, _, release, kwargs = _fixture(tmp_path)

    handoff = publish_release(**kwargs)

    assert handoff["dataset_role"] == DATASET_ROLE
    assert handoff["source_rows"] == 6
    assert handoff["split_counts"] == {split: 1 for split in SPLITS}
    for split in SPLITS:
        rows = [
            json.loads(line)
            for line in (release / f"minutes_alignment/{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(rows) == 1
        assert set(rows[0]) == {"prompt", "response"}
        sidecar = json.loads(
            (release / f"minutes_alignment/manifests/{split}.jsonl").read_text()
        )
        assert sidecar["source_index"] == 2
        assert sidecar["validator_b"]["used_for_selection"] is True
        assert sidecar["validator_b"]["mean_score"] == 8.0
        assert sidecar["tokens"]["total_tokens"] <= 4096
    audit = json.loads((release / "audits/data_quality.json").read_text())
    assert audit["pass_rows"] == 3
    assert audit["reject_rows"] == 3
    assert audit["validator_b_used_for_selection"] is True
    assert audit["integrity"]["pass_reject_exact_source_partition"] is True


def test_recomputes_validator_b_instead_of_trusting_pass_flags(tmp_path: Path) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = rows[1]["validator_b"]["result"]
    for dimension in result["dimensions"].values():
        dimension["score"] = 5
    # Keep every cached/top-level pass field forged as true.
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="recomputed gate|mean_score drift"):
        publish_release(**kwargs)


def test_rejects_source_analysis_that_is_not_exact_chk1_final_answer(
    tmp_path: Path,
) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["source_analysis"] += " altered"
    rows[1]["source_analysis_sha256"] = sha256_text(rows[1]["source_analysis"])
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="not exact chk1 final answer"):
        publish_release(**kwargs)


def test_rejects_terminal_status_that_ignores_style_gate(tmp_path: Path) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = rows[1]
    for dimension in row["validator_b"]["result"]["dimensions"].values():
        dimension["score"] = 5
    row["validator_b"].update(
        machine_pass=False,
        mean_score=5.0,
        min_score=5,
        reasons=["validator_b_mean_below_7", "validator_b_dimension_below_6"],
    )
    row["validator_b"]["result"].update(
        machine_pass=False, overall_pass=False, mean_score=5.0, min_score=5
    )
    # The terminal incorrectly remains PASS.
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)

    with pytest.raises(
        PublicationError,
        match="selected raw report differs|terminal/status consistency failure",
    ):
        publish_release(**kwargs)


def test_rejects_secret_and_overlength_completion(tmp_path: Path) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["generation"]["provider"]["api_key"] = "sk-secret-secret-secret"
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)
    with pytest.raises(PublicationError, match="credential material"):
        publish_release(**kwargs)

    _, acquisition, _, kwargs = _fixture(tmp_path / "long")
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = rows[1]
    reasoning = "R" * 5000
    row["teacher_response_analysis"] = reasoning
    row["teacher_response_analysis_sha256"] = sha256_text(reasoning)
    row["validator_a"]["result"]["source_claims"][0]["reasoning_evidence"] = reasoning
    a_projection = row["validator_a"]["provider"][
        publisher.generator.ROLE_VALIDATOR_A_PRIMARY
    ]["request_projection"]
    a_projection["field_value_sha256"]["teacher_response_analysis"] = sha256_text(
        json.dumps(reasoning, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    row["sft_response"] = reasoning + BOUNDARY + row["rewritten_minutes"]
    row["response_sha256"] = sha256_text(row["sft_response"])
    _jsonl(path, rows)
    _jsonl(
        acquisition / "sft_candidate/train.jsonl",
        [{"prompt": row["student_prompt"], "response": row["sft_response"]}],
    )
    manifest = json.loads(
        (acquisition / "manifests/train.jsonl").read_text(encoding="utf-8")
    )
    manifest["response_sha256"] = row["response_sha256"]
    _jsonl(acquisition / "manifests/train.jsonl", [manifest])
    _refresh_acquisition(acquisition)
    with pytest.raises(
        PublicationError,
        match=(
            "selected raw report differs|exceeds 4,096 tokens|"
            "terminal candidate differs from selected rewrite cache"
        ),
    ):
        publish_release(**kwargs)


def test_release_verifier_detects_sealed_byte_drift(tmp_path: Path) -> None:
    _, _, release, kwargs = _fixture(tmp_path)
    publish_release(**kwargs)
    manifest_sha = sha256_file(release / "release_manifest.json")
    assert (
        verify_release(release, expected_manifest_sha256=manifest_sha)["quality_status"]
        == "passed"
    )
    path = release / "minutes_alignment/train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(PublicationError, match="SHA-256 mismatch"):
        verify_release(release, expected_manifest_sha256=manifest_sha)


def test_rejects_unbacked_terminal_fidelity_repair_sequence(
    tmp_path: Path,
) -> None:
    source_rows, acquisition, release, kwargs = _fixture(tmp_path)
    terminal_path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in terminal_path.read_text().splitlines()]
    row = rows[0]
    row.update(
        terminal_status="GENERATION_QUALITY_REJECT",
        rejection_stage="fidelity_repair_deterministic_gate",
        rejection_reasons=["numeric_multiset_mismatch"],
    )
    analysis = row["source_analysis"]
    provided = source_rows["train"][0]["provided_data"]
    primary = _source_report(analysis, provided, passed=True)
    row["source_audit"] = {
        "complete": True,
        "machine_pass": True,
        "reasons": [],
        "contract_repair_used": False,
        "result": {
            "primary": primary,
            "adjudication": None,
            "contract_repair": None,
        },
        "provider": {
            publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY: _provider(
                publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY,
                {"source_analysis": analysis, "provided_data": provided},
            )
        },
    }
    initial_reasoning = "Preserve the original claim and quantities."
    initial_answer = (
        "A deliberately omitted initial rewrite used only for audit history."
    )
    a_result = _validator_a_result(
        analysis, initial_reasoning, initial_answer, passed=False
    )
    a_reasons = [
        "validator_a_source_claim_not_entailed",
        "validator_a_reasoning_claim_not_covered",
        "validator_a_reported_issues",
    ]
    a_provider = _provider(
        publisher.generator.ROLE_VALIDATOR_A_PRIMARY,
        {
            "source_analysis": analysis,
            "teacher_response_analysis": initial_reasoning,
            "rewritten_minutes": initial_answer,
        },
    )
    row["generation"] = {
        "selected_attempt": "fidelity_repair",
        "fidelity_repair_used": True,
        "style_repair_used": False,
        "deterministic_validation": {
            "machine_pass": False,
            "reasons": ["numeric_multiset_mismatch"],
            "diagnostics": {},
        },
        "provider": {
            publisher.generator.ROLE_REWRITE_PRIMARY: _provider(
                publisher.generator.ROLE_REWRITE_PRIMARY,
                {"analysis": analysis},
            ),
            publisher.generator.ROLE_REWRITE_FIDELITY_REPAIR: _provider(
                publisher.generator.ROLE_REWRITE_FIDELITY_REPAIR,
                {
                    "analysis": analysis,
                    "repair_feedback": {"reason_codes": a_reasons},
                },
            ),
        },
    }
    row["validator_a"] = {
        "complete": True,
        "machine_pass": False,
        "reasons": a_reasons,
        "result": a_result,
        "provider": {publisher.generator.ROLE_VALIDATOR_A_PRIMARY: a_provider},
    }
    row["repair_history"] = [
        {
            "repair_type": "fidelity",
            "trigger_stage": "validator_a_primary",
            "attempt": "primary",
            "reason_codes": a_reasons,
            "validator_a": a_result,
            "provider": {publisher.generator.ROLE_VALIDATOR_A_PRIMARY: a_provider},
        }
    ]
    _jsonl(terminal_path, rows)
    rejection_rows = [
        json.loads(line)
        for line in (acquisition / "audits/rejections.jsonl").read_text().splitlines()
    ]
    rejection_rows[0] = row
    _jsonl(acquisition / "audits/rejections.jsonl", rejection_rows)
    _jsonl(
        acquisition / "audits/repair_history.jsonl",
        [
            {
                "sample_id": row["sample_id"],
                "split": "train",
                "terminal_status": "GENERATION_QUALITY_REJECT",
                "events": row["repair_history"],
            }
        ],
    )
    receipt = json.loads((acquisition / "source_admission_receipt.json").read_text())
    receipt["status_counts"]["train"] = {"PASS": 2}
    _json(acquisition / "source_admission_receipt.json", receipt)
    final = json.loads((acquisition / "final_summary.json").read_text())
    final["status_counts"]["train"] = {
        "PASS": 1,
        "GENERATION_QUALITY_REJECT": 1,
    }
    final["provider_identities"]["roles_observed"] = sorted(
        {
            *final["provider_identities"]["roles_observed"],
            publisher.generator.ROLE_REWRITE_FIDELITY_REPAIR,
        }
    )
    _json(acquisition / "final_summary.json", final)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="provider_cache must be an object"):
        publish_release(**kwargs)
    assert not release.exists()


def test_requires_nonempty_pass_split_and_exact_terminal_partition(
    tmp_path: Path,
) -> None:
    source_rows, acquisition, _, kwargs = _fixture(tmp_path)
    reference_bank = official_reference.deserialize_official_reference_bank(
        (acquisition / "official_pre_action_reference_bank.jsonl").read_bytes()
    )
    path = acquisition / "terminal/test.jsonl"
    rows = [
        _terminal(source_rows["test"][0], reference_bank, passed=False),
        _terminal(source_rows["test"][1], reference_bank, passed=False),
    ]
    for source, terminal_row in zip(source_rows["test"], rows):
        _materialize_terminal_provider_caches(
            acquisition,
            source=source,
            terminal=terminal_row,
            reference_bank=reference_bank,
            reference_bank_sha256=sha256_file(
                acquisition / "official_pre_action_reference_bank.jsonl"
            ),
        )
        _materialize_terminal_cache(
            acquisition,
            source=source,
            terminal=terminal_row,
            reference_bank_sha256=sha256_file(
                acquisition / "official_pre_action_reference_bank.jsonl"
            ),
        )
    _jsonl(path, rows)
    _jsonl(acquisition / "sft_candidate/test.jsonl", [])
    _jsonl(acquisition / "manifests/test.jsonl", [])
    receipt = json.loads((acquisition / "source_admission_receipt.json").read_text())
    receipt["status_counts"]["test"] = {"SOURCE_QUALITY_REJECT": 2}
    _json(acquisition / "source_admission_receipt.json", receipt)
    final = json.loads((acquisition / "final_summary.json").read_text())
    final["status_counts"]["test"] = {"SOURCE_QUALITY_REJECT": 2}
    _json(acquisition / "final_summary.json", final)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="split is empty: test"):
        publish_release(**kwargs)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda row: row.update(
                rewritten_minutes=row["rewritten_minutes"].replace("at 2", "at 3")
            ),
            "numeric_multiset_mismatch",
        ),
        (
            lambda row: row.update(
                rewritten_minutes=row["rewritten_minutes"] + "\nSecond paragraph."
            ),
            "rewritten_minutes_not_single_paragraph",
        ),
        (
            lambda row: row.update(rewritten_minutes=row["source_analysis"]),
            "rewritten_minutes_exactly_copies_source_analysis",
        ),
        (
            lambda row: row.update(
                rewritten_minutes=row["rewritten_minutes"] + " Participants agreed."
            ),
            "attribution_set_mismatch",
        ),
        (
            lambda row: row.update(
                rewritten_minutes=row["rewritten_minutes"] + " Source: [1]."
            ),
            "rewritten_minutes_contains_citation",
        ),
        (
            lambda row: row.update(
                teacher_response_analysis=row["teacher_response_analysis"] + " <think>"
            ),
            "teacher_response_analysis_control_markers",
        ),
    ],
)
def test_recomputes_deterministic_text_gates_from_terminal_text(
    tmp_path: Path, mutate, reason: str
) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows[1])
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match=reason):
        publish_release(**kwargs)


def test_rejects_tampered_source_audit_exact_span_and_issue_partition(
    tmp_path: Path,
) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["source_audit"]["result"]["primary"]["claims"][0]["analysis_span"] = (
        "not an exact source span"
    )
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="analysis span is not exact"):
        publish_release(**kwargs)

    _, acquisition, _, kwargs = _fixture(tmp_path / "partition")
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["source_audit"]["result"]["primary"]["blocking_issues"][0][
        "evidence_span"
    ] = rows[0]["source_audit"]["result"]["primary"]["claims"][0]["analysis_span"]
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="blocking evidence is not exact"):
        publish_release(**kwargs)


def test_rejects_tampered_validator_a_exact_span_and_local_gate(tmp_path: Path) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["validator_a"]["result"]["source_claims"][0]["rewrite_evidence"] = (
        "not an exact rewrite span"
    )
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="rewrite_evidence.*not an exact span"):
        publish_release(**kwargs)

    _, acquisition, _, kwargs = _fixture(tmp_path / "misclassified")
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = rows[1]["validator_a"]["result"]
    issue = "The draft contains an incorrect numerical attribution"
    result["reported_issues"] = [issue]
    result["ignored_operational_issues"] = [issue]
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="local verdict/reasons drift"):
        publish_release(**kwargs)


def test_rejects_provider_projection_and_fingerprint_tampering(tmp_path: Path) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    projection = rows[1]["generation"]["provider"][
        publisher.generator.ROLE_REWRITE_PRIMARY
    ]["request_projection"]
    projection["field_value_sha256"]["analysis"] = "0" * 64
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="value binding drift"):
        publish_release(**kwargs)

    _, acquisition, _, kwargs = _fixture(tmp_path / "fingerprint")
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["validator_a"]["provider"][publisher.generator.ROLE_VALIDATOR_A_PRIMARY][
        "system_fingerprint"
    ] = "drifted-fingerprint"
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)

    with pytest.raises(
        PublicationError, match="receipt differs from provider cache|identity drift"
    ):
        publish_release(**kwargs)


def test_rejects_cross_meeting_validator_b_evidence(tmp_path: Path) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    reference_bank = official_reference.deserialize_official_reference_bank(
        (acquisition / "official_pre_action_reference_bank.jsonl").read_bytes()
    )
    wrong = reference_bank.reference_for_meeting("2026-01-01").paragraphs[0]
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    dimension = next(iter(rows[1]["validator_b"]["result"]["dimensions"].values()))
    dimension["official_evidence"] = [
        {
            "paragraph_id": wrong.paragraph_id,
            "exact_span": "economic activity remained steady",
            "style_feature_code": "INSTITUTIONAL_REGISTER",
        }
    ]
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="outside the corresponding meeting"):
        publish_release(**kwargs)


def test_rejects_prompt_contract_and_tokenizer_pin_drift(tmp_path: Path) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    contract_path = acquisition / "prompt_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["student"]["tokenizer_file_sha256"]["tokenizer.json"] = "0" * 64
    _json(contract_path, contract)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="tokenizer provenance drift"):
        publish_release(**kwargs)

    empty_tokenizer = tmp_path / "empty-default-pin"
    empty_tokenizer.mkdir()
    with pytest.raises(PublicationError, match="artifact is missing or unsafe"):
        publisher._validate_exact_tokenizer_files(
            empty_tokenizer,
            expected_tokenizer_path=publisher.DEFAULT_TOKENIZER_PATH,
        )


@pytest.mark.parametrize("drift", ["version", "digest"])
def test_rejects_sealed_tokenizer_runtime_contract_drift(
    tmp_path: Path, drift: str
) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    contract_path = acquisition / "prompt_contract.json"
    contract = json.loads(contract_path.read_text())
    runtime = contract["student"]["tokenizer_runtime_contract"]
    if drift == "version":
        runtime["library_versions"]["transformers"] = "0.0.0-drift"
    else:
        runtime["runtime_digest"] = "0" * 64
    _json(contract_path, contract)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="tokenizer provenance drift"):
        publish_release(**kwargs)


def test_release_verifier_rejects_tokenizer_runtime_digest_drift(
    tmp_path: Path,
) -> None:
    _, _, release, kwargs = _fixture(tmp_path)
    publish_release(**kwargs)
    manifest_path = release / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tokenizer_runtime_contract"]["runtime_digest"] = "0" * 64
    _json(manifest_path, manifest)

    with pytest.raises(PublicationError, match="tokenizer runtime contract drift"):
        verify_release(release, expected_manifest_sha256=None)


def test_rejects_bulk_preflight_prompt_contract_drift(tmp_path: Path) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    contract_path = acquisition / "prompt_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["bulk_preflight_gate"]["selected_rows_must_reach_terminal_pass"] = False
    _json(contract_path, contract)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="bulk preflight gate contract drift"):
        publish_release(**kwargs)


def test_repair_history_rejects_official_evidence_leakage() -> None:
    with pytest.raises(PublicationError, match="leaks corresponding official"):
        publisher._validate_repair_history(
            [{"official_evidence": "leaked"}],
            label="repair_history",
            fidelity_repair_used=False,
            style_repair_used=True,
            forbidden_reference_texts=[],
            forbidden_reference_ids=[],
            provider_audit=publisher._new_provider_audit(),
            source_analysis="source",
            generation_provider={},
            official_reference=SimpleNamespace(),
        )


def test_rejects_self_consistent_raw_and_roster_reference_tampering(
    tmp_path: Path,
) -> None:
    _, _, _, kwargs = _fixture(tmp_path)
    roster_path = kwargs["official_roster_path"]
    roster_rows = [json.loads(line) for line in roster_path.read_text().splitlines()]
    raw_path = Path(roster_rows[0]["raw_minutes_csv"])
    raw_text = raw_path.read_text(encoding="utf-8")
    raw_path.write_text(
        raw_text.replace("balanced conditions", "materially different conditions"),
        encoding="utf-8",
    )
    roster_rows[0]["raw_minutes_csv_sha256"] = sha256_file(raw_path)
    _jsonl(roster_path, roster_rows)
    kwargs["expected_official_roster_sha256"] = sha256_file(roster_path)

    with pytest.raises(PublicationError, match="does not rebuild exactly"):
        publish_release(**kwargs)


def test_rejects_legacy_v1_cache_even_with_v2_terminal_exports(tmp_path: Path) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    _json(
        acquisition / "cache/rewrite_primary/legacy.json",
        {"binding": {"schema_version": "paper-chk2-chk1-provider-cache-v1"}},
    )

    with pytest.raises(PublicationError, match="legacy v1 cache"):
        publish_release(**kwargs)


def test_recomputes_no_comparable_passage_as_hard_reject(tmp_path: Path) -> None:
    _, acquisition, _, _ = _fixture(tmp_path)
    terminal = json.loads(
        (acquisition / "terminal/train.jsonl").read_text().splitlines()[1]
    )
    record = terminal["validator_b"]
    result = record["result"]
    result["comparison_status"] = "no_comparable_passage"
    result["passage_matches"] = []
    for dimension in result["dimensions"].values():
        dimension["official_evidence"] = []
    result["machine_pass"] = False
    result["overall_pass"] = False
    record["machine_pass"] = False
    record["reasons"] = ["validator_b_no_comparable_passage"]
    bank = official_reference.deserialize_official_reference_bank(
        (acquisition / "official_pre_action_reference_bank.jsonl").read_bytes()
    )
    reference = bank.reference_for_meeting(terminal["meeting_date"])

    complete, passed, mean_score, min_score, comparison_status = (
        publisher._validate_validator_b(
            record,
            label="validator_b",
            answer=terminal["rewritten_minutes"],
            reference=reference,
            provider_audit=publisher._new_provider_audit(),
        )
    )
    assert (complete, passed, mean_score, min_score, comparison_status) == (
        True,
        False,
        8.0,
        8,
        "no_comparable_passage",
    )


def test_rejects_dimension_evidence_from_an_unmatched_same_meeting_paragraph(
    tmp_path: Path,
) -> None:
    _, acquisition, _, _ = _fixture(tmp_path)
    terminal = json.loads(
        (acquisition / "terminal/train.jsonl").read_text().splitlines()[1]
    )
    bank = official_reference.deserialize_official_reference_bank(
        (acquisition / "official_pre_action_reference_bank.jsonl").read_bytes()
    )
    reference = bank.reference_for_meeting(terminal["meeting_date"])
    extra = SimpleNamespace(
        paragraph_id="same-meeting-unmatched-paragraph",
        text=(
            "Staff members separately reviewed market functioning and reported "
            "that liquidity conditions remained orderly."
        ),
    )
    augmented_reference = SimpleNamespace(
        meeting_date=reference.meeting_date,
        paragraphs=[*reference.paragraphs, extra],
    )
    record = terminal["validator_b"]
    first_dimension = record["result"]["dimensions"][STYLE_DIMENSIONS[0]]
    first_dimension["official_evidence"] = [
        {
            "paragraph_id": extra.paragraph_id,
            "exact_span": "liquidity conditions remained orderly",
            "style_feature_code": "INSTITUTIONAL_REGISTER",
        }
    ]
    with pytest.raises(PublicationError, match="not from a passage-matched paragraph"):
        publisher._validate_validator_b(
            record,
            label="validator_b",
            answer=terminal["rewritten_minutes"],
            reference=augmented_reference,
            provider_audit=publisher._new_provider_audit(),
        )


def test_source_contract_repair_receipt_binds_raw_provider_responses() -> None:
    analysis = "Economic activity remained steady with the observed index at 1."
    provided = '{"evidence":"Economic activity remained steady at index 1."}'
    normalized = _source_report(analysis, provided, passed=True)
    invalid_raw = '{"unexpected":"transport shape"}'
    replacement_raw = json.dumps(
        {
            "claims": normalized["claims"],
            "blocking_issues": [],
            "overall_pass": True,
        },
        separators=(",", ":"),
    )
    trigger_codes = ["source_audit_content_keys"]
    primary_provider = _provider(
        publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY,
        {"source_analysis": analysis, "provided_data": provided},
    )
    primary_provider["raw_content_sha256"] = sha256_text(invalid_raw)
    repair_provider = _provider(
        publisher.generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR,
        {
            "source_analysis": analysis,
            "provided_data": provided,
            "target_report_type": "primary",
            "invalid_report": invalid_raw,
            "contract_error_codes": trigger_codes,
        },
    )
    repair_provider["raw_content_sha256"] = sha256_text(replacement_raw)
    record = {
        "complete": True,
        "machine_pass": True,
        "reasons": [],
        "contract_repair_used": True,
        "result": {
            "primary": normalized,
            "adjudication": None,
            "contract_repair": {
                "target_report_type": "primary",
                "trigger_contract_error_codes": trigger_codes,
                "invalid_report_sha256": sha256_text(invalid_raw),
                "replacement_report_sha256": sha256_text(replacement_raw),
                "contract_exhausted": False,
                "contract_exhaustion": None,
            },
        },
        "provider": {
            publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY: primary_provider,
            publisher.generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR: repair_provider,
        },
    }
    provider_audit = publisher._new_provider_audit()
    provider_audit["cache_raw_content"] = {
        (
            publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY,
            primary_provider["response_id"],
        ): invalid_raw,
        (
            publisher.generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR,
            repair_provider["response_id"],
        ): replacement_raw,
    }
    assert publisher._validate_source_audit_record(
        record,
        label="source_audit",
        source_analysis=analysis,
        provided_data=provided,
        provider_audit=provider_audit,
    )
    record["result"]["contract_repair"]["replacement_report_sha256"] = "0" * 64
    with pytest.raises(PublicationError, match="raw contract-repair receipt SHA drift"):
        publisher._validate_source_audit_record(
            record,
            label="source_audit",
            source_analysis=analysis,
            provided_data=provided,
            provider_audit=provider_audit,
        )


def test_style_repair_trigger_replays_primary_b_and_safe_feedback(
    tmp_path: Path,
) -> None:
    _, acquisition, _, _ = _fixture(tmp_path)
    terminal = json.loads(
        (acquisition / "terminal/train.jsonl").read_text().splitlines()[1]
    )
    answer = terminal["rewritten_minutes"]
    generation_provider = terminal["generation"]["provider"]
    rewrite_provider = generation_provider[publisher.generator.ROLE_REWRITE_PRIMARY]
    raw_rewrite = json.dumps({"answer": answer}, separators=(",", ":"))
    rewrite_provider["raw_content_sha256"] = sha256_text(raw_rewrite)
    b_provider = terminal["validator_b"]["provider"][
        publisher.generator.ROLE_VALIDATOR_B_PRIMARY
    ]
    raw_b = {
        key: value
        for key, value in terminal["validator_b"]["result"].items()
        if key
        in {
            "comparison_status",
            "passage_matches",
            "dimensions",
            "critical_style_errors",
            "overall_pass",
        }
    }
    for dimension in raw_b["dimensions"].values():
        dimension["score"] = 5
    raw_b["overall_pass"] = False
    raw_b_text = json.dumps(raw_b, sort_keys=True, separators=(",", ":"))
    b_provider["raw_content_sha256"] = sha256_text(raw_b_text)
    safe_feedback = {
        "dimensions": {
            name: {
                "score": dimension["score"],
                "candidate_evidence": dimension["candidate_evidence"],
                "issue_codes": dimension["issue_codes"],
                "action_codes": dimension["action_codes"],
            }
            for name, dimension in raw_b["dimensions"].items()
        },
        "critical_style_errors": [],
    }
    event = {
        "repair_type": "style",
        "trigger_stage": "validator_b_primary",
        "attempt": "primary",
        "reason_codes": [
            "validator_b_mean_below_7",
            "validator_b_dimension_below_6",
        ],
        "validator_b": {
            "machine_pass": False,
            "mean_score": 5.0,
            "min_score": 5,
            "style_feedback": safe_feedback,
            "contract_repair_used": False,
            "contract_repair": None,
        },
        "provider": {
            publisher.generator.ROLE_VALIDATOR_B_PRIMARY: b_provider,
        },
    }
    bank = official_reference.deserialize_official_reference_bank(
        (acquisition / "official_pre_action_reference_bank.jsonl").read_bytes()
    )
    reference = bank.reference_for_meeting(terminal["meeting_date"])
    official_payload = {
        "meeting_date": reference.meeting_date,
        "paragraphs": [
            {
                "paragraph_id": paragraph.paragraph_id,
                "section_name": paragraph.section_name,
                "text": paragraph.text,
            }
            for paragraph in reference.paragraphs
        ],
    }
    provider_audit = publisher._new_provider_audit()
    provider_audit["cache_raw_content"] = {
        (
            publisher.generator.ROLE_REWRITE_PRIMARY,
            rewrite_provider["response_id"],
        ): raw_rewrite,
        (
            publisher.generator.ROLE_VALIDATOR_B_PRIMARY,
            b_provider["response_id"],
        ): raw_b_text,
    }
    assert (
        publisher._validate_style_repair_trigger_snapshot(
            event=event,
            generation_provider=generation_provider,
            reference=reference,
            official_payload=official_payload,
            provider_audit=provider_audit,
            label="style_repair",
        )
        == answer
    )
    critical_raw = json.loads(raw_b_text)
    critical_raw["critical_style_errors"] = [
        {
            "error_code": "NON_MINUTES_GENRE",
            "candidate_evidence": answer,
        }
    ]
    critical_text = json.dumps(critical_raw, sort_keys=True, separators=(",", ":"))
    b_provider["raw_content_sha256"] = sha256_text(critical_text)
    provider_audit["cache_raw_content"][
        (
            publisher.generator.ROLE_VALIDATOR_B_PRIMARY,
            b_provider["response_id"],
        )
    ] = critical_text
    event["reason_codes"].append("validator_b_critical_style_error")
    safe_feedback["critical_style_errors"] = critical_raw["critical_style_errors"]
    with pytest.raises(PublicationError, match="low-score-only repair-eligible"):
        publisher._validate_style_repair_trigger_snapshot(
            event=event,
            generation_provider=generation_provider,
            reference=reference,
            official_payload=official_payload,
            provider_audit=provider_audit,
            label="style_repair",
        )
    b_provider["raw_content_sha256"] = sha256_text(raw_b_text)
    provider_audit["cache_raw_content"][
        (
            publisher.generator.ROLE_VALIDATOR_B_PRIMARY,
            b_provider["response_id"],
        )
    ] = raw_b_text
    event["reason_codes"].remove("validator_b_critical_style_error")
    safe_feedback["critical_style_errors"] = []
    safe_feedback["dimensions"][STYLE_DIMENSIONS[0]]["candidate_evidence"] = [
        "not an exact candidate span"
    ]
    with pytest.raises(PublicationError, match="safe style feedback differs"):
        publisher._validate_style_repair_trigger_snapshot(
            event=event,
            generation_provider=generation_provider,
            reference=reference,
            official_payload=official_payload,
            provider_audit=provider_audit,
            label="style_repair",
        )


def test_publisher_accepts_and_replays_a_and_b_contract_repairs(
    tmp_path: Path,
) -> None:
    source_rows, acquisition, release, kwargs = _fixture(tmp_path)
    reference_bank = official_reference.deserialize_official_reference_bank(
        (acquisition / "official_pre_action_reference_bank.jsonl").read_bytes()
    )
    terminal_path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in terminal_path.read_text().splitlines()]
    source = source_rows["train"][1]
    terminal = rows[1]
    _enable_validator_contract_repairs(
        acquisition=acquisition,
        source=source,
        terminal=terminal,
        reference_bank=reference_bank,
    )
    _jsonl(terminal_path, rows)
    _materialize_terminal_cache(
        acquisition,
        source=source,
        terminal=terminal,
        reference_bank_sha256=sha256_file(
            acquisition / "official_pre_action_reference_bank.jsonl"
        ),
    )
    _add_final_provider_roles(
        acquisition,
        publisher.generator.ROLE_VALIDATOR_A_PRIMARY_CONTRACT_REPAIR,
        publisher.generator.ROLE_VALIDATOR_B_PRIMARY_CONTRACT_REPAIR,
    )
    _refresh_acquisition(acquisition)

    handoff = publish_release(**kwargs)
    assert handoff["split_counts"] == {split: 1 for split in SPLITS}
    assert release.is_dir()
    a_repair_cache = json.loads(
        (
            acquisition
            / "cache"
            / publisher.generator.ROLE_VALIDATOR_A_PRIMARY_CONTRACT_REPAIR
            / f"{sha256_text(source['sample_id'])}.json"
        ).read_text()
    )
    assert a_repair_cache["request_projection"]["official_text_policy"] == "forbidden"
    assert (
        "corresponding_official_minutes_pre_action"
        not in a_repair_cache["request_projection"]["allowed_fields"]
    )


def test_publisher_recomputes_contract_repair_trigger_codes(
    tmp_path: Path,
) -> None:
    source_rows, acquisition, _, kwargs = _fixture(tmp_path)
    reference_bank = official_reference.deserialize_official_reference_bank(
        (acquisition / "official_pre_action_reference_bank.jsonl").read_bytes()
    )
    terminal_path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in terminal_path.read_text().splitlines()]
    source = source_rows["train"][1]
    terminal = rows[1]
    _enable_validator_contract_repairs(
        acquisition=acquisition,
        source=source,
        terminal=terminal,
        reference_bank=reference_bank,
    )
    terminal["validator_a"]["result"]["contract_repair"][
        "trigger_contract_error_codes"
    ] = ["validator_a_overall_pass_boolean"]
    _jsonl(terminal_path, rows)
    _materialize_terminal_cache(
        acquisition,
        source=source,
        terminal=terminal,
        reference_bank_sha256=sha256_file(
            acquisition / "official_pre_action_reference_bank.jsonl"
        ),
    )
    _add_final_provider_roles(
        acquisition,
        publisher.generator.ROLE_VALIDATOR_A_PRIMARY_CONTRACT_REPAIR,
        publisher.generator.ROLE_VALIDATOR_B_PRIMARY_CONTRACT_REPAIR,
    )
    _refresh_acquisition(acquisition)

    with pytest.raises(
        PublicationError,
        match="trigger contract codes differ from invalid raw report",
    ):
        publish_release(**kwargs)


def test_publisher_rejects_canonical_cache_path_swap(tmp_path: Path) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    caches = sorted((acquisition / "cache/rewrite_primary").glob("*.json"))
    assert len(caches) >= 2
    first = caches[0].read_bytes()
    second = caches[1].read_bytes()
    caches[0].write_bytes(second)
    caches[1].write_bytes(first)

    with pytest.raises(PublicationError, match="provider cache path/binding drift"):
        publish_release(**kwargs)


def test_publisher_rejects_user_prompt_binding_tamper(tmp_path: Path) -> None:
    source_rows, acquisition, _, kwargs = _fixture(tmp_path)
    source = source_rows["train"][1]
    cache_path = (
        acquisition
        / "cache/rewrite_primary"
        / f"{sha256_text(source['sample_id'])}.json"
    )
    cache = json.loads(cache_path.read_text())
    cache["binding"]["user_prompt_sha256"] = "0" * 64
    cache["binding_sha256"] = sha256_text(
        json.dumps(
            cache["binding"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    _json(cache_path, cache)

    with pytest.raises(
        PublicationError,
        match="generator terminal/cache replay failed: terminal provider cache binding mismatch",
    ):
        publish_release(**kwargs)


def test_publisher_binds_terminal_candidate_to_selected_rewrite_raw(
    tmp_path: Path,
) -> None:
    source_rows, acquisition, _, kwargs = _fixture(tmp_path)
    source = source_rows["train"][1]
    terminal_path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in terminal_path.read_text().splitlines()]
    terminal = rows[1]
    role = publisher.generator.ROLE_REWRITE_PRIMARY
    cache_path = (
        acquisition / "cache" / role / f"{sha256_text(source['sample_id'])}.json"
    )
    cache = json.loads(cache_path.read_text())
    cache["provider_response"]["raw_reasoning"] = (
        "A different cached reasoning trajectory that remains nonempty."
    )
    cache["provider_response"]["raw_content"] = json.dumps(
        {
            "answer": (
                "A different cached Minutes paragraph remained deliberately detached "
                "from the terminal candidate while retaining enough words for transport."
            )
        },
        separators=(",", ":"),
    )
    cache["raw_reasoning_sha256"] = sha256_text(
        cache["provider_response"]["raw_reasoning"]
    )
    cache["raw_content_sha256"] = sha256_text(cache["provider_response"]["raw_content"])
    _json(cache_path, cache)
    terminal["generation"]["provider"][role] = publisher.generator._provider_record(
        publisher.generator.ProviderResponse.from_dict(cache["provider_response"]),
        cache,
    )
    _jsonl(terminal_path, rows)
    _materialize_terminal_cache(
        acquisition,
        source=source,
        terminal=terminal,
        reference_bank_sha256=sha256_file(
            acquisition / "official_pre_action_reference_bank.jsonl"
        ),
    )
    _refresh_acquisition(acquisition)

    with pytest.raises(
        PublicationError,
        match="terminal candidate differs from selected rewrite cache",
    ):
        publish_release(**kwargs)


def test_publisher_rejects_cache_backed_orphan_validator_role(
    tmp_path: Path,
) -> None:
    source_rows, acquisition, _, kwargs = _fixture(tmp_path)
    source = source_rows["train"][1]
    terminal_path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in terminal_path.read_text().splitlines()]
    terminal = rows[1]
    row = SimpleNamespace(
        sample_id=source["sample_id"],
        split=source["split"],
        source_analysis=source["analysis"],
        source_analysis_sha256=sha256_text(source["analysis"]),
        provided_data=source["provided_data"],
        provided_data_sha256=sha256_text(source["provided_data"]),
        meeting_date=source["meeting_date"],
    )
    candidate = SimpleNamespace(
        teacher_response_analysis=terminal["teacher_response_analysis"],
        rewritten_minutes=terminal["rewritten_minutes"],
    )
    result = terminal["validator_a"]["result"]
    raw = {
        key: result[key]
        for key in (
            "source_claims",
            "rewrite_claims",
            "reasoning_issues",
            "bidirectional_entailment",
            "reasoning_compatible",
            "issues",
            "overall_pass",
        )
    }
    _materialize_provider_cache(
        acquisition,
        source=source,
        role=publisher.generator.ROLE_VALIDATOR_A_STYLE_REPAIR,
        system_prompt=publisher.generator.VALIDATOR_A_SYSTEM_PROMPT,
        user_prompt=publisher.generator._validator_a_user_prompt(row, candidate),
        raw_content=raw,
        raw_reasoning="Orphan but otherwise valid Validator-A report.",
        provider_container=terminal["validator_a"]["provider"],
        reference_bank_sha256=sha256_file(
            acquisition / "official_pre_action_reference_bank.jsonl"
        ),
    )
    _jsonl(terminal_path, rows)
    _materialize_terminal_cache(
        acquisition,
        source=source,
        terminal=terminal,
        reference_bank_sha256=sha256_file(
            acquisition / "official_pre_action_reference_bank.jsonl"
        ),
    )
    _add_final_provider_roles(
        acquisition, publisher.generator.ROLE_VALIDATOR_A_STYLE_REPAIR
    )
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="provider invocation sequence drift"):
        publish_release(**kwargs)


def _install_source_contract_reject(
    *, acquisition: Path, source: dict, reference_bank
) -> dict:
    class ExhaustedSourceBackend:
        def generate(self, *, role, **kwargs):
            del kwargs
            if role not in {
                publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY,
                publisher.generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR,
            }:
                raise AssertionError(role)
            return publisher.generator.ProviderResponse(
                raw_reasoning="Source verifier attempted the required report.",
                raw_content=json.dumps(
                    {"answer": "still not a source-audit report"},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                response_id=f"exhausted-{role}-{source['sample_id']}",
                returned_model=publisher.generator.MODEL,
                system_fingerprint="fixed-fingerprint",
                finish_reason="stop",
                created=None,
                usage={},
            )

    placeholder = "0" * 64
    row = publisher.generator.PreparedRow(
        sample_id=source["sample_id"],
        split=source["split"],
        source_split=source["source_split"],
        split_index=source["source_index"],
        source_line_number=source["source_index"],
        generation_manifest_line_number=source["source_index"],
        meeting_date=source["meeting_date"],
        atomic_topic=source["atomic_topic"],
        section_style_id=source["section_style_id"],
        prompt="",
        provided_data=source["provided_data"],
        source_analysis=source["analysis"],
        prompt_sha256=placeholder,
        provided_data_sha256=sha256_text(source["provided_data"]),
        source_analysis_sha256=sha256_text(source["analysis"]),
        candidate_response_sha256=placeholder,
        source_answer_sha256=sha256_text(source["analysis"]),
        source_response_sha256=placeholder,
        generation_manifest_row_sha256=placeholder,
        source_row_sha256=placeholder,
    )
    scratch = acquisition.parent / f"scratch-{source['sample_id']}"
    reference_path = acquisition / "official_pre_action_reference_bank.jsonl"
    terminal = publisher.generator._process_terminal(
        row,
        output=scratch,
        tokenizer=FakeDeepSeekTokenizer(),
        reference_bank=reference_bank,
        official_reference_bank_sha256=sha256_file(reference_path),
        backend=ExhaustedSourceBackend(),
        identity=publisher.generator.ProviderIdentityRegistry(),
        environment={publisher.generator.API_KEY_ENV: "test"},
        config=publisher.generator.ProviderConfig(),
        code_sha256=publisher.generator._implementation_contract()["composite_sha256"],
    )
    assert terminal["terminal_status"] == "SOURCE_AUDIT_CONTRACT_REJECT"
    stale_adjudication = (
        acquisition
        / "cache"
        / publisher.generator.ROLE_SOURCE_AUDIT_ADJUDICATION
        / f"{sha256_text(source['sample_id'])}.json"
    )
    stale_adjudication.unlink(missing_ok=True)
    for role in (
        publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY,
        publisher.generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR,
        "terminal",
    ):
        source_path = (
            scratch / "cache" / role / f"{sha256_text(source['sample_id'])}.json"
        )
        target_path = acquisition / "cache" / role / source_path.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    terminal_path = acquisition / f"terminal/{source['split']}.jsonl"
    rows = [json.loads(line) for line in terminal_path.read_text().splitlines()]
    rows[source["source_index"] - 1] = terminal
    _jsonl(terminal_path, rows)
    rejection_path = acquisition / "audits/rejections.jsonl"
    rejections = [json.loads(line) for line in rejection_path.read_text().splitlines()]
    _jsonl(
        rejection_path,
        [
            terminal if item["sample_id"] == source["sample_id"] else item
            for item in rejections
        ],
    )
    receipt_path = acquisition / "source_admission_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    counts = receipt["status_counts"][source["split"]]
    counts["SOURCE_QUALITY_REJECT"] -= 1
    if counts["SOURCE_QUALITY_REJECT"] == 0:
        del counts["SOURCE_QUALITY_REJECT"]
    counts["SOURCE_AUDIT_CONTRACT_REJECT"] = 1
    receipt_roles = set(receipt["provider_identities"]["roles_observed"])
    receipt_roles.add(publisher.generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR)
    receipt["provider_identities"]["roles_observed"] = sorted(receipt_roles)
    _json(receipt_path, receipt)
    final_path = acquisition / "final_summary.json"
    final = json.loads(final_path.read_text())
    final_counts = final["status_counts"][source["split"]]
    final_counts["SOURCE_QUALITY_REJECT"] -= 1
    if final_counts["SOURCE_QUALITY_REJECT"] == 0:
        del final_counts["SOURCE_QUALITY_REJECT"]
    final_counts["SOURCE_AUDIT_CONTRACT_REJECT"] = 1
    _json(final_path, final)
    _add_final_provider_roles(
        acquisition, publisher.generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR
    )
    _refresh_acquisition(acquisition)
    return terminal


def test_publisher_accepts_source_contract_exhaustion_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    source_rows, acquisition, release, kwargs = _fixture(tmp_path)
    reference_bank = official_reference.deserialize_official_reference_bank(
        (acquisition / "official_pre_action_reference_bank.jsonl").read_bytes()
    )
    source = source_rows["train"][0]
    _install_source_contract_reject(
        acquisition=acquisition,
        source=source,
        reference_bank=reference_bank,
    )
    handoff = publish_release(**kwargs)
    assert handoff["split_counts"] == {split: 1 for split in SPLITS}
    assert release.is_dir()

    source_rows, acquisition, _release, kwargs = _fixture(tmp_path / "tamper")
    reference_bank = official_reference.deserialize_official_reference_bank(
        (acquisition / "official_pre_action_reference_bank.jsonl").read_bytes()
    )
    source = source_rows["train"][0]
    terminal = _install_source_contract_reject(
        acquisition=acquisition,
        source=source,
        reference_bank=reference_bank,
    )
    terminal["source_audit"]["result"]["primary"] = {"tampered": True}
    terminal_path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in terminal_path.read_text().splitlines()]
    rows[0] = terminal
    _jsonl(terminal_path, rows)
    _materialize_terminal_cache(
        acquisition,
        source=source,
        terminal=terminal,
        reference_bank_sha256=sha256_file(
            acquisition / "official_pre_action_reference_bank.jsonl"
        ),
    )
    rejection_path = acquisition / "audits/rejections.jsonl"
    rejections = [json.loads(line) for line in rejection_path.read_text().splitlines()]
    _jsonl(
        rejection_path,
        [
            terminal if item["sample_id"] == source["sample_id"] else item
            for item in rejections
        ],
    )
    _refresh_acquisition(acquisition)
    with pytest.raises(
        PublicationError, match="source terminal exhausted normalized report drift"
    ):
        publish_release(**kwargs)


@pytest.mark.parametrize(
    ("validator", "expected_status"),
    [
        ("a", "VALIDATOR_A_CONTRACT_REJECT"),
        ("b", "VALIDATOR_B_CONTRACT_REJECT"),
    ],
)
def test_publisher_accepts_validator_contract_exhaustion(
    tmp_path: Path, validator: str, expected_status: str
) -> None:
    source_rows, acquisition, release, kwargs = _fixture(tmp_path)
    reference_bank = official_reference.deserialize_official_reference_bank(
        (acquisition / "official_pre_action_reference_bank.jsonl").read_bytes()
    )
    source = source_rows["train"][0]
    pass_template = _terminal(source, reference_bank, passed=True)

    class ExhaustedValidatorBackend:
        def generate(self, *, role, user_prompt, **call_kwargs):
            del call_kwargs
            if role == publisher.generator.ROLE_SOURCE_AUDIT_PRIMARY:
                normalized = _source_report(
                    source["analysis"], source["provided_data"], passed=True
                )
                raw = {
                    "claims": normalized["claims"],
                    "blocking_issues": [],
                    "overall_pass": True,
                }
                reasoning = "Source-only verification."
            elif role == publisher.generator.ROLE_REWRITE_PRIMARY:
                raw = {"answer": pass_template["rewritten_minutes"]}
                reasoning = pass_template["teacher_response_analysis"]
            elif role == publisher.generator.ROLE_VALIDATOR_A_PRIMARY:
                if validator == "a":
                    raw = {"answer": "not a Validator-A report"}
                else:
                    normalized = _validator_a_result(
                        source["analysis"],
                        pass_template["teacher_response_analysis"],
                        pass_template["rewritten_minutes"],
                    )
                    raw = {
                        key: normalized[key]
                        for key in (
                            "source_claims",
                            "rewrite_claims",
                            "reasoning_issues",
                            "bidirectional_entailment",
                            "reasoning_compatible",
                            "issues",
                            "overall_pass",
                        )
                    }
                reasoning = "Fidelity verification."
            elif role in {
                publisher.generator.ROLE_VALIDATOR_A_PRIMARY_CONTRACT_REPAIR,
                publisher.generator.ROLE_VALIDATOR_B_PRIMARY,
                publisher.generator.ROLE_VALIDATOR_B_PRIMARY_CONTRACT_REPAIR,
            }:
                raw = {"answer": "still not a validator report"}
                reasoning = "Attempted validator report repair."
            else:
                raise AssertionError(role)
            return publisher.generator.ProviderResponse(
                raw_reasoning=reasoning,
                raw_content=json.dumps(raw, sort_keys=True, separators=(",", ":")),
                response_id=f"{role}-{source['sample_id']}",
                returned_model=publisher.generator.MODEL,
                system_fingerprint="fixed-fingerprint",
                finish_reason="stop",
                created=None,
                usage={},
            )

    placeholder = "0" * 64
    row = publisher.generator.PreparedRow(
        sample_id=source["sample_id"],
        split=source["split"],
        source_split=source["source_split"],
        split_index=source["source_index"],
        source_line_number=source["source_index"],
        generation_manifest_line_number=source["source_index"],
        meeting_date=source["meeting_date"],
        atomic_topic=source["atomic_topic"],
        section_style_id=source["section_style_id"],
        prompt="",
        provided_data=source["provided_data"],
        source_analysis=source["analysis"],
        prompt_sha256=placeholder,
        provided_data_sha256=sha256_text(source["provided_data"]),
        source_analysis_sha256=sha256_text(source["analysis"]),
        candidate_response_sha256=placeholder,
        source_answer_sha256=sha256_text(source["analysis"]),
        source_response_sha256=placeholder,
        generation_manifest_row_sha256=placeholder,
        source_row_sha256=placeholder,
    )
    scratch = acquisition.parent / "validator-contract-scratch"
    reference_sha = sha256_file(
        acquisition / "official_pre_action_reference_bank.jsonl"
    )
    terminal = publisher.generator._process_terminal(
        row,
        output=scratch,
        tokenizer=FakeDeepSeekTokenizer(),
        reference_bank=reference_bank,
        official_reference_bank_sha256=reference_sha,
        backend=ExhaustedValidatorBackend(),
        identity=publisher.generator.ProviderIdentityRegistry(),
        environment={publisher.generator.API_KEY_ENV: "test"},
        config=publisher.generator.ProviderConfig(),
        code_sha256=publisher.generator._implementation_contract()["composite_sha256"],
    )
    assert terminal["terminal_status"] == expected_status
    stale = (
        acquisition
        / "cache"
        / publisher.generator.ROLE_SOURCE_AUDIT_ADJUDICATION
        / f"{sha256_text(source['sample_id'])}.json"
    )
    stale.unlink(missing_ok=True)
    provider_roles = set(terminal["source_audit"]["provider"])
    provider_roles.update(terminal["generation"]["provider"])
    provider_roles.update(terminal["validator_a"].get("provider", {}))
    provider_roles.update(terminal["validator_b"].get("provider", {}))
    for role in (*provider_roles, "terminal"):
        source_path = (
            scratch / "cache" / role / f"{sha256_text(source['sample_id'])}.json"
        )
        target_path = acquisition / "cache" / role / source_path.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    terminal_path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in terminal_path.read_text().splitlines()]
    rows[0] = terminal
    _jsonl(terminal_path, rows)
    rejection_path = acquisition / "audits/rejections.jsonl"
    rejections = [json.loads(line) for line in rejection_path.read_text().splitlines()]
    _jsonl(
        rejection_path,
        [
            terminal if item["sample_id"] == source["sample_id"] else item
            for item in rejections
        ],
    )
    if validator == "b":
        validator_b_path = acquisition / "audits/validator_b.jsonl"
        b_rows = [
            json.loads(line) for line in validator_b_path.read_text().splitlines()
        ]
        b_rows.append(
            {
                "sample_id": source["sample_id"],
                "split": source["split"],
                "terminal_status": expected_status,
                "validator_b": terminal["validator_b"],
            }
        )
        b_rows.sort(key=lambda item: (SPLITS.index(item["split"]), item["sample_id"]))
        _jsonl(validator_b_path, b_rows)
    receipt_path = acquisition / "source_admission_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    counts = receipt["status_counts"]["train"]
    counts["SOURCE_QUALITY_REJECT"] -= 1
    if counts["SOURCE_QUALITY_REJECT"] == 0:
        del counts["SOURCE_QUALITY_REJECT"]
    counts["PASS"] += 1
    receipt_roles = set(receipt["provider_identities"]["roles_observed"])
    receipt_roles.update(provider_roles)
    receipt["provider_identities"]["roles_observed"] = sorted(receipt_roles)
    _json(receipt_path, receipt)
    final_path = acquisition / "final_summary.json"
    final = json.loads(final_path.read_text())
    final_counts = final["status_counts"]["train"]
    final_counts["SOURCE_QUALITY_REJECT"] -= 1
    if final_counts["SOURCE_QUALITY_REJECT"] == 0:
        del final_counts["SOURCE_QUALITY_REJECT"]
    final_counts[expected_status] = 1
    _json(final_path, final)
    _add_final_provider_roles(acquisition, *provider_roles)
    _refresh_acquisition(acquisition)
    handoff = publish_release(**kwargs)
    assert handoff["split_counts"] == {split: 1 for split in SPLITS}
    assert release.is_dir()
