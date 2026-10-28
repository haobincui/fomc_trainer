from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from jobs.eval import reanalyse_paper_chk2_meeting_bootstrap as subject


def fixture_rows():
    samples = []
    # Unequal numbers of inputs per meeting, and unequal partition sizes.
    for split, meetings in (("train", (2, 1)), ("validation", (1,)), ("test", (3,))):
        for m, n in enumerate(meetings):
            for r in range(n):
                samples.append({"sample_id": f"{split}-{m}-{r}", "meeting_id": f"{split}-{m}",
                                "split": split, "prompt_sha256": "a" * 64, "reference_sha256": "b" * 64})
    rows = []
    for i, sample in enumerate(samples):
        for k in range(3):
            replicate_seed = 100 + k
            seed = int.from_bytes(hashlib.sha256(f"{replicate_seed}|{sample['sample_id']}|{k}".encode()).digest()[:8], "big") % (2**31)
            for m, model in enumerate(subject.MODELS):
                rows.append({**sample, "model_id": model, "replicate_id": k,
                             "replicate_seed": replicate_seed, "row_seed": seed,
                             "score_contract_sha256": "c" * 64,
                             **{policy: {metric: i * .02 + m * .1 + k * .01 + t * .003
                                         for t, metric in enumerate(subject.METRICS)} for policy in subject.POLICIES}})
    return samples, rows


def panel():
    return subject.build_panel(*fixture_rows())


def test_meeting_equal_weight_and_fixed_split_allocation():
    data = panel()
    values = subject.meeting_values(data)
    expected = np.mean([data.scores[..., indices, :].mean(axis=(-2, -1)) for indices in data.meeting_rows], axis=0)
    np.testing.assert_allclose(values.mean(axis=-1), expected, atol=1e-15)
    assert not np.allclose(expected, data.scores.mean(axis=(-2, -1)))
    draws, plans = subject.meeting_bootstrap(data, draws=31, seed=64)
    for split in subject.SPLITS:
        positions = data.split_positions[split]
        assert plans[split].shape == (31, len(positions))
        assert set(plans[split].ravel()).issubset(set(positions))
        np.testing.assert_allclose(draws[split], values[..., plans[split]].mean(axis=-1), atol=1e-15)
    combined = np.concatenate([plans[split] for split in subject.SPLITS], axis=1)
    np.testing.assert_allclose(draws["pooled"], values[..., combined].mean(axis=-1), atol=1e-15)


def test_main_bootstrap_invariant_to_independent_within_row_paired_permutations():
    data = panel()
    altered = copy.deepcopy(data)
    rng = np.random.default_rng(412)
    for row in range(len(data.samples)):
        permutation = rng.permutation(data.scores.shape[-1])
        altered.scores[..., row, :] = data.scores[..., row, :][..., permutation]
    before, plans = subject.meeting_bootstrap(data, draws=61, seed=183)
    after, changed_plans = subject.meeting_bootstrap(altered, draws=61, seed=183)
    for scope in before:
        np.testing.assert_allclose(before[scope], after[scope], atol=1e-15)
    for split in plans:
        np.testing.assert_array_equal(plans[split], changed_plans[split])


def test_models_stay_paired_in_both_analyses():
    data = panel()
    main, _ = subject.meeting_bootstrap(data, draws=79, seed=27)
    generation = subject.generation_bootstrap(data, draws=79, seed=28)
    for values in [*main.values(), generation]:
        np.testing.assert_allclose(values[:, 1] - values[:, 0], .1, atol=1e-15)
        np.testing.assert_allclose(values[:, 2] - values[:, 1], .1, atol=1e-15)


def test_generation_uses_independent_input_plans_and_meeting_weights(monkeypatch):
    data = panel()
    recorded = []
    original = subject.row_replicate_plan

    def recording(rng, **kwargs):
        value = original(rng, **kwargs)
        recorded.append(value)
        return value

    monkeypatch.setattr(subject, "row_replicate_plan", recording)
    actual = subject.generation_bootstrap(data, draws=83, seed=21)
    assert len(recorded) == len(data.samples)
    assert all(not np.array_equal(recorded[0], plan) for plan in recorded[1:])
    expected = np.zeros_like(actual)
    for indices in data.meeting_rows:
        for row in indices:
            expected += data.scores[..., row, recorded[row]].mean(axis=-1) / (len(data.meetings) * len(indices))
    np.testing.assert_allclose(actual, expected, atol=1e-15)


@pytest.mark.parametrize("failure", ["duplicate", "missing", "pairing", "binding", "nonfinite"])
def test_invalid_pairings_or_scores_fail_closed(failure):
    samples, rows = fixture_rows()
    if failure == "duplicate":
        rows.append(rows[0])
    elif failure == "missing":
        rows.pop()
    elif failure == "pairing":
        rows[1]["row_seed"] += 1
    elif failure == "binding":
        rows[0]["meeting_id"] = "wrong"
    else:
        rows[0]["raw_best_effort"]["mpnet_cosine"] = float("nan")
    with pytest.raises(ValueError):
        subject.build_panel(samples, rows)


def test_refuses_to_overwrite_historical_or_existing_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for output in (source, source / "nested"):
        with pytest.raises(ValueError, match="new directory outside"):
            subject.run(source, output, draws=10, seed=1)
