import tempfile
import unittest
from pathlib import Path

from jobs.eval.eval_checkpoint_generation import (
    LONG_TEXT_POLICY,
    BERTScoreBackend,
    MPNetCosineBackend,
    chunk_text_sentence_boundary,
    effective_text_token_budget,
)


class FakeTokenizer:
    """Whitespace tokenizer with reversible IDs and a pinned model limit."""

    def __init__(self, *, model_max_length: int, special_tokens: int) -> None:
        self.model_max_length = model_max_length
        self.special_tokens = special_tokens
        self.encode_calls: list[dict[str, object]] = []
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        if pair:
            raise AssertionError("Long-text scoring only tokenizes single texts")
        return self.special_tokens

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool = False,
    ) -> list[int]:
        self.encode_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
            }
        )
        if truncation:
            raise AssertionError("The long-text policy must never request truncation")
        result = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 10
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            result.append(self._token_to_id[token])
        if add_special_tokens:
            return ([-1] * self.special_tokens) + result
        return result

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        if skip_special_tokens or clean_up_tokenization_spaces:
            raise AssertionError("Token slices must be decoded without cleanup")
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)


class RecordingBERTChunkScorer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    def __call__(self, candidates, references):
        self.calls.append((list(candidates), list(references)))
        size = len(candidates)
        return {
            "bertscore_precision": [0.8] * size,
            "bertscore_recall": [0.6] * size,
            "bertscore_f1": [1.0] * size,
        }


class RecordingCosineChunkScorer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    def __call__(self, candidates, references):
        self.calls.append((list(candidates), list(references)))
        return [0.6] * len(candidates)


class TestSentenceBoundaryChunking(unittest.TestCase):
    def test_reserves_special_tokens_and_greedily_packs_sentences(self):
        tokenizer = FakeTokenizer(model_max_length=7, special_tokens=2)

        budget, total_limit, reserve = effective_text_token_budget(tokenizer)
        chunks = chunk_text_sentence_boundary(
            "one two. three four. five six.",
            tokenizer,
        )

        self.assertEqual((budget, total_limit, reserve), (5, 7, 2))
        self.assertEqual(
            [(chunk.text, chunk.token_count) for chunk in chunks],
            [("one two. three four.", 4), ("five six.", 2)],
        )
        self.assertTrue(
            all(chunk.token_count + reserve <= total_limit for chunk in chunks)
        )
        self.assertTrue(
            all(call["truncation"] is False for call in tokenizer.encode_calls)
        )

    def test_overlong_sentence_is_token_split_without_token_loss(self):
        tokenizer = FakeTokenizer(model_max_length=6, special_tokens=2)
        source = "one two three four five six seven eight nine"

        chunks = chunk_text_sentence_boundary(source, tokenizer)

        self.assertEqual([chunk.token_count for chunk in chunks], [4, 4, 1])
        reconstructed_tokens = [
            token for chunk in chunks for token in chunk.text.split()
        ]
        self.assertEqual(reconstructed_tokens, source.split())
        self.assertNotIn(9, [chunk.token_count for chunk in chunks])

    def test_explicit_limit_cannot_exceed_tokenizer_limit(self):
        tokenizer = FakeTokenizer(model_max_length=8, special_tokens=2)
        self.assertEqual(
            effective_text_token_budget(tokenizer, max_length=20),
            (6, 8, 2),
        )
        self.assertEqual(
            effective_text_token_budget(tokenizer, max_length=5),
            (3, 5, 2),
        )


class TestFrozenLongTextBackends(unittest.TestCase):
    def test_bertscore_ordinal_zip_longest_missing_side_is_weighted_zero(self):
        tokenizer = FakeTokenizer(model_max_length=6, special_tokens=2)
        scorer = RecordingBERTChunkScorer()
        with tempfile.TemporaryDirectory() as tmp:
            backend = BERTScoreBackend(
                Path(tmp),
                "a" * 64,
                verify_checksum=False,
                tokenizer=tokenizer,
                chunk_scorer=scorer,
            )
            scores = backend.score(
                [
                    "a b c d. e f g h.",
                    "",
                ],
                [
                    "r s t u.",
                    "v w x y.",
                ],
            )

        # Only document 0's ordinal-0 pair has text on both sides.  Its score
        # receives weight 4; ordinal-1 receives weight 4 and contributes zero.
        self.assertEqual(
            scorer.calls,
            [(["a b c d."], ["r s t u."])],
        )
        self.assertEqual(scores["bertscore_precision"], [0.4, 0.0])
        self.assertEqual(scores["bertscore_recall"], [0.3, 0.0])
        self.assertEqual(scores["bertscore_f1"], [0.5, 0.0])

        metadata = backend.semantic_metadata()
        self.assertEqual(metadata["long_text_policy"], LONG_TEXT_POLICY)
        audit = metadata["chunk_audit"]
        self.assertEqual(audit["candidate_chunk_counts"], [2, 0])
        self.assertEqual(audit["reference_chunk_counts"], [1, 1])
        self.assertEqual(audit["ordinal_pair_counts"], [2, 1])
        self.assertEqual(audit["document_weight_tokens"], [8, 4])
        self.assertEqual(audit["scored_nonempty_chunk_pairs"], 1)
        self.assertEqual(audit["missing_side_zero_chunk_pairs"], 2)
        self.assertFalse(audit["silent_truncation"])

    def test_mpnet_uses_the_same_chunking_and_weighting_contract(self):
        tokenizer = FakeTokenizer(model_max_length=6, special_tokens=2)
        scorer = RecordingCosineChunkScorer()
        with tempfile.TemporaryDirectory() as tmp:
            backend = MPNetCosineBackend(
                Path(tmp),
                "b" * 64,
                verify_checksum=False,
                tokenizer=tokenizer,
                chunk_scorer=scorer,
            )
            scores = backend.score(
                ["a b c d. e f g h.", ""],
                ["r s t u.", "v w x y."],
            )

        self.assertEqual(scores, [0.3, 0.0])
        self.assertEqual(scorer.calls, [(["a b c d."], ["r s t u."])])
        metadata = backend.semantic_metadata()
        self.assertEqual(metadata["long_text_policy"], LONG_TEXT_POLICY)
        self.assertEqual(
            metadata["chunk_audit"]["effective_content_token_budget"],
            4,
        )
        self.assertEqual(
            metadata["chunk_audit"]["missing_side_zero_chunk_pairs"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
