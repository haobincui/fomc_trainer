import unittest

from open_r1.validator.cos.cos_calc import cosine_similarity_calc


class DummyEmbeddingModel:
    def get_embeddings(self, sentences):
        return sentences

    def get_similarities(self, x, y=None):
        if y is None:
            return [[1.0 for _ in x] for _ in x]
        return [1.0 if left == right else 0.5 for left, right in zip(x, y)]


class TestCosineSimilarityCalc(unittest.TestCase):
    def test_single_string_inputs(self):
        model = DummyEmbeddingModel()
        result = cosine_similarity_calc(
            target="The policy rate was raised.",
            generated="Interest rates increased.",
            model_wrapper=model,
        )
        self.assertEqual(result, 0.5)

    def test_list_inputs_average_pairwise_scores(self):
        model = DummyEmbeddingModel()
        result = cosine_similarity_calc(
            target=["same", "different"],
            generated=["same", "other"],
            model_wrapper=model,
        )
        self.assertEqual(result, 0.75)


if __name__ == "__main__":
    unittest.main()
