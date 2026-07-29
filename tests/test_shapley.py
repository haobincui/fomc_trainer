import unittest

from open_r1.validator.shapley.shapley_calc import shapley_value_calc


def length_utility(text, **kwargs):
    return len(text)


class TestShapleyValueCalc(unittest.TestCase):
    def test_positive_marginal_contribution(self):
        shapley = shapley_value_calc(
            subset_with_p="abcdef",
            subset="abc",
            utility_function=length_utility,
        )
        self.assertEqual(shapley, 3.0)

    def test_negative_marginal_contribution(self):
        shapley = shapley_value_calc(
            subset_with_p="short",
            subset="much longer baseline",
            utility_function=length_utility,
        )
        self.assertLess(shapley, 0.0)


if __name__ == "__main__":
    unittest.main()
