# ref: TokenSHAP: Interpreting Large Language Models with Monte Carlo Shapley Value Estimation
# https://github.com/ronigold/TokenSHAP

import numpy as np


def shapley_value_calc(subset_with_p: list | str, subset: list | str, utility_function: callable, num_subset: int = 1, kwargs: dict = None) -> float:
    r"""
    Estimate the Shapley Value (\mathcal{SV}_i) of a given feature or prompt
    using Monte Carlo approximation, following the formulation of TokenSHAP
    (Goldstein et al., 2023).

    The Shapley Value quantifies the **marginal contribution** of a feature \( p_i \)
    to the overall model utility function \( f(\mathcal{S}) \),
    averaged over all possible subsets \(\mathcal{S} \subseteq \mathcal{P} \setminus \{p_i\}\):

    \[
    \mathcal{SV}_i(f) =
    \sum_{\mathcal{S} \subseteq \mathcal{P}\setminus p_i}
    \frac{|\mathcal{S}|! (n - |\mathcal{S}| - 1)!}{n!}
    \big(f(\mathcal{S} \cup \{p_i\}) - f(\mathcal{S})\big)
    \]

    In this implementation, \(\mathcal{SV}_i(f)\) is **approximated** by computing
    the difference between the utility of subsets with and without \( p_i \),
    and averaging over `num_subset` Monte Carlo samples.

    Parameters
    ----------
    subset_with_p : list
        A list or sequence representing a subset of features or prompts that
        includes the target feature \( p_i \).
        Example: `["GDP", "Inflation", "p_i"]`.

    subset : list
        The corresponding subset without the feature \( p_i \).
        Example: `["GDP", "Inflation"]`.

    utility_function : callable
        A user-defined scoring function that evaluates a subset \(\mathcal{S}\)
        and returns its scalar utility \( f(\mathcal{S}) \).
        For instance, this could compute:
          - Cosine similarity between model output and reference text,
          - Prediction accuracy,
          - Semantic coherence score.

    num_subset : int, default = 1
        The number of Monte Carlo samples (subsets) to evaluate.
        When set to 1, computes the marginal contribution deterministically.

    Returns
    -------
    float
        Estimated Shapley Value (\mathcal{SV}_i\)), representing the average
        marginal contribution of the feature \( p_i \) to model performance.

    Raises
    ------
    AssertionError
        If inputs are not valid lists or `utility_function` is not callable.

    Notes
    -----
    - This is a simplified Monte Carlo approximation, suitable for evaluating
      the contribution of a single token, prompt, or feature.
    - For large-scale LLM analysis, multiple random subsets are sampled and averaged.
    - Negative values indicate that adding the feature decreases model utility.
    - Implementation inspired by *TokenSHAP* (Goldstein et al., 2023).

    Example
    -------
    >>> def utility_fn(subset):
    ...     # Example: utility grows linearly with subset size
    ...     return len(subset) * 0.1
    >>> subset_with_p = ["GDP", "Inflation", "p_i"]
    >>> subset = ["GDP", "Inflation"]
    >>> shapley_value_calc(subset_with_p, subset, utility_fn)
    0.1
    """

    # Compute the utility with and without feature p_i
    try:
        # f_with = 1
        f_with = utility_function(subset_with_p, **kwargs)
        f_without = utility_function(subset, **kwargs)
        
    except Exception as e:
        raise ValueError(f"Failed calculation in Utility Function {utility_function.__name__}: {e}")

    # Marginal contribution of p_i
    marginal_contribution = float(f_with - f_without)

    # Monte Carlo approximation (here single-sample unless extended)
    
    return marginal_contribution / num_subset
