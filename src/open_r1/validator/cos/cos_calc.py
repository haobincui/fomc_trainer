import pandas as pd
import logging
from open_r1.validator.cos.embedding_model import EmbeddingModel




def cosine_similarity_calc(target: list|str, generated: list|str, model_wrapper: EmbeddingModel) -> float:
    """
    Calculate the average cosine similarity between target and generated texts
    using a specified embedding model.

    This function supports both string and list inputs. If a single string is
    provided, it will be automatically wrapped as a one-element list. Each
    pair of corresponding elements in `target` and `generated` is converted
    into vector embeddings using `model_wrapper.get_embeddings()`, and their
    cosine similarity is computed via `model_wrapper.get_similarities()`.
    The function then returns the average cosine similarity across all pairs.

    Parameters
    ----------
    target : list[str] | str
        The reference or ground-truth text(s) used for comparison.
        Can be a single string or a list of strings.

    generated : list[str] | str
        The model-generated text(s) to be evaluated against the target(s).
        Must have the same length as `target`.

    model_wrapper : EmbeddingModel
        A wrapper object providing two key methods:
            - `get_embeddings(text_list: list[str]) -> np.ndarray`
            - `get_similarities(emb1: np.ndarray, emb2: np.ndarray) -> list[float]`
        It handles both embedding generation and cosine similarity calculation.

    Returns
    -------
    float
        The mean cosine similarity between the target and generated text pairs.
        The result ranges from -1.0 to 1.0, where higher values indicate greater
        semantic similarity.

    Raises
    ------
    AssertionError
        If the lengths of `target` and `generated` do not match.

    Notes
    -----
    - The function takes the **minimum** value of each cosine similarity vector
      (via `min(cos)`) when multiple embeddings are returned, ensuring robustness
      against outlier tokens or multi-vector encodings.
    - The final score is the arithmetic mean of all pairwise cosine similarities.

    Example
    -------
    >>> cosine_similarity_calc(
    ...     target=["The policy rate was raised."],
    ...     generated=["Interest rates increased."],
    ...     model_wrapper=my_embedding_model
    ... )
    0.87
    """

    coses = 0
    if isinstance(target, str):
        target = [target]
    if isinstance(generated, str):
        generated = [generated]
    assert len(target) == len(generated), "target and generated must be same length"
    for t, g in zip(target, generated):
        target_embedding = model_wrapper.get_embeddings([t])
        generated_embedding = model_wrapper.get_embeddings([g])
        # logging.info(f"target embedding device: {target_embedding.device}")
        # logging.info(f"generated embedding device: {generated_embedding.device}")

        cos = model_wrapper.get_similarities(target_embedding, generated_embedding)
        coses += min(cos)

    return coses / len(target)

