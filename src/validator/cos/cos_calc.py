import pandas as pd

from validator.cos.embedding_model import EmbeddingModel


def cosine_similarity_calc(target: list|str, generated: list|str, model_wrapper: EmbeddingModel) -> float:
    coses = 0
    if isinstance(target, str):
        target = [target]
    if isinstance(generated, str):
        generated = [generated]
    assert len(target) == len(generated), "target and generated must be same length"
    for t, g in zip(target, generated):
        target_embedding = model_wrapper.get_embeddings([t])
        generated_embedding = model_wrapper.get_embeddings([g])

        cos = model_wrapper.get_similarities(target_embedding, generated_embedding)
        coses += min(cos)

    return coses / len(target)

