



from open_r1.validator.cos.cos_calc import cosine_similarity_calc
from open_r1.validator.cos.embedding_model import EmbeddingModel

model = EmbeddingModel("models/DeepSeek-R1-Distill-Llama-8B")

res = cosine_similarity_calc(target=["The policy rate was raised."],
                             generated=["Interest rates increased."],
                             model_wrapper=model)
# 0.660442054271698
print(res)