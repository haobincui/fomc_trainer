



from open_r1.validator.cos.cos_calc import cosine_similarity_calc
from open_r1.validator.cos.embedding_model import EmbeddingModel
from open_r1.validator.shapley.shapley_calc import shapley_value_calc

model = EmbeddingModel("models/DeepSeek-R1-Distill-Llama-8B")
target = "The policy rate was raised."
generated = "Interest rates increased."

res = cosine_similarity_calc(target=target,
                             generated=generated,
                             model_wrapper=model)
# 0.660442054271698
print(res)


shapley = shapley_value_calc(target, generated, cosine_similarity_calc, kwargs={"generated": target, "model_wrapper": model})
print(shapley)

