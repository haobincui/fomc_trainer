import torch
import torch.nn as nn
import json
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class EmbeddingModel:
    def __init__(self, model_path=None, bs=None, dump_dir="embedding_dump"):
        """
        :param dump_dir: directory to save all model dump info
        """
        self.model, self.tokenizer = self.load_model(model_path)
        self.bs = bs
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.dump_dir = dump_dir
        os.makedirs(dump_dir, exist_ok=True)
        self.dump_id = 0  # 自增编号避免覆盖

    def _save_json(self, obj, filename):
        """Save Python object as pretty JSON."""
        with open(os.path.join(self.dump_dir, filename), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def _save_numpy(self, tensor, filename):
        """Save tensor as .npy file (precision-preserving)."""
        arr = tensor.detach().cpu().float().numpy()
        np.save(os.path.join(self.dump_dir, filename), arr)

    def _save_text_tensor(self, tensor, filename):
        """Save tensor as readable text."""
        arr = tensor.detach().cpu().float().numpy()
        with open(os.path.join(self.dump_dir, filename), "w", encoding="utf-8") as f:
            np.set_printoptions(threshold=np.inf, linewidth=200)
            f.write(np.array2string(arr, separator=", "))

    def load_model(self, model_path):
        model = AutoModel.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def emb_mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return pooled

    def dump_forward_pass(self, sentences, encoded_input, model_output, embeddings):
        """Dump EVERYTHING useful for debug."""
        dump_id = self.dump_id
        self.dump_id += 1

        # 1. 原始输入句子
        self._save_json({"sentences": sentences}, f"{dump_id}_sentences.json")

        # 2. tokens & input IDs
        tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in encoded_input["input_ids"]]
        self._save_json({
            "input_ids": encoded_input["input_ids"].tolist(),
            "tokens": tokens
        }, f"{dump_id}_tokens.json")

        # 3. last_hidden_state (H)
        self._save_numpy(model_output.last_hidden_state, f"{dump_id}_last_hidden_state.npy")

        # 4. 保存 mean pooling 前的 hidden states（人类可读）
        self._save_text_tensor(model_output.last_hidden_state, f"{dump_id}_hidden_state.txt")

        # 5. past_key_values
        pkv_info = []
        # key shape = (batch_size, num_heads, seq_len, head_dim)
        # hidden_size=num_heads×head_dim
        # hidden_size = 4096 = 32 x 128
        # but GQA: reduce num_heades fro 32 to 8 in llama
        # therefore, Q size = 32 x 128; K size = 8 x 128; V size = 8 x 128
        for i, (k, v) in enumerate(model_output.past_key_values):
            pkv_info.append({
                "layer": i,
                "key_shape": list(k.shape),
                "value_shape": list(v.shape),
            })
            self._save_numpy(k, f"{dump_id}_pkv_key_layer{i}.npy")
            self._save_numpy(v, f"{dump_id}_pkv_value_layer{i}.npy")

        self._save_json(pkv_info, f"{dump_id}_pkv_summary.json")

        # 6. sentence embedding（mean pooling结果）
        self._save_numpy(embeddings, f"{dump_id}_embedding.npy")

    def get_embeddings(self, sentences):
        embeddings = torch.tensor([], device="cuda")

        batches = [sentences] if self.bs is None else \
                  [sentences[i:i+self.bs] for i in range(0, len(sentences), self.bs)]

        for batch in batches:
            print(f"👉 Model input batch: {batch}")

            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            encoded_input = encoded_input.to("cuda")

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            pooled = self.emb_mean_pooling(model_output, encoded_input["attention_mask"]).cuda()

            # 🔥 Dump everything from this forward pass
            self.dump_forward_pass(batch, encoded_input, model_output, pooled)

            embeddings = torch.cat((embeddings, pooled), dim=0)

        return embeddings

    def get_similarities(self, x, y=None):
        if y is None:
            num_samples = x.shape[0]
            sim = [[0]*num_samples for _ in range(num_samples)]
            for row in range(num_samples):
                sim[row][:row+1] = self.cos(x[row].repeat(row+1,1), x[:row+1]).tolist()
            return sim
        else:
            return self.cos(x, y).tolist()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def plot_token_contribution_heatmap(tokens, hidden_states_file, output_file):
    """
    Generate a heatmap showing token contribution to the sentence embedding.

    Parameters
    ----------
    tokens : list[str]
        List of token strings
    hidden_states_file : str
        Path to saved last_hidden_state.npy
    output_file : str
        Output image file path
    """
    # Load hidden states (T × D)
    H = np.load(hidden_states_file)  # shape: (1, T, D)
    H = H[0]  # remove batch dim

    # mean pooling embedding (same logic as your code)
    E = H.mean(axis=0)  # shape: (D,)

    # Compute cosine similarity between each token embedding and sentence embedding
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    contributions = np.array([cosine(H[i], E) for i in range(len(tokens))])

    # Create heatmap
    plt.figure(figsize=(max(10, len(tokens) * 0.6), 3))
    sns.heatmap(
        contributions[np.newaxis, :],
        cmap="YlOrRd",
        annot=False,
        cbar=True,
        xticklabels=tokens,
        yticklabels=["Contribution"],
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"🔥 Heatmap saved to: {output_file}")



if __name__ == "__main__":
    target = ["Federal fund rate will be increased by 0.25% next week."]
    generated = ["The federal reserve is expected to raise interest rates by a quarter point in the upcoming meeting."]
    model_path = "models/DeepSeek-R1-Distill-Llama-8B"
    emb_model = EmbeddingModel(model_path=model_path, bs=None, dump_dir="emb_output")
    target_emb = emb_model.get_embeddings(target)
    generated_emb = emb_model.get_embeddings(generated)
    cos_sim = emb_model.get_similarities(target_emb, generated_emb)
    print("Cosine Similarity:", cos_sim)

    import json

    with open("emb_output/0_tokens.json") as f:
        tokens = json.load(f)["tokens"][0]  # 第1条输入

    plot_token_contribution_heatmap(
        tokens=tokens,
        hidden_states_file="emb_output/0_last_hidden_state.npy",
        output_file="emb_output/0_token_heatmap.png"
    )

    with open("emb_output/1_tokens.json") as f:
        tokens = json.load(f)["tokens"][0]  # 第1条输入

    plot_token_contribution_heatmap(
        tokens=tokens,
        hidden_states_file="emb_output/1_last_hidden_state.npy",
        output_file="emb_output/1_token_heatmap.png"
    )


    


