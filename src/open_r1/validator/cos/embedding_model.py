# Ref: SemScore: Automated Evaluation of Instruction-Tuned LLMs based on Semantic Textual Similarity: [https://github.com/geronimi73/semscore]

from functools import lru_cache

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

class EmbeddingModel:
    """
    A lightweight wrapper for computing sentence embeddings and cosine similarities
    using a transformer-based language model (e.g., LLaMA, RoBERTa, or BERT variants).

    This class supports batched embedding generation and provides utilities
    for both pairwise and matrix-based cosine similarity computations.
    It is designed for evaluating semantic similarity between texts,
    such as in FOMC sentence alignment or generated vs. target text comparison tasks.

    Attributes
    ----------
    model : transformers.PreTrainedModel
        The pre-trained transformer model loaded from `model_path`.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer corresponding to the loaded transformer model.
    bs : int
        Batch size used during embedding computation.
    cos : torch.nn.CosineSimilarity
        Cosine similarity module initialized on dimension 1.

    Methods
    -------
    load_model(model_path)
        Load the transformer model and tokenizer from the specified path.
    emb_mean_pooling(model_output, attention_mask)
        Compute mean-pooled sentence embeddings from model outputs.
    get_embeddings(sentences)
        Generate dense vector embeddings for a list of input sentences.
    get_similarities(x, y=None)
        Compute cosine similarity between embedding vectors.
    """

    def __init__(self, model_path=None, bs=None):
        """
        Initialize the EmbeddingModel with a pre-trained transformer model.

        Parameters
        ----------
        model_path : str, optional
            Path to the pre-trained transformer model (Hugging Face format).
            If None, a default model must be set manually.
        bs : int, default=8
            Batch size for embedding generation. Use None for single-pass processing.
        """
        self.model, self.tokenizer = self.load_model(model_path)
        self.bs = bs
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.device = next(self.model.parameters()).device

    def load_model(self, model_path):
        """
        Load a transformer model and its tokenizer from the given path.

        Parameters
        ----------
        model_path : str
            Path to a Hugging Face pre-trained model.

        Returns
        -------
        model : transformers.PreTrainedModel
            The loaded transformer model in evaluation mode.
        tokenizer : transformers.PreTrainedTokenizer
            The corresponding tokenizer for the loaded model.

        Notes
        -----
        - Model is automatically mapped to available devices using `device_map="auto"`.
        - Uses bfloat16 precision for efficiency.
        - The tokenizer's pad token is aligned with the EOS token.
        """
        model = AutoModel.from_pretrained(
            model_path,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def emb_mean_pooling(self, model_output, attention_mask):
        """
        Compute mean-pooled sentence embeddings from token-level embeddings.

        Parameters
        ----------
        model_output : torch.Tensor
            Model output from a forward pass (`last_hidden_state` expected as first element).
        attention_mask : torch.Tensor
            Binary mask used to average only non-padding tokens.

        Returns
        -------
        torch.Tensor
            Sentence-level embedding vectors of shape (batch_size, hidden_dim).
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, sentences):
        """
        Generate embeddings for a list of sentences using mean pooling.

        Parameters
        ----------
        sentences : list[str]
            List of input sentences to encode.

        Returns
        -------
        torch.Tensor
            Tensor of sentence embeddings with shape (num_sentences, hidden_dim).

        Notes
        -----
        - Sentences are processed in batches defined by `self.bs`.
        - Embeddings are moved to GPU (`cuda`) for faster similarity computation.
        - Outputs are concatenated along the first dimension.
        """
        embeddings = []

        if self.bs is None:
            batches = [sentences]
        else:
            batches = [sentences[i:i + self.bs] for i in range(0, len(sentences), self.bs)]

        for batch in batches:
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            encoded_input = encoded_input.to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            batch_embeddings = self.emb_mean_pooling(model_output, encoded_input['attention_mask']).to(self.device)
            embeddings.append(batch_embeddings)

        return torch.cat(embeddings, dim=0)

    def get_similarities(self, x, y=None):
        """
        Compute cosine similarities between embedding vectors.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of embeddings with shape (n, d).
        y : torch.Tensor, optional
            Tensor of embeddings with shape (m, d). If None, computes pairwise
            similarities within `x` (matrix form).

        Returns
        -------
        list[list[float]] or list[float]
            - If `y` is None: returns a 2D list of pairwise similarities among all rows of `x`.
            - If `y` is provided: returns a list of cosine similarities between corresponding pairs.

        Notes
        -----
        - When `y` is None`, this performs a lower-triangular similarity computation
          to reduce redundant operations.
        - Uses PyTorch’s `nn.CosineSimilarity` for efficient GPU computation.
        """
        if y is None:
            num_samples = x.shape[0]
            similarities = [[0 for _ in range(num_samples)] for _ in range(num_samples)]
            for row in tqdm(range(num_samples)):
                similarities[row][0:row + 1] = self.cos(x[row].repeat(row + 1, 1), x[0:row + 1]).tolist()
            return similarities
        else:
            return self.cos(x, y).tolist()


@lru_cache(maxsize=4)
def get_cached_embedding_model(model_path: str, bs: int | None = None) -> EmbeddingModel:
    return EmbeddingModel(model_path=model_path, bs=bs)
