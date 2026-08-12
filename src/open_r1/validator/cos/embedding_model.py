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

    def __init__(
        self,
        model_path=None,
        bs=None,
        max_tokens: int | None = None,
        long_text_policy: str = "model-default",
    ):
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
        if max_tokens is not None and max_tokens < 3:
            raise ValueError("max_tokens must be at least 3 when provided")
        if long_text_policy not in {"model-default", "error", "truncate", "chunk-mean"}:
            raise ValueError(f"Unsupported long_text_policy={long_text_policy!r}")
        if long_text_policy != "model-default" and max_tokens is None:
            raise ValueError(
                "max_tokens is required when long_text_policy is not model-default"
            )
        self.max_tokens = max_tokens
        self.long_text_policy = long_text_policy
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
        if self.long_text_policy == "chunk-mean":
            return self._get_chunk_mean_embeddings(sentences)

        embeddings = []

        if self.bs is None:
            batches = [sentences]
        else:
            batches = [sentences[i:i + self.bs] for i in range(0, len(sentences), self.bs)]

        for batch in batches:
            tokenizer_kwargs = {
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
            }
            if self.max_tokens is not None:
                tokenizer_kwargs["max_length"] = self.max_tokens
                if self.long_text_policy == "error":
                    lengths = [
                        len(
                            self.tokenizer.encode(
                                sentence,
                                add_special_tokens=True,
                                truncation=False,
                            )
                        )
                        for sentence in batch
                    ]
                    if any(length > self.max_tokens for length in lengths):
                        raise ValueError(
                            "Embedding input exceeds max_tokens under the error "
                            f"policy: max_observed={max(lengths)}, "
                            f"max_tokens={self.max_tokens}"
                        )
            encoded_input = self.tokenizer(batch, **tokenizer_kwargs)
            encoded_input = encoded_input.to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            batch_embeddings = self.emb_mean_pooling(model_output, encoded_input['attention_mask']).to(self.device)
            embeddings.append(batch_embeddings)

        return torch.cat(embeddings, dim=0)

    def _get_chunk_mean_embeddings(self, sentences):
        """Encode every token using bounded contiguous chunks.

        Each chunk is contextualized independently.  Special tokens remain in
        the model input but are excluded from pooling; chunk vectors are then
        weighted by their content-token counts.  This avoids silent truncation
        while bounding activation memory for long document comparisons.
        """

        if self.max_tokens is None:
            raise ValueError("chunk-mean requires max_tokens")
        special_token_count = self.tokenizer.num_special_tokens_to_add(pair=False)
        payload_capacity = self.max_tokens - special_token_count
        if payload_capacity < 1:
            raise ValueError(
                "max_tokens leaves no room for content after tokenizer special tokens"
            )

        document_embeddings = []
        for sentence in sentences:
            token_ids = self.tokenizer.encode(
                sentence,
                add_special_tokens=False,
                truncation=False,
            )
            if not token_ids:
                raise ValueError("Cannot embed an empty token sequence")

            weighted_sum = None
            total_content_tokens = 0
            for start in range(0, len(token_ids), payload_capacity):
                payload = token_ids[start : start + payload_capacity]
                build_with_special_tokens = getattr(
                    self.tokenizer,
                    "build_inputs_with_special_tokens",
                    None,
                )
                if callable(build_with_special_tokens):
                    input_ids = build_with_special_tokens(payload)
                    special_mask = self.tokenizer.get_special_tokens_mask(
                        payload,
                        already_has_special_tokens=False,
                    )
                elif special_token_count == 0:
                    # Some tokenizer classes (including the pinned DeepSeek
                    # Llama tokenizer) declare that no special tokens are
                    # added but do not expose build_inputs_with_special_tokens.
                    input_ids = list(payload)
                    special_mask = [0] * len(payload)
                else:
                    raise ValueError(
                        "Tokenizer cannot construct chunk inputs with its "
                        "declared special tokens"
                    )
                if len(input_ids) != len(special_mask):
                    raise ValueError(
                        "Tokenizer returned inconsistent input and special-token masks"
                    )
                encoded_input = {
                    "input_ids": torch.tensor(
                        [input_ids],
                        dtype=torch.long,
                        device=self.device,
                    ),
                    "attention_mask": torch.ones(
                        (1, len(input_ids)),
                        dtype=torch.long,
                        device=self.device,
                    ),
                }
                pooling_mask = torch.tensor(
                    [[0 if value else 1 for value in special_mask]],
                    dtype=torch.long,
                    device=self.device,
                )
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                chunk_embedding = self.emb_mean_pooling(
                    model_output,
                    pooling_mask,
                )
                content_tokens = int(pooling_mask.sum().item())
                weighted_chunk = chunk_embedding * content_tokens
                weighted_sum = (
                    weighted_chunk
                    if weighted_sum is None
                    else weighted_sum + weighted_chunk
                )
                total_content_tokens += content_tokens

            if weighted_sum is None or total_content_tokens < 1:
                raise ValueError("No content tokens remained after chunking")
            document_embeddings.append(weighted_sum / total_content_tokens)

        return torch.cat(document_embeddings, dim=0)

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
def get_cached_embedding_model(
    model_path: str,
    bs: int | None = None,
    max_tokens: int | None = None,
    long_text_policy: str = "model-default",
) -> EmbeddingModel:
    return EmbeddingModel(
        model_path=model_path,
        bs=bs,
        max_tokens=max_tokens,
        long_text_policy=long_text_policy,
    )
