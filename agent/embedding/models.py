# -*- coding: utf-8 -*-
import logging
from abc import ABC, abstractmethod
from typing import List, Union, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ModeType = Literal["query", "passage"]

# Mapping for shorthand aliases to full HuggingFace paths
MODEL_ASSET_MAP = {
    "all": "sentence-transformers/all-MiniLM-L6-v2",
    "nli": "sentence-transformers/nli-distilroberta-base-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2"
}

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: Union[str, List[str]], mode: ModeType = "passage") -> np.ndarray:
        pass

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

class SBertEmbeddingModel(BaseEmbeddingModel):
    """
    Handles SBert family models: all-MiniLM, nli-distilroberta, and mpnet.
    """
    def __init__(self, model_alias_or_path: str = "all", local_files_only: bool = False):
        # Resolve alias if provided, else use the raw path
        self.model_name = MODEL_ASSET_MAP.get(model_alias_or_path, model_alias_or_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"🚀 Loading SBert model: {self.model_name} on {self.device}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device, local_files_only=local_files_only)
        except Exception as e:
            logger.warning(f"Standard load failed, attempting manual assembly for {self.model_name}: {e}")
            word_embedding_model = models.Transformer(self.model_name, model_args={'local_files_only': local_files_only})
            pooling_mode = 'cls' if 'simcse' in self.model_name.lower() else 'mean'
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)

    def create_embedding(self, text: Union[str, List[str]], mode: ModeType = "passage") -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)

class BGEEmbeddingModel(BaseEmbeddingModel):
    """Implementation for BAAI General Embedding (BGE) models with instructions."""
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", local_files_only: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device, local_files_only=local_files_only)
        self.query_instruction = "Represent this sentence for searching relevant passages: "

    def create_embedding(self, text: Union[str, List[str]], mode: ModeType = "passage") -> np.ndarray:
        if isinstance(text, str): text = [text]
        
        # Apply instruction prefix for queries
        processed_text = [self.query_instruction + t.strip() if mode == "query" else t.strip() for t in text]
        return self.model.encode(processed_text, normalize_embeddings=True, show_progress_bar=False)

class NomicEmbeddingModel(BaseEmbeddingModel):
    """Implementation for Nomic-Embed-Text using raw Transformers."""
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", local_files_only: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
        self.model.to(self.device).eval()

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def create_embedding(self, text: Union[str, List[str]], mode: ModeType = "passage") -> np.ndarray:
        if isinstance(text, str): text = [text]
        prefix = "search_query: " if mode == "query" else "search_document: "
        processed_text = [prefix + t.strip() for t in text]

        inputs = self.tokenizer(processed_text, padding=True, truncation=True, return_tensors="pt", max_length=8192).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)
        return self._to_numpy(normalized)

class DecoderOnlyEmbeddingModel(BaseEmbeddingModel):
    """General implementation for Decoder-only models (e.g., Qwen, Gemma)."""
    def __init__(self, model_name: str, prefix_map: Optional[dict] = None, local_files_only: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
        self.model.to(self.device).eval()
        self.prefix_map = prefix_map or {}

    def create_embedding(self, text: Union[str, List[str]], mode: ModeType = "passage") -> np.ndarray:
        if isinstance(text, str): text = [text]
        prefix = self.prefix_map.get(mode, "")
        processed_text = [prefix + t.strip() for t in text]

        inputs = self.tokenizer(processed_text, padding=True, truncation=True, return_tensors="pt", max_length=8192).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Masked mean pooling
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            normalized = F.normalize(pooled, p=2, dim=1)
        return self._to_numpy(normalized)

class QwenEmbeddingModel(DecoderOnlyEmbeddingModel):
    """Qwen-specific embedding model (no default prefix needed for current versions)."""
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", local_files_only: bool = False):
        super().__init__(model_name, prefix_map={}, local_files_only=local_files_only)

class GemmaEmbeddingModel(DecoderOnlyEmbeddingModel):
    """Gemma-specific embedding model with mandatory prefixes."""
    def __init__(self, model_name: str = "google/embeddinggemma-300m", local_files_only: bool = False):
        super().__init__(
            model_name, 
            prefix_map={"query": "query: ", "passage": "passage: "}, 
            local_files_only=local_files_only
        )

# Example Usage
if __name__ == "__main__":
    # Initialize model
    model_handler = SBertEmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    
    # Run inference
    sentences = ["The solar system is vast", "Jupiter is a gas giant"]
    embeddings = model_handler.create_embedding(sentences)
    
    # Calculate Similarity
    similarity = np.dot(embeddings[0], embeddings[1])
    print(f"Vector Shape: {embeddings.shape}")
    print(f"Cosine Similarity: {similarity:.4f}")