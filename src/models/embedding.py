from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


class BGEEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str = "mps",
        max_length: int = 384,
        normalize: bool = True,
        query_instruction: str = "",
    ):
        self.device = torch.device(
            "mps"
            if device == "mps" and torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_len = max_length
        self.normalize = normalize
        self.query_instruction = query_instruction

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            inputs = self.tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**inputs)
            # mean pool
            last_hidden = out.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked = last_hidden * attention_mask
            sum_emb = masked.sum(dim=1)
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            mean_emb = sum_emb / lengths
            if self.normalize:
                mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)
            embs.append(mean_emb.cpu().numpy())
        return np.vstack(embs)

    def encode_passages(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.encode_texts(texts, batch_size)

    def encode_queries(self, queries: List[str], batch_size: int = 64) -> np.ndarray:
        if self.query_instruction:
            queries = [self.query_instruction + q for q in queries]
        return self.encode_texts(queries, batch_size)


