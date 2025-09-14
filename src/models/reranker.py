from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class BGEReranker:
    def __init__(self, model_name: str, device: str = "mps", max_length: int = 384):
        self.device = torch.device(
            "mps"
            if device == "mps" and torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()
        self.max_len = max_length

    @torch.no_grad()
    def score(self, query: str, passages: List[str], batch_size: int = 32) -> np.ndarray:
        scores = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            inputs = self.tok(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs).logits.view(-1)
            scores.append(logits.detach().cpu().numpy())
        return np.concatenate(scores, axis=0)


