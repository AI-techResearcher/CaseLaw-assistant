from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class JinaEmbeddingWrapper:
    def __init__(self, model_name="jinaai/jina-embeddings-v3", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Loading Jina embedding model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
        print("✅ Jina model loaded.")

    def embed_query(self, query: str) -> list[float]:
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        normalized = F.normalize(embeddings, p=2, dim=1)
        return normalized[0].cpu().tolist()