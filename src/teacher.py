import requests
import torch


class TeacherAPI:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self._meta = None

    def get_meta(self):
        if self._meta is None:
            response = requests.get(f"{self.base_url}/meta", timeout=30)
            response.raise_for_status()
            self._meta = response.json()
        return self._meta

    def predict_logits(self, sample_ids, device):
        response = requests.post(
            f"{self.base_url}/predict",
            json={"sample_ids": sample_ids},
            timeout=300,
        )
        response.raise_for_status()
        logits = response.json()["logits"]
        return torch.tensor(logits, dtype=torch.float32, device=device)
