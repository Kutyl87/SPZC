import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import ViTForImageClassification

from src.data import CIFAR100WithIds, build_transforms


class TeacherService:
    def __init__(self, data_dir, model_name, device):
        transform = build_transforms()
        self.datasets = {
            "train": CIFAR100WithIds(
                split="train",
                root=data_dir,
                download=True,
                transform=transform,
            ),
            "test": CIFAR100WithIds(
                split="test",
                root=data_dir,
                download=True,
                transform=transform,
            ),
        }
        self.model = ViTForImageClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.cache = {}

    def predict(self, sample_ids):
        missing = [sample_id for sample_id in sample_ids if sample_id not in self.cache]
        if missing:
            images = []
            for sample_id in missing:
                split, index = sample_id.split(":")
                image, _, _ = self.datasets[split][int(index)]
                images.append(image)
            batch = torch.stack(images).to(self.device)
            with torch.no_grad():
                logits = self.model(pixel_values=batch).logits.cpu().tolist()
            for sample_id, sample_logits in zip(missing, logits):
                self.cache[sample_id] = sample_logits
        return [self.cache[sample_id] for sample_id in sample_ids]


def create_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(content_length)
            sample_ids = json.loads(payload)["sample_ids"]
            logits = service.predict(sample_ids)
            body = json.dumps({"logits": logits}).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    return Handler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-name", default="google/vit-base-patch16-224")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    service = TeacherService(
        data_dir=args.data_dir,
        model_name=args.model_name,
        device=device,
    )
    server = ThreadingHTTPServer((args.host, args.port), create_handler(service))
    server.serve_forever()


if __name__ == "__main__":
    main()
