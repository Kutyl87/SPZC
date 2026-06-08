import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import ViTForImageClassification
from torchvision.datasets import Imagenette

from src.data import build_datasets


def normalize_label(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def build_imagenette_indices(id2label, wnids):
    normalized_labels = {
        index: normalize_label(label)
        for index, label in id2label.items()
    }
    indices = []
    for wnid in wnids:
        aliases = Imagenette._WNID_TO_CLASS[wnid]
        alias_indices = []
        for alias in aliases:
            normalized_alias = normalize_label(alias)
            for index, label in normalized_labels.items():
                if normalized_alias in label:
                    alias_indices.append(index)
        if not alias_indices:
            raise ValueError(f"Could not map wnid {wnid} to ImageNet class")
        indices.append(min(alias_indices))
    return indices


class TeacherService:
    def __init__(self, data_dir, model_name, dataset_name, device):
        train_dataset, eval_dataset = build_datasets(dataset_name, data_dir)
        self.datasets = {
            train_dataset.split: train_dataset,
            eval_dataset.split: eval_dataset,
        }
        self.model = ViTForImageClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.cache = {}
        self.meta = {
            "dataset_name": dataset_name,
            "num_labels": self.model.config.num_labels,
            "subset_indices": None,
        }
        if dataset_name == "imagenette":
            self.meta["subset_indices"] = build_imagenette_indices(
                self.model.config.id2label,
                train_dataset.wnids,
            )

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
        def do_GET(self):
            if self.path != "/meta":
                self.send_error(404)
                return

            body = json.dumps(service.meta).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
    parser.add_argument("--dataset", choices=["cifar100", "imagenette"], default="cifar100")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    service = TeacherService(
        data_dir=args.data_dir,
        model_name=args.model_name,
        dataset_name=args.dataset,
        device=device,
    )
    server = ThreadingHTTPServer((args.host, args.port), create_handler(service))
    server.serve_forever()


if __name__ == "__main__":
    main()
