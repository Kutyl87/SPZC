import argparse
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from src.data import get_cifar100_loaders
from src.student import SmallCNN
from src.teacher import TeacherAPI
from src.utils import distillation_loss, set_seed


def train_one_epoch(student, teacher, loader, optimizer, device, temperature, alpha):
    student.train()
    total_loss = 0.0
    total_teacher_match = 0.0

    for images, _, sample_ids in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        teacher_logits = teacher.predict_logits(list(sample_ids), device)

        student_logits = student(images)
        loss = distillation_loss(
            student_logits,
            teacher_logits,
            temperature=temperature,
            alpha=alpha,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_teacher_match += (
            (student_logits.argmax(dim=1) == teacher_logits.argmax(dim=1))
            .float()
            .mean()
            .item()
            * images.size(0)
        )

    size = len(loader.dataset)
    return total_loss / size, total_teacher_match / size


@torch.no_grad()
def evaluate(student, teacher, loader, device):
    student.eval()
    total_teacher_match = 0.0

    for images, _, sample_ids in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        teacher_logits = teacher.predict_logits(list(sample_ids), device)
        student_logits = student(images)

        total_teacher_match += (
            (student_logits.argmax(dim=1) == teacher_logits.argmax(dim=1))
            .float()
            .mean()
            .item()
            * images.size(0)
        )

    size = len(loader.dataset)
    return total_teacher_match / size


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN student with ViT teacher.")
    parser.add_argument("--data-dir", default="data", help="Directory for CIFAR-100.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for checkpoints.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--teacher-api-url", default="http://127.0.0.1:8000")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_cifar100_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    teacher = TeacherAPI(base_url=args.teacher_api_url)
    student = SmallCNN(num_classes=1000).to(device)
    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    best_match = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_match = train_one_epoch(
            student,
            teacher,
            train_loader,
            optimizer,
            device,
            args.temperature,
            args.alpha,
        )
        teacher_match = evaluate(student, teacher, test_loader, device)

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"train_teacher_match={train_match:.4f} "
            f"teacher_match={teacher_match:.4f}"
        )

        if teacher_match > best_match:
            best_match = teacher_match
            torch.save(student.state_dict(), f"{args.output_dir}/best_student.pt")


if __name__ == "__main__":
    main()
