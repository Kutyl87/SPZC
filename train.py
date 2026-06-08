import argparse
import json
import datetime
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm
import os
import numpy as np

from torch.utils.data import Subset, DataLoader
from scipy.stats import entropy
import torch.nn.functional as F
from src.data import get_data_loaders, get_data_loaders_with_split
from src.student import build_student
from src.teacher import TeacherAPI
from src.utils import accuracy_from_subset_logits, distillation_loss, set_seed, hard_label_loss


def train_one_epoch(student, teacher, loader, optimizer, device, temperature, alpha, report_acc, class_indices, loss_type="distillation"):
    student.train()
    total_loss = 0.0
    total_teacher_match = 0.0
    total_student_accuracy = 0.0
    total_teacher_accuracy = 0.0

    for images, labels, sample_ids in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        teacher_logits = teacher.predict_logits(list(sample_ids), device)

        student_logits = student(images)
        loss = F.cross_entropy(student_logits, labels)

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
        if report_acc and class_indices is not None:
            total_student_accuracy += accuracy_from_subset_logits(student_logits, labels, class_indices) * images.size(0)
            total_teacher_accuracy += accuracy_from_subset_logits(teacher_logits, labels, class_indices) * images.size(0)
    
    size = len(loader.dataset)
    metrics = {
        "loss": total_loss / size,
        "teacher_match": total_teacher_match / size,
    }
    if report_acc and class_indices is not None:
        metrics["student_accuracy"] = total_student_accuracy / size
        metrics["teacher_accuracy"] = total_teacher_accuracy / size
    return metrics

def steal_one_epoch(student, teacher, loader, optimizer, device, temperature, alpha, report_acc, class_indices, loss_type="distillation"):
    student.train()
    total_loss = 0.0
    total_teacher_match = 0.0
    total_student_accuracy = 0.0
    total_teacher_accuracy = 0.0

    for images, labels, sample_ids in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        teacher_logits = teacher.predict_logits(list(sample_ids), device)

        student_logits = student(images)
        loss = 0
        if loss_type == "distillation":
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                temperature=temperature,
                alpha=alpha,
            )
        elif loss_type == "hard_label":
            loss = hard_label_loss(
                student_logits,
                teacher_logits
            )
        else:
            raise KeyError("Loss label not found.")
        

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
        if report_acc and class_indices is not None:
            total_student_accuracy += accuracy_from_subset_logits(student_logits, labels, class_indices) * images.size(0)
            total_teacher_accuracy += accuracy_from_subset_logits(teacher_logits, labels, class_indices) * images.size(0)

    size = len(loader.dataset)
    metrics = {
        "loss": total_loss / size,
        "teacher_match": total_teacher_match / size,
    }
    if report_acc and class_indices is not None:
        metrics["student_accuracy"] = total_student_accuracy / size
        metrics["teacher_accuracy"] = total_teacher_accuracy / size
    return metrics


def get_least_confident_loader(student, unlabelled_loader, device, budget=1000):
    student.eval()
    all_entropies = []
    
    with torch.no_grad():
        for images, labels, sample_ids in tqdm(unlabelled_loader, desc="check_entropy", leave=False):
            images = images.to(device)
            student_logits = student(images)
            
            probs = torch.softmax(student_logits, dim=1)
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
            
            all_entropies.extend(entropy.cpu().numpy())

    all_entropies = np.array(all_entropies)
    highest_ent_indexes = np.argsort(all_entropies)[::-1]
    
    selected_indices = highest_ent_indexes
    # selected_indices = highest_ent_indexes[:budget]

    original_dataset = unlabelled_loader.dataset
    hard_subset = Subset(original_dataset, selected_indices)
    

    # new dataset with 
    subset_loader = DataLoader(
        hard_subset, 
        batch_size=unlabelled_loader.batch_size, 
        shuffle=True, 
        num_workers=unlabelled_loader.num_workers
    )

    return subset_loader


@torch.no_grad()
def evaluate(student, teacher, loader, device, report_acc, class_indices):
    student.eval()
    total_teacher_match = 0.0
    total_student_accuracy = 0.0
    total_teacher_accuracy = 0.0

    for images, labels, sample_ids in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        teacher_logits = teacher.predict_logits(list(sample_ids), device)
        student_logits = student(images)

        total_teacher_match += (
            (student_logits.argmax(dim=1) == teacher_logits.argmax(dim=1))
            .float()
            .mean()
            .item()
            * images.size(0)
        )
        if report_acc and class_indices is not None:
            total_student_accuracy += accuracy_from_subset_logits(student_logits, labels, class_indices) * images.size(0)
            total_teacher_accuracy += accuracy_from_subset_logits(teacher_logits, labels, class_indices) * images.size(0)

    size = len(loader.dataset)
    metrics = {
        "teacher_match": total_teacher_match / size,
    }
    if report_acc and class_indices is not None:
        metrics["student_accuracy"] = total_student_accuracy / size
        metrics["teacher_accuracy"] = total_teacher_accuracy / size
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN student with ViT teacher.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", choices=["cifar100", "imagenette"], default="cifar100")
    parser.add_argument("--output-dir", default="outputs", help="Directory for checkpoints and results.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--model", choices=["small_cnn", "resnet18"], default="small_cnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--teacher-api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--report-accuracies", action="store_true")
    parser.add_argument("--run-label", default=None, help="Optional label for the results file.")
    parser.add_argument("--loss-type", choices=["distillation", "hard_label"], default="distillation")
    parser.add_argument("--least-confident", action="store_true", help="It will should be run at the resnet18 model with Imagenette dataset, will train on 4/5 train dataset and 1/5 with least entropy will use to steal from api with destilation loss.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    steal_loader, test_loader = get_data_loaders(
    dataset_name=args.dataset,
    data_dir=args.data_dir,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    )

    if args.least_confident:
        train_loader, unlabelled_loader, test_loader = get_data_loaders_with_split(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_ratios=[4, 1]
        )

    teacher = TeacherAPI(base_url=args.teacher_api_url)
    teacher_meta = teacher.get_meta()
    if teacher_meta["dataset_name"] != args.dataset:
        raise ValueError(
            f"Dataset mismatch: train.py uses {args.dataset}, teacher_api uses {teacher_meta['dataset_name']}"
        )
    class_indices = teacher_meta["subset_indices"]
    student = build_student(args.model, num_classes=teacher_meta["num_labels"]).to(device)
    optimizer = optim.Adam(student.parameters(), lr=args.lr)


    results = {"train" : [], "steal" : []}
    path = Path(f"{args.output_dir}/{args.loss_type}")
    path.mkdir(parents=True, exist_ok=True)
    best_match = 0.0
    #train model first on normal data 
    if args.least_confident:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                student,
                teacher,
                train_loader,
                optimizer,
                device,
                args.temperature,
                args.alpha,
                args.report_accuracies,
                class_indices,
                args.loss_type
            )
            eval_metrics = evaluate(
                student,
                teacher,
                test_loader,
                device,
                args.report_accuracies,
                class_indices,
            )

            epoch_entry = {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_teacher_match": round(train_metrics["teacher_match"], 6),
                "teacher_match": round(eval_metrics["teacher_match"], 6),
            }
            if args.report_accuracies and class_indices is not None:
                epoch_entry["train_teacher_acc"] = round(train_metrics["teacher_accuracy"], 6)
                epoch_entry["train_student_acc"] = round(train_metrics["student_accuracy"], 6)
                epoch_entry["val_teacher_acc"] = round(eval_metrics["teacher_accuracy"], 6)
                epoch_entry["val_student_acc"] = round(eval_metrics["student_accuracy"], 6)
            results["train"].append(epoch_entry)

            message = (
                f"epoch={epoch} "
                f"dataset={args.dataset} "
                f"model={args.model} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_teacher_match={train_metrics['teacher_match']:.4f} "
                f"teacher_match={eval_metrics['teacher_match']:.4f}"
            )
            if args.report_accuracies and class_indices is not None:
                message += (
                    f" train_teacher_acc={train_metrics['teacher_accuracy']:.4f} "
                    f"train_student_acc={train_metrics['student_accuracy']:.4f} "
                    f"val_teacher_acc={eval_metrics['teacher_accuracy']:.4f} "
                    f"val_student_acc={eval_metrics['student_accuracy']:.4f}"
                )
            elif args.report_accuracies:
                message += " label_acc=unsupported_for_dataset"
            print(message)

            if eval_metrics["student_accuracy"] > best_match:
                best_match = eval_metrics["student_accuracy"]
                path = Path(f"{args.output_dir}/least_confident")
                path.mkdir(parents=True, exist_ok=True)
                torch.save(student.state_dict(), path / f"best_{args.dataset}_{args.model}.pt")
        steal_loader = get_least_confident_loader(student, unlabelled_loader, device, budget=None)


    total_queries = len(steal_loader.dataset) * args.epochs
    best_match = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = steal_one_epoch(
            student,
            teacher,
            steal_loader,
            optimizer,
            device,
            args.temperature,
            args.alpha,
            args.report_accuracies,
            class_indices,
            args.loss_type
        )
        eval_metrics = evaluate(
            student,
            teacher,
            test_loader,
            device,
            args.report_accuracies,
            class_indices,
        )

        epoch_entry = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_teacher_match": round(train_metrics["teacher_match"], 6),
            "teacher_match": round(eval_metrics["teacher_match"], 6),
        }
        if args.report_accuracies and class_indices is not None:
            epoch_entry["train_teacher_acc"] = round(train_metrics["teacher_accuracy"], 6)
            epoch_entry["train_student_acc"] = round(train_metrics["student_accuracy"], 6)
            epoch_entry["val_teacher_acc"] = round(eval_metrics["teacher_accuracy"], 6)
            epoch_entry["val_student_acc"] = round(eval_metrics["student_accuracy"], 6)
        results["steal"].append(epoch_entry)

        message = (
            f"epoch={epoch} "
            f"dataset={args.dataset} "
            f"model={args.model} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_teacher_match={train_metrics['teacher_match']:.4f} "
            f"teacher_match={eval_metrics['teacher_match']:.4f}"
        )
        if args.report_accuracies and class_indices is not None:
            message += (
                f" train_teacher_acc={train_metrics['teacher_accuracy']:.4f} "
                f"train_student_acc={train_metrics['student_accuracy']:.4f} "
                f"val_teacher_acc={eval_metrics['teacher_accuracy']:.4f} "
                f"val_student_acc={eval_metrics['student_accuracy']:.4f}"
            )
        elif args.report_accuracies:
            message += " label_acc=unsupported_for_dataset"
        print(message)

        if eval_metrics["teacher_match"] > best_match:
            best_match = eval_metrics["teacher_match"]
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(student.state_dict(), path / f"best_{args.dataset}_{args.model}.pt")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.run_label or f"{args.dataset}_{args.model}"
    results_file = path / f"results_{label}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({
            "metadata": {
                "dataset": args.dataset,
                "model": args.model,
                "epochs": args.epochs,
                "temperature": args.temperature,
                "alpha": args.alpha,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "teacher_api_url": args.teacher_api_url,
                "total_queries": total_queries,
                "report_accuracies": args.report_accuracies,
            },
            "metrics": results,
        }, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
