import random

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distillation_loss(student_logits, teacher_logits, temperature=2.0, alpha=1.0):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return alpha * soft_loss


def accuracy_from_logits(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def accuracy_from_subset_logits(logits, labels, class_indices):
    subset_logits = logits[:, class_indices]
    return (subset_logits.argmax(dim=1) == labels).float().mean().item()
