import torch


def micro_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes standard top-1 (micro) accuracy.
    preds: Tensor of shape (N,) with predicted class indices
    labels: Tensor of shape (N,) with ground-truth class indices

    Returns:
        accuracy float in [0,1]
    """
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total


def balanced_accuracy(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """
    Computes macro (balanced) accuracy: average recall over all classes.
    Classes with zero examples are skipped.

    Returns:
        balanced accuracy float in [0,1]
    """
    recalls = []
    for cls in range(num_classes):
        mask = (labels == cls)
        count = mask.sum().item()
        if count == 0:
            continue
        cls_correct = (preds[mask] == cls).sum().item()
        recalls.append(cls_correct / count)
    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


def per_class_recall(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    """
    Computes recall for each class.

    Returns:
        dict mapping class index -> recall float (0-1), or None if no samples
    """
    recalls = {}
    for cls in range(num_classes):
        mask = (labels == cls)
        count = mask.sum().item()
        if count == 0:
            recalls[cls] = None
        else:
            cls_correct = (preds[mask] == cls).sum().item()
            recalls[cls] = cls_correct / count
    return recalls
