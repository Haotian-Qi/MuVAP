import torch


def ava_active_speaker_map(scores, labels):
    scores = torch.as_tensor(scores, dtype=torch.float64).flatten()
    labels = torch.as_tensor(labels, device=scores.device).bool().flatten()
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    if not torch.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinite values")

    positives = labels.sum()
    if not len(labels) or positives == 0:
        return 0.0

    order = torch.argsort(scores, descending=True, stable=True)
    true_positives = labels[order].cumsum(0)
    precision = true_positives / torch.arange(1, len(labels) + 1, dtype=torch.float64, device=scores.device)
    recall = true_positives / positives
    precision = torch.cat([precision.new_zeros(1), precision, precision.new_zeros(1)])
    recall = torch.cat([recall.new_zeros(1), recall, recall.new_ones(1)])
    precision = torch.flip(torch.cummax(torch.flip(precision, dims=[0]), dim=0).values, dims=[0])
    indices = torch.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = ((recall[indices] - recall[indices - 1]) * precision[indices]).sum()
    return 100 * average_precision.item()
