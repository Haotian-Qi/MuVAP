import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticProber:
    def __init__(self):
        self.scaler = StandardScaler()
        # `penalty="l2"` is the default and was deprecated in scikit-learn 1.8;
        # naming it explicitly stops working in 1.10.
        self.clf = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=42
        )

    def fit_predict(self, X_train, y_train, X_test):
        """Predictions on the test events, or None if either pool is too small.

        Returning the predictions rather than one score lets the caller report
        the probe with the same metrics as the zero-shot readout.
        """
        if len(y_train) < 10 or len(X_test) < 10:
            return None

        self.clf.fit(self.scaler.fit_transform(X_train), y_train)
        return self.clf.predict(self.scaler.transform(X_test))


def get_param_groups(named_parameters, weight_decay):
    """Decay matrices only.

    Norm gains, biases, and embedding tables are 1-D; shrinking them toward
    zero costs capacity without the regularizing effect it has on weight
    matrices. Frozen parameters are dropped so they never reach the optimizer.
    """
    decay, no_decay = [], []
    for _, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def get_adamw_optimizer(params, lr, weight_decay):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def get_cosine_warmup_scheduler(optimizer, total_steps, warmup_ratio=0.1, min_lr=None):
    total_steps = max(1, int(total_steps))
    warmup_steps = min(int(total_steps * warmup_ratio), total_steps - 1)
    decay_steps = max(1, total_steps - warmup_steps)

    max_lr = optimizer.param_groups[0]["lr"]
    min_lr = max_lr * 0.01 if min_lr is None else min_lr

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=decay_steps, eta_min=min_lr
    )

    if warmup_steps == 0:
        return cosine

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


def get_optimizer_cfg(optimizer, scheduler=None):
    cfg = {"optimizer": optimizer}
    if scheduler:
        cfg["lr_scheduler"] = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
    return cfg
