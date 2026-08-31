import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class LogisticProber:
    def __init__(self):
        self.scaler = StandardScaler()
        # `penalty="l2"` is the default and was deprecated in scikit-learn 1.8;
        # naming it explicitly stops working in 1.10.
        self.clf = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=42
        )

    def fit_and_score(self, X_train, y_train, X_test, y_test):
        if len(y_train) < 10 or len(y_test) < 10:
            return 0.0

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.clf.fit(X_train_s, y_train)
        y_pred = self.clf.predict(X_test_s)

        return f1_score(y_test, y_pred, average="macro")


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
