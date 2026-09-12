"""Frozen-feature Ridge and MLP probes fit on train, selected on validation."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

RIDGE_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
MLP_SEEDS = (0, 1, 2)
METRIC_NAMES = ("vrmse", "r2", "pearson_r", "mse", "log_mse")


def r2_score(prediction, target):
    p, y = (np.asarray(v, dtype=np.float64) for v in (prediction, target))
    total = np.square(y - y.mean()).sum()
    return float(1 - np.square(p - y).sum() / total) if total > 0 else float("nan")


def metrics(prediction, target, split="test"):
    p, y = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (prediction, target))
    if (
        p.shape != y.shape
        or not len(y)
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("non-finite or misaligned predictions and targets")
    pc, yc = p - p.mean(), y - y.mean()
    denom = np.linalg.norm(pc) * np.linalg.norm(yc)
    mse = float(np.mean((p - y) ** 2))
    variance = float(np.mean(yc**2))
    values = dict(
        vrmse=math.sqrt(mse / variance) if variance > 0 else float("nan"),
        r2=1 - mse / variance if variance > 0 else float("nan"),
        pearson_r=float(pc @ yc / denom) if denom > 0 else float("nan"),
        mse=mse,
        log_mse=math.log10(mse + 1e-300),
    )
    return {f"{split}_{k}": v for k, v in values.items()} | dict(
        metric_status="constant_target"
        if variance == 0
        else "constant_prediction"
        if denom == 0
        else "ok"
    )


def _standardize_fit(features, target):
    mean = features.mean(0)
    std = features.std(0, unbiased=False).clamp_min(1e-8)
    target = target.double()
    ym, ys = target.mean(), target.std(unbiased=False).clamp_min(1e-8)
    return (features - mean) / std, ((target - ym) / ys).float(), mean, std, ym, ys


class _ProbeMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, inputs):
        return self.network(inputs).squeeze(-1)


def predict(fit, features):
    """Replay the same fitted state for validation, clean test, and noise."""
    x = torch.tensor(np.asarray(features), dtype=torch.float32)
    if fit["kind"] == "ridge":
        return (
            ((x.double() - fit["mean"]) / fit["std"]) @ fit["weights"]
            + fit["target_mean"]
        ).numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = (x.to(device) - fit["mean"].to(device)) / fit["std"].to(device)
    predictions = []
    with torch.no_grad():
        for state in fit["states"]:
            model = _ProbeMLP(x.shape[1], fit["hidden"], fit["dropout"]).to(device)
            model.load_state_dict(state)
            model.eval()
            predictions.append(
                model(x).double().cpu() * fit["target_std"] + fit["target_mean"]
            )
    return torch.stack(predictions).mean(0).numpy()


def _inputs(features, targets, valid_features, valid_targets):
    x, xv = (np.asarray(v, dtype=np.float32) for v in (features, valid_features))
    y, yv = (
        {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in m.items()}
        for m in (targets, valid_targets)
    )
    if (
        x.ndim != 3
        or xv.ndim != 3
        or x.shape[1:] != xv.shape[1:]
        or not len(x)
        or not len(xv)
    ):
        raise ValueError(
            "features must have matching nonempty (samples,layers,dimensions)"
        )
    if not y or y.keys() != yv.keys():
        raise ValueError("training and validation targets differ")
    if (
        any(
            len(a) != len(f) or not np.isfinite(a).all()
            for f, m in ((x, y), (xv, yv))
            for a in m.values()
        )
        or not np.isfinite(x).all()
        or not np.isfinite(xv).all()
    ):
        raise ValueError("non-finite or misaligned fitting data")
    return x, y, xv, yv


def _result(layers, include_depth=True):
    valid = [row for row in layers if np.isfinite(row["valid_vrmse"])]
    return dict(
        status="ok" if valid else "undefined_validation_vrmse",
        selected_layer=min(valid, key=lambda r: r["valid_vrmse"])["layer"]
        if valid
        else None,
        layers=layers,
        include_depth=include_depth,
    )


def fit_ridge_many(
    features, targets, valid_features, valid_targets, alphas=RIDGE_ALPHAS
):
    """Share a training eigendecomposition across targets and penalties per layer."""
    x, y, xv, yv = _inputs(features, targets, valid_features, valid_targets)
    if not alphas or any(a <= 0 or not np.isfinite(a) for a in alphas):
        raise ValueError("ridge penalties must be positive")
    names = list(y)
    yt = torch.tensor(np.column_stack(list(y.values())), dtype=torch.float64)
    ym = yt.mean(0)
    output = {name: [] for name in names}
    for layer in range(x.shape[1]):
        train, valid = (torch.tensor(f[:, layer], dtype=torch.float64) for f in (x, xv))
        mean, std = train.mean(0), train.std(0, unbiased=False)
        std = torch.where(std > 1e-12, std, torch.ones_like(std))
        train, valid = (train - mean) / std, (valid - mean) / std
        ev, vec = torch.linalg.eigh(train.T @ train)
        projected = vec.T @ (train.T @ (yt - ym))
        choices = {name: [] for name in names}
        for alpha in alphas:
            weights = vec @ (
                projected / (ev.clamp_min(0)[:, None] + float(alpha) * len(train))
            )
            predictions = (valid @ weights + ym).numpy()
            for i, name in enumerate(names):
                fit = dict(
                    kind="ridge",
                    mean=mean,
                    std=std,
                    target_mean=ym[i],
                    weights=weights[:, i].clone(),
                )
                choices[name].append(
                    dict(
                        layer=layer,
                        alpha=float(alpha),
                        fit=fit,
                        **metrics(predictions[:, i], yv[name], "valid"),
                    )
                )
        for name in names:
            finite = [r for r in choices[name] if np.isfinite(r["valid_vrmse"])]
            output[name].append(
                min(finite, key=lambda r: r["valid_vrmse"])
                if finite
                else choices[name][0]
            )
    return {name: _result(layers) for name, layers in output.items()}


def fit_mlp(
    features,
    target,
    valid_features,
    valid_target,
    max_steps=2000,
    min_steps=150,
    include_depth=True,
    candidate_layers=None,
    seeds=MLP_SEEDS,
):
    x, y, xv, yv = _inputs(features, {"y": target}, valid_features, {"y": valid_target})
    if not 1 <= min_steps <= max_steps or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("invalid MLP stopping limits or seeds")
    if np.var(yv["y"]) == 0:
        return _result([], include_depth)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.tensor(y["y"], dtype=torch.float64, device=device)
    layers = []
    for layer in range(x.shape[1]) if candidate_layers is None else candidate_layers:
        train = torch.tensor(x[:, layer], device=device)
        train, yn, mean, std, ym, ys = _standardize_fit(train, y)
        valid = (torch.tensor(xv[:, layer], device=device) - mean) / std
        target_valid = ((torch.tensor(yv["y"], device=device) - ym) / ys).float()
        states, steps = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            model = _ProbeMLP(x.shape[2]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            best, best_step, best_state = float("inf"), 0, None
            for step in range(1, max_steps + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                F.mse_loss(model(train), yn).backward()
                optimizer.step()
                if step % 20 == 0 or step == max_steps:
                    model.eval()
                    with torch.no_grad():
                        loss = float(F.mse_loss(model(valid), target_valid))
                    if step >= min_steps and loss < best:
                        best, best_step = loss, step
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                    elif step >= min_steps and step - best_step >= 100:
                        break
            if best_state is None:
                raise ValueError("MLP produced no finite validation checkpoint")
            states.append(best_state)
            steps.append(best_step)
        fit = dict(
            kind="mlp",
            states=states,
            mean=mean.cpu(),
            std=std.cpu(),
            target_mean=ym.cpu(),
            target_std=ys.cpu(),
            hidden=128,
            dropout=0.1,
        )
        layers.append(
            dict(
                layer=layer,
                selected_steps=steps,
                fit=fit,
                **metrics(predict(fit, xv[:, layer]), yv["y"], "valid"),
            )
        )
    return _result(layers, include_depth)


def selected_family(ridge, mlp):
    candidates = [
        r
        for r in (ridge, mlp)
        if r.get("valid_vrmse") is not None and np.isfinite(r["valid_vrmse"])
    ]
    return min(candidates, key=lambda r: r["valid_vrmse"]) if candidates else None
