"""Validation-selected Ridge and MLP probes with reusable fitted state."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.evaluation.protocol import trajectory_folds

RIDGE_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def r2_score(prediction, target):
    p, y = (
        torch.tensor(np.asarray(prediction), dtype=torch.float64),
        torch.tensor(np.asarray(target), dtype=torch.float64),
    )
    total = (y - y.mean()).square().sum()
    return float(1 - (p - y).square().sum() / total) if total > 0 else float("nan")


def metrics(prediction, target):
    p, y = (
        np.asarray(prediction, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
    )
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("non-finite or misaligned predictions and targets")
    pc, yc = p - p.mean(), y - y.mean()
    denom = np.linalg.norm(pc) * np.linalg.norm(yc)
    mse = float(np.mean((p - y) ** 2))
    return dict(
        test_r2=r2_score(p, y),
        test_pearson_r=float(pc @ yc / denom) if denom > 0 else float("nan"),
        test_mse=mse,
        test_log_mse=math.log10(mse + 1e-300),
        metric_status="constant_target"
        if np.dot(yc, yc) == 0
        else "constant_prediction"
        if denom == 0
        else "ok",
    )


def _standardize_fit(features, target):
    mean = features.mean(0)
    std = features.std(0, unbiased=False).clamp_min(1e-8)
    # Preserve small physical variations before casting normalized MLP targets.
    target = target.double()
    ym, ys = target.mean(), target.std(unbiased=False).clamp_min(1e-8)
    return (features - mean) / std, ((target - ym) / ys).float(), mean, std, ym, ys


class _ProbeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def _select_mlp_steps_and_score(
    features: torch.Tensor,
    target: torch.Tensor,
    fold_ids: np.ndarray,
    *,
    seed: int,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    max_steps: int,
    min_steps: int,
    patience: int,
    check_every: int,
) -> dict:
    """Select stopping time on one validation fold and retain its R2."""
    if min_steps > max_steps:
        raise ValueError("min_steps cannot exceed max_steps")
    monitor = torch.as_tensor(fold_ids == (seed % 5), device=features.device)
    train = ~monitor
    if monitor.sum() == 0 or train.sum() == 0:
        raise ValueError("MLP inner split is empty")
    x_train, y_train, mean, std, target_mean, target_std = _standardize_fit(
        features[train], target[train]
    )
    x_monitor = (features[monitor] - mean) / std
    y_monitor = ((target[monitor].double() - target_mean) / target_std).float()
    torch.manual_seed(seed)
    model = _ProbeMLP(features.shape[1], hidden, dropout).to(features.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    best_r2 = float("-inf")
    best_step = min_steps
    last_improvement = 0
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        if step % check_every == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                monitor_prediction = model(x_monitor)
                monitor_loss = float(F.mse_loss(monitor_prediction, y_monitor))
            if step >= min_steps and monitor_loss < best_loss:
                best_loss = monitor_loss
                best_r2 = r2_score(
                    monitor_prediction.detach().cpu(), y_monitor.detach().cpu()
                )
                best_step = step
                last_improvement = step
            elif step >= min_steps and step - last_improvement >= patience:
                break
    return {
        "selected_steps": int(best_step),
        "valid_r2": float(best_r2),
        "monitor_fold": int(seed % 5),
    }


def _ridge_fit(features, targets, alpha):
    x, y = torch.as_tensor(features).double(), torch.as_tensor(targets).double()
    mean, std = x.mean(0), x.std(0, unbiased=False)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    x = (x - mean) / std
    ym = y.mean(0)
    gram = x.T @ x + float(alpha) * len(x) * torch.eye(x.shape[1], dtype=x.dtype)
    return dict(
        kind="ridge",
        mean=mean,
        std=std,
        target_mean=ym,
        weights=torch.linalg.solve(gram, x.T @ (y - ym)),
    )


def predict(fit, features):
    """Identical numerical path for clean scoring and later noise evaluation."""
    x = torch.tensor(np.asarray(features), dtype=torch.float32)
    if fit["kind"] == "ridge":
        return (
            ((x.double() - fit["mean"]) / fit["std"]) @ fit["weights"]
            + fit["target_mean"]
        ).numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    x = (x - fit["mean"].to(device)) / fit["std"].to(device)
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


def _check_inputs(x, targets, samples):
    x = np.asarray(x, dtype=np.float32)
    y = {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in targets.items()}
    if x.ndim != 3 or len(x) != len(samples) or not np.isfinite(x).all():
        raise ValueError("features must be finite (samples,layers,dimensions)")
    if not y or any(len(v) != len(x) or not np.isfinite(v).all() for v in y.values()):
        raise ValueError("targets and features must align and be finite")
    return x, y


def fit_ridge_many(features, targets, samples, alphas=RIDGE_ALPHAS):
    """Share layer/fold eigendecompositions across all physical targets.

    Only validation arrays enter this function. Its output contains a fitted
    probe at each layer, so depth and clean-fit noise require no additional fit.
    """
    x, target_map = _check_inputs(features, targets, samples)
    names = list(target_map)
    y = torch.tensor(np.column_stack(list(target_map.values())), dtype=torch.float64)
    x = torch.tensor(x, dtype=torch.float64)
    folds = trajectory_folds(samples)
    if not alphas or any(a <= 0 or not np.isfinite(a) for a in alphas):
        raise ValueError("ridge penalties must be positive")
    cv = np.full((x.shape[1], len(alphas), len(names), len(folds)), np.nan)
    for layer in range(x.shape[1]):
        for f, fold in enumerate(folds):
            train, held = x[fold["fit"], layer], x[fold["select"], layer]
            mean, std = train.mean(0), train.std(0, unbiased=False)
            std = torch.where(std > 1e-12, std, torch.ones_like(std))
            train, held = (train - mean) / std, (held - mean) / std
            ev, vec = torch.linalg.eigh(train.T @ train)
            ev = ev.clamp_min(0)
            yf, yh = y[fold["fit"]], y[fold["select"]]
            ym = yf.mean(0)
            projected = vec.T @ (train.T @ (yf - ym))
            total = (yh - yh.mean(0)).square().sum(0)
            for a, alpha in enumerate(alphas):
                pred = (
                    held
                    @ (vec @ (projected / (ev[:, None] + float(alpha) * len(train))))
                    + ym
                )
                score = torch.where(
                    total > 0, 1 - (pred - yh).square().sum(0) / total, torch.nan
                )
                cv[layer, a, :, f] = score.numpy()
    valid = np.isfinite(cv)
    counts = valid.sum(axis=-1)
    scores = np.divide(
        np.where(valid, cv, 0).sum(axis=-1),
        counts,
        out=np.full(cv.shape[:-1], np.nan),
        where=counts > 0,
    )
    output = {}
    for i, name in enumerate(names):
        if not np.isfinite(scores[:, :, i]).any():
            output[name] = dict(
                status="undefined_validation_r2", selected_layer=None, layers=[]
            )
            continue
        layers = []
        for layer in range(x.shape[1]):
            a = int(np.nanargmax(scores[layer, :, i]))
            fit = _ridge_fit(x[:, layer], y[:, i], alphas[a])
            layers.append(
                dict(
                    layer=layer,
                    alpha=float(alphas[a]),
                    valid_cv_r2=float(scores[layer, a, i]),
                    fit=fit,
                )
            )
        selected = max(layers, key=lambda row: row["valid_cv_r2"])
        output[name] = dict(
            status="ok", selected_layer=selected["layer"], layers=layers
        )
    return output


def fit_mlp(
    features,
    target,
    samples,
    *,
    max_steps=2000,
    min_steps=150,
    include_depth=True,
    candidate_layers=None,
):
    x, targets = _check_inputs(features, {"y": target}, samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, device=device)
    y = torch.tensor(targets["y"], dtype=torch.float64, device=device)
    folds = trajectory_folds(samples)
    ids = np.empty(len(samples), dtype=int)
    for f, fold in enumerate(folds):
        ids[fold["select"]] = f
    validation_curve = []
    for layer in range(x.shape[1]) if candidate_layers is None else candidate_layers:
        selected = [
            _select_mlp_steps_and_score(
                x[:, layer],
                y,
                ids,
                seed=seed,
                hidden=128,
                dropout=0.1,
                lr=0.01,
                weight_decay=1e-4,
                max_steps=max_steps,
                min_steps=min_steps,
                patience=100,
                check_every=20,
            )
            for seed in range(5)
        ]
        scores = np.asarray([r["valid_r2"] for r in selected])
        if np.isfinite(scores).any():
            validation_curve.append(
                dict(
                    layer=layer,
                    valid_cv_r2=float(scores[np.isfinite(scores)].mean()),
                    valid_fold_r2=scores.tolist(),
                    selected_steps=[r["selected_steps"] for r in selected],
                )
            )
    if not validation_curve:
        return dict(status="undefined_validation_r2", selected_layer=None, layers=[])
    best = max(validation_curve, key=lambda row: row["valid_cv_r2"])
    layers = []
    for selection in validation_curve if include_depth else [best]:
        layer = selection["layer"]
        train, target_norm, mean, std, ym, ys = _standardize_fit(x[:, layer], y)
        states = []
        for seed, steps in enumerate(selection["selected_steps"]):
            torch.manual_seed(seed)
            model = _ProbeMLP(x.shape[2]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            for _ in range(steps):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                F.mse_loss(model(train), target_norm).backward()
                optimizer.step()
            states.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
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
        layers.append({**selection, "fit": fit})
    return dict(
        status="ok",
        selected_layer=best["layer"],
        layers=layers,
        validation_curve=validation_curve,
        include_depth=include_depth,
    )


def selected_family(ridge, mlp):
    candidates = [
        r for r in (ridge, mlp) if np.isfinite(r.get("valid_cv_r2", float("nan")))
    ]
    # Input order makes exact ties prefer Ridge.
    return max(candidates, key=lambda r: r["valid_cv_r2"]) if candidates else None
