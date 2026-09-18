"""Frozen-feature probes: Ridge, attentive heads, and metadata-only MLPs.

Every probe is fit on train, has its stopping state and regularization chosen
on validation, and is scored once on test. Attentive probes read the complete
frozen token sequence produced from the eight input context frames; no future
frame, target clip, or target value ever enters a probe input.

Each public fit function covers exactly one encoder output so that callers can
stage a single float16 token shard at a time instead of holding every layer.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.vit import sincos_3d

RIDGE_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
MLP_SEEDS = (0,)
METRIC_NAMES = ("vrmse", "r2", "pearson_r", "mse", "log_mse")
ATTENTIVE = dict(
    heads=8,
    ffn_hidden=96,
    blocks=1,
    dropout=0.0,
    lr=1e-3,
    weight_decay=0.01,
    warmup_epochs=2,
    seed=0,
)
MLP_HIDDEN = 128
MLP_DROPOUT = 0.1


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def scored(predictions, targets, target_std, split="valid", joint=False):
    """Metrics in each target's own units plus its train-standardized MSE.

    The standardized MSE is the metric the cited governing-parameter experiment
    averages over its two jointly predicted outputs, so it is recorded for
    every probe and is the selection criterion of the joint fits.
    """
    per_output = {}
    for name, prediction in predictions.items():
        p = np.asarray(prediction, dtype=np.float64).reshape(-1)
        y = np.asarray(targets[name], dtype=np.float64).reshape(-1)
        per_output[name] = metrics(p, y, split) | {
            f"{split}_normalized_mse": float(
                np.mean(((p - y) / float(target_std[name])) ** 2)
            )
        }
    if not joint:
        (only,) = per_output.values()
        return only
    return {
        "outputs": per_output,
        f"{split}_normalized_mse": float(
            np.mean([v[f"{split}_normalized_mse"] for v in per_output.values()])
        ),
    }


def select_layers(entries, include_depth=True, criterion="valid_vrmse"):
    """Freeze the encoder output with the best validation score."""
    entries = list(entries)
    usable = [e for e in entries if np.isfinite(e.get(criterion, float("nan")))]
    return dict(
        status="ok" if usable else "undefined_validation_vrmse",
        selected_layer=min(usable, key=lambda e: e[criterion])["layer"]
        if usable
        else None,
        layers=entries,
        include_depth=include_depth,
        criterion=criterion,
    )


def _layer_data(train, targets, valid, valid_targets):
    x, xv = (np.asarray(v, dtype=np.float32) for v in (train, valid))
    y, yv = (
        {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in m.items()}
        for m in (targets, valid_targets)
    )
    if (
        x.ndim != 2
        or xv.ndim != 2
        or x.shape[1] != xv.shape[1]
        or not len(x)
        or not len(xv)
    ):
        raise ValueError("layer features must be nonempty, matching (samples,features)")
    if not y or y.keys() != yv.keys():
        raise ValueError("training and validation targets differ")
    if not np.isfinite(x).all() or not np.isfinite(xv).all():
        raise ValueError("non-finite layer features")
    for features, mapping in ((x, y), (xv, yv)):
        for values in mapping.values():
            if len(values) != len(features) or not np.isfinite(values).all():
                raise ValueError("non-finite or misaligned fitting targets")
    return x, y, xv, yv


def _standardize(values):
    mean = values.mean(0)
    return mean, values.std(0, unbiased=False).clamp_min(1e-12)


class _ProbeMLP(nn.Module):
    def __init__(self, in_dim, hidden=MLP_HIDDEN, dropout=MLP_DROPOUT):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, inputs):
        return self.network(inputs).squeeze(-1)


class AttentiveProbe(nn.Module):
    """One cross-attention block over the complete frozen context sequence.

    The global form owns a single learned query. Because that query is shared
    across the batch, the key and value projections commute with the attention
    contraction: the key bias is constant over tokens and cancels in the
    softmax, and the value projection is linear in the attention-weighted token
    mean. The factorized form below computes exactly the same function as
    standard single-query cross-attention while never multiplying the 384x384
    projections against every context token, which is what makes a full depth
    sweep affordable.

    The local form keeps the ordinary formulation: its queries are the sampled
    frozen tokens plus a fixed spatiotemporal coordinate encoding, they share
    one projected key/value sequence, and they never attend to one another.
    """

    def __init__(
        self,
        dim=384,
        heads=ATTENTIVE["heads"],
        ffn_hidden=ATTENTIVE["ffn_hidden"],
        outputs=1,
        local=False,
    ):
        super().__init__()
        if dim % heads or outputs < 1:
            raise ValueError("attention heads must divide the token dimension")
        self.dim, self.heads, self.local = dim, heads, local
        self.norm_context = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm_summary = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden), nn.GELU(), nn.Linear(ffn_hidden, dim)
        )
        self.head = nn.Linear(dim, outputs)
        self.query = None if local else nn.Parameter(torch.zeros(1, 1, dim))
        self.apply(self._init_weights)
        if self.query is not None:
            nn.init.trunc_normal_(self.query, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def _pool(self, context, queries):
        head_dim = self.dim // self.heads
        x = self.norm_context(context)
        if self.local:
            batch, tokens, _ = x.shape
            q = (
                self.q_proj(queries)
                .reshape(batch, -1, self.heads, head_dim)
                .transpose(1, 2)
            )
            k, v = (
                projection(x)
                .reshape(batch, tokens, self.heads, head_dim)
                .transpose(1, 2)
                for projection in (self.k_proj, self.v_proj)
            )
            pooled = (
                F.scaled_dot_product_attention(q, k, v)
                .transpose(1, 2)
                .reshape(batch, -1, self.dim)
            )
            return queries, pooled
        query = self.q_proj(self.query).reshape(self.heads, head_dim)
        keys = torch.einsum(
            "hj,hjd->hd", query, self.k_proj.weight.reshape(self.heads, head_dim, -1)
        )
        weights = torch.einsum("bnd,hd->bhn", x, keys).mul(head_dim**-0.5).softmax(-1)
        summary = torch.einsum("bhn,bnd->bhd", weights, x)
        pooled = torch.einsum(
            "hjd,bhd->bhj",
            self.v_proj.weight.reshape(self.heads, head_dim, -1),
            summary,
        ) + self.v_proj.bias.reshape(self.heads, head_dim)
        return self.query.expand(x.shape[0], -1, -1), pooled.reshape(
            x.shape[0], 1, self.dim
        )

    def forward(self, context, queries=None):
        """(batch, queries, outputs); a global probe has exactly one query."""
        base, pooled = self._pool(context, queries)
        summary = base + self.proj(pooled)
        summary = summary + self.ffn(self.norm_summary(summary))
        return self.head(summary)


def local_queries(context, positions, coordinates):
    """Normalized frozen token at each location plus its fixed coordinate code.

    Raw token scale grows with encoder depth, so adding the fixed sin-cos
    coordinate to an unnormalized token would make the requested location
    progressively less legible at deeper outputs. A parameter-free
    normalization of the gathered token alone fixes the query scale and the
    relative weight of the coordinate at every encoder output. Keys and values
    still carry the complete unnormalized context, and the queries remain
    mutually independent.
    """
    index = positions.unsqueeze(-1).expand(-1, -1, context.shape[-1])
    token = torch.gather(context, 1, index)
    return F.layer_norm(token, token.shape[-1:]) + coordinates[positions]


def predict(fit, features):
    """Replay a fitted Ridge or metadata-MLP state on any split."""
    x = torch.tensor(np.asarray(features), dtype=torch.float32)
    if fit["kind"] == "ridge":
        standardized = ((x.double() - fit["mean"]) / fit["std"]) @ fit["weights"]
        values = (standardized * fit["target_std"] + fit["target_mean"]).numpy()
        return values[:, 0] if values.shape[1] == 1 else values
    if fit["kind"] != "mlp":
        raise ValueError(f"{fit['kind']} probes are replayed by predict_attentive")
    dev = device()
    x = (x.to(dev) - fit["mean"].to(dev)) / fit["std"].to(dev)
    predictions = []
    with torch.no_grad():
        for state in fit["states"]:
            model = _ProbeMLP(x.shape[1], fit["hidden"], fit["dropout"]).to(dev)
            model.load_state_dict(state)
            model.eval()
            predictions.append(
                model(x).double().cpu() * fit["target_std"] + fit["target_mean"]
            )
    return torch.stack(predictions).mean(0).numpy()


def predict_attentive(fit, context, positions=None, batch_size=64, groups=None):
    """Replay a frozen attentive probe over staged float16 context tokens.

    `groups` names the contexts that share one label, as the governing
    parameters do across a trajectory's sampled windows: each real context is
    predicted on its own and the grouped predictions are then averaged.
    """
    if (positions is None) == fit["local"]:
        raise ValueError("local attentive probes require token positions")
    dev = context.device if torch.is_tensor(context) else device()
    model = AttentiveProbe(
        dim=fit["dim"],
        heads=fit["heads"],
        ffn_hidden=fit["ffn_hidden"],
        outputs=len(fit["outputs"]),
        local=fit["local"],
    ).to(dev)
    model.load_state_dict(fit["state"])
    model.eval()
    coordinates = (
        None
        if not fit["local"]
        else sincos_3d(fit["dim"], *fit["grid"]).to(dev)
    )
    index = (
        None
        if positions is None
        else torch.as_tensor(np.asarray(positions), dtype=torch.long, device=dev)
    )
    mean = torch.tensor(fit["target_mean"], dtype=torch.float64)
    std = torch.tensor(fit["target_std"], dtype=torch.float64)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(context), batch_size):
            stop = min(start + batch_size, len(context))
            batch = _stage(context[start:stop], dev)
            queries = (
                None
                if index is None
                else local_queries(batch, index[start:stop], coordinates)
            )
            outputs.append(model(batch, queries).double().cpu() * std + mean)
    values = torch.cat(outputs).numpy()
    return values if groups is None else group_mean(values, groups)


def attentive_predictions(fit, context, positions=None, batch_size=64, groups=None):
    """Flat per-output predictions aligned with flattened target arrays."""
    values = predict_attentive(fit, context, positions, batch_size, groups)
    return {
        name: values[..., i].reshape(-1) for i, name in enumerate(fit["outputs"])
    }


def group_mean(values, groups):
    """Average the predictions of every context that shares one label."""
    return np.stack([np.asarray(values[ids]).mean(axis=0) for ids in groups])


def _stage(batch, dev):
    if torch.is_tensor(batch):
        return batch.to(dev).float()
    return torch.from_numpy(np.asarray(batch)).to(dev).float()


def fit_ridge_layer(
    layer, train, targets, valid, valid_targets, alphas=RIDGE_ALPHAS, joint=False
):
    """One encoder output of Ridge, sharing an eigendecomposition over penalties.

    With `joint=True` a single two-output map and a single penalty are selected
    on the averaged standardized validation MSE, matching the cited
    governing-parameter experiment. Otherwise every target is fit and selected
    independently.
    """
    x, y, xv, yv = _layer_data(train, targets, valid, valid_targets)
    if not alphas or any(a <= 0 or not np.isfinite(a) for a in alphas):
        raise ValueError("ridge penalties must be positive")
    names = list(y)
    if joint and len(names) < 2:
        raise ValueError("a joint governing ridge needs both regime parameters")
    yt = torch.tensor(np.column_stack([y[name] for name in names]), dtype=torch.float64)
    target_mean, target_std = _standardize(yt)
    standardized = (yt - target_mean) / target_std
    train_x, valid_x = (torch.tensor(f, dtype=torch.float64) for f in (x, xv))
    mean, std = train_x.mean(0), train_x.std(0, unbiased=False)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    train_x, valid_x = (train_x - mean) / std, (valid_x - mean) / std
    values, vectors = torch.linalg.eigh(train_x.T @ train_x)
    projected = vectors.T @ (train_x.T @ standardized)
    stds = {name: float(target_std[i]) for i, name in enumerate(names)}
    criterion = "valid_normalized_mse" if joint else "valid_vrmse"

    def solution(weights, columns):
        fit = dict(
            kind="ridge",
            mean=mean,
            std=std,
            target_mean=target_mean[columns],
            target_std=target_std[columns],
            weights=weights[:, columns].clone(),
        )
        predictions = np.asarray(predict(fit, xv)).reshape(len(xv), len(columns))
        return fit, {
            names[column]: predictions[:, i] for i, column in enumerate(columns)
        }

    candidates = []
    for alpha in alphas:
        weights = vectors @ (
            projected / (values.clamp_min(0)[:, None] + float(alpha) * len(train_x))
        )
        if joint:
            fit, predictions = solution(weights, list(range(len(names))))
            candidates.append(
                dict(layer=layer, alpha=float(alpha), fit=fit)
                | scored(predictions, yv, stds, joint=True)
            )
            continue
        choices = {}
        for column, name in enumerate(names):
            fit, predictions = solution(weights, [column])
            choices[name] = dict(layer=layer, alpha=float(alpha), fit=fit) | scored(
                predictions, yv, stds
            )
        candidates.append(choices)
    if joint:
        return _best(candidates, criterion)
    return {
        name: _best([choice[name] for choice in candidates], criterion)
        for name in names
    }


def _best(candidates, criterion):
    usable = [c for c in candidates if np.isfinite(c.get(criterion, float("nan")))]
    return min(usable, key=lambda c: c[criterion]) if usable else candidates[0]


def fit_mlp_layer(
    layer,
    train,
    target,
    valid,
    valid_target,
    max_steps=2000,
    min_steps=150,
    seeds=MLP_SEEDS,
):
    """The metadata-only baseline: one 128-unit ReLU hidden layer, dropout 0.1."""
    x, y, xv, yv = _layer_data(train, {"y": target}, valid, {"y": valid_target})
    if not 1 <= min_steps <= max_steps or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("invalid MLP stopping limits or seeds")
    dev = device()
    features = torch.tensor(x, device=dev)
    mean, std = _standardize(features)
    features = (features - mean) / std
    labels = torch.tensor(y["y"], dtype=torch.float64, device=dev)
    target_mean, target_std = labels.mean(), labels.std(unbiased=False).clamp_min(1e-12)
    normalized = ((labels - target_mean) / target_std).float()
    valid_features = (torch.tensor(xv, device=dev) - mean) / std
    valid_normalized = (
        (torch.tensor(yv["y"], device=dev) - target_mean) / target_std
    ).float()
    states, steps = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = _ProbeMLP(x.shape[1]).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        best, best_step, best_state = float("inf"), 0, None
        for step in range(1, max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            F.mse_loss(model(features), normalized).backward()
            optimizer.step()
            if step % 20 == 0 or step == max_steps:
                model.eval()
                with torch.no_grad():
                    loss = float(F.mse_loss(model(valid_features), valid_normalized))
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
        target_mean=target_mean.cpu(),
        target_std=target_std.cpu(),
        hidden=MLP_HIDDEN,
        dropout=MLP_DROPOUT,
    )
    return dict(layer=layer, selected_steps=steps, fit=fit) | scored(
        {"y": predict(fit, xv)}, yv, {"y": float(target_std)}
    )


def _attentive_stack(targets, samples, queries):
    names = list(targets)
    stacked = np.stack(
        [
            np.asarray(targets[name], dtype=np.float64).reshape(samples, queries)
            for name in names
        ],
        axis=-1,
    )
    if not np.isfinite(stacked).all():
        raise ValueError("non-finite attentive probe targets")
    return names, stacked


def _attentive_loss(
    model, context, labels, positions, coordinates, batch_size, groups=None
):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(context), batch_size):
            stop = min(start + batch_size, len(context))
            batch = _stage(context[start:stop], labels.device)
            queries = (
                None
                if positions is None
                else local_queries(batch, positions[start:stop], coordinates)
            )
            outputs.append(model(batch, queries))
    predicted = torch.cat(outputs)
    if groups is not None:
        predicted = torch.stack(
            [
                predicted[
                    torch.as_tensor(ids, dtype=torch.long, device=predicted.device)
                ].mean(0)
                for ids in groups
            ]
        )
    return float(F.mse_loss(predicted, labels))

def _warmup_inverse_sqrt(step, warmup_steps):
    """Cap-independent learning-rate multiplier for one-based optimizer updates."""
    update = step + 1
    return (
        update / warmup_steps
        if update <= warmup_steps
        else math.sqrt(warmup_steps / update)
    )


def fit_attentive_layer(
    layer,
    context,
    targets,
    valid_context,
    valid_targets,
    positions=None,
    valid_positions=None,
    grid=None,
    epochs=100,
    batch_size=32,
    seed=ATTENTIVE["seed"],
    min_epochs=1,
    patience=None,
    valid_groups=None,
    joint=False,
):
    """Train one attentive probe on a single staged encoder output.

    `context` and `valid_context` are the complete frozen token sequences of
    the eight-frame input clips. Supplying `positions` selects the
    location-conditioned local probe, whose sampled queries are processed
    together in one forward pass and never attend to one another.

    `valid_groups` names the validation contexts that share one label. The
    governing parameters are constant along a trajectory, so the probe trains
    on every real sampled window with that trajectory's labels repeated, and
    validation predicts each real window and averages the grouped predictions
    before scoring. Token sequences are never averaged into a synthetic
    context.
    """
    local = positions is not None
    if local and (valid_positions is None or grid is None):
        raise ValueError("local attentive probes need token positions and a grid")
    if local and valid_groups is not None:
        raise ValueError("local attentive targets are per token, never grouped")
    if context.ndim != 3 or valid_context.ndim != 3:
        raise ValueError("attentive probes need (samples,tokens,features) context")
    if local and np.asarray(positions).shape[1] != np.asarray(valid_positions).shape[1]:
        raise ValueError("train and validation query counts differ")
    if (
        epochs < 1
        or batch_size < 1
        or not 1 <= min_epochs <= epochs
        or (patience is not None and patience < 1)
    ):
        raise ValueError(
            "attentive training needs positive epochs/batch/patience and "
            "min_epochs within the epoch cap"
        )
    queries = int(np.asarray(positions).shape[1]) if local else 1
    names, y = _attentive_stack(targets, len(context), queries)
    _, yv = _attentive_stack(
        valid_targets,
        len(valid_context) if valid_groups is None else len(valid_groups),
        queries,
    )
    if local and len(names) != 1:
        raise ValueError("local attentive probes predict one scalar per location")
    dim = int(context.shape[-1])
    dev = context.device if torch.is_tensor(context) else device()
    flat = y.reshape(-1, y.shape[-1])
    mean, deviation = flat.mean(0), np.clip(flat.std(0), 1e-12, None)
    labels = torch.tensor((y - mean) / deviation, dtype=torch.float32, device=dev)
    valid_labels = torch.tensor(
        (yv - mean) / deviation, dtype=torch.float32, device=dev
    )
    index, valid_index = (
        None
        if not local
        else torch.as_tensor(np.asarray(p), dtype=torch.long, device=dev)
        for p in (positions, valid_positions)
    )
    coordinates = None if not local else sincos_3d(dim, *grid).to(dev)
    torch.manual_seed(seed)
    model = AttentiveProbe(dim=dim, outputs=len(names), local=local).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ATTENTIVE["lr"], weight_decay=ATTENTIVE["weight_decay"]
    )
    per_epoch = math.ceil(len(context) / batch_size)
    warmup = max(ATTENTIVE["warmup_epochs"] * per_epoch, 1)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _warmup_inverse_sqrt(step, warmup),
    )
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state = float("inf"), 0, None
    valid_loss_curve = []
    trained_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(context), generator=generator).to(dev)
        for start in range(0, len(context), batch_size):
            batch = order[start : start + batch_size]
            tokens = _stage(context[batch], dev)
            optimizer.zero_grad(set_to_none=True)
            inputs = (
                None if not local else local_queries(tokens, index[batch], coordinates)
            )
            F.mse_loss(model(tokens, inputs), labels[batch]).backward()
            optimizer.step()
            schedule.step()
        model.eval()
        loss = _attentive_loss(
            model,
            valid_context,
            valid_labels,
            valid_index,
            coordinates,
            batch_size,
            valid_groups,
        )
        valid_loss_curve.append(loss)
        trained_epochs = epoch
        if loss < best:
            best, best_epoch = loss, epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        if (
            patience is not None
            and epoch >= min_epochs
            and epoch - best_epoch >= patience
        ):
            break
    if best_state is None:
        raise ValueError("attentive probe produced no finite validation epoch")
    fit = dict(
        kind="attentive",
        local=local,
        dim=dim,
        heads=ATTENTIVE["heads"],
        ffn_hidden=ATTENTIVE["ffn_hidden"],
        outputs=names,
        grid=None if not local else list(grid),
        queries=queries,
        state=best_state,
        target_mean=mean.tolist(),
        target_std=deviation.tolist(),
    )
    return dict(
        layer=layer,
        selected_epoch=best_epoch,
        selected_step=best_epoch * per_epoch,
        trained_epochs=trained_epochs,
        trained_steps=trained_epochs * per_epoch,
        valid_loss_curve=valid_loss_curve,
        fit=fit,
    ) | scored(
        attentive_predictions(
            fit, valid_context, valid_positions, groups=valid_groups
        ),
        {name: yv[..., i] for i, name in enumerate(names)},
        {name: float(deviation[i]) for i, name in enumerate(names)},
        joint=joint,
    )
