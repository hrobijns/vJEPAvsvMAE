"""Fit on training data, freeze validation choices, then score held-out data.

Both stages iterate encoder outputs in the outer loop so that exactly one
float16 token shard per representation is staged on the compute device at a
time. Probe inputs are built only from cached eight-frame input contexts and
their metadata; target arrays are read as labels and never as features.
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.evaluation.artifacts import (
    Artifact,
    canonical_hash,
    finite_json,
    provenance,
    seal,
    staged_directory,
    write_json,
)
from src.evaluation.cache import open_caches
from src.evaluation.features import shard_name
from src.evaluation.probes import (
    ATTENTIVE,
    METRIC_NAMES,
    MLP_DROPOUT,
    MLP_HIDDEN,
    MLP_SEEDS,
    RIDGE_ALPHAS,
    attentive_predictions,
    device,
    fit_attentive_layer,
    fit_mlp_layer,
    fit_ridge_layer,
    metrics,
    predict,
    scored,
    select_layers,
)
from src.evaluation.protocol import (
    Protocol,
    governing_names,
    governing_values,
    position_metadata,
    regime_metadata,
    trajectory_groups,
)
from src.physics.systems import SYSTEMS

REPRESENTATIONS = ("pooled", "token")
GOVERNING = "governing"
METADATA_METHODS = {
    "pooled": "regime_time_mlp",
    "token": "regime_time_position_mlp",
}


def _probe_layers(n_layers):
    if n_layers < 1:
        raise ValueError("feature artifact has no encoder outputs")
    return tuple(range(n_layers))


def _selected_targets(system, requested=None):
    """Validate and preserve the requested physical-target order."""
    targets = list(system.targets if requested is None else requested)
    unknown = [target for target in targets if target not in system.targets]
    if not targets or unknown or len(set(targets)) != len(targets):
        raise ValueError(
            f"physical targets must be a non-empty unique subset of "
            f"{list(system.targets)}; got {targets}"
        )
    return targets

def cell_id(row):
    keys = ("family", "representation", "target_offset", "target", "method", "sigma")
    return "|".join(f"{k}={row[k]}" for k in keys if k in row)


def _features_and_caches(feature_dir, cache_root, splits):
    caches = open_caches(cache_root, splits)
    features = [Artifact(Path(feature_dir) / split, "features") for split in splits]
    for split, feature, cache in zip(splits, features, caches):
        if feature.manifest["split"] != split or feature.manifest["caches"] != {
            split: cache.manifest["sha256"]
        }:
            raise ValueError("feature/cache split identity mismatch")
        if feature.manifest["protocol"] != cache.manifest["protocol"]:
            raise ValueError("feature/cache protocol mismatch")
        for key in ("checkpoint", "provenance"):
            if feature.manifest[key] != features[0].manifest[key]:
                raise ValueError(f"feature {key} differs across splits")
    return features, caches


def _stage_layer(array, dev, chunk=64):
    """Copy exactly one encoder output onto the compute device."""
    tensor = torch.empty(array.shape, dtype=torch.float16, device=dev)
    for start in range(0, len(array), chunk):
        stop = min(start + chunk, len(array))
        tensor[start:stop] = torch.from_numpy(np.array(array[start:stop])).to(dev)
    return tensor


def _sampled_tokens(context, positions):
    """The requested frozen tokens, flattened exactly like the local targets."""
    index = torch.as_tensor(
        np.asarray(positions), dtype=torch.long, device=context.device
    )
    gathered = torch.gather(
        context, 1, index.unsqueeze(-1).expand(-1, -1, context.shape[-1])
    )
    return gathered.reshape(-1, context.shape[-1]).float().cpu().numpy()


def _governing_window_labels(samples, dataset):
    """Each sampled window carries its own trajectory's governing labels."""
    values = [governing_values(dataset, row["parameters"]) for row in samples]
    return {
        name: np.array([value[name] for value in values])
        for name in governing_names(dataset)
    }


class Split:
    """One split's labels, metadata and lazily staged encoder outputs."""

    def __init__(self, feature, cache, protocol):
        self.feature, self.cache, self.protocol = feature, cache, protocol
        self.grid = tuple(cache.manifest["grid"])
        system = SYSTEMS[protocol.dataset]
        self.samples = {r: cache.json(f"{r}/samples.json") for r in REPRESENTATIONS}
        self.positions = np.array(cache.array("token/positions.npy"), dtype=np.int64)
        self.targets = {
            representation: {
                (offset, target): np.asarray(
                    cache.array(f"{representation}/offset{offset}_{target}.npy")
                )
                for offset in protocol.target_offsets
                for target in system.targets
            }
            for representation in REPRESENTATIONS
        }
        self.metadata = {
            "pooled": regime_metadata(self.samples["pooled"], protocol.dataset),
            "token": position_metadata(
                self.samples["token"], self.positions, self.grid, protocol.dataset
            ),
        }
        self.groups, trajectories = trajectory_groups(self.samples["pooled"])
        # Per trajectory for reporting, per sampled window for attentive fits.
        self.governing = _governing_window_labels(trajectories, protocol.dataset)
        self.governing_windows = _governing_window_labels(
            self.samples["pooled"], protocol.dataset
        )
        shards = feature.manifest["shards"]
        self.n_layers = shards["layers"]
        if (
            shards["dtype"] != "float16"
            or list(feature.manifest["grid"]) != list(self.grid)
            or any(
                shards[representation][0] != len(self.samples[representation])
                or shards[representation][1] != int(np.prod(self.grid))
                for representation in REPRESENTATIONS
            )
            or self.positions.shape
            != (len(self.samples["token"]), protocol.token_samples)
        ):
            raise ValueError("feature shard geometry disagrees with the cache")

    def pooled(self, layer):
        features = self.feature.array("pooled/pooled.npy")
        if len(features) != len(self.samples["pooled"]) or features.shape[1] != self.n_layers:
            raise ValueError("feature/sample identity mismatch")
        return np.asarray(features[:, layer], dtype=np.float32)

    def sampled(self, layer):
        """Gather only the fixed local probe tokens without staging a full shard."""
        shard = self.feature.array(f"token/{shard_name(layer)}")
        rows = np.arange(len(self.positions))[:, None]
        gathered = shard[rows, self.positions]
        return np.asarray(gathered, dtype=np.float32).reshape(-1, gathered.shape[-1])

    def regime_features(self, layer):
        features = self.pooled(layer)
        return np.stack([features[ids].mean(axis=0) for ids in self.groups])

    def shard(self, representation, layer, dev):
        return _stage_layer(
            self.feature.array(f"{representation}/{shard_name(layer)}"), dev
        )

    def flat_targets(self, representation, targets=None):
        wanted = (
            {target for _, target in self.targets[representation]}
            if targets is None
            else set(targets)
        )
        return {
            f"{offset}:{target}": values.reshape(-1)
            for (offset, target), values in self.targets[representation].items()
            if target in wanted
        }


def _entry_row(entry, output=None):
    """A validation-curve row: never the fitted state, one output at a time."""
    row = {k: v for k, v in entry.items() if k not in ("fit", "outputs")}
    if output is not None:
        row.update(entry["outputs"][output])
    return row


def _validation_row(result, base, method, fit_key, output=None):
    row = {
        **base,
        "method": method,
        "status": result["status"],
        "selected_layer": result["selected_layer"],
        "fit_key": fit_key,
    }
    curve = [_entry_row(entry, output) for entry in result["layers"]]
    row["validation_curve"] = curve
    if result["selected_layer"] is None:
        row.update(
            valid_vrmse=float("nan"),
            valid_r2=float("nan"),
            metric_status="undefined_validation_vrmse",
        )
    else:
        entry = next(r for r in curve if r["layer"] == result["selected_layer"])
        row.update({k: v for k, v in entry.items() if k != "layer"})
        if "alpha" in entry:
            row["selected_alpha"] = entry["alpha"]
    return row


def _rows(result, plan, fit_key, scores=None):
    """One row per reported output, with test metrics when scores are supplied."""
    rows = []
    for output in plan["outputs"] or [None]:
        base = plan["base"] if output is None else {**plan["base"], "target": output}
        row = _validation_row(result, base, plan["method"], fit_key, output)
        if scores is not None:
            depth = []
            for entry in result["layers"]:
                if entry["layer"] not in scores:
                    continue
                block = scores[entry["layer"]]
                depth.append(
                    _entry_row(entry, output)
                    | (block if output is None else block["outputs"][output])
                )
            if result["selected_layer"] is None:
                row.update({f"test_{name}": float("nan") for name in METRIC_NAMES})
            else:
                selected = next(
                    d for d in depth if d["layer"] == result["selected_layer"]
                )
                row.update({k: v for k, v in selected.items() if k != "layer"})
            row["depth_curve"] = depth if result["include_depth"] else []
        if plan["shared"]:
            row["shared"] = True
        row["cell_id"] = cell_id(row)
        rows.append(row)
    return rows


def _persistence_rows(split, protocol, dataset, targets, test=False):
    rows = []
    for representation in REPRESENTATIONS:
        for offset in protocol.target_offsets:
            if not offset:
                continue
            for target in targets:
                row = dict(
                    family="physics",
                    representation=representation,
                    target_offset=offset,
                    target=target,
                    method="persistence",
                    shared=True,
                    status="ok",
                    selected_layer=None,
                    fit_key=None,
                    **metrics(
                        split.targets[representation][0, target].reshape(-1),
                        split.targets[representation][offset, target].reshape(-1),
                        "test" if test else "valid",
                    ),
                )
                row["cell_id"] = cell_id(row)
                rows.append(row)
    return rows


def _probe_settings(
    protocol, probe_layers, n_layers, metadata_widths, tuning, physical_targets
):
    return dict(
        ridge_alphas=list(RIDGE_ALPHAS),
        mlp_max_steps=tuning["mlp_max_steps"],
        mlp_min_steps=tuning["mlp_min_steps"],
        mlp_hidden=MLP_HIDDEN,
        mlp_dropout=MLP_DROPOUT,
        mlp_lr=0.01,
        mlp_weight_decay=1e-4,
        probe_seeds=list(MLP_SEEDS),
        mlp_predictions="single_seed",
        metadata_methods=dict(METADATA_METHODS),
        physical_targets=list(physical_targets),
        metadata_inputs_pooled=metadata_widths["pooled"],
        metadata_inputs_token=metadata_widths["token"],
        governing_only=tuning["governing_only"],
        feature_mlp=tuning["feature_mlp"],
        attentive=tuning["attentive"],
        attentive_epochs=tuning["attentive_epochs"],
        attentive_batch_size=tuning["attentive_batch_size"],
        attentive_min_epochs=tuning["attentive_min_epochs"],
        attentive_patience=tuning["attentive_patience"],
        attentive_blocks=ATTENTIVE["blocks"],
        attentive_heads=ATTENTIVE["heads"],
        attentive_ffn_hidden=ATTENTIVE["ffn_hidden"],
        attentive_dropout=ATTENTIVE["dropout"],
        attentive_lr=ATTENTIVE["lr"],
        attentive_weight_decay=ATTENTIVE["weight_decay"],
        attentive_warmup_epochs=ATTENTIVE["warmup_epochs"],
        attentive_schedule="linear_warmup_inverse_sqrt",
        attentive_seed=ATTENTIVE["seed"],
        attentive_context="all_input_context_tokens",
        attentive_queries=dict(
            pooled="learned_global",
            token="normalized_frozen_token_plus_coordinate",
        ),
        attentive_local_queries=protocol.token_samples,
        governing_targets=list(governing_names(protocol.dataset)),
        governing_fit="joint_two_output_ridge_scalar_mlp",
        governing_standardization="train_only",
        governing_selection_metric="valid_normalized_mse",
        regime_samples=dict(
            ridge="trajectory_averaged_pooled_features",
            mlp="trajectory_averaged_pooled_features",
            attentive="per_window_context_then_averaged_predictions",
        ),
        probe_layers=list(probe_layers),
        probe_outputs=[
            "final_norm" if layer == n_layers - 1 else f"block_{layer + 1}"
            for layer in probe_layers
        ],
        selection_metric="valid_vrmse",
    )


def _replace(write, path):
    """Publish partial work only once it is completely on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f"{path.name}.writing")
    write(staging)
    os.replace(staging, path)


class PartialFits:
    """Per-encoder-output probe fits kept beside the unfinished output.

    A full depth sweep of attentive probes is far too long to restart from
    zero, so each completed encoder output and the metadata-only stage are
    published atomically. Resuming is allowed only against byte-identical
    inputs: the checkpoint, every cache and feature hash, the feature
    extraction provenance, the probe settings, and this module's own
    provenance all bind the saved state. Nothing here ever masks a failed
    scientific cell; a cell that raises simply leaves its output unpublished.
    """

    def __init__(self, output, identity):
        self.root = Path(output).with_name(f".{Path(output).name}.partial-fits")
        self.identity = identity
        self.digest = canonical_hash(identity)
        recorded = self.root / "identity.json"
        if recorded.is_file():
            saved = json.loads(recorded.read_text())
            if saved.get("sha256") != self.digest:
                raise ValueError(
                    f"partial probe fits at {self.root} were produced from "
                    "different inputs, settings or code; remove them to refit"
                )
        else:
            _replace(
                lambda path: path.write_text(
                    json.dumps(
                        {**finite_json(identity), "sha256": self.digest},
                        indent=2,
                        sort_keys=True,
                    )
                ),
                recorded,
            )

    def _path(self, stage):
        return self.root / f"{stage}.pt"

    def completed(self, stage):
        return self._path(stage).is_file()

    def load(self, stage):
        return torch.load(self._path(stage), map_location="cpu", weights_only=False)

    def publish(self, stage, payload):
        _replace(lambda path: torch.save(payload, path), self._path(stage))

    def discard(self):
        shutil.rmtree(self.root, ignore_errors=True)


def fit_probes(
    feature_dir,
    cache_root,
    output,
    mlp_max_steps=2000,
    mlp_min_steps=150,
    attentive_epochs=100,
    attentive_batch_size=32,
    attentive_min_epochs=15,
    attentive_patience=10,
    physical_targets=None,
    feature_mlp=False,
    attentive=True,
    governing_only=False,
):
    features, caches = _features_and_caches(feature_dir, cache_root, ("train", "valid"))
    protocol = Protocol.from_dict(caches[0].manifest["protocol"])
    train, valid = (Split(f, c, protocol) for f, c in zip(features, caches))
    probe_layers = _probe_layers(train.n_layers)
    system = SYSTEMS[protocol.dataset]
    physical_targets = _selected_targets(system, physical_targets)
    governing = list(governing_names(protocol.dataset))
    dev = device()
    settings = _probe_settings(
        protocol,
        probe_layers,
        train.n_layers,
        {r: int(train.metadata[r].shape[1]) for r in REPRESENTATIONS},
        dict(
            mlp_max_steps=mlp_max_steps,
            mlp_min_steps=mlp_min_steps,
            governing_only=governing_only,
            feature_mlp=feature_mlp,
            attentive=attentive,
            attentive_epochs=attentive_epochs,
            attentive_batch_size=attentive_batch_size,
            attentive_min_epochs=attentive_min_epochs,
            attentive_patience=attentive_patience,
        ),
        physical_targets,
    )
    caches_by_split = {c.manifest["split"]: c.manifest["sha256"] for c in caches}
    features_by_split = {f.manifest["split"]: f.manifest["sha256"] for f in features}
    partial = PartialFits(
        output,
        dict(
            protocol=protocol.to_dict(),
            checkpoint=train.feature.manifest["checkpoint"],
            caches=caches_by_split,
            features=features_by_split,
            feature_provenance=train.feature.manifest["provenance"],
            probe_settings=settings,
            provenance=provenance(),
        ),
    )
    entries, plans = defaultdict(dict), {}

    def plan(base, method, criterion="valid_vrmse", include_depth=True, shared=False, outputs=None):
        key = cell_id({**base, "method": method})
        plans[key] = dict(
            base=dict(base),
            method=method,
            criterion=criterion,
            include_depth=include_depth,
            shared=shared,
            outputs=outputs,
        )
        return key

    def restore(stage):
        resumed = partial.load(stage)
        plans.update(resumed["plans"])
        for key, entry in resumed["entries"].items():
            entries[key][entry["layer"]] = entry
        print(f"resumed {stage} from {partial.root}", flush=True)

    for layer in probe_layers:
        stage = f"layer{layer}"
        if partial.completed(stage):
            restore(stage)
            continue
        done = {}
        for representation in REPRESENTATIONS:
            local = representation == "token"
            if governing_only and local:
                # The governing-parameter gate reads pooled features only.
                continue
            context = train.shard(representation, layer, dev) if attentive else None
            valid_context = (
                valid.shard(representation, layer, dev) if attentive else None
            )
            if local:
                x = (
                    _sampled_tokens(context, train.positions)
                    if attentive
                    else train.sampled(layer)
                )
                xv = (
                    _sampled_tokens(valid_context, valid.positions)
                    if attentive
                    else valid.sampled(layer)
                )
            else:
                x, xv = train.pooled(layer), valid.pooled(layer)
            ridge = (
                None
                if governing_only
                else fit_ridge_layer(
                    layer,
                    x,
                    train.flat_targets(representation, physical_targets),
                    xv,
                    valid.flat_targets(representation, physical_targets),
                )
            )
            for offset in () if governing_only else protocol.target_offsets:
                for target in physical_targets:
                    base = dict(
                        family="physics",
                        representation=representation,
                        target_offset=offset,
                        target=target,
                    )
                    done[plan(base, "ridge")] = ridge[f"{offset}:{target}"]
                    if feature_mlp:
                        done[plan(base, "mlp")] = fit_mlp_layer(
                            layer,
                            x,
                            train.targets[representation][offset, target].reshape(-1),
                            xv,
                            valid.targets[representation][offset, target].reshape(-1),
                            max_steps=mlp_max_steps,
                            min_steps=mlp_min_steps,
                        )
                    if attentive:
                        done[plan(base, "attentive")] = fit_attentive_layer(
                            layer,
                            context,
                            {target: train.targets[representation][offset, target]},
                            valid_context,
                            {target: valid.targets[representation][offset, target]},
                            positions=train.positions if local else None,
                            valid_positions=valid.positions if local else None,
                            grid=train.grid if local else None,
                            epochs=attentive_epochs,
                            batch_size=attentive_batch_size,
                            min_epochs=attentive_min_epochs,
                            patience=attentive_patience,
                        )
                    print(
                        f"fit output {layer} {representation} offset {offset} {target}",
                        flush=True,
                    )
            del x, xv
            if not local:
                base = dict(
                    family="regime",
                    representation="pooled",
                    target_offset=0,
                    target=GOVERNING,
                )
                regime_x = train.regime_features(layer)
                valid_regime_x = valid.regime_features(layer)
                key = plan(
                    base,
                    "ridge",
                    criterion="valid_normalized_mse",
                    outputs=governing,
                )
                done[key] = fit_ridge_layer(
                    layer,
                    regime_x,
                    train.governing,
                    valid_regime_x,
                    valid.governing,
                    joint=True,
                )
                if feature_mlp:
                    for target in governing:
                        target_base = {**base, "target": target}
                        done[plan(target_base, "mlp")] = fit_mlp_layer(
                            layer,
                            regime_x,
                            train.governing[target],
                            valid_regime_x,
                            valid.governing[target],
                            max_steps=mlp_max_steps,
                            min_steps=mlp_min_steps,
                        )
                if attentive:
                    key = plan(
                        base,
                        "attentive",
                        criterion="valid_normalized_mse",
                        outputs=governing,
                    )
                    # Every real sampled window is its own training example; the
                    # trajectory's labels repeat and predictions average by trajectory.
                    done[key] = fit_attentive_layer(
                        layer,
                        context,
                        train.governing_windows,
                        valid_context,
                        valid.governing,
                        valid_groups=valid.groups,
                        epochs=attentive_epochs,
                        batch_size=attentive_batch_size,
                        min_epochs=attentive_min_epochs,
                        patience=attentive_patience,
                        joint=True,
                    )
                del regime_x, valid_regime_x
                print(f"fit output {layer} governing parameters", flush=True)
            if attentive:
                del context, valid_context
        partial.publish(stage, dict(entries=done, plans=dict(plans)))
        for key, entry in done.items():
            entries[key][layer] = entry

    if governing_only:
        pass
    elif partial.completed("metadata"):
        restore("metadata")
    else:
        done = {}
        for representation in REPRESENTATIONS:
            method = METADATA_METHODS[representation]
            for offset in protocol.target_offsets:
                for target in physical_targets:
                    base = dict(
                        family="physics",
                        representation=representation,
                        target_offset=offset,
                        target=target,
                    )
                    key = plan(base, method, include_depth=False, shared=True)
                    done[key] = fit_mlp_layer(
                        0,
                        train.metadata[representation],
                        train.targets[representation][offset, target].reshape(-1),
                        valid.metadata[representation],
                        valid.targets[representation][offset, target].reshape(-1),
                        max_steps=mlp_max_steps,
                        min_steps=mlp_min_steps,
                    )
            print(f"fit {method}", flush=True)
        partial.publish("metadata", dict(entries=done, plans=dict(plans)))
        for key, entry in done.items():
            entries[key][0] = entry

    rows, fitted = [], {}
    for key, by_layer in entries.items():
        spec = plans[key]
        result = select_layers(
            [by_layer[layer] for layer in sorted(by_layer)],
            spec["include_depth"],
            spec["criterion"],
        ) | {"plan": spec}
        fitted[key] = result
        rows.extend(_rows(result, spec, key))
    if not governing_only:
        rows.extend(
            _persistence_rows(valid, protocol, protocol.dataset, physical_targets)
        )
    with staged_directory(output) as stage:
        write_json(stage / "rows.json", rows)
        torch.save(fitted, stage / "fits.pt")
        seal(
            stage,
            "probe_fits",
            protocol=protocol.to_dict(),
            checkpoint=train.feature.manifest["checkpoint"],
            caches=caches_by_split,
            features=features_by_split,
            feature_provenance=train.feature.manifest["provenance"],
            probe_settings=settings,
        )
    partial.discard()
    return Path(output)


def _scores_wanted(result, layer):
    if result["selected_layer"] is None:
        return False
    return result["include_depth"] or result["selected_layer"] == layer


def score_probes(feature_dir, cache_root, probe_dir, output):
    """Open test exactly once, replaying each frozen validation-selected fit."""
    fits = Artifact(probe_dir, "probe_fits")
    features, caches = _features_and_caches(
        feature_dir, cache_root, ("train", "valid", "test")
    )
    feature_provenance = fits.manifest["feature_provenance"]
    for feature, cache in zip(features, caches):
        split = cache.manifest["split"]
        if (
            feature.manifest["checkpoint"] != fits.manifest["checkpoint"]
            or feature.manifest["provenance"] != feature_provenance
        ):
            raise ValueError(
                "test checkpoint or feature-extraction code differs from fitted probes"
            )
        if split != "test" and (
            fits.manifest["caches"][split] != cache.manifest["sha256"]
            or fits.manifest["features"][split] != feature.manifest["sha256"]
        ):
            raise ValueError("fitting inputs changed before test scoring")
    protocol = Protocol.from_dict(fits.manifest["protocol"])
    test = Split(features[-1], caches[-1], protocol)
    physical_targets = _selected_targets(
        SYSTEMS[protocol.dataset],
        fits.manifest["probe_settings"].get("physical_targets"),
    )
    fitted = torch.load(fits.file("fits.pt"), map_location="cpu", weights_only=False)
    validation = fits.json("rows.json")
    if not fitted or not validation:
        raise ValueError("probe fits record no validation-selected cells")
    if any(result["selected_layer"] is None for result in fitted.values()) and not any(
        result["selected_layer"] is not None for result in fitted.values()
    ):
        raise ValueError("no cell reached a usable validation selection")
    probe_layers = fits.manifest["probe_settings"]["probe_layers"]
    grouped = defaultdict(list)
    for key, result in fitted.items():
        base, method = result["plan"]["base"], result["plan"]["method"]
        kind = (
            "regime"
            if base["family"] == "regime"
            else "metadata"
            if method in METADATA_METHODS.values()
            else method
        )
        grouped[kind, base["representation"]].append((key, result))
    dev = device()
    scores = defaultdict(dict)
    for layer in probe_layers:
        for representation in REPRESENTATIONS:
            ridge_cells = [
                (k, r)
                for k, r in grouped["ridge", representation]
                if _scores_wanted(r, layer)
            ]
            mlp_cells = [
                (k, r)
                for k, r in grouped["mlp", representation]
                if _scores_wanted(r, layer)
            ]
            attentive_cells = [
                (k, r)
                for k, r in grouped["attentive", representation]
                if _scores_wanted(r, layer)
            ]
            regime_cells = [
                (k, r)
                for k, r in grouped["regime", representation]
                if _scores_wanted(r, layer)
            ]
            if not (ridge_cells or mlp_cells or attentive_cells or regime_cells):
                continue
            local = representation == "token"
            needs_context = bool(
                attentive_cells
                or any(r["plan"]["method"] == "attentive" for _, r in regime_cells)
            )
            context = (
                test.shard(representation, layer, dev) if needs_context else None
            )
            targets = test.flat_targets(representation)
            feature_cells = ridge_cells + mlp_cells
            if feature_cells:
                x = test.sampled(layer) if local else test.pooled(layer)
                for key, result in feature_cells:
                    base = result["plan"]["base"]
                    name = f"{base['target_offset']}:{base['target']}"
                    entry = next(
                        e for e in result["layers"] if e["layer"] == layer
                    )
                    target_std = entry["fit"]["target_std"]
                    if torch.is_tensor(target_std) and target_std.ndim:
                        target_std = target_std[0]
                    scores[key][layer] = scored(
                        {base["target"]: predict(entry["fit"], x)},
                        {base["target"]: targets[name]},
                        {base["target"]: float(target_std)},
                        split="test",
                    )
                del x
            for key, result in attentive_cells:
                base = result["plan"]["base"]
                name = f"{base['target_offset']}:{base['target']}"
                entry = next(e for e in result["layers"] if e["layer"] == layer)
                fit = entry["fit"]
                scores[key][layer] = scored(
                    attentive_predictions(
                        fit, context, test.positions if local else None
                    ),
                    {base["target"]: targets[name]},
                    {base["target"]: float(fit["target_std"][0])},
                    split="test",
                )
            if regime_cells:
                regime_x = test.regime_features(layer)
                for key, result in regime_cells:
                    entry = next(e for e in result["layers"] if e["layer"] == layer)
                    fit = entry["fit"]
                    names = result["plan"]["outputs"]
                    if names:
                        stds = {
                            name: float(fit["target_std"][i])
                            for i, name in enumerate(names)
                        }
                        if fit["kind"] == "ridge":
                            values = np.asarray(predict(fit, regime_x)).reshape(
                                len(regime_x), len(names)
                            )
                            predictions = {
                                name: values[:, i] for i, name in enumerate(names)
                            }
                        else:
                            # Predict windows, then average them per trajectory.
                            predictions = attentive_predictions(
                                fit, context, groups=test.groups
                            )
                        scores[key][layer] = scored(
                            predictions,
                            test.governing,
                            stds,
                            split="test",
                            joint=True,
                        )
                    else:
                        name = result["plan"]["base"]["target"]
                        scores[key][layer] = scored(
                            {name: predict(fit, regime_x)},
                            {name: test.governing[name]},
                            {name: float(fit["target_std"])},
                            split="test",
                        )
                del regime_x
            if context is not None:
                del context
        for key, result in grouped["metadata", "pooled"] + grouped[
            "metadata", "token"
        ]:
            if layer != 0 or not _scores_wanted(result, 0):
                continue
            base = result["plan"]["base"]
            representation = base["representation"]
            entry = next(e for e in result["layers"] if e["layer"] == 0)
            scores[key][0] = scored(
                {base["target"]: predict(entry["fit"], test.metadata[representation])},
                {
                    base["target"]: test.targets[representation][
                        base["target_offset"], base["target"]
                    ].reshape(-1)
                },
                {base["target"]: float(entry["fit"]["target_std"])},
                split="test",
            )
    rows, saved = [], {}
    for key, result in fitted.items():
        plan = result["plan"]
        for row in _rows(result, plan, key, scores[key]):
            rows.append(row)
            base = plan["base"]
            if (
                base["family"] == "physics"
                and base["representation"] == "pooled"
                and base["target_offset"] == 0
                and plan["method"] == "ridge"
                and result["selected_layer"] is not None
            ):
                entry = next(
                    e
                    for e in result["layers"]
                    if e["layer"] == result["selected_layer"]
                )
                saved[row["cell_id"]] = dict(
                    fit=entry["fit"],
                    layer=result["selected_layer"],
                    target=base["target"],
                    method="ridge",
                )
    rows.extend(
        _persistence_rows(
            test, protocol, protocol.dataset, physical_targets, test=True
        )
    )
    with staged_directory(output) as stage:
        write_json(stage / "rows.json", rows)
        torch.save(saved, stage / "probes.pt")
        seal(
            stage,
            "probes",
            protocol=fits.manifest["protocol"],
            checkpoint=fits.manifest["checkpoint"],
            caches={c.manifest["split"]: c.manifest["sha256"] for c in caches},
            features=features[-1].manifest["sha256"],
            probe_settings=fits.manifest["probe_settings"],
            fits=fits.manifest["sha256"],
        )
    return Path(output)


def evaluate_noise(feature_dir, cache_root, probe_dir, output):
    """Paired input corruption for the pooled linear probes.

    Noise features are mean-pooled by construction, so only the pooled Ridge
    probes can be replayed on them; an attentive head needs the full token
    sequence of the corrupted clip, which is not cached.
    """
    features, caches = _features_and_caches(feature_dir, cache_root, ("test",))
    feature, cache = features[0], caches[0]
    probes = Artifact(probe_dir, "probes")
    if (
        feature.manifest["checkpoint"] != probes.manifest["checkpoint"]
        or probes.manifest["caches"]["test"] != cache.manifest["sha256"]
    ):
        raise ValueError("noise features belong to different checkpoint or targets")
    states = torch.load(
        probes.file("probes.pt"), map_location="cpu", weights_only=False
    )
    originals = {r["cell_id"]: r for r in probes.json("rows.json")}
    protocol = Protocol.from_dict(feature.manifest["protocol"])
    rows = []
    for identity, state in states.items():
        target = np.asarray(cache.array(f"pooled/offset0_{state['target']}.npy"))
        for sigma in protocol.noise_sigmas:
            scores = [
                metrics(
                    predict(
                        state["fit"],
                        feature.array(
                            "pooled/pooled.npy"
                            if sigma == 0
                            else f"pooled/noise_{sigma:g}_{seed}.npy"
                        )[:, state["layer"]],
                    ),
                    target,
                )
                for seed in protocol.noise_seeds
            ]
            row = dict(
                family="noise",
                representation="pooled",
                target_offset=0,
                target=state["target"],
                sigma=sigma,
                method=state["method"],
                selected_layer=state["layer"],
                status="ok",
                valid_vrmse=originals[identity]["valid_vrmse"],
                valid_r2=originals[identity]["valid_r2"],
                metric_status=scores[0]["metric_status"],
            )
            for name in METRIC_NAMES:
                metric = "test_" + name
                row[metric] = float(np.mean([s[metric] for s in scores]))
                row[metric + "_per_corruption_seed"] = [s[metric] for s in scores]
                if sigma == 0:
                    expected = originals[identity].get(metric)
                    if (expected is None and np.isfinite(row[metric])) or (
                        expected is not None
                        and not np.isclose(row[metric], expected, atol=1e-10, rtol=1e-8)
                    ):
                        raise ValueError(
                            f"zero-noise replay differs: {identity} {metric}"
                        )
            rows.append(row)
    for row in rows:
        row["cell_id"] = cell_id(row)
    with staged_directory(output) as stage:
        write_json(stage / "rows.json", rows)
        seal(
            stage,
            "noise",
            protocol=protocol.to_dict(),
            checkpoint=feature.manifest["checkpoint"],
            caches=probes.manifest["caches"],
            features=feature.manifest["sha256"],
            probe_settings=probes.manifest["probe_settings"],
            noise_methods=["ridge"],
        )
    return Path(output)
