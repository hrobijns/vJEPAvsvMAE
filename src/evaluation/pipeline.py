"""Fit on training data, freeze validation choices, then score held-out data."""

from pathlib import Path
import numpy as np
import torch

from src.evaluation.artifacts import Artifact, seal, staged_directory, write_json
from src.evaluation.cache import open_caches
from src.evaluation.probes import (
    RIDGE_ALPHAS,
    MLP_SEEDS,
    METRIC_NAMES,
    fit_mlp,
    fit_ridge_many,
    metrics,
    predict,
    selected_family,
)
from src.evaluation.protocol import (
    Protocol,
    nuisance_basis,
    position_basis,
    trajectory_average,
)
from src.physics.systems import SYSTEMS


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


def _data(feature, cache, representation):
    protocol = Protocol.from_dict(cache.manifest["protocol"])
    samples = cache.json(f"{representation}/samples.json")
    x = np.asarray(feature.array(f"{representation}.npy"))
    if len(x) != len(samples):
        raise ValueError("feature/sample identity mismatch")
    if representation == "token":
        if x.ndim != 4 or x.shape[1] != protocol.token_samples:
            raise ValueError("invalid token features")
        basis = position_basis(
            cache.array("token/positions.npy"), cache.manifest["grid"]
        )
        x = x.reshape(-1, *x.shape[-2:])
    else:
        basis = nuisance_basis(samples, protocol.dataset)
    targets = {
        f"{offset}:{target}": np.asarray(
            cache.array(f"{representation}/offset{offset}_{target}.npy")
        ).reshape(-1)
        for offset in protocol.target_offsets
        for target in SYSTEMS[protocol.dataset].targets
    }
    return x, basis, targets


def _validation_row(fitted, base, method):
    row = {
        **base,
        "method": method,
        "status": fitted["status"],
        "selected_layer": fitted["selected_layer"],
    }
    curve = [
        {k: v for k, v in entry.items() if k != "fit"} for entry in fitted["layers"]
    ]
    row["validation_curve"] = curve
    if fitted["selected_layer"] is None:
        row.update(
            valid_vrmse=float("nan"),
            valid_r2=float("nan"),
            metric_status="undefined_validation_vrmse",
        )
    else:
        entry = next(r for r in curve if r["layer"] == fitted["selected_layer"])
        row.update({k: v for k, v in entry.items() if k != "layer"})
        if "alpha" in entry:
            row["selected_alpha"] = entry["alpha"]
    return row


def _score_fit(fitted, test_features, target, base, method):
    row = _validation_row(fitted, base, method)
    if fitted["selected_layer"] is None:
        row.update({f"test_{m}": float("nan") for m in METRIC_NAMES})
        row["depth_curve"] = []
        return row, None
    depth, selected_fit = [], None
    for entry in fitted["layers"]:
        if (
            not fitted.get("include_depth", True)
            and entry["layer"] != fitted["selected_layer"]
        ):
            continue
        score = {k: v for k, v in entry.items() if k != "fit"} | metrics(
            predict(entry["fit"], test_features[:, entry["layer"]]), target
        )
        depth.append(score)
        if entry["layer"] == fitted["selected_layer"]:
            selected_fit = entry["fit"]
            row.update({k: v for k, v in score.items() if k != "layer"})
    row["depth_curve"] = depth if fitted.get("include_depth", True) else []
    return row, selected_fit


def _regime_data(feature, cache):
    samples = cache.json("pooled/samples.json")
    x, samples = trajectory_average(feature.array("pooled.npy"), samples)
    system = SYSTEMS[cache.manifest["protocol"]["dataset"]]
    y = np.stack([system.regime_values(r["parameters"]) for r in samples])
    names = [("log10_" if system.log_parameters else "") + p for p in system.parameters]
    return x, {name: y[:, i] for i, name in enumerate(names)}


def fit_probes(feature_dir, cache_root, output, mlp_max_steps=2000, mlp_min_steps=150):
    features, caches = _features_and_caches(feature_dir, cache_root, ("train", "valid"))
    train_feature, valid_feature = features
    train, valid = caches
    protocol = Protocol.from_dict(train.manifest["protocol"])
    rows, fitted = [], {}

    def record(fit, base, method, shared=False):
        row = _validation_row(fit, base, method)
        row["cell_id"] = cell_id(row)
        if shared:
            row["shared"] = True
        rows.append(row)
        fitted[row["cell_id"]] = fit
        return row

    for representation in ("pooled", "token"):
        x, b, y = _data(train_feature, train, representation)
        xv, bv, yv = _data(valid_feature, valid, representation)
        ridge = fit_ridge_many(x, y, xv, yv)
        controls = fit_ridge_many(b[:, None, :], y, bv[:, None, :], yv)
        combined = None
        if representation == "pooled":
            combined = fit_ridge_many(
                np.concatenate(
                    [x, np.repeat(b[:, None, :], x.shape[1], axis=1)], axis=2
                ),
                y,
                np.concatenate(
                    [xv, np.repeat(bv[:, None, :], xv.shape[1], axis=1)], axis=2
                ),
                yv,
            )
        for offset in protocol.target_offsets:
            for target in SYSTEMS[protocol.dataset].targets:
                key = f"{offset}:{target}"
                base = dict(
                    family="physics",
                    representation=representation,
                    target_offset=offset,
                    target=target,
                )
                rr = record(ridge[key], base, "ridge")
                mlp = fit_mlp(
                    x,
                    y[key],
                    xv,
                    yv[key],
                    max_steps=mlp_max_steps,
                    min_steps=mlp_min_steps,
                )
                mr = record(mlp, base, "mlp")
                chosen = selected_family(rr, mr)
                selected = {
                    **(chosen or rr),
                    "method": "selected",
                    "selected_method": chosen["method"] if chosen else None,
                }
                selected["cell_id"] = cell_id(selected)
                rows.append(selected)
                record(
                    controls[key],
                    base,
                    "regime_time" if representation == "pooled" else "position",
                    shared=True,
                )
                if combined is not None:
                    record(combined[key], base, "ridge_plus_control")
                if offset:
                    row = {
                        **base,
                        "method": "persistence",
                        "shared": True,
                        "status": "ok",
                        **metrics(yv[f"0:{target}"], yv[key], "valid"),
                    }
                    rows.append(row | {"cell_id": cell_id(row)})
                print(f"fit {representation} offset {offset} {target}", flush=True)
    x, y = _regime_data(train_feature, train)
    xv, yv = _regime_data(valid_feature, valid)
    ridge = fit_ridge_many(x, y, xv, yv)
    for name in y:
        base = dict(
            family="regime", representation="pooled", target_offset=0, target=name
        )
        record(ridge[name], base, "ridge")
        record(
            fit_mlp(
                x,
                y[name],
                xv,
                yv[name],
                max_steps=mlp_max_steps,
                min_steps=mlp_min_steps,
            ),
            base,
            "mlp",
        )
    with staged_directory(output) as stage:
        write_json(stage / "rows.json", rows)
        torch.save(fitted, stage / "fits.pt")
        seal(
            stage,
            "probe_fits",
            protocol=protocol.to_dict(),
            checkpoint=train_feature.manifest["checkpoint"],
            caches={c.manifest["split"]: c.manifest["sha256"] for c in caches},
            features={f.manifest["split"]: f.manifest["sha256"] for f in features},
            probe_settings=dict(
                ridge_alphas=list(RIDGE_ALPHAS),
                mlp_max_steps=mlp_max_steps,
                mlp_min_steps=mlp_min_steps,
                mlp_hidden=128,
                mlp_dropout=0.1,
                mlp_lr=0.01,
                mlp_weight_decay=1e-4,
                probe_seeds=list(MLP_SEEDS),
                mlp_predictions="ensemble_mean",
                selection_metric="valid_vrmse",
            ),
        )
    return Path(output)


def score_probes(feature_dir, cache_root, probe_dir, selection, output):
    fits = Artifact(probe_dir, "probe_fits")
    choice = Artifact(selection, "checkpoint_selection")
    winners = choice.json("selections.json")
    if not any(
        r["fit_sha256"] == fits.manifest["sha256"]
        and r["checkpoint_sha256"] == fits.manifest["checkpoint"]["sha256"]
        for r in winners
    ):
        raise ValueError("probe fits were not selected for test evaluation")
    features, caches = _features_and_caches(
        feature_dir, cache_root, ("train", "valid", "test")
    )
    for feature, cache in zip(features, caches):
        split = cache.manifest["split"]
        if (
            feature.manifest["checkpoint"] != fits.manifest["checkpoint"]
            or feature.manifest["provenance"] != fits.manifest["provenance"]
        ):
            raise ValueError(
                "test checkpoint or analysis code differs from fitted probes"
            )
        if split != "test" and (
            fits.manifest["caches"][split] != cache.manifest["sha256"]
            or fits.manifest["features"][split] != feature.manifest["sha256"]
        ):
            raise ValueError("fitting inputs changed before test scoring")
    feature, cache = features[-1], caches[-1]
    fitted = torch.load(fits.file("fits.pt"), map_location="cpu", weights_only=False)
    validation = fits.json("rows.json")
    data = {rep: _data(feature, cache, rep) for rep in ("pooled", "token")}
    regime_x, regime_y = _regime_data(feature, cache)
    rows, saved = [], {}
    for original in validation:
        method = original["method"]
        if method == "selected":
            continue
        base = {
            k: original[k]
            for k in ("family", "representation", "target_offset", "target")
        }
        if base["family"] == "regime":
            x, y = regime_x, regime_y[base["target"]]
        else:
            x, b, targets = data[base["representation"]]
            y = targets[f"{base['target_offset']}:{base['target']}"]
            if method == "persistence":
                row = {**original, **metrics(targets[f"0:{base['target']}"], y)}
                rows.append(row)
                continue
            if method in ("regime_time", "position"):
                x = b[:, None, :]
            elif method == "ridge_plus_control":
                x = np.concatenate(
                    [x, np.repeat(b[:, None, :], x.shape[1], axis=1)], axis=2
                )
        row, state = _score_fit(fitted[original["cell_id"]], x, y, base, method)
        if original.get("shared"):
            row["shared"] = True
        row["cell_id"] = cell_id(row)
        rows.append(row)
        if (
            base["family"] == "physics"
            and base["representation"] == "pooled"
            and base["target_offset"] == 0
            and method in ("ridge", "mlp")
            and state is not None
        ):
            saved[row["cell_id"]] = dict(
                fit=state,
                layer=row["selected_layer"],
                target=base["target"],
                method=method,
            )
    lookup = {r["cell_id"]: r for r in rows}
    for original in validation:
        if original["method"] == "selected":
            key = cell_id(original | {"method": original["selected_method"] or "ridge"})
            rows.append(
                {k: v for k, v in lookup[key].items() if k != "depth_curve"}
                | {
                    "method": "selected",
                    "selected_method": original["selected_method"],
                    "cell_id": original["cell_id"],
                }
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
            features=feature.manifest["sha256"],
            probe_settings=fits.manifest["probe_settings"],
            fits=fits.manifest["sha256"],
            selection=choice.manifest["sha256"],
            selection_policy=choice.manifest["policy"],
        )
    return Path(output)


def evaluate_noise(feature_dir, cache_root, probe_dir, output):
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
    clean_rows = probes.json("rows.json")
    originals = {r["cell_id"]: r for r in clean_rows}
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
                            "pooled.npy"
                            if sigma == 0
                            else f"noise_{sigma:g}_{seed}.npy"
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
    for clean in clean_rows:
        if (
            clean["family"] == "physics"
            and clean["representation"] == "pooled"
            and clean["target_offset"] == 0
            and clean["method"] == "selected"
            and clean.get("selected_method")
        ):
            rows.extend(
                [
                    r
                    | {
                        "method": "selected",
                        "selected_method": clean["selected_method"],
                    }
                    for r in rows
                    if r["target"] == clean["target"]
                    and r["method"] == clean["selected_method"]
                ]
            )
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
            selection=probes.manifest["selection"],
            selection_policy=probes.manifest["selection_policy"],
            probe_settings=probes.manifest["probe_settings"],
        )
    return Path(output)
