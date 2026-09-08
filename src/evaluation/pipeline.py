"""One clean-fit probing workflow for all supported systems."""

from pathlib import Path

import numpy as np
import torch

from src.evaluation.artifacts import Artifact, seal, staged_directory, write_json
from src.evaluation.cache import open_caches
from src.evaluation.probes import (
    RIDGE_ALPHAS,
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
    keys = ("family", "representation", "gap", "target", "method", "sigma")
    return "|".join(f"{k}={row[k]}" for k in keys if k in row)


def _features_and_caches(feature_dir, cache_root):
    feature = Artifact(feature_dir, "features")
    valid, test = open_caches(cache_root)
    if feature.manifest["caches"] != {
        c.manifest["split"]: c.manifest["sha256"] for c in (valid, test)
    }:
        raise ValueError("features belong to different target caches")
    if feature.manifest["protocol"] != valid.manifest["protocol"]:
        raise ValueError("feature/cache protocol mismatch")
    return feature, valid, test


def _score_fit(fitted, test_features, target, base, method):
    row = {**base, "method": method, "status": fitted["status"]}
    if fitted["selected_layer"] is None:
        row.update(
            selected_layer=None,
            valid_cv_r2=float("nan"),
            test_r2=float("nan"),
            test_pearson_r=float("nan"),
            test_mse=float("nan"),
            metric_status="undefined_validation_r2",
            depth_curve=[],
        )
        return row, None
    depth = []
    for entry in fitted["layers"]:
        prediction = predict(entry["fit"], test_features[:, entry["layer"]])
        depth.append(
            {k: v for k, v in entry.items() if k != "fit"} | metrics(prediction, target)
        )
    selection = next(
        entry
        for entry in fitted["layers"]
        if entry["layer"] == fitted["selected_layer"]
    )
    chosen = next(
        entry for entry in depth if entry["layer"] == fitted["selected_layer"]
    )
    row.update({k: v for k, v in chosen.items() if k != "layer"})
    row.update(
        selected_layer=chosen["layer"],
        depth_curve=depth if fitted.get("include_depth", True) else [],
        validation_curve=fitted.get("validation_curve", []),
    )
    if "alpha" in chosen:
        row["selected_alpha"] = chosen["alpha"]
    return row, selection["fit"]


def fit_probes(feature_dir, cache_root, output, mlp_max_steps=2000, mlp_min_steps=150):
    if not 1 <= mlp_min_steps <= mlp_max_steps:
        raise ValueError("MLP steps must satisfy 1 <= min_steps <= max_steps")
    feature, valid, test = _features_and_caches(feature_dir, cache_root)
    protocol = Protocol.from_dict(valid.manifest["protocol"])
    system = SYSTEMS[protocol.dataset]
    rows, saved = [], {}
    for representation in ("pooled", "token"):
        vs = valid.json(f"{representation}/samples.json")
        ts = test.json(f"{representation}/samples.json")
        xv = np.asarray(feature.array(f"valid/{representation}.npy"))
        xt = np.asarray(feature.array(f"test/{representation}.npy"))
        if len(xv) != len(vs) or len(xt) != len(ts):
            raise ValueError("feature/sample identity mismatch")
        if representation == "token":
            if xv.ndim != 4 or xt.shape[1:] != xv.shape[1:]:
                raise ValueError("invalid token features")
            count = xv.shape[1]
            vs = [row for row in vs for _ in range(count)]
            ts = [row for row in ts for _ in range(count)]
            xv = xv.reshape(-1, *xv.shape[-2:])
            xt = xt.reshape(-1, *xt.shape[-2:])
            bv = position_basis(
                valid.array("token/positions.npy"), valid.manifest["grid"]
            )
            bt = position_basis(
                test.array("token/positions.npy"), test.manifest["grid"]
            )
        else:
            bv = nuisance_basis(vs, protocol.dataset)
            bt = nuisance_basis(ts, protocol.dataset)
        yv = {
            f"{gap}:{target}": np.asarray(
                valid.array(f"{representation}/gap{gap}_{target}.npy")
            ).reshape(-1)
            for gap in protocol.gaps
            for target in system.targets
        }
        yt = {
            f"{gap}:{target}": np.asarray(
                test.array(f"{representation}/gap{gap}_{target}.npy")
            ).reshape(-1)
            for gap in protocol.gaps
            for target in system.targets
        }
        ridge = fit_ridge_many(xv, yv, vs)
        controls = fit_ridge_many(bv[:, None, :], yv, vs)
        combined = None
        if representation == "pooled":
            cv = np.concatenate(
                [xv, np.repeat(bv[:, None, :], xv.shape[1], axis=1)], axis=2
            )
            ct = np.concatenate(
                [xt, np.repeat(bt[:, None, :], xt.shape[1], axis=1)], axis=2
            )
            combined = fit_ridge_many(cv, yv, vs)
        for gap in protocol.gaps:
            for target in system.targets:
                key = f"{gap}:{target}"
                base = dict(
                    family="physics",
                    representation=representation,
                    gap=gap,
                    target=target,
                )
                ridge_row, ridge_fit = _score_fit(
                    ridge[key], xt, yt[key], base, "ridge"
                )
                mlp = fit_mlp(
                    xv,
                    yv[key],
                    vs,
                    max_steps=mlp_max_steps,
                    min_steps=mlp_min_steps,
                    include_depth=representation == "pooled",
                )
                mlp_row, mlp_fit = _score_fit(mlp, xt, yt[key], base, "mlp")
                rows.extend((ridge_row, mlp_row))
                selection = selected_family(ridge_row, mlp_row)
                if selection:
                    selected = {
                        k: v for k, v in selection.items() if k != "depth_curve"
                    }
                    selected.update(
                        method="selected", selected_method=selection["method"]
                    )
                    rows.append(selected)
                else:
                    rows.append({**ridge_row, "method": "selected"})
                control, _ = _score_fit(
                    controls[key],
                    bt[:, None, :],
                    yt[key],
                    base,
                    "regime_time" if representation == "pooled" else "position",
                )
                control["shared"] = True
                rows.append(control)
                if combined is not None:
                    rows.append(
                        _score_fit(
                            combined[key], ct, yt[key], base, "ridge_plus_control"
                        )[0]
                    )
                if gap > 0:
                    rows.append(
                        {
                            **base,
                            "method": "persistence",
                            "shared": True,
                            "status": "ok",
                            **metrics(yt[f"0:{target}"], yt[key]),
                        }
                    )
                if gap == 0 and representation == "pooled":
                    for row, fit in ((ridge_row, ridge_fit), (mlp_row, mlp_fit)):
                        if fit is not None:
                            saved[cell_id(row)] = dict(
                                fit=fit,
                                layer=row["selected_layer"],
                                target=target,
                                method=row["method"],
                            )
                print(f"fit {representation} gap {gap} {target}", flush=True)
    # Trajectory-averaged features for time-invariant regime parameters.
    vs = valid.json("pooled/samples.json")
    ts = test.json("pooled/samples.json")
    xv, vs = trajectory_average(feature.array("valid/pooled.npy"), vs)
    xt, ts = trajectory_average(feature.array("test/pooled.npy"), ts)
    yv = np.stack([system.regime_values(row["parameters"]) for row in vs])
    yt = np.stack([system.regime_values(row["parameters"]) for row in ts])
    names = [("log10_" if system.log_parameters else "") + p for p in system.parameters]
    fits = fit_ridge_many(xv, {name: yv[:, i] for i, name in enumerate(names)}, vs)
    for i, name in enumerate(names):
        base = dict(family="regime", representation="pooled", gap=0, target=name)
        ridge_row, _ = _score_fit(fits[name], xt, yt[:, i], base, "ridge")
        mlp = fit_mlp(
            xv,
            yv[:, i],
            vs,
            max_steps=mlp_max_steps,
            min_steps=mlp_min_steps,
            include_depth=False,
            # Preserve the workshop regime check: use the Ridge-selected layer.
            # Physical targets above select MLP layers independently.
            candidate_layers=()
            if ridge_row["selected_layer"] is None
            else (ridge_row["selected_layer"],),
        )
        mlp_row, _ = _score_fit(mlp, xt, yt[:, i], base, "mlp")
        rows.extend((ridge_row, mlp_row))
    for row in rows:
        row["cell_id"] = cell_id(row)
    with staged_directory(output) as stage:
        write_json(stage / "rows.json", rows)
        torch.save(saved, stage / "probes.pt")
        seal(
            stage,
            "probes",
            protocol=protocol.to_dict(),
            checkpoint=feature.manifest["checkpoint"],
            caches=feature.manifest["caches"],
            features=feature.manifest["sha256"],
            probe_settings=dict(
                ridge_alphas=list(RIDGE_ALPHAS),
                mlp_max_steps=mlp_max_steps,
                mlp_min_steps=mlp_min_steps,
                mlp_hidden=128,
                mlp_dropout=0.1,
                mlp_lr=0.01,
                mlp_weight_decay=1e-4,
                probe_seeds=list(range(5)),
            ),
        )
    return Path(output)


def evaluate_noise(feature_dir, cache_root, probe_dir, output):
    feature, valid, test = _features_and_caches(feature_dir, cache_root)
    probes = Artifact(probe_dir, "probes")
    if (
        probes.manifest["features"] != feature.manifest["sha256"]
        or probes.manifest["caches"] != feature.manifest["caches"]
    ):
        raise ValueError("saved probes belong to different features/caches")
    protocol = Protocol.from_dict(feature.manifest["protocol"])
    states = torch.load(
        probes.file("probes.pt"), map_location="cpu", weights_only=False
    )
    clean_rows = probes.json("rows.json")
    originals = {row["cell_id"]: row for row in clean_rows}
    rows = []
    for identity, state in states.items():
        target = np.asarray(test.array(f"pooled/gap0_{state['target']}.npy"))
        for sigma in protocol.noise_sigmas:
            scores = []
            for seed in protocol.noise_seeds:
                name = (
                    "test/pooled.npy"
                    if sigma == 0
                    else f"test/noise_{sigma:g}_{seed}.npy"
                )
                x = feature.array(name)
                scores.append(
                    metrics(predict(state["fit"], x[:, state["layer"]]), target)
                )
            row = dict(
                family="noise",
                representation="pooled",
                gap=0,
                target=state["target"],
                sigma=sigma,
                method=state["method"],
                selected_layer=state["layer"],
                status="ok",
                valid_cv_r2=originals[identity]["valid_cv_r2"],
                metric_status=scores[0]["metric_status"],
            )
            for metric in ("test_r2", "test_pearson_r", "test_mse", "test_log_mse"):
                values = np.asarray([s[metric] for s in scores], dtype=float)
                row[metric] = float(values.mean())
                row[metric + "_per_corruption_seed"] = values.tolist()
                if sigma == 0:
                    expected = originals[identity].get(metric)
                    if expected is None:
                        if np.isfinite(row[metric]):
                            raise ValueError("zero-noise metric became defined")
                    elif not np.isclose(row[metric], expected, atol=1e-10, rtol=1e-8):
                        raise ValueError(
                            f"zero-noise replay differs: {identity} {metric}"
                        )
            rows.append(row)
    for clean in clean_rows:
        if (
            clean["family"] == "physics"
            and clean["representation"] == "pooled"
            and clean["gap"] == 0
            and clean["method"] == "selected"
            and clean.get("selected_method")
        ):
            for row in list(rows):
                if (
                    row["target"] == clean["target"]
                    and row["method"] == clean["selected_method"]
                ):
                    rows.append(
                        {
                            **row,
                            "method": "selected",
                            "selected_method": clean["selected_method"],
                        }
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
            caches=feature.manifest["caches"],
            features=feature.manifest["sha256"],
            selection=probes.manifest["sha256"],
            probe_settings=probes.manifest["probe_settings"],
        )
    return Path(output)
