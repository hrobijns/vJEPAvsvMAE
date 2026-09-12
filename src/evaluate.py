"""Prepare, extract, probe, and report frozen encoders on The Well."""

import argparse
import json

import yaml

from src.evaluation.cache import prepare_cache
from src.evaluation.features import extract_features
from src.evaluation.pipeline import evaluate_noise, fit_probes, score_probes
from src.evaluation.selection import select_checkpoints
from src.evaluation.protocol import Protocol
from src.evaluation.reporting import aggregate, plot
from src.models.checkpoints import load_encoder
from src.objectives import OBJECTIVES


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-checkpoints")
    validate.add_argument("checkpoints", nargs="+")
    cache = sub.add_parser("prepare-cache")
    cache.add_argument("--base", required=True)
    cache.add_argument(
        "--config",
        required=True,
        help="evaluation YAML with dataset and explicit frame_limit",
    )
    cache.add_argument("--split", choices=("train", "valid", "test"), required=True)
    cache.add_argument("--cache-root", required=True)
    extract = sub.add_parser("extract-features")
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--cache-root", required=True)
    extract.add_argument("--feature-root", required=True)
    extract.add_argument("--batch-size", type=int, default=4)
    extract.add_argument("--split", choices=("train", "valid", "test"), required=True)
    extract.add_argument("--include-noise", action="store_true")
    fit = sub.add_parser("fit-probes")
    fit.add_argument("--feature-dir", required=True)
    fit.add_argument("--cache-root", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--mlp-max-steps", type=int, default=2000)
    fit.add_argument("--mlp-min-steps", type=int, default=150)
    selection = sub.add_parser("select-checkpoints")
    selection.add_argument("paths", nargs="+")
    selection.add_argument("--output", required=True)
    score = sub.add_parser("score-probes")
    for name in ("feature-dir", "cache-root", "probe-dir", "selection", "output"):
        score.add_argument("--" + name, required=True)
    noise = sub.add_parser("evaluate-noise")
    noise.add_argument("--feature-dir", required=True)
    noise.add_argument("--cache-root", required=True)
    noise.add_argument("--probe-dir", required=True)
    noise.add_argument("--output", required=True)
    combine = sub.add_parser("aggregate")
    combine.add_argument("paths", nargs="+")
    combine.add_argument("--output", required=True)
    combine.add_argument("--kind", choices=("probes", "noise"), default="probes")
    combine.add_argument("--objectives", nargs="+", default=list(OBJECTIVES))
    combine.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    figures = sub.add_parser("plot")
    figures.add_argument("--aggregate-dir", required=True)
    figures.add_argument("--output", required=True)
    figures.add_argument("--metric", choices=("vrmse", "r2"), default="vrmse")
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    command = args.pop("command")
    if command == "validate-checkpoints":
        print(
            json.dumps(
                [load_encoder(path)[2] for path in args["checkpoints"]], indent=2
            )
        )
        return
    if command == "prepare-cache":
        with open(args.pop("config")) as handle:
            config = yaml.safe_load(handle)
        if (
            not isinstance(config, dict)
            or "frame_limit" not in config
            or set(config) - set(Protocol.__dataclass_fields__)
        ):
            raise ValueError(
                "evaluation config must declare frame_limit and only supported protocol fields"
            )
        args["protocol"] = Protocol.from_dict(config)
    functions = {
        "prepare-cache": prepare_cache,
        "extract-features": extract_features,
        "fit-probes": fit_probes,
        "select-checkpoints": select_checkpoints,
        "score-probes": score_probes,
        "evaluate-noise": evaluate_noise,
        "aggregate": aggregate,
        "plot": plot,
    }
    print(functions[command](**args))


if __name__ == "__main__":
    main()
