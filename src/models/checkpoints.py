"""Load encoder weights using the shape and architecture stored with them."""

from dataclasses import asdict

import torch

from src.data.well import ClipSpec
from src.evaluation.artifacts import canonical_hash, sha256_file
from src.models.vit import build_encoder


def load_encoder(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    spec = ClipSpec(**payload["spec"])
    if min(asdict(spec).values()) < 1 or config["objective_name"] not in (
        "jepa",
        "mae",
    ):
        raise ValueError("invalid encoder checkpoint")
    encoder = build_encoder(spec, config["encoder"])
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.eval()
    metadata = dict(
        dataset=config["data"]["dataset_name"],
        objective=config["objective_name"],
        seed=int(config["seed"]),
        step=int(payload["step"]),
        spec=asdict(spec),
        encoder=config["encoder"],
        sha256=sha256_file(path),
        training_protocol={
            "frame_limit": config["data"].get("frame_limit", "unrecorded"),
            "valid_stride": config["data"].get("valid_stride", "unrecorded"),
            "batch_size": config.get("optim", {}).get("batch_size"),
            "total_steps": config.get("optim", {}).get("total_steps"),
        },
        config_sha256=canonical_hash(config),
        training_identity=payload.get("training_identity"),
    )
    return encoder, config, metadata
