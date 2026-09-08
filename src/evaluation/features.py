"""Unmasked layer features and deterministic, paired input corruption."""

from pathlib import Path

import numpy as np
import torch

from src.evaluation.artifacts import seal, staged_directory
from src.evaluation.cache import open_caches
from src.evaluation.protocol import Protocol
from src.models.checkpoints import load_encoder


@torch.no_grad()
def layerwise_features(encoder, clip, token_positions=None):
    x = encoder.tokenize(clip)
    states = []
    for block in encoder.blocks:
        x = block(x)
        states.append(x.float())
    states.append(encoder.norm(x).float())
    pooled = torch.stack([state.mean(1) for state in states], dim=1).cpu()
    if token_positions is None:
        return pooled, None
    positions = torch.tensor(
        np.array(token_positions), dtype=torch.long, device=x.device
    )
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(x.shape[0], -1)
    selected = [
        torch.gather(state, 1, positions.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        for state in states
    ]
    return pooled, torch.stack(selected, dim=2).cpu()


def paired_noise_batch(clips, sigma, corruption_seed, sample_indices):
    if len(clips) != len(sample_indices):
        raise ValueError("noise sample IDs do not align")
    output = clips.clone()
    if sigma == 0:
        return output
    for i, sample in enumerate(sample_indices):
        seed = int(
            np.random.SeedSequence(
                [20260825, int(corruption_seed), int(sample)]
            ).generate_state(1)[0]
        )
        generator = torch.Generator(device=clips.device).manual_seed(seed)
        output[i].add_(
            torch.randn(
                clips[i].shape,
                dtype=clips.dtype,
                device=clips.device,
                generator=generator,
            ),
            alpha=float(sigma),
        )
    return output


def extract_features(checkpoint, cache_root, feature_root, batch_size=4):
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    caches = open_caches(cache_root)
    protocol = Protocol.from_dict(caches[0].manifest["protocol"])
    encoder, config, metadata = load_encoder(checkpoint)
    expected = dict(
        n_channels=len(caches[0].manifest["source"]["channels"]),
        n_frames=protocol.n_frames,
        height=caches[0].manifest["source"]["shape"][0],
        width=caches[0].manifest["source"]["shape"][1],
    )
    patch = tuple(config["encoder"][f"patch_{axis}"] for axis in ("t", "h", "w"))
    if (
        metadata["dataset"] != protocol.dataset
        or metadata["spec"] != expected
        or patch != protocol.patch
    ):
        raise ValueError("checkpoint dataset or input geometry differs from cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    destination = (
        Path(feature_root)
        / f"{metadata['objective']}_seed{metadata['seed']}_{metadata['sha256'][:12]}"
    )
    with staged_directory(destination) as stage:
        for cache in caches:
            split = cache.manifest["split"]
            output_dir = stage / split
            output_dir.mkdir()
            for kind in ("pooled", "token"):
                contexts = cache.array(f"{kind}/context.npy")
                positions = (
                    cache.array("token/positions.npy") if kind == "token" else None
                )
                # Verify sample ordering as well as the array contents.
                samples = cache.json(f"{kind}/samples.json")
                if len(samples) != len(contexts):
                    raise ValueError("context/sample length mismatch")
                shapes = (len(contexts), len(encoder.blocks) + 1, encoder.embed_dim)
                if kind == "token":
                    shapes = (len(contexts), protocol.token_samples, *shapes[1:])
                output = np.lib.format.open_memmap(
                    output_dir / f"{kind}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=shapes,
                )
                for start in range(0, len(contexts), batch_size):
                    stop = min(start + batch_size, len(contexts))
                    clip = (
                        torch.from_numpy(np.array(contexts[start:stop]))
                        .float()
                        .to(device)
                    )
                    pooled, tokens = layerwise_features(
                        encoder,
                        clip,
                        None if positions is None else positions[start:stop],
                    )
                    output[start:stop] = (
                        pooled if kind == "pooled" else tokens
                    ).numpy()
                output.flush()
            if split == "test":
                contexts = cache.array("pooled/context.npy")
                for sigma in protocol.noise_sigmas:
                    if sigma == 0:
                        continue  # The clean feature array is the exact zero-noise endpoint.
                    for seed in protocol.noise_seeds:
                        output = np.lib.format.open_memmap(
                            output_dir / f"noise_{sigma:g}_{seed}.npy",
                            mode="w+",
                            dtype=np.float32,
                            shape=(
                                len(contexts),
                                len(encoder.blocks) + 1,
                                encoder.embed_dim,
                            ),
                        )
                        for start in range(0, len(contexts), batch_size):
                            stop = min(start + batch_size, len(contexts))
                            clip = (
                                torch.from_numpy(np.array(contexts[start:stop]))
                                .float()
                                .to(device)
                            )
                            noisy = paired_noise_batch(
                                clip, sigma, seed, range(start, stop)
                            )
                            output[start:stop] = layerwise_features(encoder, noisy)[
                                0
                            ].numpy()
                        output.flush()
            print(
                f"{metadata['objective']} seed {metadata['seed']}: extracted {split}",
                flush=True,
            )
        seal(
            stage,
            "features",
            protocol=protocol.to_dict(),
            checkpoint=metadata,
            caches={
                cache.manifest["split"]: cache.manifest["sha256"] for cache in caches
            },
            batch_size=batch_size,
        )
    return destination
