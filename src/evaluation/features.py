"""Unmasked layer features and deterministic, paired input corruption.

Attentive probes need the complete frozen token sequence at every encoder
output, which cannot be held in memory for every sample and every layer at
once. Extraction therefore runs the frozen encoder once per batch and appends
each encoder output to its own float16 shard; fitting and scoring stage a
single shard at a time. Each shard is produced only from the cached
eight-frame input context, so no future frame ever reaches a probe input.
"""

from pathlib import Path

import numpy as np
import torch

from src.evaluation.artifacts import seal, staged_directory
from src.evaluation.cache import open_caches
from src.evaluation.protocol import Protocol
from src.models.checkpoints import load_encoder

FLOAT16_MAX = float(np.finfo(np.float16).max)


@torch.no_grad()
def layerwise_states(encoder, clip):
    """Every encoder output for one batch: 12 block outputs plus the final norm."""
    x = encoder.tokenize(clip)
    states = []
    for block in encoder.blocks:
        x = block(x)
        states.append(x.float())
    states.append(encoder.norm(x).float())
    return states


def pooled_features(states):
    return torch.stack([state.mean(1) for state in states], dim=1).cpu()


def _float16_safe(state, kind, layer):
    """Refuse to silently round an encoder output away in a float16 shard."""
    if not torch.isfinite(state).all():
        raise ValueError(
            f"non-finite {kind} encoder output {layer}: refusing to shard it"
        )
    observed = float(state.abs().max())
    if observed > FLOAT16_MAX:
        raise ValueError(
            f"{kind} encoder output {layer} reaches |{observed:g}|, beyond the "
            f"float16 range {FLOAT16_MAX:g}; shard it in a wider dtype"
        )
    return observed


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


def shard_name(layer):
    return f"tokens_layer{layer}.npy"


def _batches(contexts, batch_size, device):
    for start in range(0, len(contexts), batch_size):
        stop = min(start + batch_size, len(contexts))
        clip = torch.from_numpy(np.array(contexts[start:stop])).float().to(device)
        yield start, stop, clip


def extract_features(
    checkpoint, cache_root, feature_root, split, batch_size=4, include_noise=False
):
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if split not in ("train", "valid", "test") or (include_noise and split != "test"):
        raise ValueError("noise extraction is available only for test")
    cache = open_caches(cache_root, (split,))[0]
    protocol = Protocol.from_dict(cache.manifest["protocol"])
    encoder, config, metadata = load_encoder(checkpoint)
    expected = dict(
        n_channels=len(cache.manifest["source"]["channels"]),
        n_frames=protocol.n_frames,
        height=cache.manifest["source"]["shape"][0],
        width=cache.manifest["source"]["shape"][1],
    )
    patch = tuple(config["encoder"][f"patch_{axis}"] for axis in ("t", "h", "w"))
    if (
        metadata["dataset"] != protocol.dataset
        or metadata["spec"] != expected
        or patch != protocol.patch
    ):
        raise ValueError("checkpoint dataset or input geometry differs from cache")
    grid = tuple(cache.manifest["grid"])
    n_tokens = int(np.prod(grid))
    n_layers = len(encoder.blocks) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    destination = (
        Path(feature_root)
        / f"{metadata['objective']}_seed{metadata['seed']}_{metadata['sha256'][:12]}"
        / split
    )
    shapes, maxima = {}, {}
    with staged_directory(destination) as stage:
        for kind in ("pooled", "token"):
            folder = stage / kind
            folder.mkdir()
            contexts = cache.array(f"{kind}/context.npy")
            # Verify sample ordering as well as the array contents.
            samples = cache.json(f"{kind}/samples.json")
            if len(samples) != len(contexts):
                raise ValueError("context/sample length mismatch")
            shapes[kind] = [len(contexts), n_tokens, encoder.embed_dim]
            shards = [
                np.lib.format.open_memmap(
                    folder / shard_name(layer),
                    mode="w+",
                    dtype=np.float16,
                    shape=(len(contexts), n_tokens, encoder.embed_dim),
                )
                for layer in range(n_layers)
            ]
            pooled = np.lib.format.open_memmap(
                folder / "pooled.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(contexts), n_layers, encoder.embed_dim),
            )
            maxima[kind] = [0.0] * n_layers
            for start, stop, clip in _batches(contexts, batch_size, device):
                states = layerwise_states(encoder, clip)
                if states[0].shape[1] != n_tokens:
                    raise ValueError("encoder token count differs from the cache grid")
                pooled[start:stop] = pooled_features(states).numpy()
                for layer, (shard, state) in enumerate(zip(shards, states)):
                    maxima[kind][layer] = max(
                        maxima[kind][layer], _float16_safe(state, kind, layer)
                    )
                    shard[start:stop] = state.half().cpu().numpy()
            pooled.flush()
            for shard in shards:
                shard.flush()
            print(
                f"{metadata['objective']} seed {metadata['seed']}: "
                f"extracted {split}/{kind}",
                flush=True,
            )
        if include_noise:
            contexts = cache.array("pooled/context.npy")
            for sigma in protocol.noise_sigmas:
                if sigma == 0:
                    continue  # The clean array is the exact zero-noise endpoint.
                for seed in protocol.noise_seeds:
                    output = np.lib.format.open_memmap(
                        stage / "pooled" / f"noise_{sigma:g}_{seed}.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(contexts), n_layers, encoder.embed_dim),
                    )
                    for start, stop, clip in _batches(contexts, batch_size, device):
                        noisy = paired_noise_batch(clip, sigma, seed, range(start, stop))
                        output[start:stop] = pooled_features(
                            layerwise_states(encoder, noisy)
                        ).numpy()
                    output.flush()
        seal(
            stage,
            "features",
            protocol=protocol.to_dict(),
            checkpoint=metadata,
            caches={cache.manifest["split"]: cache.manifest["sha256"]},
            batch_size=batch_size,
            split=split,
            include_noise=include_noise,
            grid=list(grid),
            shards=dict(
                dtype="float16",
                layers=n_layers,
                pooled=shapes["pooled"],
                token=shapes["token"],
                max_abs=maxima,
                max_finite_abs=FLOAT16_MAX,
            ),
            pooled_dtype="float32",
            token_sample_features="gathered_from_layer_shards",
        )
    return destination.parent
