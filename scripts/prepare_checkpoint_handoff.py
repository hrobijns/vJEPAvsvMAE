#!/usr/bin/env python3
"""Copy retained encoders byte-for-byte and index their training provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.evaluation.artifacts import (
    canonical_hash,
    seal,
    sha256_file,
    staged_directory,
    write_json,
)
from src.models.checkpoints import load_encoder


def prepare(training_manifest, output):
    manifest = json.loads(Path(training_manifest).read_text())
    rows = []
    with staged_directory(output) as stage:
        for run in manifest["runs"]:
            source = Path(run["run_dir"])
            if run["seed"] != 1:
                continue
            candidates = [
                source / f"encoder_{p:03d}pct.pt" for p in (25, 50, 75, 100)
            ] + [source / "encoder_best_val.pt"]
            for path in candidates:
                encoder, config, meta = load_encoder(path)
                if (meta["dataset"], meta["objective"], meta["seed"]) != (
                    run["dataset"],
                    run["objective"],
                    run["seed"],
                ):
                    raise ValueError(f"checkpoint/run identity differs: {path}")
                if not 0 < meta["step"] <= meta["training_protocol"]["total_steps"]:
                    raise ValueError(f"invalid checkpoint step: {path}")
                tensors = {}
                for name, value in encoder.state_dict().items():
                    if not torch.isfinite(value).all():
                        raise ValueError(f"nonfinite encoder tensor: {path} {name}")
                    tensors[name] = dict(
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                        sha256=hashlib.sha256(
                            value.cpu().numpy().tobytes()
                        ).hexdigest(),
                    )
                relative = Path(run["dataset"]) / run["objective"] / path.name
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                if sha256_file(destination) != meta["sha256"]:
                    raise ValueError(f"copied checkpoint differs: {path}")
                rows.append(
                    dict(
                        path=str(relative),
                        dataset=meta["dataset"],
                        objective=meta["objective"],
                        seed=meta["seed"],
                        step=meta["step"],
                        candidate=path.stem.removeprefix("encoder_"),
                        bytes=destination.stat().st_size,
                        sha256=meta["sha256"],
                        encoder_state_sha256=canonical_hash(tensors),
                        config_sha256=meta["config_sha256"],
                        spec=meta["spec"],
                        training_protocol=meta["training_protocol"],
                        training_identity=meta["training_identity"],
                        data_exposure=meta["data_exposure"],
                    )
                )
                print(f"verified {relative}: step {meta['step']}", flush=True)
        write_json(
            stage / "index.json",
            dict(training_provenance=manifest["provenance"], checkpoints=rows),
        )
        (stage / "README.md").write_text("""# ICLR study: seed-1 encoder candidates

All retained encoders from the twelve 100,000-step training runs: three
systems and four objectives. Each run supplies the 25%, 50%, 75%, and 100%
milestones plus its minimum-pretraining-validation-loss encoder. These are
candidates, not downstream-selected final models. `index.json` records actual
steps, checksums, input geometry, exposure, and training provenance. Identical
encoder-state and configuration hashes identify equivalent candidates.

Files preserve the original float32 checkpoint payloads byte-for-byte and are
tracked by the repository's existing `checkpoints/**/*.pt` Git LFS rule.
Training continuation states and feature/probe caches are not included.

To avoid fetching every checkpoint on clone, use `GIT_LFS_SKIP_SMUDGE=1 git
clone REPOSITORY`, then fetch only a needed system from the repository root:

```bash
git lfs pull --include="checkpoints/iclr2027/seed1/shear_flow/**/*.pt"
```

Load an encoder through the repository's normal loader:

```python
from src.models.checkpoints import load_encoder
encoder, config, metadata = load_encoder(
    "checkpoints/iclr2027/seed1/shear_flow/jepa/encoder_100pct.pt"
)
```

The encoder takes normalized, channel-first eight-frame clips. Obtain input
normalization and channel ordering from `WellSource` and `normalize` in
`src.data.source`; use the matching dataset and native resolution. Global
features average tokens; local features retain the sampled token positions.
The evaluation commands and exact target definitions are in the main README
and methods documentation. All checkpoint and probe choices use validation
data before test scoring.

The payloads total approximately 5 GiB. Lossless compression saved only 7–8%
in representative checks and is not used. Confirm the repository owner's
remaining LFS storage and download allowance before uploading. Nothing in
this preparation script pushes files or changes GitHub billing settings.
""")
        seal(
            stage,
            "encoder_handoff",
            training_provenance=manifest["provenance"],
            count=len(rows),
            bytes=sum(r["bytes"] for r in rows),
        )
    return Path(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(prepare(args.training_manifest, args.output))
