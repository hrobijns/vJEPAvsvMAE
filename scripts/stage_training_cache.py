"""Reuse one full training cache per machine; serialize and resume its copy."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.artifacts import canonical_hash


def stage(base, dataset, cache_root):
    source = Path(base).expanduser() / "memmap" / dataset / "train.npy"
    metadata = source.with_suffix(".meta.json").read_bytes()
    meta = json.loads(metadata)
    if (meta.get("sha256") != canonical_hash({k: v for k, v in meta.items() if k != "sha256"})
            or not meta.get("complete") or meta.get("dataset") != dataset or meta.get("split") != "train"):
        raise ValueError("invalid or incomplete source cache metadata")
    expected = meta["array_sha256"]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("invalid cache content hash")
    local_base = Path(cache_root).expanduser().resolve() / expected
    destination = local_base / "memmap" / dataset / "train.npy"
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # All jobs on this machine share the same lock and content-addressed file.
    with (local_base / ".copy.lock").open("a") as lock:
        print(f"Waiting for cache copy lock: {local_base}", file=sys.stderr, flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            if (destination.stat().st_size != source.stat().st_size
                    or destination.with_suffix(".meta.json").read_bytes() != metadata):
                raise ValueError("existing local cache differs from the requested source")
            # The normal training constructor hashes the complete array before use.
            return dict(base_path=str(local_base), reused=True, array_sha256=expected,
                        source_identity=meta["source_identity"], seconds=time.monotonic() - started)
        partial = destination.with_suffix(".partial.npy")
        offset = partial.stat().st_size if partial.exists() else 0
        size = source.stat().st_size
        if offset > size:
            raise ValueError("partial local cache is larger than its source")
        if shutil.disk_usage(local_base).free < size - offset + 2**30:
            raise OSError("not enough local disk space for the full training cache")
        digest = hashlib.sha256()
        if offset:
            with partial.open("rb") as handle:
                for block in iter(lambda: handle.read(16 << 20), b""):
                    digest.update(block)
        print(f"Copying full cache to {destination}; resuming at {offset}/{size} bytes", file=sys.stderr, flush=True)
        with source.open("rb") as src, partial.open("ab") as dst:
            src.seek(offset)
            copied, last_report = offset, offset
            for block in iter(lambda: src.read(16 << 20), b""):
                dst.write(block)
                digest.update(block)
                copied += len(block)
                if copied - last_report >= 8 * 2**30:
                    print(f"Cache copy: {copied / 2**30:.1f}/{size / 2**30:.1f} GiB", file=sys.stderr, flush=True)
                    last_report = copied
            dst.flush()
            os.fsync(dst.fileno())
        if partial.stat().st_size != size or digest.hexdigest() != expected:
            raise ValueError("local cache copy failed its content hash check")
        destination.with_suffix(".meta.json").write_bytes(metadata)
        partial.rename(destination)
        return dict(base_path=str(local_base), reused=False, resumed_bytes=offset,
                    array_sha256=expected, source_identity=meta["source_identity"],
                    seconds=time.monotonic() - started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()
    print(stage(args.base, args.dataset, args.cache_root)["base_path"])
