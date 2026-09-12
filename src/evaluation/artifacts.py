"""Immutable, content-checked analysis artifacts."""

import hashlib
import json
import math
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SCHEMA = "well-analysis-2"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_json(value):
    """Represent undefined metrics as JSON null, never nonstandard NaN."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects completed outputs from accidental reuse.
    with path.open("x") as handle:
        json.dump(finite_json(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def provenance():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src").rglob("*.py"))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "code_sha256": canonical_hash(
            {str(p.relative_to(root)): sha256_file(p) for p in files}
        ),
    }


@contextmanager
def staged_directory(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial.", dir=destination.parent)
    )
    try:
        yield stage
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
    except BaseException:
        # Keep interrupted work for inspection; it is never a completed artifact.
        print(f"incomplete output retained at {stage}", flush=True)
        raise


def seal(root, kind, **metadata):
    root = Path(root)
    manifest = {"schema": SCHEMA, "kind": kind, **metadata, "provenance": provenance()}
    manifest["files"] = {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    manifest["sha256"] = canonical_hash(manifest)
    write_json(root / "manifest.json", manifest)
    return manifest


class Artifact:
    """Validate the manifest and each file once per consuming process."""

    def __init__(self, root, kind):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        claim = self.manifest.get("sha256")
        body = {key: value for key, value in self.manifest.items() if key != "sha256"}
        if (
            self.manifest.get("schema") != SCHEMA
            or self.manifest.get("kind") != kind
            or canonical_hash(body) != claim
        ):
            raise ValueError(f"invalid {kind} manifest: {root}")
        self.verified = set()

    def file(self, relative):
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("artifact path escapes its root")
        expected = self.manifest["files"].get(str(relative))
        if expected is None:
            raise ValueError(f"unrecorded artifact file: {relative}")
        if relative not in self.verified:
            if sha256_file(path) != expected:
                raise ValueError(f"artifact content hash mismatch: {path}")
            self.verified.add(relative)
        return path

    def array(self, relative):
        return np.load(self.file(relative), mmap_mode="r", allow_pickle=False)

    def json(self, relative):
        return json.loads(self.file(relative).read_text())
