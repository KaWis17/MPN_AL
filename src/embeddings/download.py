"""Fetch base-model weights into the Hugging Face cache.

Compute nodes of many clusters (PLGrid included) have no internet, while login nodes do. Download on
the login node, then run jobs with ``HF_HUB_OFFLINE=1``::

    python scripts/download_base_models.py phikon-v2 lunit-dino-s16 --verify
    python scripts/download_base_models.py --open        # every registered model that needs no access request

Gated models need an approved Hugging Face account and a token on the machine (``hf auth login``).
Every person has to accept such a model's terms themselves, and a cache holding gated weights may
only be shared between people who did.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .base_model import _access_error
from .registry import BaseModelSpec, list_specs, load_registry_file, resolve_spec

logger = logging.getLogger("mpn_al.download")

# Formats nobody here loads, and (when safetensors exist) the duplicate pickled copies of the weights.
_NEVER = ["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.tflite", "flax_model*", "tf_model*", "rust_model*"]
_PICKLES = ["*.bin", "*.pt", "*.pth", "*.ckpt"]


@dataclass
class DownloadResult:
    name: str
    repo: str | None
    path: str | None
    size_bytes: int
    status: str  # ok | verified | local | denied | failed | verify-failed
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "verified", "local")


def hub_repo(spec: BaseModelSpec) -> str | None:
    """The Hugging Face repository holding the weights of ``spec``; ``None`` when ``source`` is a local path."""
    source = spec.source
    if Path(source).exists():
        return None
    if spec.backend == "transformers":
        return source
    for prefix in ("hf-hub:", "hf_hub:"):
        if source.startswith(prefix):
            return source[len(prefix) :]
    import timm

    config = timm.models.get_pretrained_cfg(source)  # e.g. 'vit_small_patch14_dinov2.lvd142m' -> timm/<same name>
    repo = getattr(config, "hf_hub_id", None) if config is not None else None
    if not repo:
        raise ValueError(f"cannot tell which Hugging Face repository holds the weights of timm model {source!r}")
    return repo


def download_model(model: str | BaseModelSpec, *, cache_dir: str | Path | None = None, revision: str | None = None) -> DownloadResult:
    """Download one model into the cache; never raises for hub problems, reports them in the result."""
    spec = resolve_spec(model)
    try:
        repo = hub_repo(spec)
    except ValueError as exc:
        return DownloadResult(spec.name, None, None, 0, "failed", str(exc))
    if repo is None:
        return DownloadResult(spec.name, None, spec.source, 0, "local", "weights are a local path")

    from huggingface_hub import HfApi, snapshot_download

    try:
        files = HfApi().list_repo_files(repo, revision=revision)
        ignore = list(_NEVER)
        if any(name.endswith(".safetensors") for name in files):
            ignore += _PICKLES
        path = snapshot_download(repo, revision=revision, cache_dir=cache_dir, ignore_patterns=ignore)
    except Exception as exc:  # noqa: BLE001 - hub errors come in many types
        error = _access_error(spec, exc)
        if error is not None:
            return DownloadResult(spec.name, repo, None, 0, "denied", str(error))
        return DownloadResult(spec.name, repo, None, 0, "failed", f"{type(exc).__name__}: {exc}")
    size = sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())
    return DownloadResult(spec.name, repo, str(path), size, "ok")


def verify_offline(model: str, cache_dir: str | Path | None = None) -> tuple[bool, str]:
    """Load ``model`` with the hub disabled, on the CPU, in a fresh process: proves the cache is complete."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "list_base_models.py"
    environment = dict(os.environ, HF_HUB_OFFLINE="1")
    if cache_dir is not None:
        environment["HF_HUB_CACHE"] = str(cache_dir)
    result = subprocess.run(
        [sys.executable, str(script), "--check", model, "--device", "cpu"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    tail = (result.stdout.strip().splitlines() or result.stderr.strip().splitlines() or [""])[-1]
    return result.returncode == 0, tail if result.returncode == 0 else (result.stderr.strip()[-600:] or tail)


def main_download(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download base-model weights into the Hugging Face cache (HF_HOME).")
    parser.add_argument("models", nargs="*", help="registry names, or timm:<name> / hf:<org/repo>")
    parser.add_argument("--open", action="store_true", help="all registered models that need no access request (large: includes ViT-g models)")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face hub cache directory (default: HF_HOME)")
    parser.add_argument("--revision", help="branch, tag or commit of the repositories")
    parser.add_argument("--verify", action="store_true", help="afterwards load every model offline on the CPU")
    parser.add_argument("--registry-file", type=Path, help="YAML file with additional base-model entries")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.registry_file is not None:
        load_registry_file(args.registry_file)
    names = list(args.models) + ([spec.name for spec in list_specs() if not spec.gated] if args.open else [])
    if not names:
        parser.error("name at least one model, or use --open")

    results = []
    for name in dict.fromkeys(names):
        logger.info("downloading %s ...", name)
        result = download_model(name, cache_dir=args.cache_dir, revision=args.revision)
        if result.status == "ok" and args.verify:
            loaded, detail = verify_offline(name, args.cache_dir)
            result.status, result.detail = ("verified", detail) if loaded else ("verify-failed", detail)
        results.append(result)

    print(f"{'status':13s} {'model':16s} {'repository':44s} {'size':>9s}")
    for result in results:
        print(f"{result.status:13s} {result.name:16s} {(result.repo or '-'):44s} {result.size_bytes / 2**30:8.2f}G")
        if result.detail and not result.ok or result.status == "verify-failed":
            print(f"    {result.detail}")
    return 0 if all(result.ok for result in results) else 1
