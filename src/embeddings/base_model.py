"""Uniform access to the hidden representations of SSL base models.

Every supported base model is a Vision Transformer, loaded through ``timm`` or through
Hugging Face ``transformers``. :class:`SSLBaseModel` hides the differences (prefix tokens,
register tokens, normalisation constants, API) and returns the same structure for all of
them:

* the class token - one vector per image;
* the patch tokens of the last layer as a feature map ``[B, D, h, w]``;
* the patch tokens of any intermediate layers, same shape (used by the segmentation decoder).

Inputs are RGB float tensors in ``[0, 1]`` whose height and width are multiples of the patch
size. Cutting such inputs out of a 1024 px tile is the job of ``windows.py``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import torch
from torch import nn

if TYPE_CHECKING:
    import numpy as np

from .registry import IMAGENET_MEAN, IMAGENET_STD, POOLINGS, BaseModelSpec, resolve_spec

logger = logging.getLogger(__name__)


class BaseModelAccessError(RuntimeError):
    """The weights could not be downloaded (gated repository, missing token, no network)."""


@dataclass
class HiddenRepresentations:
    """What a base model computed for a batch of images."""

    patch_tokens: torch.Tensor  # [B, D, h, w], last layer, after the encoder's final norm
    cls_token: torch.Tensor | None = None  # [B, D]
    register_tokens: torch.Tensor | None = None  # [B, R, D]
    hidden_states: dict[int, torch.Tensor] = field(default_factory=dict)  # block index -> [B, D, h, w]

    @property
    def grid(self) -> tuple[int, int]:
        """Token grid ``(h, w)``."""
        return int(self.patch_tokens.shape[-2]), int(self.patch_tokens.shape[-1])

    def pooled(self, pooling: str = "cls") -> torch.Tensor:
        """One vector per image: ``cls``, ``mean`` of the patch tokens, or both concatenated."""
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        mean = self.patch_tokens.mean(dim=(-2, -1))
        if pooling == "mean" or self.cls_token is None:
            return mean
        if pooling == "cls":
            return self.cls_token
        return torch.cat([self.cls_token, mean], dim=-1)


class SSLBaseModel(nn.Module):
    """A frozen-by-default SSL encoder with a backend-independent interface."""

    def __init__(
        self,
        spec: BaseModelSpec,
        net: nn.Module,
        adapter: "_Adapter",
        mean: Sequence[float],
        std: Sequence[float],
    ):
        super().__init__()
        self.spec = spec
        self.net = net
        self._adapter = adapter
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self._frozen = False

    # ---- facts read from the loaded network -------------------------------------------

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def patch_size(self) -> int:
        return self._adapter.patch_size

    @property
    def embed_dim(self) -> int:
        return self._adapter.embed_dim

    @property
    def num_layers(self) -> int:
        return self._adapter.num_layers

    @property
    def num_prefix_tokens(self) -> int:
        """Class token plus register tokens that precede the patch tokens."""
        return self._adapter.num_prefix_tokens

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def output_dim(self, pooling: str | None = None) -> int:
        """Width of :meth:`embed` for the given pooling."""
        pooling = pooling or self.spec.tile_pooling
        both = pooling == "cls+mean" and self._adapter.has_cls
        return self.embed_dim * (2 if both else 1)

    def default_layers(self, count: int = 4) -> tuple[int, ...]:
        """``count`` evenly spaced block indices ending at the last block (e.g. 2, 5, 8, 11 of 12)."""
        if count > self.num_layers:
            raise ValueError(f"asked for {count} layers, but the encoder has {self.num_layers}")
        return tuple((i + 1) * self.num_layers // count - 1 for i in range(count))

    def fit_size(self, size: float) -> int:
        """Nearest input side that is a positive multiple of the patch size."""
        return max(self.patch_size, int(round(size / self.patch_size)) * self.patch_size)

    def describe(self) -> dict[str, Any]:
        """Facts of the instantiated network, for logs and run provenance."""
        return {
            "name": self.spec.name,
            "backend": self.spec.backend,
            "source": self.spec.source,
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "num_layers": self.num_layers,
            "num_prefix_tokens": self.num_prefix_tokens,
            "mean": [round(float(v), 6) for v in self.mean.flatten()],
            "std": [round(float(v), 6) for v in self.std.flatten()],
            "parameters": sum(p.numel() for p in self.net.parameters()),
            "license": self.spec.license,
        }

    # ---- training state -----------------------------------------------------------------

    def freeze(self) -> "SSLBaseModel":
        """Stop gradients and pin the encoder to eval mode (it stays there under ``.train()``)."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self._frozen = True
        self.eval()
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen

    def train(self, mode: bool = True) -> "SSLBaseModel":
        return super().train(mode and not self._frozen)

    # ---- forward ------------------------------------------------------------------------

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """``[0, 1]`` RGB -> the normalisation the encoder was trained with."""
        return (images.to(self.mean.dtype) - self.mean) / self.std

    def forward(
        self,
        images: torch.Tensor,
        layers: Iterable[int] = (),
        *,
        normalize: bool = True,
    ) -> HiddenRepresentations:
        """Encode ``images`` (``[B, 3, H, W]``, ``[0, 1]`` unless ``normalize=False``).

        ``layers`` lists the blocks whose patch tokens to return in ``hidden_states``
        (0-based, negatives count from the end). Their outputs are raw block outputs; only
        ``patch_tokens`` and ``cls_token`` pass through the encoder's final norm.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected images shaped [B, 3, H, W], got {tuple(images.shape)}")
        height, width = int(images.shape[-2]), int(images.shape[-1])
        patch = self.patch_size
        if height % patch or width % patch:
            raise ValueError(
                f"input {height}x{width} is not a multiple of the patch size {patch}; "
                "resize with fit_size() or plan the input with mpn_al.embeddings.windows"
            )
        wanted = sorted({int(layer) % self.num_layers for layer in layers})
        x = self.normalize(images) if normalize else images
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self._frozen):
            final, hidden = self._adapter.tokens(self.net, x, wanted)

        prefix = self.num_prefix_tokens
        batch, _, dim = final.shape
        grid = (height // patch, width // patch)
        patch_tokens = final[:, prefix:].reshape(batch, grid[0], grid[1], dim).permute(0, 3, 1, 2).contiguous()
        has_cls = self._adapter.has_cls
        return HiddenRepresentations(
            patch_tokens=patch_tokens,
            cls_token=final[:, 0] if has_cls else None,
            register_tokens=final[:, 1:prefix] if has_cls and prefix > 1 else None,
            hidden_states=hidden,
        )

    @torch.no_grad()
    def embed(self, images: torch.Tensor, pooling: str | None = None) -> torch.Tensor:
        """One embedding per image; ``pooling`` defaults to the model card's recommendation."""
        return self.forward(images).pooled(pooling or self.spec.tile_pooling)


def image_to_tensor(image: "np.ndarray", device: torch.device | str) -> torch.Tensor:
    """RGB uint8 ``[H, W, 3]`` -> float ``[1, 3, H, W]`` in ``[0, 1]`` on ``device``."""
    import numpy as np

    array = np.ascontiguousarray(image)
    if not array.flags.writeable:  # e.g. arrays backed by a PIL image; torch wants to own writable memory
        array = array.copy()
    return torch.from_numpy(array).to(device).permute(2, 0, 1)[None].float().div_(255.0)


# ---- backends -------------------------------------------------------------------------------


class _Adapter:
    """Reads a network's token layout and runs it; one subclass per backend."""

    patch_size: int
    embed_dim: int
    num_layers: int
    num_prefix_tokens: int
    has_cls: bool

    def tokens(self, net: nn.Module, x: torch.Tensor, layers: list[int]) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Return final tokens ``[B, prefix + N, D]`` and ``{layer: [B, D, h, w]}``."""
        raise NotImplementedError


class _TimmAdapter(_Adapter):
    def __init__(self, net: nn.Module):
        if not all(hasattr(net, name) for name in ("patch_embed", "blocks", "forward_intermediates")):
            raise TypeError(
                f"{type(net).__name__} is not a timm Vision Transformer with forward_intermediates(); "
                "only ViT-style base models are supported (timm>=1.0)"
            )
        patch = net.patch_embed.patch_size
        self.patch_size = int(patch[0] if isinstance(patch, (tuple, list)) else patch)
        self.embed_dim = int(net.embed_dim)
        self.num_layers = len(net.blocks)
        self.num_prefix_tokens = int(getattr(net, "num_prefix_tokens", 0))
        self.has_cls = getattr(net, "cls_token", None) is not None

    def tokens(self, net, x, layers):
        indices = layers or [self.num_layers - 1]
        final, intermediates = net.forward_intermediates(
            x, indices=indices, norm=False, output_fmt="NCHW", intermediates_only=False
        )
        hidden = dict(zip(indices, intermediates)) if layers else {}
        return final, hidden


class _TransformersAdapter(_Adapter):
    def __init__(self, net: nn.Module):
        config = net.config
        patch = config.patch_size
        self.patch_size = int(patch[0] if isinstance(patch, (tuple, list)) else patch)
        self.embed_dim = int(config.hidden_size)
        self.num_layers = int(config.num_hidden_layers)
        self.num_prefix_tokens = 1 + int(getattr(config, "num_register_tokens", 0) or 0)
        self.has_cls = True
        # ViTModel needs the flag for non-native sizes; DINOv2 models always interpolate.
        self._interpolate = "interpolate_pos_encoding" in inspect.signature(net.forward).parameters

    def tokens(self, net, x, layers):
        kwargs: dict[str, Any] = {"pixel_values": x, "output_hidden_states": bool(layers)}
        if self._interpolate:
            kwargs["interpolate_pos_encoding"] = True
        output = net(**kwargs)
        final = output.last_hidden_state
        batch, _, dim = final.shape
        grid = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)
        hidden = {}
        for layer in layers:
            # hidden_states[0] is the embedding output, so block `layer` is at index layer + 1.
            tokens = output.hidden_states[layer + 1][:, self.num_prefix_tokens :]
            hidden[layer] = tokens.reshape(batch, grid[0], grid[1], dim).permute(0, 3, 1, 2).contiguous()
        return final, hidden


# ---- construction ---------------------------------------------------------------------------


def build_base_model(
    model: str | BaseModelSpec,
    *,
    pretrained: bool = True,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    freeze: bool = True,
) -> SSLBaseModel:
    """Instantiate a base model by registry name, ad-hoc reference or spec.

    ``checkpoint_path`` loads local weights (for an in-house SSL encoder, or for hub models
    already downloaded to group storage) instead of the pretrained ones. ``pretrained=False``
    gives random weights and is meant for tests.
    """
    spec = resolve_spec(model)
    if spec.backend == "timm":
        net, mean, std = _build_timm(spec, pretrained, checkpoint_path)
        adapter: _Adapter = _TimmAdapter(net)
    else:
        net, mean, std = _build_transformers(spec, pretrained, checkpoint_path)
        adapter = _TransformersAdapter(net)

    base = SSLBaseModel(spec, net, adapter, mean, std)
    for label, expected, actual in (
        ("embed_dim", spec.embed_dim, base.embed_dim),
        ("patch_size", spec.patch_size, base.patch_size),
    ):
        if expected is not None and expected != actual:
            logger.warning("%s: registry says %s=%s but the loaded network has %s", spec.name, label, expected, actual)
    if freeze:
        base.freeze()
    if device is not None:
        base.to(device)
    return base


def _build_timm(spec: BaseModelSpec, pretrained: bool, checkpoint_path: str | Path | None):
    import timm

    kwargs: dict[str, Any] = {"num_classes": 0}
    kwargs.update(_resolve_objects(spec.create_kwargs))
    if spec.dynamic_img_size:
        kwargs.setdefault("dynamic_img_size", True)
    try:
        net = timm.create_model(spec.source, pretrained=pretrained and checkpoint_path is None, **kwargs)
    except Exception as exc:  # noqa: BLE001 - hub errors come in many types
        raise _access_error(spec, exc) or exc
    if checkpoint_path is not None:
        load_encoder_checkpoint(net, checkpoint_path)
    config = getattr(net, "pretrained_cfg", None) or {}
    mean = spec.mean or tuple(config.get("mean") or IMAGENET_MEAN)
    std = spec.std or tuple(config.get("std") or IMAGENET_STD)
    return net, mean, std


def _build_transformers(spec: BaseModelSpec, pretrained: bool, checkpoint_path: str | Path | None):
    try:
        from transformers import AutoConfig, AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise ImportError(f"{spec.name} needs the transformers backend: pip install 'mpn-al[hf]'") from exc

    kwargs = dict(spec.create_kwargs)
    hub_kwargs = {key: kwargs[key] for key in ("trust_remote_code",) if key in kwargs}
    try:
        if pretrained and checkpoint_path is None:
            net = AutoModel.from_pretrained(spec.source, **kwargs)
        else:
            net = AutoModel.from_config(AutoConfig.from_pretrained(spec.source, **hub_kwargs), **hub_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise _access_error(spec, exc) or exc
    if checkpoint_path is not None:
        load_encoder_checkpoint(net, checkpoint_path)

    mean, std = spec.mean, spec.std
    if mean is None or std is None:
        try:
            processor = AutoImageProcessor.from_pretrained(spec.source, **hub_kwargs)
            mean = mean or tuple(processor.image_mean)
            std = std or tuple(processor.image_std)
        except Exception:  # noqa: BLE001 - a missing processor config is not fatal
            logger.warning("%s: no image processor config found, assuming ImageNet normalisation", spec.name)
    return net, mean or IMAGENET_MEAN, std or IMAGENET_STD


def _resolve_objects(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Registry entries name layer classes as dotted strings (``timm.layers.SwiGLUPacked``)."""
    resolved = {}
    for key, value in kwargs.items():
        if key.endswith("_layer") and isinstance(value, str):
            module_name, _, attribute = value.rpartition(".")
            value = getattr(importlib.import_module(module_name), attribute)
        resolved[key] = value
    return resolved


def _access_error(spec: BaseModelSpec, exc: Exception) -> BaseModelAccessError | None:
    """Translate hub authentication failures into an actionable message; ``None`` for other errors."""
    text = f"{type(exc).__name__}: {exc}"
    markers = ("GatedRepo", "401", "403", "gated", "Unauthorized", "LocalEntryNotFound", "RepositoryNotFound")
    if not any(marker in text for marker in markers):
        return None
    steps = (
        f"request access at {spec.url or 'the model page'}, then run `hf auth login` "
        "(`huggingface-cli login` on older huggingface_hub) or set HF_TOKEN on the machine "
        "that downloads the weights"
    )
    error = BaseModelAccessError(
        f"could not download {spec.name!r} from {spec.source!r}. "
        + (f"The repository is gated: {steps}. " if spec.gated else "Check the network, HF_HOME and HF_TOKEN. ")
        + "On a cluster without internet on compute nodes, download on the login node first "
        "and run jobs with HF_HUB_OFFLINE=1. "
        f"Original error: {text}"
    )
    error.__cause__ = exc
    return error


_WRAPPER_KEYS = ("teacher", "model_ema", "ema", "model", "state_dict", "model_state_dict", "student", "encoder", "backbone")
_KEY_PREFIXES = ("module.", "backbone.", "encoder.", "model.", "net.", "teacher.", "student.")


def load_encoder_checkpoint(net: nn.Module, path: str | Path) -> dict[str, int]:
    """Load local encoder weights, tolerating the wrappers SSL training code adds.

    Checkpoints from DINO-style training nest the weights (``{"teacher": {...}}``) and prefix
    the keys (``backbone.``, ``module.``); projection heads are simply left out. Raises if no
    tensor fits the architecture.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state: Any = load_file(str(path))
    else:
        state = torch.load(str(path), map_location="cpu", weights_only=True)

    unwrapped = True
    while unwrapped:
        unwrapped = False
        for key in _WRAPPER_KEYS:
            if isinstance(state, dict) and isinstance(state.get(key), dict):
                state, unwrapped = state[key], True
                break

    target = net.state_dict()
    matched: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        stripped = True
        while stripped and key not in target:
            stripped = False
            for prefix in _KEY_PREFIXES:
                if key.startswith(prefix):
                    key, stripped = key[len(prefix) :], True
                    break
        if key in target and target[key].shape == value.shape:
            matched[key] = value

    if not matched:
        raise ValueError(f"{path}: no tensor matches {type(net).__name__}; wrong architecture or constructor arguments?")
    net.load_state_dict(matched, strict=False)
    report = {"loaded": len(matched), "missing": len(target) - len(matched), "unused": len(state) - len(matched)}
    log = logger.warning if report["missing"] else logger.info
    log("%s: loaded %d tensors, %d missing in the checkpoint, %d unused", path.name, *report.values())
    return report
