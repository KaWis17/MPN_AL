"""Registry of self-supervised (SSL) base models.

A :class:`BaseModelSpec` says *how to obtain* a base model (backend, hub id, constructor
arguments) and records what the model card states (input size, resolution, licence).
Patch size, embedding width and token layout are informational here: the loader reads the
real values from the instantiated network.

Normalisation is different: with ``mean``/``std`` left at ``None`` the loader trusts the
model's own hub config. That is right for repositories published by the model's authors,
but community ports can carry placeholder values (the Lunit port does), so an entry for a
port should state the statistics from the original release.

Model-card facts were checked against the Hugging Face cards on 2026-11-16. Licences change
and gated repositories need an approved account, so re-read the card before relying on one.

Besides the built-in names, any ViT can be addressed ad hoc:

* ``timm:<timm name or hf-hub:org/repo>``, e.g. ``timm:vit_small_patch14_dinov2.lvd142m``
* ``hf:<org/repo>`` for a `transformers` ``AutoModel``

More entries can be added from YAML with :func:`load_registry_file`
(see ``configs/base_models/example.yaml``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Published by Bioptimus for H-optimus-0 and H-optimus-1.
H_OPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
H_OPTIMUS_STD = (0.211883, 0.230117, 0.177517)
# Release notes of lunit-io/benchmark-ssl-pathology ("determined from 10% of the training samples").
LUNIT_MEAN = (0.70322989, 0.53606487, 0.66096631)
LUNIT_STD = (0.21716536, 0.26081574, 0.20723464)

BACKENDS = ("timm", "transformers")
POOLINGS = ("cls", "mean", "cls+mean")


@dataclass(frozen=True)
class BaseModelSpec:
    """How to load one SSL base model, and what its card says about it."""

    name: str
    backend: str  # "timm" | "transformers"
    source: str  # timm model name / "hf-hub:org/repo", or a Hugging Face repo id
    architecture: str = ""
    ssl_method: str = ""
    training_data: str = ""
    embed_dim: int | None = None
    patch_size: int | None = None
    num_register_tokens: int | None = None
    native_size: int = 224  # side (px) of the input the model was trained on
    native_mpp: float = 0.5  # microns per pixel of that input
    mean: tuple[float, float, float] | None = None  # None: take it from the model's own config
    std: tuple[float, float, float] | None = None
    tile_pooling: str = "cls"  # tile embedding recommended by the card: "cls" | "mean" | "cls+mean"
    create_kwargs: Mapping[str, Any] = field(default_factory=dict)
    dynamic_img_size: bool = True  # timm: interpolate position embeddings for other input sizes
    gated: bool = False  # Hugging Face access request required
    license: str = ""
    license_notes: str = ""
    url: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"{self.name}: backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.tile_pooling not in POOLINGS:
            raise ValueError(f"{self.name}: tile_pooling must be one of {POOLINGS}, got {self.tile_pooling!r}")


_NC_ND = (
    "Non-commercial research only; weights may not be redistributed and modified or derived "
    "weights (fine-tuned, LoRA-merged, distilled) may not be shared. Keeping the encoder frozen "
    "and storing only our own head avoids the question."
)

_BUILTIN: tuple[BaseModelSpec, ...] = (
    BaseModelSpec(
        name="uni2-h",
        backend="timm",
        source="hf-hub:MahmoodLab/UNI2-h",
        architecture="ViT-H/14",
        ssl_method="DINOv2",
        training_data="200M+ tiles from 300k+ H&E and IHC slides (Mass General Brigham)",
        embed_dim=1536,
        patch_size=14,
        num_register_tokens=8,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        create_kwargs={
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "no_embed_class": True,
            "mlp_layer": "timm.layers.SwiGLUPacked",
            "act_layer": "torch.nn.SiLU",
            "reg_tokens": 8,
        },
        gated=True,
        license="CC-BY-NC-ND-4.0",
        license_notes=_NC_ND,
        url="https://huggingface.co/MahmoodLab/UNI2-h",
        notes="Access needs an institutional e-mail as the primary Hugging Face address. "
        "The card does not state the magnification; 20x (0.5 um/px) is assumed as for UNI.",
    ),
    BaseModelSpec(
        name="uni",
        backend="timm",
        source="hf-hub:MahmoodLab/UNI",
        architecture="ViT-L/16",
        ssl_method="DINOv2",
        training_data="Mass-100K: 100M tiles from 100k H&E slides at 20x",
        embed_dim=1024,
        patch_size=16,
        num_register_tokens=0,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        create_kwargs={"init_values": 1e-5},
        gated=True,
        license="CC-BY-NC-ND-4.0",
        license_notes=_NC_ND,
        url="https://huggingface.co/MahmoodLab/UNI",
    ),
    BaseModelSpec(
        name="h-optimus-1",
        backend="timm",
        source="hf-hub:bioptimus/H-optimus-1",
        architecture="ViT-g/14",
        ssl_method="SSL (not named on card)",
        training_data="billions of tiles from 1M+ slides of 800k+ patients",
        embed_dim=1536,
        patch_size=14,
        mean=H_OPTIMUS_MEAN,
        std=H_OPTIMUS_STD,
        create_kwargs={"init_values": 1e-5},
        gated=True,
        license="CC-BY-NC-ND-4.0",
        license_notes=_NC_ND,
        url="https://huggingface.co/bioptimus/H-optimus-1",
        notes="1.1B parameters; the card recommends float16 autocast. Access needs an "
        "institutional e-mail.",
    ),
    BaseModelSpec(
        name="h-optimus-0",
        backend="timm",
        source="hf-hub:bioptimus/H-optimus-0",
        architecture="ViT-g/14",
        ssl_method="SSL (not named on card)",
        training_data="500k+ H&E slides",
        embed_dim=1536,
        patch_size=14,
        mean=H_OPTIMUS_MEAN,
        std=H_OPTIMUS_STD,
        create_kwargs={"init_values": 1e-5},
        gated=True,
        license="Apache-2.0",
        license_notes="Permissive: fine-tuning and sharing derived weights are allowed. "
        "Download still requires accepting the terms on Hugging Face.",
        url="https://huggingface.co/bioptimus/H-optimus-0",
        notes="1.1B parameters; the card recommends float16 autocast.",
    ),
    BaseModelSpec(
        name="h0-mini",
        backend="timm",
        source="hf-hub:bioptimus/H0-mini",
        architecture="ViT-B/14",
        ssl_method="DINOv2 distillation from H-optimus-0",
        training_data="PanCancer40M: 43M tiles from 6,093 TCGA slides",
        embed_dim=768,
        patch_size=14,
        num_register_tokens=4,
        create_kwargs={"mlp_layer": "timm.layers.SwiGLUPacked", "act_layer": "torch.nn.SiLU"},
        gated=True,
        license="CC-BY-NC-ND-4.0",
        license_notes=_NC_ND,
        url="https://huggingface.co/bioptimus/H0-mini",
        notes="86M parameters: the light option, and the encoder behind HistoPLUS. "
        "Normalisation is read from the model config.",
    ),
    BaseModelSpec(
        name="virchow2",
        backend="timm",
        source="hf-hub:paige-ai/Virchow2",
        architecture="ViT-H/14",
        ssl_method="DINOv2 (modified)",
        training_data="3.1M slides (MSKCC), tiles at 2.0, 1.0, 0.5 and 0.25 um/px",
        embed_dim=1280,
        patch_size=14,
        num_register_tokens=4,
        tile_pooling="cls+mean",
        create_kwargs={"mlp_layer": "timm.layers.SwiGLUPacked", "act_layer": "torch.nn.SiLU"},
        gated=True,
        license="CC-BY-NC-ND-4.0",
        license_notes=_NC_ND,
        url="https://huggingface.co/paige-ai/Virchow2",
        notes="Trained on four magnifications including 40x (0.25 um/px), so it can be fed "
        "native-resolution crops. Tile embedding is [class token, mean patch token] = 2560-d.",
    ),
    BaseModelSpec(
        name="virchow",
        backend="timm",
        source="hf-hub:paige-ai/Virchow",
        architecture="ViT-H/14",
        ssl_method="DINOv2",
        training_data="1.5M slides (MSKCC) at 0.5 um/px",
        embed_dim=1280,
        patch_size=14,
        num_register_tokens=0,
        tile_pooling="cls+mean",
        create_kwargs={"mlp_layer": "timm.layers.SwiGLUPacked", "act_layer": "torch.nn.SiLU"},
        gated=True,
        license="Apache-2.0",
        license_notes="Permissive (with non-binding usage guidelines); gated download.",
        url="https://huggingface.co/paige-ai/Virchow",
    ),
    BaseModelSpec(
        name="prov-gigapath",
        backend="timm",
        source="hf_hub:prov-gigapath/prov-gigapath",
        architecture="ViT-g/16",
        ssl_method="DINOv2",
        training_data="Prov-Path (Providence)",
        embed_dim=1536,
        patch_size=16,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        gated=True,
        license="Apache-2.0 (card: research use only)",
        license_notes="Tagged Apache-2.0, but the card limits the intended use to research and "
        "reproducibility and puts any deployed use out of scope.",
        url="https://huggingface.co/prov-gigapath/prov-gigapath",
        notes="Tile encoder only; the slide-level encoder is not used here.",
    ),
    BaseModelSpec(
        name="phikon-v2",
        backend="transformers",
        source="owkin/phikon-v2",
        architecture="ViT-L/16",
        ssl_method="DINOv2",
        training_data="PANCAN-XL: 456M tiles at 20x from 60k H&E slides",
        embed_dim=1024,
        patch_size=16,
        num_register_tokens=0,
        gated=False,
        license="Owkin non-commercial licence",
        license_notes="Non-commercial. The card itself points to a LoRA fine-tuning notebook; "
        "read the licence PDF before sharing derived weights.",
        url="https://huggingface.co/owkin/phikon-v2",
        notes="Downloads without an access request: a good first pathology model to wire up.",
    ),
    BaseModelSpec(
        name="phikon",
        backend="transformers",
        source="owkin/phikon",
        architecture="ViT-B/16",
        ssl_method="iBOT",
        training_data="40M pan-cancer tiles from TCGA",
        embed_dim=768,
        patch_size=16,
        num_register_tokens=0,
        create_kwargs={"add_pooling_layer": False},
        gated=False,
        license="Owkin non-commercial licence",
        license_notes="Non-commercial; the card says the model can be fine-tuned.",
        url="https://huggingface.co/owkin/phikon",
    ),
    BaseModelSpec(
        name="midnight-12k",
        backend="transformers",
        source="kaiko-ai/midnight",
        architecture="ViT-g/14",
        ssl_method="DINOv2 (modified)",
        training_data="12k TCGA slides (FFPE)",
        embed_dim=1536,
        patch_size=14,
        num_register_tokens=0,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        tile_pooling="cls+mean",
        gated=False,
        license="MIT",
        license_notes="Permissive: no restriction on fine-tuning or on sharing derived weights.",
        url="https://huggingface.co/kaiko-ai/midnight",
        notes="The only large pathology model here that is both ungated and permissively licensed.",
    ),
    BaseModelSpec(
        name="hibou-l",
        backend="transformers",
        source="histai/hibou-L",
        architecture="ViT-L/14",
        ssl_method="DINOv2 with registers",
        training_data="1.2B tiles (private)",
        embed_dim=1024,
        patch_size=14,
        create_kwargs={"trust_remote_code": True},
        gated=True,
        license="Apache-2.0",
        license_notes="Permissive; gated download.",
        url="https://huggingface.co/histai/hibou-L",
        notes="Loads the authors' model code from the hub (trust_remote_code). A CellViT decoder "
        "trained on this encoder exists (histai/cellvit-hibou-l, CC-BY-NC-SA-4.0).",
    ),
    BaseModelSpec(
        name="lunit-dino-s16",
        backend="timm",
        source="hf-hub:1aurent/vit_small_patch16_224.lunit_dino",
        architecture="ViT-S/16",
        ssl_method="DINO",
        training_data="19M TCGA patches",
        embed_dim=384,
        patch_size=16,
        num_register_tokens=0,
        mean=LUNIT_MEAN,
        std=LUNIT_STD,
        gated=False,
        license="lunit-non-commercial",
        license_notes="Non-commercial, research only.",
        url="https://huggingface.co/1aurent/vit_small_patch16_224.lunit_dino",
        notes="21M parameters: small enough for CPU smoke tests and laptops. The normalisation is "
        "the one in Lunit's release notes; the config of this timm port carries 0.5/0.5 instead.",
    ),
    BaseModelSpec(
        name="dinov2-s14",
        backend="timm",
        source="vit_small_patch14_dinov2.lvd142m",
        architecture="ViT-S/14",
        ssl_method="DINOv2",
        training_data="LVD-142M natural images (not histology)",
        embed_dim=384,
        patch_size=14,
        num_register_tokens=0,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        gated=False,
        license="Apache-2.0",
        license_notes="Permissive.",
        url="https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m",
        notes="General-domain reference point: shows how much the pathology pre-training adds. "
        "Trained at 518 px on photographs; it is fed 224 px windows here, like the pathology models.",
    ),
)

_REGISTRY: dict[str, BaseModelSpec] = {}


def register(spec: BaseModelSpec, *, overwrite: bool = False) -> BaseModelSpec:
    """Add a spec under ``spec.name`` (names are case-insensitive)."""
    key = spec.name.lower()
    if key in _REGISTRY and not overwrite:
        raise ValueError(f"base model {spec.name!r} is already registered")
    _REGISTRY[key] = spec
    return spec


def list_specs() -> list[BaseModelSpec]:
    """All registered specs, in registration order."""
    return list(_REGISTRY.values())


def resolve_spec(model: str | BaseModelSpec) -> BaseModelSpec:
    """Turn a registry name or an ad-hoc ``timm:``/``hf:`` reference into a spec."""
    if isinstance(model, BaseModelSpec):
        return model
    key = model.lower()
    if key in _REGISTRY:
        return _REGISTRY[key]
    prefix, _, source = model.partition(":")
    if source and prefix.lower() == "timm":
        return BaseModelSpec(name=model, backend="timm", source=source, notes="ad-hoc timm reference")
    if source and prefix.lower() == "hf":
        return BaseModelSpec(name=model, backend="transformers", source=source, notes="ad-hoc Hugging Face reference")
    known = ", ".join(sorted(_REGISTRY))
    raise KeyError(
        f"unknown base model {model!r}. Registered: {known}. "
        "Ad-hoc references: 'timm:<timm name or hf-hub:org/repo>' or 'hf:<org/repo>'."
    )


def load_registry_file(path: str | Path, *, overwrite: bool = True) -> list[BaseModelSpec]:
    """Register every entry of a YAML file shaped ``{base_models: [{name: ..., ...}, ...]}``."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    entries = document.get("base_models")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a top-level 'base_models' list")
    known_fields = {f.name for f in fields(BaseModelSpec)}
    specs = []
    for entry in entries:
        unknown = set(entry) - known_fields
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)} in entry {entry.get('name')!r}")
        entry = dict(entry)
        for key in ("mean", "std"):
            if entry.get(key) is not None:
                entry[key] = tuple(float(v) for v in entry[key])
        specs.append(register(BaseModelSpec(**entry), overwrite=overwrite))
    return specs


for _spec in _BUILTIN:
    register(_spec)
