"""The configuration files shipped in the repository must load."""

from pathlib import Path

import pytest
import torch

from mpn_al.embeddings import BaseModelSpec, build_base_model, load_registry_file, resolve_spec
from mpn_al.segmenter import LabelSchema

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted((CONFIGS / "labels").glob("*.yaml")), ids=lambda p: p.name)
def test_label_schemas_load(path):
    schema = LabelSchema.from_yaml(path)
    assert schema.num_classes >= 1 and schema.classes[0].id == 1
    assert LabelSchema.from_dict(schema.to_dict()) == schema  # what a head checkpoint stores


def test_megakaryocyte_schema_maps_the_qupath_classes():
    schema = LabelSchema.from_yaml(CONFIGS / "labels" / "megakaryocyte_binary.yaml")
    assert schema.role_of("meg_atypical")[1].name == "megakaryocyte"
    assert schema.role_of("megakaryocyte")[0] == "class"
    assert schema.role_of("uncertain")[0] == "ignore" and schema.role_of("ROI")[0] == "roi"
    assert schema.role_of("lymphocyte")[0] == "unknown" and schema.role_of(None)[0] == "unknown"


def test_label_schema_rejects_gaps_and_duplicate_names():
    with pytest.raises(ValueError, match="class ids must be 1..K"):
        LabelSchema.from_dict({"classes": [{"id": 2, "name": "a"}]})
    with pytest.raises(ValueError, match="used more than once"):
        LabelSchema.from_dict({"classes": [{"name": "a", "aliases": ["b"]}, {"name": "b"}]})


def test_example_registry_file_registers_an_inhouse_encoder():
    (spec,) = load_registry_file(CONFIGS / "base_models" / "example.yaml")
    assert resolve_spec("inhouse-vits16") is spec and spec.backend == "timm"
    model = build_base_model(spec, pretrained=False)
    assert (model.embed_dim, model.patch_size) == (384, 16)


def test_dynamic_input_size_does_not_change_the_native_forward():
    """Model cards of H-optimus create the network without dynamic sizing; we enable it. Same numbers at 224 px."""
    torch.manual_seed(0)
    fixed = build_base_model(
        BaseModelSpec(name="fixed", backend="timm", source="vit_tiny_patch16_224", dynamic_img_size=False),
        pretrained=False,
    )
    dynamic = build_base_model("timm:vit_tiny_patch16_224", pretrained=False)
    dynamic.net.load_state_dict(fixed.net.state_dict())
    images = torch.rand(2, 3, 224, 224)
    assert torch.allclose(fixed.embed(images), dynamic.embed(images), atol=1e-6)
    assert torch.allclose(fixed(images).patch_tokens, dynamic(images).patch_tokens, atol=1e-6)
    with pytest.raises(Exception):
        fixed(torch.rand(1, 3, 448, 448))  # only the dynamic variant accepts other sizes
    assert dynamic(torch.rand(1, 3, 448, 448)).grid == (28, 28)
