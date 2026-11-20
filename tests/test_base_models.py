"""Base-model registry, loading and hidden representations."""

import pytest
import torch

from mpn_al.embeddings import BaseModelSpec, build_base_model, list_specs, load_registry_file, resolve_spec
from mpn_al.embeddings.base_model import _access_error, load_encoder_checkpoint


def test_registry_lists_the_candidate_models():
    names = {spec.name for spec in list_specs()}
    assert {"uni2-h", "h-optimus-1", "h-optimus-0", "virchow2", "phikon-v2", "midnight-12k"} <= names
    for spec in list_specs():
        assert spec.license, f"{spec.name} must state its licence"
        assert spec.url.startswith("https://"), spec.name


def test_models_with_their_own_statistics_do_not_fall_back_to_the_hub_config():
    """The timm port of Lunit DINO ships 0.5/0.5 in its config; the model was trained with other values."""
    lunit = resolve_spec("lunit-dino-s16")
    assert lunit.mean == pytest.approx((0.70322989, 0.53606487, 0.66096631))
    assert lunit.std == pytest.approx((0.21716536, 0.26081574, 0.20723464))
    for name in ("h-optimus-0", "h-optimus-1"):
        assert resolve_spec(name).mean == pytest.approx((0.707223, 0.578729, 0.703617))
        assert resolve_spec(name).std == pytest.approx((0.211883, 0.230117, 0.177517))


def test_resolve_is_case_insensitive_and_accepts_adhoc_references():
    assert resolve_spec("UNI2-H").source == "hf-hub:MahmoodLab/UNI2-h"
    adhoc = resolve_spec("timm:vit_small_patch14_dinov2.lvd142m")
    assert (adhoc.backend, adhoc.source) == ("timm", "vit_small_patch14_dinov2.lvd142m")
    assert resolve_spec("hf:owkin/phikon-v2").backend == "transformers"
    with pytest.raises(KeyError, match="unknown base model"):
        resolve_spec("no-such-model")


def test_registry_file_adds_entries(tmp_path):
    path = tmp_path / "extra.yaml"
    path.write_text(
        "base_models:\n"
        "  - name: inhouse-vits16\n"
        "    backend: timm\n"
        "    source: vit_small_patch16_224\n"
        "    mean: [0.7, 0.6, 0.7]\n"
        "    std: [0.2, 0.2, 0.2]\n",
        encoding="utf-8",
    )
    (spec,) = load_registry_file(path)
    assert resolve_spec("inhouse-vits16") is spec
    assert spec.mean == (0.7, 0.6, 0.7)
    path.write_text("base_models:\n  - {name: bad, backend: timm, source: x, typo_field: 1}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_registry_file(path)


def test_hidden_representations_have_the_documented_shapes(tiny_model):
    assert (tiny_model.patch_size, tiny_model.embed_dim, tiny_model.num_layers) == (16, 192, 12)
    assert tiny_model.default_layers() == (2, 5, 8, 11)
    images = torch.rand(2, 3, 160, 224)  # not the native 224x224: exercises dynamic input size
    output = tiny_model(images, layers=tiny_model.default_layers())
    assert output.grid == (10, 14)
    assert output.patch_tokens.shape == (2, 192, 10, 14)
    assert output.cls_token.shape == (2, 192)
    assert sorted(output.hidden_states) == [2, 5, 8, 11]
    assert all(h.shape == (2, 192, 10, 14) for h in output.hidden_states.values())
    assert output.pooled("cls+mean").shape == (2, 384)
    assert tiny_model.output_dim("cls+mean") == 384


def test_last_hidden_state_matches_patch_tokens_up_to_the_final_norm(tiny_model):
    images = torch.rand(1, 3, 224, 224)
    output = tiny_model(images, layers=[-1])
    raw = output.hidden_states[11].permute(0, 2, 3, 1)
    normed = tiny_model.net.norm(raw).permute(0, 3, 1, 2)
    assert torch.allclose(normed, output.patch_tokens, atol=1e-5)


def test_embedding_equals_the_plain_timm_forward(tiny_model):
    """The wrapper must not change what the encoder computes at its native size."""
    images = torch.rand(2, 3, 224, 224)
    expected = tiny_model.net(tiny_model.normalize(images))  # timm: pooled class token for num_classes=0
    assert torch.allclose(tiny_model.embed(images, "cls"), expected, atol=1e-5)


def test_input_must_be_a_multiple_of_the_patch_size(tiny_model):
    with pytest.raises(ValueError, match="multiple of the patch size"):
        tiny_model(torch.rand(1, 3, 230, 224))
    assert tiny_model.fit_size(230) == 224
    assert tiny_model.fit_size(3) == 16


def test_frozen_encoder_stays_in_eval_mode_and_gets_no_gradients(tiny_model):
    assert tiny_model.frozen
    tiny_model.train()
    assert not tiny_model.training and not tiny_model.net.training
    images = torch.rand(1, 3, 224, 224, requires_grad=True)
    assert not tiny_model(images).patch_tokens.requires_grad
    assert all(not p.requires_grad for p in tiny_model.parameters())


def test_register_tokens_are_separated_from_patch_tokens():
    model = build_base_model(
        BaseModelSpec(
            name="tiny-reg4",
            backend="timm",
            source="vit_tiny_patch16_224",
            create_kwargs={"reg_tokens": 4, "no_embed_class": True},
        ),
        pretrained=False,
    )
    assert model.num_prefix_tokens == 5
    output = model(torch.rand(1, 3, 224, 224))
    assert output.register_tokens.shape == (1, 4, 192)
    assert output.patch_tokens.shape == (1, 192, 14, 14)


def test_local_checkpoint_with_ssl_wrappers_is_loaded(tmp_path, tiny_model):
    """DINO-style checkpoints nest the weights and prefix the keys."""
    state = {f"backbone.{k}": v + 1.0 for k, v in tiny_model.net.state_dict().items()}
    state["head.projection.weight"] = torch.zeros(3, 3)  # projection head: must be ignored
    path = tmp_path / "ssl_checkpoint.pth"
    torch.save({"teacher": state, "epoch": 7}, path)

    loaded = build_base_model("timm:vit_tiny_patch16_224", pretrained=False, checkpoint_path=path)
    for key, value in tiny_model.net.state_dict().items():
        assert torch.equal(loaded.net.state_dict()[key], value + 1.0), key

    torch.save({"model": {"unrelated": torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match="no tensor matches"):
        load_encoder_checkpoint(loaded.net, path)


def test_transformers_backend_matches_the_timm_interface(tmp_path):
    transformers = pytest.importorskip("transformers")
    config = transformers.ViTConfig(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, intermediate_size=128, image_size=224, patch_size=16
    )
    transformers.ViTModel(config, add_pooling_layer=False).save_pretrained(tmp_path / "vit")
    spec = BaseModelSpec(
        name="tiny-hf",
        backend="transformers",
        source=str(tmp_path / "vit"),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        create_kwargs={"add_pooling_layer": False},
    )
    model = build_base_model(spec)
    assert (model.patch_size, model.embed_dim, model.num_layers, model.num_prefix_tokens) == (16, 64, 4, 1)
    output = model(torch.rand(2, 3, 160, 192), layers=model.default_layers())
    assert output.patch_tokens.shape == (2, 64, 10, 12)
    assert output.cls_token.shape == (2, 64)
    assert sorted(output.hidden_states) == [0, 1, 2, 3]
    last = model.net.layernorm(output.hidden_states[3].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    assert torch.allclose(last, output.patch_tokens, atol=1e-5)


def test_gated_repository_errors_explain_what_to_do():
    spec = resolve_spec("uni2-h")
    error = _access_error(spec, RuntimeError("401 Client Error: GatedRepoError"))
    assert "request access at https://huggingface.co/MahmoodLab/UNI2-h" in str(error)
    assert _access_error(spec, ValueError("shape mismatch")) is None
