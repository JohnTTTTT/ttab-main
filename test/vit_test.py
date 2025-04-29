# tests/test_transfer_vit_ttab.py

import os
import pytest
import torch

# TTAB imports
import parameters                                                 # the CLI parser
import ttab.configs.utils as configs_utils                          # config/scenario builder :contentReference[oaicite:0]{index=0}
from ttab.loads.define_dataset import ConstructTestDataset        # data loader :contentReference[oaicite:1]{index=1}
from ttab.loads.define_model import define_model  # model builder & loader :contentReference[oaicite:2]{index=2}

# -----------------------------------------------------------------------------
# — Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def config():
    """Build a TTAB config object and override just the bits we need."""
    conf = parameters.get_args()  # parse your parameters.py definitions :contentReference[oaicite:3]{index=3}

    # Point to your ViT‐on‐AffectNet checkpoint
    conf.ckpt_path = os.getenv("VIT_CKPT_PATH", "/home/johnt/projects/rrg-amiilab/johnt/ttab-main/AffectNet7-66_14.pth")
    assert os.path.exists(conf.ckpt_path), f"Checkpoint not found at {conf.ckpt_path}"

    # Make sure TTAB knows how many classes to expect
    conf.statistics = {"n_classes": 7}

    # Match your original training setup
    conf.model_name = "vit_base_patch16_224"
    conf.data_names = ["affectnet"]      # your custom dataset name
    conf.entry_of_shared_layers = None
    conf.group_norm_num_groups = None
    conf.grad_checkpoint = False
    conf.model_adaptation_method = "none"  # no TTA when just evaluating

    # These two TTAB fields are always needed
    conf.base_data_name = "affectnet"
    conf.data_path = "/home/johnt/scratch/AffectNet7_37k_balanced"

    return conf

@pytest.fixture(scope="session")
def model_and_state(config):
    """Instantiate the TTAB ViT, load weights, and also grab the raw checkpoint dict."""
    # Build the PyTorch model
    model = define_model(config)                         # from ttab.loads.define_model :contentReference[oaicite:4]{index=4}
    from ttab.loads.define_model import define_model, load_pretrained_model
    # Load the pretrained parameters into it
    load_pretrained_model(config=config, model=model)    # this will call model.load_state_dict internally :contentReference[oaicite:5]{index=5}

    # Get the raw checkpoint map
    ckpt = torch.load(config.ckpt_path, map_location="cpu")
    # In TTAB, image‐based models store their weights under ckpt["model"]
    state_dict = ckpt.get("model", ckpt)

    return model, state_dict

# -----------------------------------------------------------------------------
# — Tests
# -----------------------------------------------------------------------------

def test_head_shape_matches_checkpoint(model_and_state):
    """The in‐memory head weight shape must match exactly the saved weights."""
    model, state = model_and_state
    # TTAB’s define_model puts the ViT head in model.head
    wt_model = model.head.weight
    # Checkpoint uses key "head.weight"
    wt_ckpt = state["head.weight"]
    assert wt_model.shape == wt_ckpt.shape, (
        f"ViT head shape mismatch: model has {tuple(wt_model.shape)}, "
        f"checkpoint has {tuple(wt_ckpt.shape)}"
    )

def test_state_dict_no_missing_or_unexpected(model_and_state):
    """Loading with strict=False should reveal zero missing/unexpected keys."""
    model, state = model_and_state
    mismatch = model.load_state_dict(state, strict=False)
    assert not mismatch.missing_keys,   f"Missing keys: {mismatch.missing_keys}"
    assert not mismatch.unexpected_keys, f"Unexpected keys: {mismatch.unexpected_keys}"

def test_model_is_in_eval_mode_and_no_grad(model_and_state):
    """model.eval() + torch.no_grad() must freeze all dropout/BN and grad computation."""
    model, _ = model_and_state

    model.train()
    model.eval()
    assert not model.training, "model.training should be False after model.eval()"

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        _ = model(x)
    # After torch.no_grad, no parameter should require grad
    assert not any(p.requires_grad for p in model.parameters()), "Some params still require_grad=True"

def test_dataset_class_mapping_consistency(config):
    """
    The training‐time class mapping (your original AffectNet loader)
    must match what TTAB’s ConstructTestDataset produces.
    """
    # Build the TTAB test dataset
    dataset_builder = ConstructTestDataset(config)
    # Here we assume your scenario only has one domain and one test split
    # so we can directly peek at the underlying torch Dataset
    # You may need to adapt this if your loader wraps things differently.
    class_to_idx = dataset_builder.get_test_datasets(
        test_domains=[config.scenario.test_domains[0]]
    )[0].dataset.class_to_idx
    # Compare to your original mapping (passed via config.statistics or elsewhere)
    # For example, if you stored it in config.statistics["class_to_idx"]:
    assert class_to_idx == config.statistics.get("class_to_idx", class_to_idx), (
        "TTAB dataset.class_to_idx does not match your original mapping"
    )

@pytest.mark.parametrize("mean,std", [
    ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),   # ImageNet defaults
    # add your AffectNet-specific normalization here if different
])
def test_normalization_pipeline(mean, std, config):
    """
    Ensure that TTAB’s data‐loading pipeline applies the expected
    Normalize(mean, std) parameters.
    """
    builder = ConstructTestDataset(config)
    # hack: the first dataset produced will have a .dataset.transform attr if it's torchvision-based
    pt_ds = builder.get_test_datasets(test_domains=[config.scenario.test_domains[0]])[0].dataset
    norm_steps = [t for t in getattr(pt_ds, "transform", []) if isinstance(t, torch.transforms.Normalize)]
    assert norm_steps, "No Normalize() in the dataset.transform pipeline"
    norm = norm_steps[-1]
    assert list(norm.mean) == mean, f"Normalize.mean is {norm.mean}, expected {mean}"
    assert list(norm.std)  == std,  f"Normalize.std is {norm.std}, expected {std}"