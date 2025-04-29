#!/usr/bin/env python3
# check_transfer.py

import os
import sys
import torch

# 1) Ensure we can import the repo root
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import parameters                             # your CLI parser
from ttab.loads.define_model import define_model
# delay this import to avoid the circular-import trap
# from ttab.loads.define_model import load_pretrained_model
from ttab.loads.define_dataset import ConstructTestDataset
from torchvision import transforms

def fail(msg):
    print("  ✗", msg)

def ok(msg):
    print("  ✓", msg)

def main():
    print("\nConfiguring TTAB model and data…")
    cfg = parameters.get_args()
    cfg.ckpt_path   = os.environ.get("VIT_CKPT_PATH", "/home/johnt/projects/rrg-amiilab/johnt/ttab-main/AffectNet7-66_14.pth")
    cfg.base_data_name = "affectnet"
    cfg.data_names     = ["affectnet"]
    cfg.data_path      = os.environ.get("AFFECTNET_PATH", "/home/johnt/scratch/AffectNet7_37k_balanced")
    cfg.statistics     = {"n_classes": 7, "class_to_idx": None}
    cfg.model_name     = "vit_base_patch16_224"
    cfg.model_adaptation_method = "none"

    if not os.path.isfile(cfg.ckpt_path):
        print(f"❗️ Checkpoint not found at {cfg.ckpt_path}")
        sys.exit(1)

    # 1) Instantiate & load weights
    print("\n1) Loading model and checkpoint…")
    model = define_model(cfg)
    from ttab.loads.define_model import load_pretrained_model
    load_pretrained_model(cfg, model)
    raw = torch.load(cfg.ckpt_path, map_location="cpu")
    state = raw.get("model", raw)

    # 2) Test head shape
    print("\n2) Testing classifier-head shape…")
    head_wt = model.head.weight
    ckpt_wt = state.get("head.weight")
    if ckpt_wt is None:
        fail("  – no key 'head.weight' in checkpoint")
    elif tuple(head_wt.shape) != tuple(ckpt_wt.shape):
        fail(f"  – model {tuple(head_wt.shape)} vs ckpt {tuple(ckpt_wt.shape)}")
    else:
        ok("head shape matches")

    # 3) Test strict=False state_dict load
    print("\n3) Testing strict=False state_dict load…")
    mismatch = model.load_state_dict(state, strict=False)
    if mismatch.missing_keys or mismatch.unexpected_keys:
        fail(f"  – missing_keys={mismatch.missing_keys}, unexpected_keys={mismatch.unexpected_keys}")
    else:
        ok("no missing/unexpected keys")

    # 4) Test eval() + no_grad()
    print("\n4) Testing eval mode & no_grad…")
    model.train()
    model.eval()
    if model.training:
        fail("  – model.training is still True after eval()")
    else:
        ok("model.training is False")

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        _ = model(x)
    if any(p.requires_grad for p in model.parameters()):
        fail("  – some parameters still require_grad=True under no_grad")
    else:
        ok("no parameters require grad")

    # 5) Dataset class mapping
    print("\n5) Testing dataset.class_to_idx consistency…")
    builder = ConstructTestDataset(cfg)
    # this returns a list of AdaptationDataset wrappers; the .dataset is a Torchvision Dataset
    test_ds = builder.get_test_datasets(test_domains=[cfg.scenario.test_domains[0]])[0].dataset
    mapping = getattr(test_ds, "class_to_idx", None)
    orig_map = cfg.statistics.get("class_to_idx")
    if mapping is None:
        fail("  – test_ds has no attribute class_to_idx")
    elif orig_map and mapping != orig_map:
        fail(f"  – mapping {mapping} != original {orig_map}")
    else:
        ok("class_to_idx present and consistent (or no original map to compare)")

    # 6) Normalization parameters
    print("\n6) Testing Normalize(mean,std) in pipeline…")
    transforms_list = []
    if hasattr(test_ds, "transform"):
        # torchvision Compose: test_ds.transform.transforms is a list
        transforms_list = getattr(test_ds.transform, "transforms", [])
    found = [t for t in transforms_list if isinstance(t, transforms.Normalize)]
    if not found:
        fail("  – no torchvision.transforms.Normalize in dataset.transform")
    else:
        norm = found[-1]
        exp_mean = [0.485, 0.456, 0.406]
        exp_std  = [0.229, 0.224, 0.225]
        if list(norm.mean) != exp_mean or list(norm.std) != exp_std:
            fail(f"  – got mean={norm.mean}, std={norm.std}; expected {exp_mean}/{exp_std}")
        else:
            ok("Normalize(mean,std) matches expected values")

    print("\n✅ Done. Adjust any FAILs above and re-run until everything is ✓.\n")

if __name__ == "__main__":
    main()
