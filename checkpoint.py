import torch
ckpt = torch.load(
    "/home/johnt/projects/rrg-amiilab/johnt/ttab-main/vit_large_patch16_224.pth",
    map_location="cpu"
)
# Depending on how save_model was implemented, your weights might all be under ckpt['model']:
state = ckpt.get("model", ckpt)  

print("All keys in your checkpoint:\n", list(state.keys())[:10], "…")
print("Classification‑head keys:")
print([k for k in state.keys() if any(x in k for x in ["head", "fc", "classifier"])])