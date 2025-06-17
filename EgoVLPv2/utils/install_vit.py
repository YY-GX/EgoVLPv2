# save_vit_hf_to_pth.py

import torch
from transformers import ViTModel

# Load Hugging Face ViT-base-patch16
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# Convert to state_dict format used by EgoVLPv2
state_dict = model.state_dict()

# Save it
torch.save(state_dict, "./checkpoints/jx_vit_base_p16_224-80ecf9dd.pth")

print("Saved EgoVLP-compatible ViT weights to checkpoints/")
