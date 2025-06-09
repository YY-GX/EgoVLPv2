import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import glob
import cv2
import csv

from model.model import FrozenInTime

# ==== Settings ====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = './checkpoints/EgoVLPv2_smallproj.pth'
CLIPS_DIR = './clips_egoclip'
# PROMPTS = ["a person is walking", "a person is sitting", "a person is standing"]
# PROMPTS = ["walk around", "sit on the chair", "stand up"]
PROMPTS = ["looks around", "walks around", "sits in the house"]

# ==== Preprocessing ====
IMG_SIZE = 224
NUM_FRAMES = 128  # sample N frames per video
CLIP_DURATION = 5  # seconds per clip

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# # ==== Load Model ====
# print("Loading model...")
# ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
# ckpt['config']['arch']['args']['video_params']['num_frames'] = NUM_FRAMES
# model = FrozenInTime(**ckpt['config']['arch']['args'])
# model.load_state_dict(ckpt['state_dict'], strict=False)
# model = model.to(DEVICE)
# model.eval()

# ==== Load Model ====
print("Loading model...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# Ensure num_frames is set BEFORE model instantiation if it's used in __init__
# (The current code sets it correctly, but re-confirm placement)
ckpt_args = ckpt['config']['arch']['args']
ckpt_args['video_params']['num_frames'] = NUM_FRAMES

# --- CRITICAL FIX: Force projection_dim to 256 ---
# The smallproj checkpoint uses 256 for its projection layers
ckpt_args['projection_dim'] = 256
# Also, ensure 'projection' is set to 'minimal' if not already
if 'projection' not in ckpt_args:
    ckpt_args['projection'] = 'minimal' # Or whatever the config for smallproj uses

# --- PROBLEM FIX: Dummy config object to prevent errors ---
# The 'config' argument in FrozenInTime's __init__ is a global/ConfigParser object
# that's used for various internal parameters.
# While we don't have the full ConfigParser, we can provide a dummy that has essential keys.
# Check the config used by the official `multinode_train_epic.py` (configs/eval/epic.json)
# for values for these parameters, e.g., 'hidden_size', 'num_layers', 'num_fuse_block', 'use_checkpoint'
from types import SimpleNamespace
dummy_config = SimpleNamespace(
    vocab_size=50265, # Standard for RoBERTa tokenizer
    hidden_size=768,  # RoBERTa base hidden size
    num_layers=12,    # RoBERTa base layers
    num_heads=12,     # RoBERTa base heads
    mlp_ratio=4,      # Common MLP ratio for transformers
    drop_rate=0.1,    # Common dropout rate
    input_image_embed_size=768, # ViT base embed_dim
    input_text_embed_size=768,  # RoBERTa base hidden size
    num_fuse_block=4, # This is crucial for cross-modal layers, check official config
    use_checkpoint=False # Set to False if you don't use torch.utils.checkpoint
)
# If task_names is also needed as a key:
dummy_config.task_names = 'EgoNCE' # Or 'EgoNCE_ITM_MLM' if you want to initialize those heads

# Now pass it explicitly
model = FrozenInTime(
    **ckpt_args,
    config=dummy_config, # Pass the dummy config
    task_names='EgoNCE' # Explicitly set task to EgoNCE for simpler inference, this avoids ITM/MLM init
)

model.load_state_dict(ckpt['state_dict'], strict=False)
model = model.to(DEVICE)
model.eval()












# ==== Load Tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(ckpt['config']['arch']['args']['text_params']['model'])

with torch.no_grad():
    print("Encoding prompts...")
    text_inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_embeds = model.compute_text(text_inputs).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

# ==== Helper ====
def load_video_frames(path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, frame_count - 1, num_frames).astype(int)
    frames = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(transform(img))

    cap.release()
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))

    frames = torch.stack(frames)
    frames = frames.unsqueeze(0).to(DEVICE)
    return frames

# ==== Inference & Logging ====
print("Running inference...")
clip_paths = sorted(glob.glob(os.path.join(CLIPS_DIR, '*.mp4')))
os.makedirs("runs", exist_ok=True)
csv_path = os.path.join("runs", "action_segments.csv")

prev_label = None
start_time = 0
results = []

for i, clip_path in enumerate(tqdm(clip_paths)):
    video_frames = load_video_frames(clip_path)
    video_embeds = model.compute_video(video_frames).float()
    video_embeds = F.normalize(video_embeds, dim=-1)

    sim = torch.matmul(video_embeds, text_embeds.T)
    probs = F.softmax(sim, dim=-1).cpu().detach().numpy()[0]
    pred_idx = probs.argmax()
    pred_label = PROMPTS[pred_idx]

    print(f"{os.path.basename(clip_path)} → {pred_label}")
    print(f"Probabilities: {[round(p, 4) for p in probs]}")

    if pred_label != prev_label:
        if prev_label is not None:
            end_time = i * CLIP_DURATION
            results.append([f"{start_time:.1f}s", f"{end_time:.1f}s", prev_label])
        start_time = i * CLIP_DURATION
        prev_label = pred_label

# Log final segment
results.append([f"{start_time:.1f}s", f"{(len(clip_paths)) * CLIP_DURATION:.1f}s", prev_label])

# Save to CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Start Time", "End Time", "Action"])
    writer.writerows(results)

print(f"\nSaved action segments to: {csv_path}")
