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

from model.model import FrozenInTime # Assuming this import path is correct

# ==== Settings ====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = './checkpoints/EgoVLPv2_smallproj.pth'
CLIPS_DIR = './clips_egoclip' # Ensure this points to your 5-second clips
PROMPTS = ["#C C looks around", "#C C walks around", "#C C sits in the house"]
# PROMPTS = ["walking", "sitting", "standing"]
# PROMPTS = ["cat", "washing machine", "skydiving"]


# ==== Preprocessing ====
IMG_SIZE = 224
NUM_FRAMES = 128 # Sample N frames per video
CLIP_DURATION = 5 # seconds per clip

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==== Load Model ====
print("Loading model...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# Extract and prepare arguments from the checkpoint's config
ckpt_args = ckpt['config']['arch']['args']
ckpt_args['video_params']['num_frames'] = NUM_FRAMES

# --- CRITICAL FIX: Force projection_dim to 256 for EgoVLPv2_smallproj ---
ckpt_args['projection_dim'] = 256
# Ensure 'projection' type is set, usually 'minimal' for these models
if 'projection' not in ckpt_args:
    ckpt_args['projection'] = 'minimal'

# --- PROBLEM FIX: Provide a dummy config dictionary for FrozenInTime's internal use ---
# This dictionary contains parameters that FrozenInTime expects to find
# within its 'config' object for initializing sub-modules like cross-attention heads.
# These values are typical for roberta-base and the EgoVLP architecture.
dummy_config = {
    'vocab_size': 50265, # Standard for RoBERTa tokenizer
    'hidden_size': 768,  # RoBERTa base hidden size
    'num_layers': 12,    # RoBERTa base layers
    'num_heads': 12,     # RoBERTa base heads
    'mlp_ratio': 4,      # Common MLP ratio for transformers
    'drop_rate': 0.1,    # Common dropout rate
    'input_image_embed_size': 768, # ViT base embed_dim
    'input_text_embed_size': 768,  # RoBERTa base hidden size (same as hidden_size for RoBERTa)
    'num_fuse_block': 4, # Crucial for cross-modal layers; typical value from EgoVLP
    'use_checkpoint': False # Set to False unless you're using torch.utils.checkpoint
}

# The `FrozenInTime` class also expects a `task_names` key in its `config` dictionary if
# it's going to initialize ITM/MLM heads or similar.
dummy_config['task_names'] = 'EgoNCE' # Setting it to EgoNCE for basic embedding extraction

# Instantiate the model with the prepared arguments and the dummy config
model = FrozenInTime(
    **ckpt_args,
    config=dummy_config, # Pass the dictionary directly
    task_names='EgoNCE' # Explicitly set task to EgoNCE for simpler inference path
)

# Load the pre-trained weights
missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
print("Missing keys:", missing)
model = model.to(DEVICE)
model.eval()

# ==== Load Tokenizer ====
print("Using tokenizer:", ckpt['config']['arch']['args']['text_params']['model'])
tokenizer = AutoTokenizer.from_pretrained(ckpt['config']['arch']['args']['text_params']['model'])

with torch.no_grad():
    print("Encoding prompts...")
    text_inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_embeds = model.compute_text(text_inputs).float()
    print("text_embeds std dev:", text_embeds.std().item())
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
    # Handle cases where video might be too short to sample NUM_FRAMES
    if len(frames) == 0: # If no frames were read at all (e.g., corrupt file)
        print(f"Warning: No frames read from {path}. Skipping this clip.")
        return None
    if len(frames) < num_frames:
        # Pad with the last frame if not enough frames were sampled
        frames += [frames[-1]] * (num_frames - len(frames))

    frames = torch.stack(frames)
    frames = frames.unsqueeze(0).to(DEVICE)
    return frames

# ==== Inference & Logging ====
print("Running inference...")
# CRITICAL CHANGE: Use 'clip_*.mp4' to only process the generated 5-second clips
clip_paths = sorted(glob.glob(os.path.join(CLIPS_DIR, 'clip_*.mp4')))

# DEBUG prints to verify clip paths
print(f"DEBUG: CLIPS_DIR is set to: {CLIPS_DIR}")
print(f"DEBUG: Found {len(clip_paths)} clips matching 'clip_*.mp4'.")
# print(f"DEBUG: First 5 clip paths: {clip_paths[:5]}") # Uncomment for more verbose debugging

os.makedirs("runs", exist_ok=True)
csv_path = os.path.join("runs", "action_segments.csv")

prev_label = None
start_time = 0
results = []

if not clip_paths:
    print(f"Error: No clips found in {CLIPS_DIR} matching 'clip_*.mp4'. Please check path and filenames.")
else:
    for i, clip_path in enumerate(tqdm(clip_paths)):
        video_frames = load_video_frames(clip_path)

        # Skip if no frames could be loaded from the video (e.g., corrupt file)
        if video_frames is None:
            continue

        video_embeds = model.compute_video(video_frames).float()
        video_embeds = F.normalize(video_embeds, dim=-1)

        text_sims = torch.matmul(text_embeds, text_embeds.T)
        print("Prompt-to-prompt similarity matrix:")
        print(text_sims.cpu().numpy())

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