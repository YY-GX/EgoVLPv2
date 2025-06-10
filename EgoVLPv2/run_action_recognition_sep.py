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
import argparse

from parse_config import ConfigParser
from model.model import FrozenInTime

# ====== Config and Checkpoint Loading ======
print("Loading config and checkpoint...")

CONFIG_PATH = './configs/pt/egoclip.json'
CHECKPOINT_PATH = './checkpoints/EgoVLPv2_smallproj.pth'
CLIPS_DIR = './clips_egoclip'
PROMPTS = ["walking", "sitting", "standing"]

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True)
custom_args = parser.parse_args(args=['--config', CONFIG_PATH])
config = ConfigParser(custom_args)

ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
ckpt_args = ckpt['config']['arch']['args']
ckpt_args['video_params']['num_frames'] = 128  # Match video loading below

if 'projection_dim' not in ckpt_args:
    ckpt_args['projection_dim'] = 256
if 'projection' not in ckpt_args:
    ckpt_args['projection'] = 'minimal'

# Inject missing keys into config
if 'use_checkpoint' not in ckpt['config']:
    ckpt['config']['use_checkpoint'] = False
if 'task_names' not in ckpt['config']:
    ckpt['config']['task_names'] = 'EgoNCE'

model = FrozenInTime(
    **ckpt_args,
    config=ckpt['config'],
    task_names=ckpt['config']['task_names']
)

missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
print("Missing keys:", missing)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(DEVICE)
model.eval()

# ====== Tokenizer and Prompt Encoding ======
print("Using tokenizer:", ckpt_args['text_params']['model'])
tokenizer = AutoTokenizer.from_pretrained(ckpt_args['text_params']['model'])

with torch.no_grad():
    print("Encoding prompts...")
    text_inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_embeds = model.compute_text(text_inputs).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

# ====== Preprocessing Settings ======
IMG_SIZE = 224
NUM_FRAMES = 128
CLIP_DURATION = 5

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    if len(frames) == 0:
        print(f"Warning: No frames read from {path}. Skipping.")
        return None
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))

    frames = torch.stack(frames).unsqueeze(0).to(DEVICE)
    return frames

# ====== Inference and Logging ======
print("Running inference...")
clip_paths = sorted(glob.glob(os.path.join(CLIPS_DIR, 'clip_*.mp4')))
print(f"DEBUG: Found {len(clip_paths)} clips in {CLIPS_DIR}")

os.makedirs("runs", exist_ok=True)
csv_path = os.path.join("runs", "action_segments.csv")

prev_label = None
start_time = 0
results = []

for i, clip_path in enumerate(tqdm(clip_paths)):
    video_frames = load_video_frames(clip_path)
    if video_frames is None:
        continue

    with torch.no_grad():
        video_embeds = model.compute_video(video_frames).float()
        video_embeds = F.normalize(video_embeds, dim=-1)
        sim = torch.matmul(video_embeds, text_embeds.T)
        probs = F.softmax(sim, dim=-1).cpu().numpy()[0]
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

results.append([f"{start_time:.1f}s", f"{(len(clip_paths)) * CLIP_DURATION:.1f}s", prev_label])

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Start Time", "End Time", "Action"])
    writer.writerows(results)

print(f"Saved action segments to: {csv_path}")
