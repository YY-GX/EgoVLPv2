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
CLIPS_DIR = './clips'
# PROMPTS = ["a person is walking", "a person is sitting", "a person is standing"]
PROMPTS = ["walk around", "sit on the chair", "stand up"]


# ==== Preprocessing ====
IMG_SIZE = 224
NUM_FRAMES = 128  # sample N frames per video
CLIP_DURATION = 5  # seconds per clip

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==== Load Model ====
print("Loading model...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
ckpt['config']['arch']['args']['video_params']['num_frames'] = NUM_FRAMES
model = FrozenInTime(**ckpt['config']['arch']['args'])
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
