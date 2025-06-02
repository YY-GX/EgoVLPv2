# run_action_recognition_with_print.py

import os
import csv
import glob
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import cv2

from model.model import FrozenInTime

# === Settings ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = './checkpoints/EgoVLPv2_smallproj.pth'
CLIPS_DIR = './sliding_clips'
OUTPUT_CSV = 'action_segments.csv'
PROMPTS = ["The person is walking forward", "The person sits down on a chair", "The person is standing still"]
PROMPTS = ["The person is walking", "The person sits down on a chair", "The person is standing"]

IMG_SIZE = 224
NUM_FRAMES = 60  # 2s * 30fps
STRIDE_SEC = 1

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load Model ===
print("Loading model...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
ckpt['config']['arch']['args']['video_params']['num_frames'] = NUM_FRAMES
model = FrozenInTime(**ckpt['config']['arch']['args'])
model.load_state_dict(ckpt['state_dict'], strict=False)
model = model.to(DEVICE)
model.eval()

# === Load Tokenizer and Prompts ===
tokenizer = AutoTokenizer.from_pretrained(ckpt['config']['arch']['args']['text_params']['model'])
with torch.no_grad():
    text_inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_embeds = model.compute_text(text_inputs).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

# === Load Clip as Frames ===
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

    frames = torch.stack(frames)           # [T, 3, H, W]
    frames = frames.unsqueeze(0).to(DEVICE)  # [1, T, 3, H, W]
    return frames

# === Run Inference ===
print("Running action recognition...")
clip_paths = sorted(glob.glob(os.path.join(CLIPS_DIR, '*.mp4')))

previous_label = None
timestamps = []

for i, clip_path in enumerate(tqdm(clip_paths)):
    video_frames = load_video_frames(clip_path)
    with torch.no_grad():
        video_embeds = model.compute_video(video_frames).float()
        video_embeds = F.normalize(video_embeds, dim=-1)
        sim = torch.matmul(video_embeds, text_embeds.T)

        probs = F.softmax(sim, dim=-1)  # [1, num_prompts]
        confidence, pred_idx = torch.max(probs, dim=-1)

        pred_idx = sim.argmax(dim=-1).item()
        pred_label = PROMPTS[pred_idx]

    # Log and print if action changed
    if pred_label != previous_label:
        print(f"probabilities: {probs.cpu().numpy()}, confidence: {confidence.item():.4f}, predicted label: {pred_label}")
        seconds = i * STRIDE_SEC  # since stride is STRIDE_SEC
        print(f"[{seconds:.1f}s] {pred_label}")
        timestamps.append((seconds, pred_label))
        previous_label = pred_label

# === Write to CSV ===
with open(OUTPUT_CSV, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['start_time_sec', 'action'])
    writer.writerows(timestamps)

print(f"\nDone. Action segments saved to {OUTPUT_CSV}")
