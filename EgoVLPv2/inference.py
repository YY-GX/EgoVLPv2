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
from datetime import timedelta

from model.model import FrozenInTime

# ==== Settings ====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = './checkpoints/EgoVLPv2_smallproj.pth'
CLIPS_DIR = './clips'
PROMPTS = [
    "The person is walking forward",
    "The person is standing still",
    "The person is sitting down on a chair"
]

IMG_SIZE = 224
NUM_FRAMES = 90  # 3 seconds at 30fps
STRIDE_FRAMES = 30  # 1 second stride

# ==== Transform ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== Load model ====
print("Loading model...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
ckpt['config']['arch']['args']['video_params']['num_frames'] = NUM_FRAMES
model = FrozenInTime(**ckpt['config']['arch']['args'])
model.load_state_dict(ckpt['state_dict'], strict=False)
model = model.to(DEVICE)
model.eval()

# ==== Tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(ckpt['config']['arch']['args']['text_params']['model'])

with torch.no_grad():
    print("Encoding prompts...")
    text_inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_embeds = model.compute_text(text_inputs).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

# ==== Helper: sliding window loader ====
def sliding_window_video(path, window_frames=NUM_FRAMES, stride_frames=STRIDE_FRAMES):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    windows = []
    for start in range(0, total_frames - window_frames + 1, stride_frames):
        frames = []
        for idx in range(start, start + window_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(transform(img))
        if len(frames) == window_frames:
            tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)
            timestamp = timedelta(seconds=start / fps)
            windows.append((timestamp, tensor))
    cap.release()
    return windows

# ==== Inference loop ====
print("Running sliding-window inference...")
clip_paths = sorted(glob.glob(os.path.join(CLIPS_DIR, '*.mp4')))
for clip_path in tqdm(clip_paths):
    windows = sliding_window_video(clip_path)
    last_label = None
    print(f"\n--- {os.path.basename(clip_path)} ---")
    for timestamp, frames in windows:
        with torch.no_grad():
            video_embeds = model.compute_video(frames).float()
            video_embeds = F.normalize(video_embeds, dim=-1)
            sim = torch.matmul(video_embeds, text_embeds.T)  # [1, num_prompts]
            pred_idx = sim.argmax(dim=-1).item()
            label = PROMPTS[pred_idx]
            if label != last_label:
                print(f"{str(timestamp)[:-3]} → {label}")
                last_label = label
