import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import cv2

from model.model import FrozenInTime

# ==== Environment ====
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==== Settings ====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = './checkpoints/EgoVLPv2_smallproj.pth'
VIDEO_PATH = './clips/yy_1.vrs.mp4'
PROMPTS = ["The person is walking forward", "The person is standing still", "The person sits down on a chair"]
OUTPUT_CSV = 'action_segments.csv'

IMG_SIZE = 224
FPS = 30
WINDOW_SECONDS = 3
STEP_SECONDS = 0.5
NUM_FRAMES = int(FPS * WINDOW_SECONDS)
STEP_FRAMES = int(FPS * STEP_SECONDS)

# ==== Transforms ====
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
model = model.to(DEVICE).eval()

# ==== Tokenizer & Prompt Embeddings ====
print("Encoding prompts...")
tokenizer = AutoTokenizer.from_pretrained(ckpt['config']['arch']['args']['text_params']['model'])
with torch.no_grad():
    text_inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_embeds = model.compute_text(text_inputs).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

# ==== Load video ====
print(f"Reading video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_buffer = []

for _ in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_buffer.append(transform(img))

cap.release()
print(f"Loaded {len(frame_buffer)} frames")

# ==== Sliding window inference ====
segments = []
prev_label = None

print("Running sliding window inference...")
for start_idx in tqdm(range(0, len(frame_buffer) - NUM_FRAMES + 1, STEP_FRAMES)):
    end_idx = start_idx + NUM_FRAMES
    clip = frame_buffer[start_idx:end_idx]
    video_tensor = torch.stack(clip).unsqueeze(0).to(DEVICE)  # [1, T, 3, H, W]

    with torch.no_grad():
        video_embeds = model.compute_video(video_tensor).float()
        video_embeds = F.normalize(video_embeds, dim=-1)
        sim = torch.matmul(video_embeds, text_embeds.T)
        pred_idx = sim.argmax(dim=-1).item()
        pred_label = PROMPTS[pred_idx]

    timestamp = start_idx / FPS
    if pred_label != prev_label:
        segments.append((timestamp, pred_label))
        prev_label = pred_label

# ==== Save results ====
with open(OUTPUT_CSV, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['start_time_sec', 'action'])
    for t, action in segments:
        writer.writerow([round(t, 2), action])

print(f"\n✓ Done. Action segments saved to: {OUTPUT_CSV}")
