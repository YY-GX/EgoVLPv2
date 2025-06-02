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

from model.model import FrozenInTime

# ==== Settings ====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = './checkpoints/EgoVLPv2_smallproj.pth'
CLIPS_DIR = './clips'
PROMPTS = ["a person is walking", "a person is sitting", "a person is standing"]

# ==== Preprocessing (you can tweak this) ====
IMG_SIZE = 224
NUM_FRAMES = 4  # sample N frames per video

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet
                         std=[0.229, 0.224, 0.225])
])

# ==== Load Model ====
print("Loading model...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model = FrozenInTime(**ckpt['config']['arch']['args'])
model.load_state_dict(ckpt['state_dict'], strict=False)
model = model.to(DEVICE)
model.eval()

# ==== Load Tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(ckpt['config']['arch']['args']['text_params']['model'])

with torch.no_grad():
    print("Encoding prompts...")
    text_inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_embeds = model.compute_text(text_inputs).float()  # [3, D]
    text_embeds = F.normalize(text_embeds, dim=-1)

# ==== Helper: Sample N evenly spaced frames from video ====
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
        frames.append(transform(img))  # shape: [3, H, W]

    cap.release()

    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))

    frames = torch.stack(frames)           # [T, 3, H, W]
    frames = frames.unsqueeze(0).to(DEVICE)  # [1, T, 3, H, W]
    return frames

# ==== Inference on clips ====
print("Running inference...")
clip_paths = sorted(glob.glob(os.path.join(CLIPS_DIR, '*.mp4')))

for clip_path in tqdm(clip_paths):
    video_frames = load_video_frames(clip_path)  # [1, T, C, H, W]
    print("video_frames.shape:", video_frames.shape)
    video_embeds = model.compute_video(video_frames).float()   # [1, D]
    video_embeds = F.normalize(video_embeds, dim=-1)

    # cosine sim
    sim = torch.matmul(video_embeds, text_embeds.T)  # [1, 3]
    pred_idx = sim.argmax(dim=-1).item()
    pred_label = PROMPTS[pred_idx]

    print(f"{os.path.basename(clip_path)} → {pred_label}")
