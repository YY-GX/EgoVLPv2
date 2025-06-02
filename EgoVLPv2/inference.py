import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import cv2
import multiprocessing as mp

from model.model import FrozenInTime

# ==== Config ====
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = './checkpoints/EgoVLPv2_smallproj.pth'
CLIPS_DIR = './clips'
LOG_CSV = 'inference_results.csv'
PROMPTS = [
    "The person is walking forward",
    "The person is standing still",
    "The person sits down on a chair"
]
IMG_SIZE = 224
NUM_FRAMES = 90
WINDOW_SIZE_S = 3
FPS = 30
SLIDE_STRIDE = FPS  # stride of 1s

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== Global Model and Token Embeds ====
model = None
text_embeds = None


def init_model():
    global model, text_embeds
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    ckpt['config']['arch']['args']['video_params']['num_frames'] = NUM_FRAMES
    model = FrozenInTime(**ckpt['config']['arch']['args'])
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(DEVICE).eval()

    tokenizer = AutoTokenizer.from_pretrained(ckpt['config']['arch']['args']['text_params']['model'])
    text_inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        text_embeds = model.compute_text(text_inputs).float()
        text_embeds = F.normalize(text_embeds, dim=-1)


def extract_frames(path):
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in range(frame_count):
        success, frame = cap.read()
        if not success: break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(transform(img))
    cap.release()
    return torch.stack(frames)  # [T, 3, H, W]


def infer_one_clip(path):
    global model, text_embeds
    frames = extract_frames(path)
    T = frames.shape[0]
    results = []

    for start in range(0, T - NUM_FRAMES + 1, SLIDE_STRIDE):
        clip = frames[start:start + NUM_FRAMES]
        clip = clip.unsqueeze(0).to(DEVICE)  # [1, T, 3, H, W]
        with torch.no_grad():
            video_embed = model.compute_video(clip).float()
            video_embed = F.normalize(video_embed, dim=-1)
            sim = torch.matmul(video_embed, text_embeds.T)
            pred_idx = sim.argmax(dim=-1).item()
            results.append((start / FPS, PROMPTS[pred_idx]))

    # Collapse to change points
    collapsed = []
    prev_label = None
    for ts, label in results:
        if label != prev_label:
            collapsed.append((ts, label))
            prev_label = label

    return [(os.path.basename(path), ts, label) for ts, label in collapsed]


def main():
    init_model()
    clip_paths = sorted(glob.glob(os.path.join(CLIPS_DIR, '*.mp4')))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        all_results = list(tqdm(pool.imap(infer_one_clip, clip_paths), total=len(clip_paths)))

    flat_results = [item for sublist in all_results for item in sublist]
    df = pd.DataFrame(flat_results, columns=["clip", "start_time_s", "pred_label"])
    df.to_csv(LOG_CSV, index=False)
    print(f"Saved results to {LOG_CSV}")


if __name__ == '__main__':
    main()
