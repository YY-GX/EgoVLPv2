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
PROMPTS = ["#C C looks around", "#C C walks around", "#C C sits in the house"]
CLIP_DURATION = 5
NUM_FRAMES = 128
IMG_SIZE = 224

# ==== Preprocessing ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==== Load Model ====
print("Loading model...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
ckpt_args = ckpt['config']['arch']['args']
ckpt_args['video_params']['num_frames'] = NUM_FRAMES

# Use exact config to avoid projection mismatches
ckpt_args = ckpt['config']['arch']['args']  # already a dict
ckpt_args['video_params']['num_frames'] = NUM_FRAMES

model = FrozenInTime(
    **ckpt_args,
    config=ckpt['config'],  # pass as-is
    task_names='EgoNCE'  # hardcoded or fallback
)



missing_keys, _ = model.load_state_dict(ckpt['state_dict'], strict=False)
print("Missing keys:", missing_keys)

model = model.to(DEVICE)
model.eval()

# ==== Load Tokenizer ====
print("Using tokenizer:", ckpt_args['text_params']['model'])
tokenizer = AutoTokenizer.from_pretrained(ckpt_args['text_params']['model'])

with torch.no_grad():
    print("Encoding prompts...")
    text_inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_embeds = model.compute_text(text_inputs).float()
    print("text_embeds std dev:", text_embeds.std().item())
    text_embeds = F.normalize(text_embeds, dim=-1)

# ==== Frame Loader ====
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

# ==== Inference ====
print("Running inference...")
clip_paths = sorted(glob.glob(os.path.join(CLIPS_DIR, 'clip_*.mp4')))
print(f"DEBUG: Found {len(clip_paths)} clips.")

os.makedirs("runs", exist_ok=True)
csv_path = os.path.join("runs", "action_segments.csv")

prev_label = None
start_time = 0
results = []

if not clip_paths:
    print("Error: No clips found.")
else:
    for i, clip_path in enumerate(tqdm(clip_paths)):
        video_frames = load_video_frames(clip_path)
        if video_frames is None:
            continue
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

    results.append([f"{start_time:.1f}s", f"{(len(clip_paths)) * CLIP_DURATION:.1f}s", prev_label])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Start Time", "End Time", "Action"])
        writer.writerows(results)

    print(f"\nSaved action segments to: {csv_path}")
