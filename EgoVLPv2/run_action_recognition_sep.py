import os
import glob
import csv
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from model.model import FrozenInTime
from parse_config import ConfigParser
import cv2

# ==== Settings ====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG_PATH = './configs/pt/egoclip.json'
CHECKPOINT_PATH = './checkpoints/EgoVLPv2_smallproj.pth'
CLIPS_DIR = './clips_egoclip'
PROMPTS = ["#C C looks around", "#C C walks around", "#C C sits in the house"]
CLIP_DURATION = 5
NUM_FRAMES = 128
IMG_SIZE = 224

# ==== Transforms ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==== Load Config and Checkpoint ====
print("Loading config and checkpoint...")
config = ConfigParser.from_json(CONFIG_PATH)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# Inject required keys if missing
if 'use_checkpoint' not in config['config']:
    config['config']['use_checkpoint'] = False
if 'task_names' not in config['config']:
    config['config']['task_names'] = 'EgoNCE'

# Extract model args from config
arch_args = config['arch']['args']
arch_args['video_params']['num_frames'] = NUM_FRAMES

# Instantiate model
model = FrozenInTime(
    **arch_args,
    config=config['config'],
    task_names=config['config']['task_names']
)

missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model = model.to(DEVICE).eval()

# ==== Tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(arch_args['text_params']['model'])

with torch.no_grad():
    print("Encoding prompts...")
    text_inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_embeds = model.compute_text(text_inputs).float()
    print("text_embeds std dev:", text_embeds.std().item())
    text_embeds = F.normalize(text_embeds, dim=-1)

# ==== Load Clip Frames ====
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
        return None
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    return torch.stack(frames).unsqueeze(0).to(DEVICE)

# ==== Inference ====
print("Running inference...")
clip_paths = sorted(glob.glob(os.path.join(CLIPS_DIR, 'clip_*.mp4')))
os.makedirs("runs", exist_ok=True)
csv_path = os.path.join("runs", "action_segments.csv")

prev_label = None
start_time = 0
results = []

if not clip_paths:
    print("No clips found.")
else:
    for i, clip_path in enumerate(tqdm(clip_paths)):
        video_frames = load_video_frames(clip_path)
        if video_frames is None:
            continue

        with torch.no_grad():
            video_embeds = model.compute_video(video_frames).float()
            video_embeds = F.normalize(video_embeds, dim=-1)
            sim = torch.matmul(video_embeds, text_embeds.T)
            probs = F.softmax(sim, dim=-1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
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
