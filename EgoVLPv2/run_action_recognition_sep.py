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

from model.model import FrozenInTime  # Make sure this import path is correct

from collections import OrderedDict

def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from state_dict keys (used when model was saved with DataParallel).
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict



# ==== Settings ====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = './checkpoints/EgoVLPv2_smallproj.pth'
CLIPS_DIR = './clips_egoclip'
PROMPTS = ["walking", "sitting", "standing"]
IMG_SIZE = 224
NUM_FRAMES = 128
CLIP_DURATION = 5

# ==== Preprocessing ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==== Load Model ====
print("Loading model...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

ckpt['state_dict'] = remove_module_prefix(ckpt['state_dict'])

ckpt_args = ckpt['config']['arch']['args']
ckpt_args['video_params']['num_frames'] = NUM_FRAMES
ckpt_args['projection_dim'] = 256
if 'projection' not in ckpt_args:
    ckpt_args['projection'] = 'minimal'

dummy_config = {
    'vocab_size': 50265,
    'hidden_size': 768,
    'num_layers': 12,
    'num_heads': 12,
    'mlp_ratio': 4,
    'drop_rate': 0.1,
    'input_image_embed_size': 768,
    'input_text_embed_size': 768,
    'num_fuse_block': 4,
    'use_checkpoint': False,
    'task_names': 'EgoNCE'
}

model = FrozenInTime(
    **ckpt_args,
    config=dummy_config,
    task_names='EgoNCE'
)

missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model = model.to(DEVICE)
model.eval()

# ==== Tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(ckpt_args['text_params']['model'])

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
    if len(frames) == 0:
        print(f"Warning: No frames read from {path}. Skipping this clip.")
        return None
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))

    frames = torch.stack(frames)
    frames = frames.unsqueeze(0).to(DEVICE)
    return frames

# ==== Inference ====
print("Running inference...")
clip_paths = sorted(glob.glob(os.path.join(CLIPS_DIR, 'clip_*.mp4')))
print(f"DEBUG: CLIPS_DIR is set to: {CLIPS_DIR}")
print(f"DEBUG: Found {len(clip_paths)} clips matching 'clip_*.mp4'.")

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

    results.append([f"{start_time:.1f}s", f"{(len(clip_paths)) * CLIP_DURATION:.1f}s", prev_label])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Start Time", "End Time", "Action"])
        writer.writerows(results)

    print(f"\nSaved action segments to: {csv_path}")
