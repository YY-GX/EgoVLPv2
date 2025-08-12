# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import argparse
import collections
import transformers
import shutil
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model_epic_charades as module_arch
from parse_config import ConfigParser
from utils.util import replace_nested_dict_item
from data_loader.data_loader import dataset_loader
from data_loader.transforms import init_video_transform_dict

# --- Argument parsing (imitate multinode_train_epic.py) ---
parser = argparse.ArgumentParser(description='EgoVLPv2 Evaluation Script')
parser.add_argument('--task_names', default='EgoNCE_ITM_MLM', type=str, help='Task_Names')
parser.add_argument('-c', '--config', required=True, type=str, help='config file path (default: None)')
parser.add_argument('--model_ckpt', required=True, type=str, help='Path to model_best.pth checkpoint')
parser.add_argument('--output_dir', required=True, type=str, help='Directory to save confusion matrix and mis-predicted examples')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--annotation_mode', type=str, default='mode2', choices=['mode1', 'mode2'], help='Annotation split mode to use (mode1 or mode2)')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size for evaluation (overrides config)')
parser.add_argument('--num_workers', type=int, default=None, help='Number of DataLoader workers (overrides config)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# --- Config handling (imitate multinode_train_epic.py) ---
with open(args.config, 'r') as f:
    config_json = json.load(f)
config_json['annotation_mode'] = args.annotation_mode
meta_dir = config_json['data_loader']['args']['meta_dir']
if '{annotation_mode}' in meta_dir:
    config_json['data_loader']['args']['meta_dir'] = meta_dir.replace('{annotation_mode}', args.annotation_mode)

# Create a simple config object instead of using ConfigParser
class SimpleConfig:
    def __init__(self, config_dict):
        self._config = config_dict
    def __getitem__(self, key):
        return self._config[key]
    def initialize(self, name, module, *args, **kwargs):
        module_name = self._config[name]['type']
        module_args = dict(self._config[name]['args'])
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

config = SimpleConfig(config_json)

# --- Tokenizer ---
tokenizer = transformers.AutoTokenizer.from_pretrained(
    config['arch']['args']['text_params']['model'],
    TOKENIZERS_PARALLELISM=False
)

# --- Model ---
model = config.initialize('arch', module_arch)
checkpoint = torch.load(args.model_ckpt, map_location=args.device)

# Handle different checkpoint formats
if 'state_dict' in checkpoint:
    # Training checkpoint with metadata
    state_dict = checkpoint['state_dict']
    print(f"Loaded training checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
else:
    # Direct model weights
    state_dict = checkpoint
    print("Loaded direct model weights")

# Load the state dict with strict=False to handle missing keys
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
if missing_keys:
    print(f"Warning: Missing keys: {len(missing_keys)}")
if unexpected_keys:
    print(f"Warning: Unexpected keys: {len(unexpected_keys)}")

model = model.to(args.device)
model.eval()

# --- Test dataloader (imitate test loader logic) ---
dl_args = config['data_loader']['args']
dataset_name = dl_args['dataset_name']
text_params = dl_args['text_params']
video_params = dl_args['video_params']
data_dir = dl_args['data_dir']
meta_dir = dl_args['meta_dir']
split = 'test'
reader = dl_args['reader']
batch_size = args.batch_size if args.batch_size is not None else dl_args['batch_size']
num_workers = args.num_workers if args.num_workers is not None else dl_args['num_workers']
shuffle = False

tsfm_dict = init_video_transform_dict()
tsfm = tsfm_dict[split]
test_dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, meta_dir, split, tsfm, reader=reader)
from torch.utils.data import DataLoader
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=True
)
dataset_obj = test_loader.dataset

action_texts = ["sitting", "walking", "standing", "upstair", "downstair"]
num_classes = len(action_texts)
action_tokens = tokenizer(action_texts, padding=True, truncation=True, return_tensors='pt')
action_tokens = {k: v.to(args.device) for k, v in action_tokens.items()}
with torch.no_grad():
    # Check if model is wrapped in DataParallel
    if hasattr(model, 'module'):
        action_embeddings = model.module.compute_text(action_tokens)  # [num_classes, embed_dim]
    else:
        action_embeddings = model.compute_text(action_tokens)  # [num_classes, embed_dim]

all_predictions = []
all_labels = []
video_ids = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Evaluating'):
        # Move tensors to device
        batch['text'] = {k: v.to(args.device, non_blocking=True) for k, v in batch['text'].items()}
        batch['video'] = batch['video'].to(args.device, non_blocking=True)
        batch['label'] = batch['label'].to(args.device, non_blocking=True)

        # Get video embeddings
        if hasattr(model, 'module'):
            ret = model.module.infer(batch, return_embeds=True, task_names="Dual", ret={})
        else:
            ret = model.infer(batch, return_embeds=True, task_names="Dual", ret={})
        video_embeds = ret['video_embeds']  # [batch_size, embed_dim]

        # Compute distances and predictions
        distances = torch.cdist(video_embeds, action_embeddings)  # [batch_size, num_classes]
        predictions = torch.argmin(distances, dim=1)

        all_predictions.append(predictions.cpu())
        all_labels.append(batch['label'].cpu())

        # Extract video IDs
        meta = batch['meta']
        if isinstance(meta, list):
            if len(meta) > 0 and isinstance(meta[0], dict) and 'paths' in meta[0]:
                video_ids.extend([m['paths'] for m in meta])
            else:
                video_ids.extend([str(m) for m in meta])
        elif isinstance(meta, dict) and 'paths' in meta:
            video_ids.append(meta['paths'])
        else:
            video_ids.append(str(meta))

if all_predictions:
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    true_labels = all_labels.numpy()
    pred_labels = all_predictions.numpy()
    correct = (all_predictions == all_labels).sum().item()
    total = all_predictions.size(0)
    accuracy = correct / total if total > 0 else 0.0
    print(f"Test set accuracy: {accuracy:.4f} ({correct}/{total})")

    # Save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    np.save(os.path.join(args.output_dir, 'confusion_matrix.npy'), cm)

    # Mis-predicted stats
    mis_pred_counts = {i: sum((true_labels == i) & (pred_labels != i)) for i in range(num_classes)}
    ranked = sorted(mis_pred_counts.items(), key=lambda x: x[1], reverse=True)
    with open(os.path.join(args.output_dir, 'mis_pred_stats.txt'), 'w') as f:
        for idx, count in ranked:
            action_name = action_texts[idx] if idx < len(action_texts) else f"action_{idx}"
            f.write(f"Action {idx} ({action_name}): {count} mis-predicted\n")

    # Save up to 5 mis-predicted videos per action
    mis_pred_examples = {i: [] for i in range(num_classes)}
    for i, (t, p, vid) in enumerate(zip(true_labels, pred_labels, video_ids)):
        if t != p and len(mis_pred_examples[t]) < 5:
            mis_pred_examples[t].append((vid, t, p))
    for action_idx, examples in mis_pred_examples.items():
        for j, (vid, t, p) in enumerate(examples):
            try:
                folder, clip = vid.split('_', 1)
                video_fp = os.path.join(dataset_obj.data_dir, folder, f"{clip}.mp4")
                dst = os.path.join(args.output_dir, f'action{action_idx}_mis{j}_true{t}_pred{p}.mp4')
                shutil.copy(video_fp, dst)
            except Exception as e:
                print(f"[Warning] Could not copy video for video_id {vid}: {e}")
else:
    print("No predictions made.") 