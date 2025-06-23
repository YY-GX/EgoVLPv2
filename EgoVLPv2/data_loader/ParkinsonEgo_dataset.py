# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pdb
import sys
import json
import pandas as pd
import numpy as np
import torch
import transformers
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import csv
from pathlib import Path

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from base.base_dataset import TextVideoDataset


class ParkinsonEgo(TextVideoDataset):
    # Define the mapping from action_label string to integer index
    ACTION_LABELS = [
        'sitting', 'walking', 'standing', 'upstair', 'downstair'
    ]
    ACTION_LABEL_TO_IDX = {label: idx for idx, label in enumerate(ACTION_LABELS)}
    IDX_TO_ACTION_LABEL = {idx: label for label, idx in ACTION_LABEL_TO_IDX.items()}

    def __init__(self, dataset_name, text_params, video_params, data_dir, meta_dir=None, split='train', tsfms=None, cut=None, subsample=None, sliding_window_stride=-1, reader='decord', neg_param=None):
        super().__init__(dataset_name=dataset_name, text_params=text_params, video_params=video_params, data_dir=data_dir, meta_dir=meta_dir, split=split, tsfms=tsfms, cut=cut, subsample=subsample, sliding_window_stride=sliding_window_stride, reader=reader, neg_param=neg_param)
        
        # Build tokenizer robustly
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")

    def _load_metadata(self):
        """Load metadata from CSV file."""
        metadata = []
        # Use meta_dir directly if provided, otherwise construct from data_dir
        if self.meta_dir:
            csv_path = os.path.join(self.meta_dir, f'{self.split}.csv')
        else:
            csv_path = os.path.join(os.path.dirname(self.data_dir), 'annotations', f'{self.split}.csv')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract source_video and clip number from video_id
                video_id = row['video_id']  # e.g., "video_0_clip_001"
                
                # Parse the video_id to get source_video and clip number
                if '_clip_' in video_id:
                    source_video = video_id.split('_clip_')[0]  # e.g., "video_0"
                    clip_num = video_id.split('_clip_')[1]  # e.g., "001"
                else:
                    # Fallback: assume it's just a clip number and use video_0
                    source_video = 'video_0'
                    clip_num = video_id
                
                # Construct video path using the source_video
                video_path = os.path.join(self.data_dir, source_video, f'clip_{clip_num}.mp4')
                
                # Add to metadata
                metadata.append({
                    'video_path': video_path,
                    'action_label': row['action_label'],
                    'start_time': row['start_time']
                })
        
        self.metadata = metadata  # Assign the metadata to self.metadata
        return metadata

    def __getitem__(self, item):
        try:
            item = item % len(self.metadata)
            sample = self.metadata[item]
            
            # Load video frames
            video_path = sample['video_path']
            video_loading = self.video_params.get('loading', 'strict')
            frame_sample = 'rand'
            if self.split == 'test':
                frame_sample = 'uniform'
            
            # Create a zero tensor as fallback
            fallback_tensor = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                                         self.video_params['input_res']])
            
            try:
                if not os.path.isfile(video_path):
                    if video_loading == 'strict':
                        raise FileNotFoundError(f"Video file not found: {video_path}")
                    frames = fallback_tensor
                else:
                    frames, idxs = self.video_reader(video_path, self.video_params['num_frames'], frame_sample)
                    if frames is None or frames.shape[0] == 0:
                        if video_loading == 'strict':
                            raise ValueError(f"No frames loaded from {video_path}")
                        frames = fallback_tensor
                    
                    # Apply video transforms
                    if self.transforms is not None:
                        if self.video_params['num_frames'] > 1:
                            frames = frames.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                            frames = self.transforms(frames)
                            frames = frames.transpose(0, 1)  # recover
                        else:
                            frames = self.transforms(frames)
                    
                    # Create final tensor with padding if needed
                    final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                                       self.video_params['input_res']])
                    final[:frames.shape[0]] = frames
                    frames = final
                    
            except Exception as e:
                if video_loading == 'strict':
                    raise ValueError(f'Video loading failed for {video_path}, video loading for this dataset is strict.') from e
                frames = fallback_tensor
            
            # Tokenize action label
            text = sample['action_label']
            text_tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.text_params['max_length'], return_tensors='pt')
            
            # Ensure all tensors are on CPU for distributed training
            frames = frames.cpu()
            text_tokens = {k: v.squeeze(0).cpu() for k, v in text_tokens.items()}  # Remove batch dimension
            
            meta_arr = {
                'raw_captions': text,
                'paths': video_path,
                'dataset': self.dataset_name,
                'start_time': sample['start_time']
            }
            
            action_label_str = sample['action_label']
            assert action_label_str in self.ACTION_LABEL_TO_IDX, f"Unknown action_label: {action_label_str}"
            label = self.ACTION_LABEL_TO_IDX[action_label_str]
            
            result = {
                'video': frames,
                'text': text_tokens,
                'meta': meta_arr,
                'label': torch.tensor(label, dtype=torch.long)
            }
            
            return result
            
        except Exception as e:
            raise  # Re-raise the exception so the worker can handle it

    def __len__(self):
        return len(self.metadata) 