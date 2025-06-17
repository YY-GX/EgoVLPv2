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
    def __init__(self, dataset_name, text_params, video_params, data_dir, meta_dir=None, split='train', tsfms=None, cut=None, subsample=None, sliding_window_stride=-1, reader='decord', neg_param=None):
        super().__init__(dataset_name=dataset_name, text_params=text_params, video_params=video_params, data_dir=data_dir, meta_dir=meta_dir, split=split, tsfms=tsfms, cut=cut, subsample=subsample, sliding_window_stride=sliding_window_stride, reader=reader, neg_param=neg_param)
        
        # Build tokenizer robustly
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")

    def _load_metadata(self):
        """Load metadata from CSV file."""
        metadata = []
        csv_path = os.path.join(os.path.dirname(self.data_dir), 'annotations', f'{self.split}.csv')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert video_id to string and pad with zeros to ensure 3 digits
                video_id = str(row['video_id']).zfill(3)
                
                # Construct video path
                video_path = os.path.join(self.data_dir, 'video_0', f'clip_{video_id}.mp4')
                
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
            # Get worker info and print immediately
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            num_workers = worker_info.num_workers if worker_info is not None else 1
            print(f"[DEBUG] Worker {worker_id}/{num_workers} starting to process item {item}", flush=True)
            
            item = item % len(self.metadata)
            sample = self.metadata[item]
            
            # Debug print at the start with worker ID and force flush
            print(f"[DEBUG] Worker {worker_id}/{num_workers} processing item {item} with sample: {sample}", flush=True)
            
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
                    print(f"[DEBUG] Worker {worker_id} Video file not found: {video_path}", flush=True)
                    if video_loading == 'strict':
                        raise FileNotFoundError(f"Video file not found: {video_path}")
                    frames = fallback_tensor
                else:
                    print(f"[DEBUG] Worker {worker_id} Loading video: {video_path}", flush=True)
                    frames, idxs = self.video_reader(video_path, self.video_params['num_frames'], frame_sample)
                    print(f"[DEBUG] Worker {worker_id} Video loaded, frames shape: {frames.shape if frames is not None else None}", flush=True)
                    if frames is None or frames.shape[0] == 0:
                        print(f"[DEBUG] Worker {worker_id} No frames loaded from {video_path}", flush=True)
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
                print(f"[DEBUG] Worker {worker_id} Error in video loading: {str(e)}", flush=True)
                if video_loading == 'strict':
                    raise ValueError(f'Video loading failed for {video_path}, video loading for this dataset is strict.') from e
                frames = fallback_tensor
            
            # Tokenize action label
            text = sample['action_label']
            text_tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.text_params['max_length'], return_tensors='pt')
            print(f"[DEBUG] Worker {worker_id} Text tokens created: {text_tokens.keys()}", flush=True)
            
            # Ensure all tensors are on CPU for distributed training
            frames = frames.cpu()
            text_tokens = {k: v.cpu() for k, v in text_tokens.items()}
            
            meta_arr = {
                'raw_captions': text,
                'paths': video_path,
                'dataset': self.dataset_name,
                'start_time': sample['start_time']
            }
            
            result = {
                'video': frames,
                'text': text_tokens,
                'meta': meta_arr
            }
            
            print(f"[DEBUG] Worker {worker_id} Returning result with keys: {result.keys()}", flush=True)
            return result
            
        except Exception as e:
            print(f"[DEBUG] Worker {worker_id} Critical error in __getitem__ for item {item}: {str(e)}", flush=True)
            print(f"[DEBUG] Worker {worker_id} Full traceback:", e.__traceback__, flush=True)
            raise  # Re-raise the exception so the worker can handle it

    def __len__(self):
        return len(self.metadata) 