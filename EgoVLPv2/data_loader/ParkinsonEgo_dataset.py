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

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from base.base_dataset import TextVideoDataset


class ParkinsonEgo(TextVideoDataset):
    def __init__(self, dataset_name, text_params, video_params, data_dir, meta_dir=None, split='train', tsfms=None, cut=None, subsample=1, sliding_window_stride=-1, reader='decord', neg_param=None):
        super().__init__(dataset_name=dataset_name, text_params=text_params, video_params=video_params, data_dir=data_dir, meta_dir=meta_dir, split=split, tsfms=tsfms, cut=cut, subsample=subsample, sliding_window_stride=sliding_window_stride, reader=reader, neg_param=neg_param)
        
        # Build tokenizer robustly
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")

    def _load_metadata(self):
        # Load the CSV file for the current split
        csv_path = os.path.join(self.meta_dir, f'{self.split}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()  # Strip whitespace from headers
        if 'video_id' not in df.columns or 'source_video' not in df.columns:
            print(f"[DEBUG] Columns in {csv_path}: {df.columns.tolist()}")
            raise KeyError(f"'video_id' or 'source_video' column not found in {csv_path}")
        
        self.metadata = []
        for _, row in df.iterrows():
            video_id = row['video_id']  # e.g., video_0_clip_066
            source_video = row['source_video']  # e.g., video_0
            # Extract the numeric part after the last underscore
            try:
                clip_num = video_id.split('_')[-1]  # e.g., '066'
                clip_name = f"clip_{clip_num}"
            except IndexError:
                print(f"[DEBUG] Unexpected video_id format: {video_id}")
                continue
            video_path = os.path.join(self.data_dir, source_video, f"{clip_name}.mp4")
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
                continue
            self.metadata.append({
                'video_path': video_path,
                'action_label': row['action_label'],
                'start_time': row['start_time'],
                'end_time': row.get('end_time', None)
            })
        if len(self.metadata) == 0:
            raise ValueError(f"No valid samples found in {csv_path}")

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata[item]
        
        # Load video frames
        video_path = sample['video_path']
        try:
            imgs, idxs = self.video_reader(video_path, self.video_params['num_frames'])
            
            # Apply video transforms
            if self.transforms is not None:
                if self.video_params['num_frames'] > 1:
                    imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                    imgs = self.transforms(imgs)
                    imgs = imgs.transpose(0, 1)  # recover
                else:
                    imgs = self.transforms(imgs)
            
            # Create final tensor with padding if needed
            final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                               self.video_params['input_res']])
            final[:imgs.shape[0]] = imgs
            
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            # Return a different sample
            return self.__getitem__((item + 1) % len(self.metadata))
        
        # Tokenize action label
        text = sample['action_label']
        text_tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.text_params['max_length'], return_tensors='pt')
        
        return {
            'video': final,
            'text': text_tokens,
            'meta': {
                'video_path': video_path,
                'action_label': text,
                'start_time': sample['start_time'],
                'end_time': sample['end_time']
            }
        }

    def __len__(self):
        return len(self.metadata) 