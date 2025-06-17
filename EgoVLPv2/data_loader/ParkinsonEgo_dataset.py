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
        
        # Build tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(text_params['model'])

    def _load_metadata(self):
        # Load the CSV file for the current split
        csv_path = os.path.join(self.meta_dir, f'{self.split}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()  # Strip whitespace from headers
        
        self.metadata = []
        for _, row in df.iterrows():
            video_path = os.path.join(self.data_dir, row['video_path'])
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
                continue
                
            self.metadata.append({
                'video_path': video_path,
                'action_label': row['action_label'],
                'start_time': row['start_time'],
                'end_time': row['end_time']
            })
            
        if len(self.metadata) == 0:
            raise ValueError(f"No valid samples found in {csv_path}")

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata[item]
        
        # Load video frames
        video_path = sample['video_path']
        try:
            frames = self.video_reader(video_path, self.video_params['num_frames'])
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            # Return a different sample
            return self.__getitem__((item + 1) % len(self.metadata))
        
        # Apply video transforms
        if self.transforms is not None:
            frames = self.transforms(frames)
        
        # Tokenize action label
        text = sample['action_label']
        text_tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.text_params['max_length'], return_tensors='pt')
        
        return {
            'video': frames,
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