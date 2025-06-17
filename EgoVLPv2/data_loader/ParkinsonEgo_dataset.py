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

from base.base_dataset import TextVideoDataset
from base.transforms import init_transform_dict, init_video_transform_dict


class ParkinsonEgo(TextVideoDataset):
    def __init__(self, dataset_name, text_params, video_params, data_dir, meta_dir=None, split='train', tsfms=None, cut=None, subsample=1, sliding_window_stride=-1, reader='decord', neg_param=None):
        super().__init__(dataset_name=dataset_name, text_params=text_params, video_params=video_params, data_dir=data_dir, meta_dir=meta_dir, split=split, tsfms=tsfms, cut=cut, subsample=subsample, sliding_window_stride=sliding_window_stride, reader=reader, neg_param=neg_param)
        
        # Build tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(text_params['model'])

    def _load_metadata(self):
        # Load the CSV file for the current split
        csv_path = os.path.join(self.meta_dir, f'{self.split}.csv')
        df = pd.read_csv(csv_path)
        
        self.metadata = []
        for _, row in df.iterrows():
            self.metadata.append({
                'video_path': os.path.join(self.data_dir, row['video_path']),
                'action_label': row['action_label'],
                'start_time': row['start_time'],
                'end_time': row['end_time']
            })

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata[item]
        
        # Load video frames
        video_path = sample['video_path']
        frames = self.video_reader(video_path, self.video_params['num_frames'])
        
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