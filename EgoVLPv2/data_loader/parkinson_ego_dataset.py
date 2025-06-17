import os
import sys
import pandas as pd
import torch
from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_transform_dict, init_video_transform_dict
import decord
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

class ParkinsonEgo(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'train.csv',
            'val': 'val.csv',
            'test': 'test.csv'
        }
        target_split_fp = split_files[self.split]
        self.metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp))
        
        # Create a mapping from action labels to indices
        self.action_to_idx = {action: idx for idx, action in enumerate(sorted(self.metadata['action_label'].unique()))}
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}
        self.num_classes = len(self.action_to_idx)

    def _get_video_path(self, sample):
        video_id = sample['video_id']
        # Handle both single video collection and multiple collections
        if os.path.exists(os.path.join(self.data_dir, 'video_0', video_id + '.mp4')):
            video_fp = os.path.join(self.data_dir, 'video_0', video_id + '.mp4')
        else:
            # Search in all video_X directories
            for d in os.listdir(self.data_dir):
                if d.startswith('video_'):
                    potential_path = os.path.join(self.data_dir, d, video_id + '.mp4')
                    if os.path.exists(potential_path):
                        video_fp = potential_path
                        break
            else:
                raise FileNotFoundError(f"Video {video_id} not found in any video_X directory")
        
        return video_fp, video_id

    def _get_caption(self, sample):
        # For action recognition, we'll use the action label as the caption
        return sample['action_label']

    def get_frame_ids(self, total_frames, num_segments, jitter=True):
        """Get frame indices for sampling."""
        if total_frames <= num_segments:
            # If video is shorter than required frames, repeat frames
            indices = list(range(total_frames))
            indices = indices * (num_segments // len(indices) + 1)
            indices = indices[:num_segments]
        else:
            # Uniform sampling with optional jitter
            indices = []
            for i in range(num_segments):
                if jitter:
                    start = int(i * total_frames / num_segments)
                    end = int((i + 1) * total_frames / num_segments)
                    indices.append(torch.randint(start, end, (1,)).item())
                else:
                    indices.append(int((i + 0.5) * total_frames / num_segments))
        return indices

    def video_loader_by_frames(self, video_fp, frame_ids):
        """Load video frames using decord."""
        try:
            vr = decord.VideoReader(video_fp)
            frames = vr.get_batch(frame_ids).numpy()
            frames = torch.from_numpy(frames).float()
            return frames
        except Exception as e:
            print(f"Error loading video {video_fp}: {e}")
            # Return zero tensor of correct shape if video loading fails
            return torch.zeros((len(frame_ids), 224, 224, 3))

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, video_id = self._get_video_path(sample)
        action_label = self._get_caption(sample)
        
        # Get action index
        action_idx = self.action_to_idx[action_label]
        
        # Load video frames
        try:
            vr = decord.VideoReader(video_fp)
            total_frames = len(vr)
        except Exception as e:
            print(f"Error getting frame count for {video_fp}: {e}")
            total_frames = 150  # Assuming 5s video at 30fps
            
        frame_ids = self.get_frame_ids(
            total_frames,
            self.video_params['num_frames'],
            jitter=(self.split == 'train')
        )
        
        imgs = self.video_loader_by_frames(video_fp, frame_ids)
        
        # Apply transforms
        if self.split in ['test', 'val']:
            crop_size = self.video_params["input_res"]
            self.transforms = transforms.Compose([
                transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size),
                transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            ])
        else:  # 'train' split
            crop_size = self.video_params["input_res"]
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
                transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            ])
            
        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.permute(3, 0, 1, 2)  # T H W C -> C T H W
                imgs = self.transforms(imgs)
                imgs = imgs.permute(1, 0, 2, 3)  # C T H W -> T C H W
            else:
                imgs = self.transforms(imgs)
                
        # Create one-hot encoded target
        target = torch.zeros(self.num_classes)
        target[action_idx] = 1
        
        meta_arr = {
            'raw_captions': action_label,
            'paths': video_id,
            'dataset': self.dataset_name,
            'action_idx': action_idx
        }
        
        data = {
            'video': imgs,
            'text': action_label,
            'meta': meta_arr,
            'target': target
        }
        
        return data

    def __len__(self):
        return len(self.metadata) 