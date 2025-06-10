# Open /mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data_loader/EpicKitchens_MIR_dataset_yy.py

import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_transform_dict, init_video_transform_dict

import torch
from PIL import Image
from torchvision import transforms
import decord
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

ASSUMED_FPS = 30


class MultiInstanceRetrieval(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'EPIC_100_retrieval_train.csv',
            'val': 'EPIC_100_retrieval_test.csv',
            'test': 'EPIC_100_retrieval_test.csv'
        }
        split_files_sentence = {
            'train': 'EPIC_100_retrieval_train_sentence.csv',
            'val': 'EPIC_100_retrieval_test_sentence.csv',
            'test': 'EPIC_100_retrieval_test_sentence.csv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp))

        target_split_sentence_fp = split_files_sentence[self.split]
        metadata_sentence = pd.read_csv(os.path.join(self.meta_dir, target_split_sentence_fp))

        if self.split == 'train':
            path_relevancy = os.path.join(self.meta_dir, 'relevancy/caption_relevancy_EPIC_100_retrieval_train.pkl')
        elif self.split in ['val', 'test']:
            path_relevancy = os.path.join(self.meta_dir, 'relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl')

        pkl_file = open(path_relevancy, 'rb')
        self.relevancy = 0.1
        self.relevancy_mat = pickle.load(pkl_file)

        self.metadata = metadata
        self.metadata_sentence = metadata_sentence

        # --- NEW: Create a mapping from string video_id to numerical index ---
        # This is crucial for passing IDs as tensors for all_gather.
        all_video_ids_in_dataset = self.metadata['video_id'].unique().tolist()
        self.video_id_to_numerical_idx = {vid_id: i for i, vid_id in enumerate(all_video_ids_in_dataset)}
        self.numerical_idx_to_video_id = {i: vid_id for vid_id, i in self.video_id_to_numerical_idx.items()}
        # --- END NEW ---

    def _get_video_path(self, sample):
        pid = sample['participant_id']
        vid = sample['video_id']

        rel_video_fp = '{}/{}.MP4'.format(pid, vid)
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)

        if not os.path.exists(full_video_fp):
            full_video_fp_lower = os.path.join(self.data_dir, pid, vid + '.mp4')
            if os.path.exists(full_video_fp_lower):
                full_video_fp = full_video_fp_lower
            else:
                raise FileNotFoundError(
                    f"Video file not found at expected path: {full_video_fp} or {full_video_fp_lower}. "
                    f"Check data_dir in config.json, CSV video_id/participant_id, and actual file names/casing.")

        return full_video_fp, rel_video_fp

    def _get_caption(self, item_idx, sample):
        caption_text = sample['narration']
        narration_id = sample['narration_id']
        return caption_text, 1, narration_id

    def get_frame_ids(self, total_frames_in_clip, num_frames_to_sample, jitter=True):
        if total_frames_in_clip == 0:
            return []

        if num_frames_to_sample > total_frames_in_clip:
            frame_ids = np.linspace(0, total_frames_in_clip - 1, num_frames_to_sample).astype(int)
        else:
            if jitter:
                seg_size = float(total_frames_in_clip) / num_frames_to_sample
                seq = []
                for i in range(num_frames_to_sample):
                    start = int(np.round(seg_size * i))
                    end = int(np.round(seg_size * (i + 1)))
                    frame_id = np.random.randint(low=start, high=(end))
                    seq.append(frame_id)
                frame_ids = np.array(seq, dtype=int)
            else:
                frame_ids = np.linspace(0, total_frames_in_clip - 1, num_frames_to_sample).astype(int)

        frame_ids = np.clip(frame_ids, 0, total_frames_in_clip - 1)
        return frame_ids.tolist()

    def video_loader_by_frames(self, video_fp, frame_ids):
        try:
            vr = decord.VideoReader(video_fp)
            frames = vr.get_batch(frame_ids).numpy()
            frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
        except (IndexError, decord.DECORDError) as error:
            print(f"Error reading video {video_fp} with frame_ids {frame_ids}. Original error: {error}")
            print("Returning black frames.")
            dummy_frame_shape = (self.video_params['input_res'], self.video_params['input_res'], 3)
            frames = [torch.zeros(dummy_frame_shape, dtype=torch.float32) for _ in range(len(frame_ids))]
            frames = torch.stack(frames, dim=0)

        return torch.stack(frames, dim=0)

    def datetime2sec(self, st):
        hh, mm, ss = st.split(':')
        return int(hh) * 3600 + int(mm) * 60 + float(ss)

    def __getitem__(self, item_idx):
        item_idx = item_idx % len(self.metadata)
        sample = self.metadata.iloc[item_idx]
        video_fp, rel_fp = self._get_video_path(sample)
        caption_text, relation_dummy, narration_id = self._get_caption(item_idx, sample)

        frame_sample_jitter = (self.split == 'train')

        try:
            vr = decord.VideoReader(video_fp)
            total_frames_in_current_5s_clip = len(vr)
        except decord.DECORDError as e:
            print(
                f"Error opening video {video_fp} for frame count: {e}. This clip might be corrupted. Returning dummy frames.")
            total_frames_in_current_5s_clip = ASSUMED_FPS * 5  # Fallback, assumes a 5-sec clip if reader fails
            frame_ids = self.get_frame_ids(
                total_frames_in_current_5s_clip,
                self.video_params['num_frames'],
                jitter=frame_sample_jitter
            )
            imgs = torch.zeros(
                (self.video_params['num_frames'], self.video_params['input_res'], self.video_params['input_res'], 3),
                dtype=torch.float32)
            # Ensure imgs has the correct shape for transforms before permute
            imgs = imgs.permute(3, 0, 1, 2)  # H W C -> C T H W if num_frames > 1
            if self.video_params['num_frames'] == 1:
                imgs = imgs.squeeze(1)  # Remove T dim if num_frames is 1 for single image transforms
            # Skip transforms for dummy frames as they are already zeros and normalized values don't make sense.
            # Just ensure final shape is T C H W
            imgs = imgs.permute(1, 0, 2, 3)  # C T H W -> T C H W

            # --- NEW: Return dummy data for a corrupted video ---
            # Ensure the dummy data has the correct expected tensor shapes
            meta_arr = {
                'raw_captions': caption_text,
                'paths': torch.tensor(self.video_id_to_numerical_idx[sample['video_id']], dtype=torch.long),
                # Numerical ID
                'dataset': self.dataset_name,
                'narration_id': narration_id,
                'is_corrupted': True  # Flag for debugging
            }
            data = {'video': imgs, 'text': caption_text, 'meta': meta_arr, 'relation': relation_dummy,
                    'item_v': item_idx, 'item_t': item_idx}
            return data
            # --- END NEW: Return dummy data ---

        frame_ids = self.get_frame_ids(
            total_frames_in_current_5s_clip,
            self.video_params['num_frames'],
            jitter=frame_sample_jitter
        )

        imgs = self.video_loader_by_frames(video_fp, frame_ids)

        if self.split in ['test', 'val']:
            crop_size = self.video_params["input_res"]
            self.transforms = transforms.Compose([
                transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size),
                transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            ])
        else:
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
                imgs = imgs.permute(2, 0, 1)  # H W C -> C H W

        meta_arr = {
            'raw_captions': caption_text,
            'paths': torch.tensor(self.video_id_to_numerical_idx[sample['video_id']], dtype=torch.long),  # Numerical ID
            'dataset': self.dataset_name,
            'narration_id': narration_id  # Pass the narration_id (query ID)
        }
        data = {'video': imgs, 'text': caption_text, 'meta': meta_arr, 'relation': relation_dummy, 'item_v': item_idx,
                'item_t': item_idx}
        return data


if __name__ == "__main__":
    # Ensure YOUR_MOCK_EK100_ROOT is correctly set if testing this block
    # For a quick test, you might hardcode paths here temporarily
    # e.g., DATA_DIR = "/path/to/your/my_simulated_ek100_data/EK100/video_ht256px"
    #        META_DIR = "/path/to/your/my_simulated_ek100_data/EK100/epic-kitchens-100-annotations/retrieval_annotations"

    kwargs = dict(
        dataset_name="MultiInstanceRetrieval",
        text_params={
            "input": "text"
        },
        video_params={
            "input_res": 224,
            "num_frames": 4,
            "loading": "lax"
        },
        data_dir="/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data/my_simulated_ek100_data/EK100/video_ht256px",
        # Example path
        meta_dir="/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data/my_simulated_ek100_data/EK100/epic-kitchens-100-annotations/retrieval_annotations",
        # Example path
        tsfms=init_video_transform_dict()['test'],
        reader='cv2_epic',
        split='test'  # Use 'test' to load your mock data
    )
    dataset = MultiInstanceRetrieval(**kwargs)
    print(f"Dataset has {len(dataset)} items.")
    if len(dataset) > 0:
        for i in range(min(10, len(dataset))):  # Print first 10 items
            item = dataset[i]
            print(
                f"Item {i}: video_id={item['meta']['paths']}, query={item['meta']['narration_id']}, text_len={len(item['text'])}")
            print(f"  Video shape: {item['video'].shape}")
            # print(item.keys()) # Original line, now print item properties
    else:
        print("Dataset is empty. Check your metadata files and paths.")