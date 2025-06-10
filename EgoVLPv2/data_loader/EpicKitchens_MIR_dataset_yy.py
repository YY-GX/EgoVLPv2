# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os  # Ensure os is imported
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

# --- ADDED: For consistent frame rate calculation ---
# Assuming a standard video frame rate of 30 FPS.
# If your videos are different, adjust this.
ASSUMED_FPS = 30


# --- END ADDED ---

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
        self.relevancy = 0.1  # This value is likely from original code, might not be used for current eval
        self.relevancy_mat = pickle.load(pkl_file)  # This is your generated numpy array

        self.metadata = metadata
        self.metadata_sentence = metadata_sentence

    def _get_video_path(self, sample):
        # sample is a pandas Series for a row from EPIC_100_retrieval_test.csv
        pid = sample['participant_id']  # e.g., 'aria_P01'
        vid = sample['video_id']  # e.g., 'aria_P01_000'

        # Construct full path: data_dir/participant_id/video_id.MP4
        # self.data_dir is now the root of participant folders (e.g., .../video_ht256px/)
        rel_video_fp = '{}/{}.MP4'.format(pid, vid)  # e.g., 'aria_P01/aria_P01_000.MP4'
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)

        # --- NEW: Add fallback for .mp4 extension and robust existence check ---
        if not os.path.exists(full_video_fp):
            full_video_fp_lower = os.path.join(self.data_dir, pid, vid + '.mp4')  # Try lowercase extension
            if os.path.exists(full_video_fp_lower):
                full_video_fp = full_video_fp_lower
            else:
                # If neither .MP4 nor .mp4 works, raise error or try other structures
                raise FileNotFoundError(
                    f"Video file not found at expected path: {full_video_fp} or {full_video_fp_lower}. "
                    f"Check data_dir in config.json, CSV video_id/participant_id, and actual file names/casing.")
        # --- END NEW ---

        return full_video_fp, rel_video_fp  # rel_video_fp is not directly used for loading, but part of original return

    def _get_caption(self, item_idx, sample):  # Renamed idx to item_idx for clarity
        # return sentence, relevancy score, idx
        # For simplicity, we directly return the narration string and its mock_narration_id (as idx)
        # Relevancy score here is just a placeholder as it's for training/sampling logic.

        caption_text = sample['narration']  # Get the actual text query from the 'narration' column
        narration_id = sample['narration_id']  # Get the unique ID for this query

        # The 'relation' (relevancy score) from original code's _get_caption is
        # for training logic (e.g. positive/negative sampling).
        # For evaluation, we mainly need the caption_text and the narration_id (as idx).
        # We'll use 1 and -1 as dummy values for relevance and original idx.
        return caption_text, 1, narration_id  # Returning narration_id as idx (string)

    # --- MODIFIED: Simplified get_frame_ids to generate 0-based indices ---
    def get_frame_ids(self, total_frames_in_clip, num_frames_to_sample, jitter=True):
        """
        Generates 0-based frame indices to sample uniformly or with jitter from a clip.
        :param total_frames_in_clip: Actual number of frames in the current 5-second video clip.
        :param num_frames_to_sample: Number of frames to sample from this clip (e.g., 16).
        :param jitter: Whether to add random jitter for training (True) or uniform sampling (False).
        :return: A list of 0-based frame IDs.
        """
        if total_frames_in_clip == 0:
            return []  # No frames to sample from an empty clip

        if num_frames_to_sample > total_frames_in_clip:
            # If requesting more frames than available, sample all available and repeat the last one
            frame_ids = np.linspace(0, total_frames_in_clip - 1, num_frames_to_sample).astype(int)
        else:
            # Uniformly sample frames
            if jitter:  # For training (usually, though not used in eval-only here)
                seg_size = float(total_frames_in_clip) / num_frames_to_sample
                seq = []
                for i in range(num_frames_to_sample):
                    start = int(np.round(seg_size * i))
                    end = int(np.round(seg_size * (i + 1)))
                    frame_id = np.random.randint(low=start, high=(end))  # exclusive high range
                    seq.append(frame_id)
                frame_ids = np.array(seq, dtype=int)
            else:  # For validation/testing (uniform sampling)
                frame_ids = np.linspace(0, total_frames_in_clip - 1, num_frames_to_sample).astype(int)

        # Ensure indices are within valid 0-based bounds for the clip
        frame_ids = np.clip(frame_ids, 0, total_frames_in_clip - 1)
        return frame_ids.tolist()

    # --- END MODIFIED get_frame_ids ---

    # --- MODIFIED: video_loader_by_frames to get total frames from decord ---
    def video_loader_by_frames(self, video_fp, frame_ids):
        """
        Loads frames from a video using decord for specified frame_ids.
        Assumes frame_ids are 0-based relative to the opened video_fp.
        """
        try:
            vr = decord.VideoReader(video_fp)
            # Ensure frame_ids are valid for this specific video reader
            # (decord's _validate_indices already checks this, but it's good practice)

            # The 'Out of bound indices' error happens here if frame_ids are too large.
            # We are now ensuring frame_ids are 0-based and within bounds of the 5-sec clip.
            frames = vr.get_batch(frame_ids).numpy()
            frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
        except (IndexError, decord.DECORDError) as error:
            # Fallback if decord still fails for some reason (e.g., truly corrupted file)
            print(f"Error reading video {video_fp} with frame_ids {frame_ids}. Original error: {error}")
            print("Returning black frames.")
            # Ensure the returned tensor matches expected dimensions (C, T, H, W for transforms)
            # Input_res and num_frames from video_params
            dummy_frame_shape = (self.video_params['input_res'], self.video_params['input_res'], 3)  # H W C
            frames = [torch.zeros(dummy_frame_shape, dtype=torch.float32) for _ in range(len(frame_ids))]
            frames = torch.stack(frames, dim=0)  # T H W C
            return frames
            # Original code raised ValueError(), which could terminate DataLoader worker.
            # By returning zeros, we let the batch proceed.

        return torch.stack(frames, dim=0)  # T H W C

    # --- END MODIFIED video_loader_by_frames ---

    def datetime2sec(self, st):
        hh, mm, ss = st.split(':')
        return int(hh) * 3600 + int(mm) * 60 + float(ss)

    # --- MODIFIED __getitem__ START ---
    def __getitem__(self, item_idx):  # Renamed item to item_idx for consistency
        item_idx = item_idx % len(self.metadata)
        sample = self.metadata.iloc[item_idx]  # Get the current sample's metadata row

        video_fp, rel_fp = self._get_video_path(sample)  # Construct path to the 5-sec video clip

        # _get_caption now returns caption_text and narration_id
        caption_text, relation_dummy, narration_id = self._get_caption(item_idx, sample)

        # Determine sampling strategy (uniform for val/test)
        frame_sample_jitter = (self.split == 'train')  # Jitter for train, uniform for val/test

        # --- CRITICAL FIX START: Calculate 0-based frame_ids for the current 5-sec clip ---
        # Get total frames in THIS specific 5-second video clip
        try:
            vr = decord.VideoReader(video_fp)
            total_frames_in_current_5s_clip = len(vr)  # decord.VideoReader supports len() for total frames
        except decord.DECORDError as e:
            print(f"Error opening video {video_fp} for frame count: {e}. Returning dummy frames.")
            total_frames_in_current_5s_clip = self.video_params[
                'num_frames']  # Fallback: Assume enough frames for sampling
            # Optionally, return dummy data immediately or try loading next item
            # For robustness, we let video_loader_by_frames handle this gracefully with zeros.

        # Generate 0-based frame IDs for the current 5-second clip
        frame_ids = self.get_frame_ids(
            total_frames_in_current_5s_clip,
            self.video_params['num_frames'],  # Number of frames to sample (e.g., 16)
            jitter=frame_sample_jitter
        )
        # If frame_ids is empty due to an empty clip, try to load a dummy batch
        if not frame_ids:
            print(f"Warning: No frame_ids generated for {video_fp}. Likely empty or corrupt. Returning black frames.")
            # Manually create a black image stack
            imgs = torch.zeros(
                (self.video_params['num_frames'], self.video_params['input_res'], self.video_params['input_res'], 3),
                dtype=torch.float32)
        else:
            imgs = self.video_loader_by_frames(video_fp, frame_ids)  # Pass 0-based frame_ids
        # --- END CRITICAL FIX ---

        # --- Video Transforms (Original Logic) ---
        if self.split in ['test', 'val']:
            crop_size = self.video_params["input_res"]
            # Define transforms with standard ImageNet normalization values for pre-trained models
            self.transforms = transforms.Compose([
                transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size),
                # Note: These mean/std are for BGR image range 0-255. Ensure your video_loader returns BGR,
                # or adjust mean/std for RGB 0-255 or 0-1 range.
                # Common Pytorch models use RGB [0-1] mean/std (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                # The values below are common for models trained on ImageNet with 0-255 BGR input.
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
                # imgs is T H W C from video_loader_by_frames
                imgs = imgs.permute(3, 0, 1, 2)  # T H W C -> C T H W (required for some video transforms)
                imgs = self.transforms(imgs)
                imgs = imgs.permute(1, 0, 2, 3)  # C T H W -> T C H W (for model input)
            else:  # Single frame image input
                imgs = self.transforms(imgs)  # Assumes imgs is H W C
                imgs = imgs.permute(2, 0, 1)  # H W C -> C H W

        # Prepare metadata for output
        # 'paths': item holds the original index which is not the video_id.
        # We need to pass the actual video_id (e.g., aria_P01_000) for custom printing later.
        meta_arr = {
            'raw_captions': caption_text,
            'paths': sample['video_id'],  # Pass the actual video_id (e.g., aria_P01_000) here
            'dataset': self.dataset_name,
            'narration_id': narration_id  # Pass the narration_id (query ID)
        }
        # Assuming `relation` is not used in eval, pass the original relation (dummy) or 0
        data = {'video': imgs, 'text': caption_text, 'meta': meta_arr, 'relation': relation_dummy, 'item_v': item_idx,
                'item_t': item_idx}
        return data
    # --- END MODIFIED __getitem__ ---


# Original __main__ block (for testing dataset loading)
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
            "num_frames": 16,  # Match config
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