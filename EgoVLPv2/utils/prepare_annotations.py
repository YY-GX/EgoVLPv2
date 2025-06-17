import os
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from pathlib import Path
import glob

def parse_timestamp(timestamp):
    """Convert timestamp string to seconds."""
    try:
        if timestamp.strip() == 'end':
            return None
        dt = datetime.strptime(timestamp.strip(), '%M:%S')
        return dt.minute * 60 + dt.second
    except ValueError:
        # Try with hours if minutes:seconds format fails
        dt = datetime.strptime(timestamp.strip(), '%H:%M:%S')
        return dt.hour * 3600 + dt.minute * 60 + dt.second

def get_action_at_time(timestamp, annotations):
    """Get the action label for a given timestamp."""
    timestamp_sec = parse_timestamp(timestamp)
    
    # Find the last action that started before this timestamp
    for i in range(len(annotations) - 1):
        current_start = parse_timestamp(annotations.iloc[i]['timestamp'])
        next_start = parse_timestamp(annotations.iloc[i + 1]['timestamp'])
        
        # Handle the case where next_start is 'end'
        if next_start is None:
            if current_start <= timestamp_sec:
                return annotations.iloc[i]['action_label']
        else:
            if current_start <= timestamp_sec < next_start:
                return annotations.iloc[i]['action_label']
    
    # If timestamp is after the last action, return the last action
    return annotations.iloc[-2]['action_label']  # -2 because the last row is 'end'

def generate_clip_annotations_for_video(raw_annotation_file, clip_duration, video_id):
    """
    Generate clip annotations for a single video.
    
    Args:
        raw_annotation_file (str): Path to the raw annotation CSV file
        clip_duration (int): Duration of each clip in seconds
        video_id (str): ID of the video (e.g., 'video_0')
    
    Returns:
        pd.DataFrame: DataFrame containing clip annotations
    """
    # Read raw annotations
    annotations = pd.read_csv(raw_annotation_file)
    # Strip whitespace from column names
    annotations.columns = annotations.columns.str.strip()
    
    # Get video duration from the 'end' timestamp
    end_timestamp = annotations.iloc[-1]['timestamp']
    if end_timestamp.strip() != 'end':
        raise ValueError(f"Last row of {raw_annotation_file} must have 'end' as timestamp")
    last_timestamp = parse_timestamp(annotations.iloc[-2]['timestamp'])  # Get timestamp of last action
    end_time = parse_timestamp(end_timestamp)  # Get the end time
    
    # Generate clip timestamps
    clip_starts = np.arange(0, end_time, clip_duration)
    
    # Create clips dataframe
    clips = []
    for start_time in clip_starts:
        # Convert start_time back to MM:SS format
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"
        
        # Get action label for this timestamp
        action_label = get_action_at_time(timestamp, annotations)
        
        # Generate clip ID
        clip_id = f"{video_id}_clip_{int(start_time):03d}"
        
        clips.append({
            'video_id': clip_id,
            'action_label': action_label,
            'start_time': timestamp,
            'source_video': video_id
        })
    
    return pd.DataFrame(clips)

def generate_combined_annotations(annotation_dir, clip_duration, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, output_dir=None):
    """
    Generate train/val/test annotation files from multiple raw timestamp annotations.
    
    Args:
        annotation_dir (str): Directory containing raw annotation CSV files
        clip_duration (int): Duration of each clip in seconds
        train_ratio (float): Ratio of clips for training
        val_ratio (float): Ratio of clips for validation
        test_ratio (float): Ratio of clips for testing
        output_dir (str): Directory to save the generated annotation files
    """
    # Find all raw annotation files
    raw_annotation_files = glob.glob(os.path.join(annotation_dir, 'video_*.csv'))
    if not raw_annotation_files:
        raise ValueError(f"No raw annotation files found in {annotation_dir}")
    
    # Generate clips for each video
    all_clips = []
    for raw_file in raw_annotation_files:
        video_id = os.path.splitext(os.path.basename(raw_file))[0]  # Get video_0, video_1, etc.
        clips_df = generate_clip_annotations_for_video(raw_file, clip_duration, video_id)
        all_clips.append(clips_df)
    
    # Combine all clips
    combined_clips = pd.concat(all_clips, ignore_index=True)
    
    # Shuffle clips
    combined_clips = combined_clips.sample(frac=1, random_state=42)
    
    # Split into train/val/test
    n_clips = len(combined_clips)
    n_train = int(n_clips * train_ratio)
    n_val = int(n_clips * val_ratio)
    
    # Create copies of the splits to avoid SettingWithCopyWarning
    train_df = combined_clips.iloc[:n_train].copy()
    val_df = combined_clips.iloc[n_train:n_train + n_val].copy()
    test_df = combined_clips.iloc[n_train + n_val:].copy()
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Combine all splits back for statistics
    all_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = annotation_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV files
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Print statistics
    print(f"\nGenerated annotation files in {output_dir}:")
    print(f"Total clips: {n_clips}")
    print(f"Train: {len(train_df)} clips")
    print(f"Val: {len(val_df)} clips")
    print(f"Test: {len(test_df)} clips")
    
    # Print statistics per video
    print("\nClips per video:")
    for video_id in combined_clips['source_video'].unique():
        video_clips = all_splits[all_splits['source_video'] == video_id]
        print(f"\n{video_id}:")
        print(f"  Total: {len(video_clips)} clips")
        print(f"  Train: {len(video_clips[video_clips['split'] == 'train'])} clips")
        print(f"  Val: {len(video_clips[video_clips['split'] == 'val'])} clips")
        print(f"  Test: {len(video_clips[video_clips['split'] == 'test'])} clips")

def main():
    parser = argparse.ArgumentParser(description='Generate train/val/test annotation files from raw timestamps')
    parser.add_argument('--annotation_dir', type=str, required=True, 
                      help='Directory containing raw annotation CSV files (video_*.csv)')
    parser.add_argument('--clip_duration', type=int, required=True, 
                      help='Duration of each clip in seconds')
    parser.add_argument('--train_ratio', type=float, default=0.7, 
                      help='Ratio of clips for training')
    parser.add_argument('--val_ratio', type=float, default=0.15, 
                      help='Ratio of clips for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15, 
                      help='Ratio of clips for testing')
    parser.add_argument('--output_dir', type=str, 
                      help='Directory to save the generated annotation files')
    
    args = parser.parse_args()
    
    generate_combined_annotations(
        args.annotation_dir,
        args.clip_duration,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.output_dir
    )

if __name__ == '__main__':
    main() 