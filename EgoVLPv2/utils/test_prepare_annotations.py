#!/usr/bin/env python3
"""
Test script for prepare_annotations.py
This script helps verify that the annotation preparation works correctly.
"""

import os
import pandas as pd
import tempfile
import shutil
from pathlib import Path

def create_test_annotations():
    """Create test annotation files for video_0 through video_5."""
    
    # Test data structure - can be extended to any number of videos
    test_annotations = {
        'video_0': [
            {'timestamp': '00:00', 'action_label': 'sitting'},
            {'timestamp': '00:30', 'action_label': 'standing'},
            {'timestamp': '01:00', 'action_label': 'walking'},
            {'timestamp': '01:30', 'action_label': 'sitting'}
        ],
        'video_1': [
            {'timestamp': '00:00', 'action_label': 'walking'},
            {'timestamp': '00:45', 'action_label': 'upstair'},
            {'timestamp': '01:15', 'action_label': 'downstair'}
        ],
        'video_2': [
            {'timestamp': '00:00', 'action_label': 'standing'},
            {'timestamp': '00:20', 'action_label': 'walking'},
            {'timestamp': '00:50', 'action_label': 'sitting'}
        ],
        'video_3': [
            {'timestamp': '00:00', 'action_label': 'upstair'},
            {'timestamp': '00:40', 'action_label': 'downstair'},
            {'timestamp': '01:00', 'action_label': 'walking'}
        ],
        'video_4': [
            {'timestamp': '00:00', 'action_label': 'sitting'},
            {'timestamp': '00:25', 'action_label': 'standing'},
            {'timestamp': '00:55', 'action_label': 'walking'}
        ],
        'video_5': [
            {'timestamp': '00:00', 'action_label': 'walking'},
            {'timestamp': '00:35', 'action_label': 'upstair'},
            {'timestamp': '01:05', 'action_label': 'downstair'}
        ],
        # Add more test videos to demonstrate flexibility
        'video_10': [
            {'timestamp': '00:00', 'action_label': 'standing'},
            {'timestamp': '00:30', 'action_label': 'walking'},
            {'timestamp': '01:00', 'action_label': 'sitting'}
        ],
        'video_15': [
            {'timestamp': '00:00', 'action_label': 'upstair'},
            {'timestamp': '00:25', 'action_label': 'downstair'},
            {'timestamp': '00:50', 'action_label': 'walking'}
        ]
    }
    
    return test_annotations

def create_test_video_structure(base_dir, video_id, num_clips=10):
    """Create test video directory structure."""
    video_dir = os.path.join(base_dir, video_id)
    os.makedirs(video_dir, exist_ok=True)
    
    # Create dummy clip files with correct naming
    for i in range(num_clips):
        clip_path = os.path.join(video_dir, f"clip_{i:03d}.mp4")
        with open(clip_path, 'w') as f:
            f.write(f"dummy video clip {i}")
    
    return video_dir

def test_prepare_annotations():
    """Test the prepare_annotations script."""
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Create test annotation files
        annotation_dir = os.path.join(temp_dir, "annotations")
        os.makedirs(annotation_dir, exist_ok=True)
        
        test_data = create_test_annotations()
        
        for video_id, annotations in test_data.items():
            annotation_file = os.path.join(annotation_dir, f"{video_id}.csv")
            df = pd.DataFrame(annotations)
            df.to_csv(annotation_file, index=False)
            print(f"Created {annotation_file} with {len(annotations)} annotations")
        
        # Create test video directory structure
        video_dir = os.path.join(temp_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        for video_id in test_data.keys():
            create_test_video_structure(video_dir, video_id, num_clips=8)
            print(f"Created video directory: {video_dir}/{video_id}")
        
        # Test the prepare_annotations script
        print("\n" + "="*50)
        print("Testing prepare_annotations.py")
        print("="*50)
        
        try:
            # Import and run the script
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            from prepare_annotations import generate_combined_annotations
            
            # Run the annotation generation
            output_dir = os.path.join(temp_dir, "output")
            generate_combined_annotations(
                annotation_dir=annotation_dir,
                clip_duration=30,  # 30 seconds per clip
                video_dir=video_dir,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                output_dir=output_dir
            )
            
            # Verify output files
            print("\n" + "="*50)
            print("Verifying output files")
            print("="*50)
            
            for split in ['train', 'val', 'test']:
                output_file = os.path.join(output_dir, f"{split}.csv")
                if os.path.exists(output_file):
                    df = pd.read_csv(output_file)
                    print(f"{split}.csv: {len(df)} clips")
                    print(f"  Columns: {list(df.columns)}")
                    print(f"  Actions: {df['action_label'].value_counts().to_dict()}")
                else:
                    print(f"ERROR: {output_file} not found!")
            
            print("\n" + "="*50)
            print("Test completed successfully!")
            print("="*50)
            
        except Exception as e:
            print(f"ERROR: Test failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_prepare_annotations() 