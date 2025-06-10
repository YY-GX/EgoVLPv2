import pandas as pd
import os
import csv
import pickle
import numpy as np

# --- Configuration ---
EGOCLIP_CSV_PATH = './egoclip.csv'  # Path to your original egoclip.csv
TARGET_EGOCLIP_VIDEO_UID = '002ad105-bd9a-4858-953e-54e88dc7587e'  # The specific video UID to extract
MOCK_PARTICIPANT_ID = 'aria_P01'  # Your chosen mock EPIC participant ID
MOCK_VIDEO_BASE_NAME = 'clip_'  # Prefix for your 5-second clips (e.g., clip_000.mp4)
CLIP_DURATION_SECONDS = 5  # Length of each generated clip

# --- IMPORTANT: Only include these specific queries ---
TARGET_PROMPTS_FOR_EVAL = [
    "#C C looks around",
    "#C C walks around",
    "#C C sits in the house."  # Make sure this matches exactly, including the period
]

# Output directory for the mock EPIC annotations
# Ensure YOUR_MOCK_EK100_ROOT is set as an environment variable OR hardcode its absolute path
YOUR_MOCK_EK100_ROOT = "/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data/my_simulated_ek100_data"
OUTPUT_METADATA_DIR = os.path.join(YOUR_MOCK_EK100_ROOT, 'EK100/epic-kitchens-100-annotations/retrieval_annotations')
OUTPUT_RELEVANCY_DIR = os.path.join(OUTPUT_METADATA_DIR, 'relevancy')
os.makedirs(OUTPUT_METADATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_RELEVANCY_DIR, exist_ok=True)

print(f"Loading egoclip.csv from: {EGOCLIP_CSV_PATH}")
try:
    df_egoclip = pd.read_csv(EGOCLIP_CSV_PATH, sep='\t', error_bad_lines=False, warn_bad_lines=True)
except Exception as e:
    print(f"Error reading egoclip.csv: {e}")
    print("Please ensure the delimiter is correct and any problematic lines are handled.")
    exit()

# Filter for the target video
df_target_video_raw = df_egoclip[df_egoclip['video_uid'] == TARGET_EGOCLIP_VIDEO_UID].copy()

if df_target_video_raw.empty:
    print(f"Error: No entries found for video_uid '{TARGET_EGOCLIP_VIDEO_UID}' in '{EGOCLIP_CSV_PATH}'.")
    exit()

# --- NEW: Filter by TARGET_PROMPTS_FOR_EVAL ---
df_target_video_filtered = df_target_video_raw[df_target_video_raw['clip_text'].isin(TARGET_PROMPTS_FOR_EVAL)].copy()

if df_target_video_filtered.empty:
    print(
        f"Warning: No relevant entries found for video_uid '{TARGET_EGOCLIP_VIDEO_UID}' with the specified TARGET_PROMPTS_FOR_EVAL.")
    print("Please check your TARGET_PROMPTS_FOR_EVAL list and egoclip.csv content.")
    exit()

video_total_duration_seconds = df_target_video_raw['video_dur'].iloc[0]  # Use raw duration from original video entry
video_total_frames = int(video_total_duration_seconds * 30)  # Assuming 30 FPS

# --- Determine the 5-second clip IDs ---
num_5s_clips = int(np.ceil(video_total_duration_seconds / CLIP_DURATION_SECONDS))
# These will be the video_ids in the CSV (e.g., clip_000, clip_001)
five_s_clip_video_ids = [f"{MOCK_VIDEO_BASE_NAME}{i:03d}" for i in range(num_5s_clips)]

print(f"Video duration: {video_total_duration_seconds:.2f}s. Generating {num_5s_clips} 5-second clips IDs.")

# --- Prepare unique queries for sentence mapping (only from filtered list) ---
unique_queries_for_generation = df_target_video_filtered['clip_text'].unique().tolist()
unique_queries_for_generation.sort()  # Sort for consistent ID assignment

# Map original query text to a unique mock narration_id (e.g., aria_P01_query_000)
query_text_to_mock_narration_id = {
    q_text: f"{MOCK_PARTICIPANT_ID}_query_{i:03d}" for i, q_text in enumerate(unique_queries_for_generation)
}
mock_narration_id_to_query_text = {v: k for k, v in query_text_to_mock_narration_id.items()}

print(f"Using {len(unique_queries_for_generation)} unique queries from egoclip.csv that match TARGET_PROMPTS_FOR_EVAL.")

# --- Generate EPIC_100_retrieval_test.csv ---
# This CSV defines all (query, clip_ID) pairs for evaluation.
epic_retrieval_test_data = []
for mock_5s_clip_id in five_s_clip_video_ids:
    for query_text in unique_queries_for_generation:  # Iterate only over filtered queries
        mock_narration_id = query_text_to_mock_narration_id[query_text]

        # Mimic EPIC-Kitchens fields for each (query, 5s-clip) pair
        # start_timestamp/stop_timestamp and frames here refer to the 5-sec clip's own boundaries.
        clip_index = int(mock_5s_clip_id.split('_')[-1])  # e.g., 000 from clip_000
        clip_start_sec = clip_index * CLIP_DURATION_SECONDS
        clip_end_sec = min((clip_index + 1) * CLIP_DURATION_SECONDS, video_total_duration_seconds)

        # Assuming EPIC frames start from 1
        clip_start_frame = int(clip_start_sec * 30) + 1
        clip_stop_frame = int(clip_end_sec * 30)

        # Adjust stop_frame if it exceeds total video frames for the very last clip
        if clip_stop_frame > video_total_frames:
            clip_stop_frame = video_total_frames
        # Ensure stop frame is at least start frame for very short last segments if rounding makes it weird
        if clip_stop_frame < clip_start_frame:
            clip_stop_frame = clip_start_frame


        def seconds_to_hms_precise(seconds):
            """Converts seconds to HH:MM:SS.mmm format"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


        epic_retrieval_test_data.append({
            'narration_id': mock_narration_id,  # This is the ID for the query itself
            'participant_id': MOCK_PARTICIPANT_ID,
            'video_id': mock_5s_clip_id,  # This is the ID for the 5-sec clip (e.g., clip_000)
            'narration_timestamp': seconds_to_hms_precise(clip_start_sec),  # Context timestamp
            'start_timestamp': seconds_to_hms_precise(clip_start_sec),
            'stop_timestamp': seconds_to_hms_precise(clip_end_sec),
            'start_frame': clip_start_frame,
            'stop_frame': clip_stop_frame,
            'narration': query_text,
            'verb': '', 'verb_class': -1,  # Dummy values
            'noun': '', 'noun_class': -1,  # Dummy values
            'all_nouns': '[]', 'all_noun_classes': '[]'  # Dummy values for lists
        })

df_epic_retrieval_test = pd.DataFrame(epic_retrieval_test_data)
df_epic_retrieval_test.to_csv(
    os.path.join(OUTPUT_METADATA_DIR, "EPIC_100_retrieval_test.csv"),
    index=False
)
print(f"\nGenerated EPIC_100_retrieval_test.csv with {len(df_epic_retrieval_test)} entries.")

# --- Generate EPIC_100_retrieval_test_sentence.csv ---
epic_sentence_data = []
for query_text in unique_queries_for_generation:  # Iterate only over filtered queries
    mock_narration_id = query_text_to_mock_narration_id[query_text]
    epic_sentence_data.append({
        'narration_id': mock_narration_id,
        'narration': query_text
    })

df_epic_sentence = pd.DataFrame(epic_sentence_data)
df_epic_sentence.to_csv(
    os.path.join(OUTPUT_METADATA_DIR, "EPIC_100_retrieval_test_sentence.csv"),
    index=False
)
print(f"Generated EPIC_100_retrieval_test_sentence.csv with {len(df_epic_sentence)} entries.")

# --- Generate caption_relevancy_EPIC_100_retrieval_test.pkl ---
# This PKL defines which query is relevant to which 5-second clip based on original annotations.
relevancy_matrix = np.zeros((len(unique_queries_for_generation), num_5s_clips), dtype=np.float32)

# Create mappings from mock_narration_id/mock_5s_clip_id to their row/column indices in the matrix
# Ensure the order matches the unique_queries_for_generation list
query_narration_id_to_idx = {id: i for i, id in enumerate(df_epic_sentence['narration_id'].tolist())}
five_s_clip_id_to_idx = {id: i for i, id in enumerate(five_s_clip_video_ids)}

# Populate the relevancy_matrix
for query_text in unique_queries_for_generation:
    mock_query_narration_id = query_text_to_mock_narration_id[query_text]
    query_row_idx = query_narration_id_to_idx[mock_query_narration_id]

    # Filter original egoclip.csv for this specific query
    matching_original_entries = df_target_video_filtered[df_target_video_filtered['clip_text'] == query_text]

    for _, original_row in matching_original_entries.iterrows():
        original_moment_start_sec = original_row['clip_start']
        original_moment_end_sec = original_row['clip_end']

        # Determine which 5-second clips this original moment overlaps with
        first_overlapping_5s_clip_idx = int(original_moment_start_sec / CLIP_DURATION_SECONDS)
        # np.ceil ensures that even a tiny overlap at the end includes the last segment
        last_overlapping_5s_clip_idx = int(np.ceil(original_moment_end_sec / CLIP_DURATION_SECONDS)) - 1

        # Edge case: if original_moment_end_sec is very small or equal to start,
        # last_overlapping_5s_clip_idx might be less than first_overlapping_5s_clip_idx.
        # Ensure at least one segment is considered if the moment is valid.
        if last_overlapping_5s_clip_idx < first_overlapping_5s_clip_idx and original_moment_start_sec < original_moment_end_sec:
            if first_overlapping_5s_clip_idx < num_5s_clips:  # Ensure it's a valid clip index
                relevancy_matrix[query_row_idx, first_overlapping_5s_clip_idx] = 1.0
        else:
            for clip_idx in range(first_overlapping_5s_clip_idx, last_overlapping_5s_clip_idx + 1):
                if 0 <= clip_idx < num_5s_clips:  # Ensure clip_idx is within bounds
                    relevancy_matrix[query_row_idx, clip_idx] = 1.0

# Save the relevancy matrix as a pickle file
pkl_filepath = os.path.join(OUTPUT_RELEVANCY_DIR, 'caption_relevancy_EPIC_100_retrieval_test.pkl')
with open(pkl_filepath, 'wb') as f:
    pickle.dump(relevancy_matrix, f)

print(f"\nGenerated relevancy PKL at: {pkl_filepath}")
print(f"Relevancy matrix shape: {relevancy_matrix.shape} (num_queries x num_5s_clips)")