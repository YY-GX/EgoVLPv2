# Save this as `generate_mock_epic_clip_retrieval_metadata.py` in your main project folder.
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

# Output directory for the mock EPIC annotations
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
df_target_video = df_egoclip[df_egoclip['video_uid'] == TARGET_EGOCLIP_VIDEO_UID].copy()
if df_target_video.empty:
    print(f"Error: No entries found for video_uid '{TARGET_EGOCLIP_VIDEO_UID}' in '{EGOCLIP_CSV_PATH}'.")
    exit()

video_total_duration_seconds = df_target_video['video_dur'].iloc[0]
video_total_frames = int(video_total_duration_seconds * 30)  # Assuming 30 FPS

# --- Determine the 5-second clip IDs ---
num_5s_clips = int(np.ceil(video_total_duration_seconds / CLIP_DURATION_SECONDS))
five_s_clip_video_ids = [f"{MOCK_PARTICIPANT_ID}_{MOCK_VIDEO_BASE_NAME}{i:03d}" for i in range(num_5s_clips)]

print(f"Video duration: {video_total_duration_seconds:.2f}s. Generating {num_5s_clips} 5-second clips IDs.")

# --- Extract unique queries from egoclip.csv for sentence mapping ---
# We'll use all unique narrations from the target video as our queries.
unique_queries_raw = df_target_video['clip_text'].unique().tolist()
unique_queries_raw.sort()  # Sort for consistent ID assignment
# Map original query text to a unique mock narration_id (e.g., aria_P01_000_0)
query_text_to_mock_narration_id = {
    q_text: f"{MOCK_PARTICIPANT_ID}_query_{i:03d}" for i, q_text in enumerate(unique_queries_raw)
}
mock_narration_id_to_query_text = {v: k for k, v in query_text_to_mock_narration_id.items()}

print(f"Found {len(unique_queries_raw)} unique queries from egoclip.csv.")

# --- Generate EPIC_100_retrieval_test.csv ---
# This CSV defines all (query, clip_ID) pairs for evaluation.
# Each row represents a test sample: "Does this query match this specific 5-second clip?"
epic_retrieval_test_data = []
for mock_5s_clip_id in five_s_clip_video_ids:
    for query_text in unique_queries_raw:
        mock_narration_id = query_text_to_mock_narration_id[query_text]

        # Mimic EPIC-Kitchens fields for each (query, 5s-clip) pair
        # start_timestamp/stop_timestamp and frames here refer to the 5-sec clip's own boundaries.
        # This is where the DataLoader will read the frames for this specific data point.
        clip_index = int(mock_5s_clip_id.split('_')[-1])  # e.g., 000 from aria_P01_clip_000
        clip_start_sec = clip_index * CLIP_DURATION_SECONDS
        clip_end_sec = min((clip_index + 1) * CLIP_DURATION_SECONDS, video_total_duration_seconds)

        # Assuming EPIC frames start from 1
        clip_start_frame = int(clip_start_sec * 30) + 1
        clip_stop_frame = int(clip_end_sec * 30)
        if clip_start_frame > clip_stop_frame:  # Handle very short last clips if needed
            clip_stop_frame = clip_start_frame  # Ensure stop_frame is at least start_frame


        def seconds_to_hms_precise(seconds):
            """Converts seconds to HH:MM:SS.mmm format"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


        epic_retrieval_test_data.append({
            'narration_id': mock_narration_id,  # This is the ID for the query itself
            'participant_id': MOCK_PARTICIPANT_ID,
            'video_id': mock_5s_clip_id,  # This is the ID for the 5-sec clip
            'narration_timestamp': seconds_to_hms_precise(clip_start_sec),  # Can use start of clip as context
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
for query_text in unique_queries_raw:
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
relevancy_data = {}

# Iterate through each query you're testing
for query_text in unique_queries_raw:
    mock_query_narration_id = query_text_to_mock_narration_id[query_text]

    # Filter original egoclip.csv for this specific query
    # (assuming `clip_start`, `clip_end` refer to ground truth moments for this text)
    matching_original_entries = df_target_video[df_target_video['clip_text'] == query_text]

    relevant_5s_clip_ids = []
    # For each original annotated segment for this query:
    for _, original_row in matching_original_entries.iterrows():
        original_moment_start_sec = original_row['clip_start']
        original_moment_end_sec = original_row['clip_end']

        # Determine which 5-second clips this original moment overlaps with
        # An original moment can span multiple 5-second clips.
        # Find the starting and ending 5-second clip indices that overlap.

        # Calculate the first 5-sec clip index that overlaps
        first_overlapping_5s_clip_idx = int(original_moment_start_sec / CLIP_DURATION_SECONDS)
        # Calculate the last 5-sec clip index that overlaps (ceil ensures even partial overlap counts)
        last_overlapping_5s_clip_idx = int(np.ceil(original_moment_end_sec / CLIP_DURATION_SECONDS)) - 1
        # Handle case where end_sec is exactly on a boundary or 0-length clips
        if last_overlapping_5s_clip_idx < 0: last_overlapping_5s_clip_idx = 0
        if first_overlapping_5s_clip_idx > last_overlapping_5s_clip_idx:
            first_overlapping_5s_clip_idx = last_overlapping_5s_clip_idx  # If it's a tiny clip fitting into one segment.

        for clip_idx in range(first_overlapping_5s_clip_idx, last_overlapping_5s_clip_idx + 1):
            if clip_idx < num_5s_clips:  # Ensure clip_idx is within bounds of actual generated clips
                relevant_5s_clip_ids.append(f"{MOCK_PARTICIPANT_ID}_{MOCK_VIDEO_BASE_NAME}{clip_idx:03d}")

    # Store relevancy for this query: map query narration_id to a list of relevant 5s clip IDs
    # The actual format for relevancy in EPIC-Kitchens is usually:
    # {query_id (int): {video_id (str): relevance_score (float)}} OR {query_id (int): list of video_id (str)}
    # The `mir_metrics_vtc` function takes `relevancy` (a dictionary) and uses it to align `similarity_matrix`
    # The functions `calculate_nDCG` and `calculate_mAP` take a matrix.
    # The `relevancy` object passed to `calculate_nDCG` and `calculate_mAP` from JPoSE is typically a 2D numpy array.
    # So we should create a 2D array of 0s and 1s.

# --- Re-thinking relevancy_data structure for the PKL: it should be a 2D numpy array ---
# Create a zero-filled numpy array: (num_queries x num_5s_clips)
relevancy_matrix = np.zeros((len(unique_queries_raw), num_5s_clips), dtype=np.float32)

# Create mappings from mock_narration_id/mock_5s_clip_id to their row/column indices in the matrix
query_narration_id_to_idx = {id: i for i, id in enumerate(query_text_to_mock_narration_id.values())}
five_s_clip_id_to_idx = {id: i for i, id in enumerate(five_s_clip_video_ids)}

# Populate the relevancy_matrix
for query_text in unique_queries_raw:
    mock_query_narration_id = query_text_to_mock_narration_id[query_text]
    query_row_idx = query_narration_id_to_idx[mock_query_narration_id]

    matching_original_entries = df_target_video[df_target_video['clip_text'] == query_text]

    for _, original_row in matching_original_entries.iterrows():
        original_moment_start_sec = original_row['clip_start']
        original_moment_end_sec = original_row['clip_end']

        first_overlapping_5s_clip_idx = int(original_moment_start_sec / CLIP_DURATION_SECONDS)
        last_overlapping_5s_clip_idx = int(np.ceil(original_moment_end_sec / CLIP_DURATION_SECONDS)) - 1
        if last_overlapping_5s_clip_idx < first_overlapping_5s_clip_idx and original_moment_start_sec < original_moment_end_sec:
            # Handle cases where clip is tiny, fits within one 5s segment but ceil makes end_idx smaller.
            # Example: 1.0 - 1.5s -> 0.2, 0.3. ceil(0.3) = 1. So 0 to 0. Needs to be just 0.
            if first_overlapping_5s_clip_idx < num_5s_clips:
                relevancy_matrix[query_row_idx, first_overlapping_5s_clip_idx] = 1.0
        else:
            for clip_idx in range(first_overlapping_5s_clip_idx, last_overlapping_5s_clip_idx + 1):
                if clip_idx < num_5s_clips:  # Ensure clip_idx is within bounds
                    relevancy_matrix[query_row_idx, clip_idx] = 1.0

# Save the relevancy matrix as a pickle file
pkl_filepath = os.path.join(OUTPUT_RELEVANCY_DIR, 'caption_relevancy_EPIC_100_retrieval_test.pkl')
with open(pkl_filepath, 'wb') as f:
    pickle.dump(relevancy_matrix, f)

print(f"\nGenerated relevancy PKL at: {pkl_filepath}")
print(f"Relevancy matrix shape: {relevancy_matrix.shape} (num_queries x num_5s_clips)")