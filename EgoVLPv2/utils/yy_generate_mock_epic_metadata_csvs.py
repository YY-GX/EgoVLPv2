import pandas as pd
import os
import csv
import pickle
import numpy as np

# --- Configuration ---
EGOCLIP_CSV_PATH = './egoclip.csv'  # Path to your original egoclip.csv
TARGET_EGOCLIP_VIDEO_UID = '002ad105-bd9a-4858-953e-54e88dc7587e'  # The specific video UID to extract
MOCK_PARTICIPANT_ID = 'aria_P01'  # Your chosen mock EPIC participant ID
MOCK_VIDEO_BASE_NAME = ''  # IMPORTANT: Leave this blank if your 5-sec clips are named '000.mp4', '001.mp4' directly.
# If they are named 'clip_000.mp4', set this to 'clip_'.
# We assume 'aria_P01_000.MP4' or 'aria_P01_clip_000.MP4' files.
CLIP_DURATION_SECONDS = 5  # Length of each generated clip

# Output root directory for the mock EPIC dataset structure
# IMPORTANT: Replace "/path/to/your/EgoVLPv2/EgoVLPv2/" with your actual project root.
YOUR_MOCK_EK100_ROOT = "/path/to/your/EgoVLPv2/EgoVLPv2/data/my_simulated_ek100_data"

OUTPUT_METADATA_DIR = "/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data/my_simulated_ek100_data"
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

# Filter for the target video UID (no query filtering here)
df_target_video = df_egoclip[df_egoclip['video_uid'] == TARGET_EGOCLIP_VIDEO_UID].copy()

if df_target_video.empty:
    print(f"Error: No entries found for video_uid '{TARGET_EGOCLIP_VIDEO_UID}' in '{EGOCLIP_CSV_PATH}'.")
    exit()

video_total_duration_seconds = df_target_video['video_dur'].iloc[0]
video_total_frames = int(video_total_duration_seconds * 30)  # Assuming 30 FPS

# --- Determine the 5-second clip IDs ---
num_5s_clips = int(np.ceil(video_total_duration_seconds / CLIP_DURATION_SECONDS))
# video_id in CSV will be like "aria_P01_000", matching the new filenames (e.g., aria_P01_000.MP4)
five_s_clip_video_ids = [f"{MOCK_PARTICIPANT_ID}_{MOCK_VIDEO_BASE_NAME}{i:03d}" for i in range(num_5s_clips)]

print(f"Video duration: {video_total_duration_seconds:.2f}s. Generating {num_5s_clips} 5-second clips IDs.")

# --- Prepare ALL unique queries from egoclip.csv for sentence mapping ---
unique_queries_raw = df_target_video['clip_text'].unique().tolist()
unique_queries_raw.sort()  # Sort for consistent ID assignment

# Map original query text to a unique mock narration_id
query_text_to_mock_narration_id = {
    q_text: f"{MOCK_PARTICIPANT_ID}_query_{i:03d}" for i, q_text in enumerate(unique_queries_raw)
}
mock_narration_id_to_query_text = {v: k for k, v in query_text_to_mock_narration_id.items()}

print(
    f"Using {len(unique_queries_raw)} unique queries from egoclip.csv for metadata generation (all available queries).")

# --- Generate EPIC_100_retrieval_test.csv ---
epic_retrieval_test_data = []
for mock_5s_clip_id in five_s_clip_video_ids:
    for query_text in unique_queries_raw:  # Iterate over ALL unique queries
        mock_narration_id = query_text_to_mock_narration_id[query_text]

        # Simplified: Always split by '_' and take the last part for the numeric index
        clip_index = int(mock_5s_clip_id.split('_')[-1])
        # ^ Adjust parsing based on whether MOCK_VIDEO_BASE_NAME is used in mock_5s_clip_id

        clip_start_sec = clip_index * CLIP_DURATION_SECONDS
        clip_end_sec = min((clip_index + 1) * CLIP_DURATION_SECONDS, video_total_duration_seconds)

        clip_start_frame = int(clip_start_sec * 30) + 1
        clip_stop_frame = int(clip_end_sec * 30)
        if clip_stop_frame > video_total_frames:
            clip_stop_frame = video_total_frames
        if clip_stop_frame < clip_start_frame:  # Ensure end frame is not before start frame (for very short last segments)
            clip_stop_frame = clip_start_frame


        def seconds_to_hms_precise(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


        epic_retrieval_test_data.append({
            'narration_id': mock_narration_id,
            'participant_id': MOCK_PARTICIPANT_ID,
            'video_id': mock_5s_clip_id,  # e.g., aria_P01_000
            'narration_timestamp': seconds_to_hms_precise(clip_start_sec),
            'start_timestamp': seconds_to_hms_precise(clip_start_sec),
            'stop_timestamp': seconds_to_hms_precise(clip_end_sec),
            'start_frame': clip_start_frame,
            'stop_frame': clip_stop_frame,
            'narration': query_text,
            'verb': '', 'verb_class': -1,
            'noun': '', 'noun_class': -1,
            'all_nouns': '[]', 'all_noun_classes': '[]'
        })

df_epic_retrieval_test = pd.DataFrame(epic_retrieval_test_data)
df_epic_retrieval_test.to_csv(
    os.path.join(OUTPUT_METADATA_DIR, "EPIC_100_retrieval_test.csv"),
    index=False
)
print(f"\nGenerated EPIC_100_retrieval_test.csv with {len(df_epic_retrieval_test)} entries (all queries).")

# --- Generate EPIC_100_retrieval_test_sentence.csv ---
epic_sentence_data = []
for query_text in unique_queries_raw:  # Iterate over ALL unique queries
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
print(f"Generated EPIC_100_retrieval_test_sentence.csv with {len(df_epic_sentence)} entries (all unique queries).")

# --- Generate caption_relevancy_EPIC_100_retrieval_test.pkl ---
relevancy_matrix = np.zeros((len(unique_queries_raw), num_5s_clips), dtype=np.float32)

# Create mappings from mock_narration_id/mock_5s_clip_id to their row/column indices in the matrix
query_narration_id_to_idx = {id: i for i, id in enumerate(df_epic_sentence['narration_id'].tolist())}
five_s_clip_id_to_idx = {id: i for i, id in enumerate(five_s_clip_video_ids)}

# Populate the relevancy_matrix
for query_text in unique_queries_raw:  # Iterate over ALL unique queries
    mock_query_narration_id = query_text_to_mock_narration_id[query_text]
    query_row_idx = query_narration_id_to_idx[mock_query_narration_id]

    # Filter original egoclip.csv for this specific query
    matching_original_entries = df_target_video[df_target_video['clip_text'] == query_text]

    for _, original_row in matching_original_entries.iterrows():
        original_moment_start_sec = original_row['clip_start']
        original_moment_end_sec = original_row['clip_end']

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