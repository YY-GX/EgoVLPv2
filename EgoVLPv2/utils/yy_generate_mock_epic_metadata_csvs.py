# Save this as `yy_generate_dummy_relevancy_pkl.py` in your main project folder.
import pickle
import os
import pandas as pd
import numpy as np  # Ensure numpy is imported

# --- Configuration ---
EGOCLIP_CSV_PATH = './egoclip.csv'  # Path to your original egoclip.csv
TARGET_EGOCLIP_VIDEO_UID = '002ad105-bd9a-4858-953e-54e88dc7587e'  # The specific video UID to extract
MOCK_EPIC_PARTICIPANT_ID = 'aria_P01'  # Your chosen mock EPIC participant ID
MOCK_EPIC_VIDEO_ID = 'aria_01'  # Your chosen mock EPIC video ID
MOCK_FULL_VIDEO_ID = f"{MOCK_EPIC_PARTICIPANT_ID}_{MOCK_EPIC_VIDEO_ID}"  # Combined ID for consistency

# Output directory for the mock EPIC annotations
YOUR_MOCK_EK100_ROOT = "/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data/my_simulated_ek100_data"
OUTPUT_RELEVANCY_DIR = os.path.join(YOUR_MOCK_EK100_ROOT,
                                    'EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy')
os.makedirs(OUTPUT_RELEVANCY_DIR, exist_ok=True)

# --- Load original egoclip.csv ---
print(f"Loading egoclip.csv from: {EGOCLIP_CSV_PATH}")
# Use error_bad_lines=False to skip problematic lines as previously discussed
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

print(f"Extracted {len(df_target_video)} annotation entries for video '{TARGET_EGOCLIP_VIDEO_UID}'.")

# --- Generate the relevancy dictionary ---
# This dictionary will map mock_narration_id to a list of (mock_full_video_id, start_time, end_time) tuples.
# This structure closely matches what would be used for moment localization ground truth.
relevancy_data = {}

# We also need the mapping from original query text to mock narration_id from generate_mock_epic_metadata_csvs.py
# It saves this in `query_mapping_info.csv`. Let's load it here.
query_mapping_info_path = os.path.join(OUTPUT_RELEVANCY_DIR, os.pardir,
                                       'query_mapping_info.csv')  # Go up one dir from 'relevancy'
df_query_mapping = pd.read_csv(query_mapping_info_path)
query_text_to_narration_id = dict(zip(df_query_mapping['original_query'], df_query_mapping['mock_narration_id']))
narration_id_to_query_text = dict(zip(df_query_mapping['mock_narration_id'], df_query_mapping['original_query']))

# Iterate through unique queries from the mapping info to build the structure
# We only add entries to the PKL for queries we've actually created in the mock CSVs.
for mock_narration_id in df_query_mapping['mock_narration_id']:
    original_query_text = narration_id_to_query_text[mock_narration_id]

    # Find all original egoclip entries for this query text
    matching_egoclip_entries = df_target_video[df_target_video['clip_text'] == original_query_text]

    relevant_moments_for_query = []
    for idx, row in matching_egoclip_entries.iterrows():
        start_time_seconds = row['clip_start']
        end_time_seconds = row['clip_end']
        # The relevancy is usually 1.0 for a ground truth segment
        relevance_score = 1.0
        relevant_moments_for_query.append(
            (MOCK_FULL_VIDEO_ID, start_time_seconds, end_time_seconds, relevance_score)
        )

    # Store it as a list of tuples associated with the mock narration ID
    relevancy_data[mock_narration_id] = relevant_moments_for_query

print(f"Generated relevancy data for {len(relevancy_data)} queries.")

# --- Save as pickle file ---
pkl_filepath = os.path.join(OUTPUT_RELEVANCY_DIR, 'caption_relevancy_EPIC_100_retrieval_test.pkl')
with open(pkl_filepath, 'wb') as f:
    pickle.dump(relevancy_data, f)

print(f"Relevancy PKL created at: {pkl_filepath}")
print("Content structure: {mock_narration_id: [(mock_video_id, clip_start_sec, clip_end_sec, relevance_score), ... ]}")