# Save this as `generate_mock_epic_metadata_csvs.py` in your main project folder.
import pandas as pd
import os
import csv

# --- Configuration ---
EGOCLIP_CSV_PATH = './egoclip.csv'  # Path to your original egoclip.csv
TARGET_EGOCLIP_VIDEO_UID = '002ad105-bd9a-4858-953e-54e88dc7587e'  # The specific video UID to extract
MOCK_EPIC_PARTICIPANT_ID = 'aria_P01'  # Your chosen mock EPIC participant ID
MOCK_EPIC_VIDEO_ID = 'aria_01'  # Your chosen mock EPIC video ID
MOCK_FULL_VIDEO_ID = f"{MOCK_EPIC_PARTICIPANT_ID}_{MOCK_EPIC_VIDEO_ID}"  # Combined ID for consistency

# Output directory for the mock EPIC annotations
YOUR_MOCK_EK100_ROOT = "/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data/my_simulated_ek100_data"
OUTPUT_METADATA_DIR = os.path.join(YOUR_MOCK_EK100_ROOT, 'EK100/epic-kitchens-100-annotations/retrieval_annotations')
os.makedirs(OUTPUT_METADATA_DIR, exist_ok=True)


# --- Helper function for time formatting ---
def seconds_to_hms(seconds):
    """Converts seconds to HH:MM:SS.mmm format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


print(f"Loading egoclip.csv from: {EGOCLIP_CSV_PATH}")
df_egoclip = pd.read_csv(EGOCLIP_CSV_PATH)

# Filter for the target video
df_target_video = df_egoclip[df_egoclip['video_uid'] == TARGET_EGOCLIP_VIDEO_UID].copy()

if df_target_video.empty:
    print(f"Error: No entries found for video_uid '{TARGET_EGOCLIP_VIDEO_UID}' in '{EGOCLIP_CSV_PATH}'.")
    exit()

video_duration_seconds = df_target_video['video_dur'].iloc[0]
# Assuming ~30 FPS for frame calculation
video_total_frames = int(video_duration_seconds * 30)

# Extract unique clip_text entries to use as queries
# It's good practice to ensure uniqueness and preserve original text.
unique_queries = df_target_video['clip_text'].unique().tolist()
# Sort them for consistent ID assignment
unique_queries.sort()

print(f"Found {len(unique_queries)} unique queries for video '{TARGET_EGOCLIP_VIDEO_UID}':")
for i, q in enumerate(unique_queries):
    print(f"  [{i}]: {q}")

# --- Generate EPIC_100_retrieval_test.csv ---
epic_retrieval_test_data = []
for i, query_text in enumerate(unique_queries):
    mock_narration_id = f"{MOCK_FULL_VIDEO_ID}_{i}"  # e.g., aria_P01_01_0

    epic_retrieval_test_data.append({
        'narration_id': mock_narration_id,
        'participant_id': MOCK_EPIC_PARTICIPANT_ID,
        'video_id': MOCK_FULL_VIDEO_ID,
        'narration_timestamp': seconds_to_hms(0.0),  # As a dummy or actual start of first query
        'start_timestamp': seconds_to_hms(0.0),
        'stop_timestamp': seconds_to_hms(video_duration_seconds),
        'start_frame': 1,
        'stop_frame': video_total_frames,
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
print(f"\nGenerated EPIC_100_retrieval_test.csv: {os.path.join(OUTPUT_METADATA_DIR, 'EPIC_100_retrieval_test.csv')}")

# --- Generate EPIC_100_retrieval_test_sentence.csv ---
epic_sentence_data = []
for i, query_text in enumerate(unique_queries):
    mock_narration_id = f"{MOCK_FULL_VIDEO_ID}_{i}"  # e.g., aria_P01_01_0
    epic_sentence_data.append({
        'narration_id': mock_narration_id,
        'narration': query_text
    })

df_epic_sentence = pd.DataFrame(epic_sentence_data)
df_epic_sentence.to_csv(
    os.path.join(OUTPUT_METADATA_DIR, "EPIC_100_retrieval_test_sentence.csv"),
    index=False
)
print(
    f"Generated EPIC_100_retrieval_test_sentence.csv: {os.path.join(OUTPUT_METADATA_DIR, 'EPIC_100_retrieval_test_sentence.csv')}")

# --- Save mapping of original query text to mock narration_id and its index ---
# This will be useful for the PKL generation and prediction printing later
query_mapping_info = [
    {'original_query': q_text, 'mock_narration_id': f"{MOCK_FULL_VIDEO_ID}_{i}", 'index': i}
    for i, q_text in enumerate(unique_queries)
]
with open(os.path.join(OUTPUT_METADATA_DIR, 'query_mapping_info.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['original_query', 'mock_narration_id', 'index'])
    for row in query_mapping_info:
        writer.writerow([row['original_query'], row['mock_narration_id'], row['index']])
print(f"Generated query_mapping_info.csv: {os.path.join(OUTPUT_METADATA_DIR, 'query_mapping_info.csv')}")