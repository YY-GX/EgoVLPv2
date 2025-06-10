# Save this as `generate_dummy_relevancy_pkl.py` in your main project folder.
import pickle
import os
import numpy as np
import pandas as pd

# --- Configuration ---
YOUR_MOCK_EK100_ROOT = "/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data/my_simulated_ek100_data"
OUTPUT_RELEVANCY_DIR = os.path.join(YOUR_MOCK_EK100_ROOT, 'EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy')
os.makedirs(OUTPUT_RELEVANCY_DIR, exist_ok=True)

# Path to the sentence CSV generated in the previous step
MOCK_SENTENCE_CSV_PATH = os.path.join(YOUR_MOCK_EK100_ROOT, 'EK100/epic-kitchens-100-annotations/retrieval_annotations', 'EPIC_100_retrieval_test_sentence.csv')

# Load the sentence CSV to get the number of queries
df_sentence = pd.read_csv(MOCK_SENTENCE_CSV_PATH, sep='\t')
num_queries = len(df_sentence)
num_videos = 1 # You only have one video: aria_P01_01

# Create a NumPy array of zeros. This represents dummy ground truth (no relevance).
# The shape is (num_queries x num_videos).
dummy_relevancy_array = np.zeros((num_queries, num_videos), dtype=np.float32)

# Save as pickle
pkl_filepath = os.path.join(OUTPUT_RELEVANCY_DIR, 'caption_relevancy_EPIC_100_retrieval_test.pkl')
with open(pkl_filepath, 'wb') as f:
    pickle.dump(dummy_relevancy_array, f)

print(f"Dummy relevancy PKL created at: {pkl_filepath}")