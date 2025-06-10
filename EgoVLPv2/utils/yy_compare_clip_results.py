import pandas as pd
import os
import re  # For regex to extract clip numbers
import csv
import numpy as np

# --- Configuration ---
# Paths to your input files
EGOCLIP_GT_CSV_PATH = './egoclip.csv'
PREDICTIONS_CSV_PATH = './eval_yy/top_clips_predictions.csv'  # Adjust path if needed
CLIP_DURATION_SECONDS = 5  # Needs to match what you used for splitting and in trainer

# --- IoU Threshold for considering a prediction "correct" ---
IOU_THRESHOLD = 0.5  # A common threshold (e.g., 50% overlap)

# --- Metrics to calculate (top-K) ---
TOP_K_VALUES = [1, 5]  # Calculate P@1, R@1, P@5, R@5


# --- Helper Functions ---

def interval_iou(interval1, interval2):
    """
    Calculates Intersection over Union (IoU) for two time intervals.
    Intervals are [start, end).
    """
    start1, end1 = interval1
    start2, end2 = interval2

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    intersection_length = max(0, intersection_end - intersection_start)
    union_length = (end1 - start1) + (end2 - start2) - intersection_length

    if union_length == 0:
        return 0.0  # Avoid division by zero if both intervals are points or empty
    return intersection_length / union_length


def parse_time_string(time_str):
    """Parses a time string like '123.0s - 128.0s' into (start_sec, end_sec)."""
    match = re.match(r'(\d+\.?\d*)s - (\d+\.?\d*)s', time_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"Could not parse time string: {time_str}")


# --- Main Comparison Logic ---

def run_comparison():
    print("--- Starting Prediction Comparison ---")

    # Load Ground Truth (egoclip.csv)
    print(f"Loading Ground Truth from: {EGOCLIP_GT_CSV_PATH}")
    try:
        # Use sep='\t' and error_bad_lines=False as used in generation
        df_gt = pd.read_csv(EGOCLIP_GT_CSV_PATH, sep='\t', error_bad_lines=False, warn_bad_lines=True)
    except Exception as e:
        print(f"Error loading GT CSV: {e}")
        print("Please ensure the path is correct and the CSV is valid.")
        return

    # Filter GT for the specific video UID used for prediction
    # This ID is hardcoded in the generation script: '002ad105-bd9a-4858-953e-54e88dc7587e'
    TARGET_EGOCLIP_VIDEO_UID = '002ad105-bd9a-4858-953e-54e88dc7587e'
    df_gt_video = df_gt[df_gt['video_uid'] == TARGET_EGOCLIP_VIDEO_UID].copy()
    if df_gt_video.empty:
        print(f"Error: No GT entries found for video_uid '{TARGET_EGOCLIP_VIDEO_UID}' in egoclip.csv.")
        return

    # Load Predictions (top_clips_predictions.csv)
    print(f"Loading Predictions from: {PREDICTIONS_CSV_PATH}")
    try:
        df_preds = pd.read_csv(PREDICTIONS_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Predictions CSV not found at {PREDICTIONS_CSV_PATH}.")
        print("Make sure your previous script successfully saved the output.")
        return
    except Exception as e:
        print(f"Error loading Predictions CSV: {e}")
        return

    unique_queries = df_preds['Query_Text'].unique()
    print(f"Comparing {len(unique_queries)} unique queries.")

    overall_metrics = {k: {f'P@{tk}': [] for tk in TOP_K_VALUES} for k in unique_queries}
    for k in unique_queries:
        for tk in TOP_K_VALUES:
            overall_metrics[k][f'R@{tk}'] = []

    # Iterate through each unique query
    for query_text in sorted(unique_queries):
        print(f"\n--- Query: '{query_text}' ---")

        # Get Ground Truth segments for this query
        gt_segments_for_query = df_gt_video[df_gt_video['clip_text'] == query_text]
        if gt_segments_for_query.empty:
            print("  No Ground Truth segments found for this query.")
            continue

        # Convert GT segments to (start_sec, end_sec) tuples
        gt_intervals = []
        for _, row in gt_segments_for_query.iterrows():
            gt_intervals.append((row['clip_start'], row['clip_end']))
        print(f"  Ground Truth ({len(gt_intervals)} segments): {gt_intervals}")

        # Get Predicted clips for this query, ordered by rank
        predicted_clips_for_query = df_preds[df_preds['Query_Text'] == query_text].sort_values(by='Rank')

        if predicted_clips_for_query.empty:
            print("  No Predicted clips found for this query.")
            continue

        print(f"  Predicted Clips ({len(predicted_clips_for_query)} shown in top 5):")

        # For each K value, assess precision and recall
        for k_val in TOP_K_VALUES:
            correct_predictions_at_k = 0
            retrieved_gt_segments = set()  # Store indices of GT segments that were retrieved

            top_k_predictions = predicted_clips_for_query.head(k_val)

            # Check overlap for each top-K prediction
            for _, pred_row in top_k_predictions.iterrows():
                pred_interval_str = pred_row['Start_Time'] + " - " + pred_row['End_Time']
                try:
                    pred_start, pred_end = parse_time_string(pred_interval_str)
                except ValueError as e:
                    print(f"    Skipping prediction due to time parsing error: {e}")
                    continue
                pred_interval = (pred_start, pred_end)

                is_correct_pred = False
                # Check this prediction against all GT segments
                for gt_idx, gt_interval in enumerate(gt_intervals):
                    iou = interval_iou(pred_interval, gt_interval)
                    if iou >= IOU_THRESHOLD:
                        correct_predictions_at_k += 1
                        retrieved_gt_segments.add(gt_idx)  # Mark this GT segment as retrieved
                        is_correct_pred = True
                        break  # Found a match, move to next prediction

                status = "Correct" if is_correct_pred else "Incorrect"
                print(
                    f"    Rank {pred_row['Rank']}: {pred_row['Clip_ID']} ({pred_interval_str}) | Score: {pred_row['Score']} | IoU >= {IOU_THRESHOLD}: {status}")

            # Calculate P@K and R@K for current k_val
            precision_at_k = correct_predictions_at_k / k_val
            recall_at_k = len(retrieved_gt_segments) / len(gt_intervals) if len(gt_intervals) > 0 else 0.0

            overall_metrics[query_text][f'P@{k_val}'].append(precision_at_k)
            overall_metrics[query_text][f'R@{k_val}'].append(recall_at_k)

            print(f"  P@{k_val}: {precision_at_k:.2f} | R@{k_val}: {recall_at_k:.2f}")

    # --- Print Overall Averages ---
    print("\n--- Overall Average Metrics (across all queries) ---")
    avg_results = {}
    for k_val in TOP_K_VALUES:
        all_p_at_k = [m[f'P@{k_val}'][0] for m in overall_metrics.values() if m[f'P@{k_val}']]
        all_r_at_k = [m[f'R@{k_val}'][0] for m in overall_metrics.values() if m[f'R@{k_val}']]

        avg_results[f'Avg P@{k_val}'] = np.mean(all_p_at_k) if all_p_at_k else 0.0
        avg_results[f'Avg R@{k_val}'] = np.mean(all_r_at_k) if all_r_at_k else 0.0

        print(f"Avg P@{k_val}: {avg_results[f'Avg P@{k_val}']:.3f}")
        print(f"Avg R@{k_val}: {avg_results[f'Avg R@{k_val}']:.3f}")

    print("\n--- Comparison Complete ---")


if __name__ == "__main__":
    run_comparison()