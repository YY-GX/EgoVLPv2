import pandas as pd
import re
from collections import Counter # Import Counter for easily counting frequencies

def analyze_clip_text(csv_file_path="egoclip.csv", delimiter='\t'):
    """
    Analyzes the 'clip_text' column of a CSV file.

    It filters for clips containing 'walk', 'stand', or 'sit' (case-insensitive, whole words)
    and then prints the top 5 most common specific clip texts for each category.

    Args:
        csv_file_path (str): The path to the egoclip.csv file.
        delimiter (str): The delimiter used in the CSV file (e.g., ',', '\t', ';').
                         Defaults to a tab ('\t') based on previous interactions.
    """
    try:
        # Load the CSV file into a pandas DataFrame, using the specified delimiter.
        # on_bad_lines='warn' will issue a warning for malformed lines instead of failing,
        # which can help debug but might skip problematic data.
        df = pd.read_csv(csv_file_path, sep=delimiter, on_bad_lines='warn')
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        print("Please make sure 'egoclip.csv' is in the same directory as the script,")
        print("or provide the full path to the file.")
        return
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file: {e}")
        print("This often means the file is not correctly formatted,")
        print("or the delimiter is not correct.")
        print(f"Currently using delimiter: '{delimiter}'")
        print("You might try changing the 'delimiter' argument to '\\t' (for tab) or ';' (for semicolon).")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV file: {e}")
        return

    # Check if the 'clip_text' column exists in the DataFrame.
    if 'clip_text' not in df.columns:
        print("Error: The 'clip_text' column was not found in the CSV file.")
        print("Please ensure your CSV has a column named 'clip_text'.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Define the regular expression to find 'walk', 'stand', or 'sit' as whole words.
    # \b   : Word boundary, ensures we match whole words (e.g., "stand" but not "understanding").
    # ()   : Grouping for 'OR' condition.
    # |    : OR operator, matches 'walk' OR 'stand' OR 'sit'.
    # case=False : Makes the search case-insensitive (e.g., will match "Walk", "STAND", "sit").
    # na=False   : Treats any NaN (Not a Number) values in 'clip_text' as not containing the regex.
    search_pattern = r'\b(walk|stand|sit)\b'

    # Filter the DataFrame to include only rows where 'clip_text' contains any of the target words.
    # We convert 'clip_text' to string type first to handle potential non-string entries gracefully.
    filtered_df = df[df['clip_text'].astype(str).str.contains(search_pattern, case=False, na=False)]

    # Initialize lists to store clip texts for each category.
    # We'll populate these lists with the original clip_text strings.
    walk_clips = []
    stand_clips = []
    sit_clips = []

    # Iterate through the 'clip_text' column of the filtered DataFrame.
    # For each clip, check if it contains 'walk', 'stand', or 'sit' and add the full text
    # to the respective category list. A single clip can appear in multiple lists
    # if it contains multiple target words.
    for text in filtered_df['clip_text'].astype(str):
        # Convert text to lowercase once for easier case-insensitive checks with re.search.
        text_lower = text.lower()

        # Check for 'walk' and add to list if found
        if re.search(r'\bwalk\b', text_lower):
            walk_clips.append(text)
        # Check for 'stand' and add to list if found
        if re.search(r'\bstand\b', text_lower):
            stand_clips.append(text)
        # Check for 'sit' and add to list if found
        if re.search(r'\bsit\b', text_lower):
            sit_clips.append(text)

    # Helper function to print the top N common clips for a given category.
    def print_top_n_clips(category_name, clips_list, n=5):
        # If the list is empty, there are no clips for this category.
        if not clips_list:
            print(f"\n--- No clips found for '{category_name}' category ---")
            return

        # Use Counter to count the occurrences of each unique clip text in the list.
        clip_counts = Counter(clips_list)
        # Get the top N most common clip texts and their counts.
        top_n_clips = clip_counts.most_common(n)

        print(f"\n--- Top {n} clips for '{category_name}' category (Total unique clips: {len(clip_counts)}, Total occurrences: {len(clips_list)}) ---")
        for text, count in top_n_clips:
            print(f"'{text}': {count} clips")
        print("---------------------------------------------")

    # Print the top 5 results for each category.
    print_top_n_clips("walk", walk_clips, n=5)
    print_top_n_clips("stand", stand_clips, n=5)
    print_top_n_clips("sit", sit_clips, n=5)

# --- How to Run the Script ---
# 1. Save your egoclip.csv file in the same directory as this Python script.
# 2. Run this Python script. It will automatically call the analyze_clip_text function.
#    You can also provide the full path to your CSV like: analyze_clip_text("/path/to/your/egoclip.csv")

# Call the function with the tab delimiter, which was identified as correct in previous steps.
analyze_clip_text(delimiter='\t')
