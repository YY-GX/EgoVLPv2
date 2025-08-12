import pandas as pd

# Paths to your CSV files
val_csv = "./annotations/mode2/val.csv"
test_csv = "./annotations/mode2/test.csv"

# Load CSVs
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

# Count class occurrences
val_counts = val_df['action_label'].value_counts().sort_index()
test_counts = test_df['action_label'].value_counts().sort_index()

# Combine into a single DataFrame for comparison
comparison = pd.DataFrame({
    'val_count': val_counts,
    'test_count': test_counts
}).fillna(0).astype(int)

print("Class distribution comparison (val vs test):")
print(comparison)
print("\nClasses only in val:", set(val_counts.index) - set(test_counts.index))
print("Classes only in test:", set(test_counts.index) - set(val_counts.index))