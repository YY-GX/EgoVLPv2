import csv

input_file = "video_6_ori.csv"
output_file = "video_6.csv"

# Set this to your desired shift (in seconds)
shift_time = -1.979  # e.g., -1 for shifting all timestamps 1 second earlier

def seconds_to_mmss(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"

with open(input_file, newline='') as csvfile_in, open(output_file, 'w', newline='') as csvfile_out:
    reader = csv.DictReader(csvfile_in)
    writer = csv.writer(csvfile_out)
    writer.writerow(['timestamp', 'action_label'])

    total_time = 0.0
    writer.writerow(['00:00', ''])  # Start at 0:00

    for row in reader:
        if row['lap'].startswith('Lap'):
            lap_seconds = float(row['seconds'])
            total_time += lap_seconds
            shifted_time = max(0.0, total_time + shift_time)
            timestamp = seconds_to_mmss(shifted_time)
            writer.writerow([timestamp, ''])  # Placeholder for action_label

print(f"Transformed file saved as {output_file}") 