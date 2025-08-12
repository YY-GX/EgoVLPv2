# Define source and destination directories (adjust if different from current location)
SOURCE_CLIPS_DIR="./clips_egoclip"
DEST_ROOT="/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data/my_simulated_ek100_data/EK100/video_ht256px"
DEST_PARTICIPANT_DIR="aria_P01"

# Delete any old, incorrect clips directory first (optional, but good for cleanliness)
rm -rf "$DEST_ROOT/$DEST_PARTICIPANT_DIR"

# Create the correct nested directory for the clips
mkdir -p "$DEST_ROOT/$DEST_PARTICIPANT_DIR"

# Copy and rename each 5-second clip
for f in "$SOURCE_CLIPS_DIR"/clip_*.mp4; do
    filename=$(basename "$f") # Gets 'clip_000.mp4'
    clip_num_ext=${filename#clip_} # Gets '000.mp4'
    new_name="${DEST_PARTICIPANT_DIR}_${clip_num_ext^^}" # Forms 'aria_P01_000.MP4' (uppercase extension)
    cp "$f" "$DEST_ROOT/$DEST_PARTICIPANT_DIR/$new_name"
done

echo "Video clips organized and renamed into: $DEST_ROOT/$DEST_PARTICIPANT_DIR/"
