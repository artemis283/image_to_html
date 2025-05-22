import os
import shutil

# Source directory (cache)
source_dir = "/root/.cache/kagglehub/datasets/vikramtiwari/pix2code/versions/1/web/all_data"

# Destination directory
dest_dir = "processed_files"

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Get all PNG files
png_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]

print(f"Found {len(png_files)} PNG files to move")

# Move each file
for png_file in png_files:
    source_path = os.path.join(source_dir, png_file)
    dest_path = os.path.join(dest_dir, png_file)
    
    try:
        shutil.copy2(source_path, dest_path)
        print(f"Copied {png_file}")
    except Exception as e:
        print(f"Error copying {png_file}: {str(e)}")

print("\nCopy complete!")