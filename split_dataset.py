import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, train_dir, test_dir, test_ratio=0.2, seed=42):
    """
    Split the dataset into training and testing sets.
    
    Args:
        source_dir (str): Directory containing the original images and HTML files
        train_dir (str): Directory to store training files
        test_dir (str): Directory to store testing files
        test_ratio (float): Ratio of test set (default: 0.2)
        seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create train and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all image files and their corresponding HTML files
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    html_files = [f.replace('.png', '.html') for f in image_files]
    
    # Create pairs of (image, html) files
    file_pairs = list(zip(image_files, html_files))
    
    # Shuffle the pairs
    random.shuffle(file_pairs)
    
    # Calculate split index
    split_idx = int(len(file_pairs) * (1 - test_ratio))
    
    # Split into train and test sets
    train_pairs = file_pairs[:split_idx]
    test_pairs = file_pairs[split_idx:]
    
    # Copy files to respective directories
    for img_file, html_file in train_pairs:
        # Copy image file
        shutil.copy2(
            os.path.join(source_dir, img_file),
            os.path.join(train_dir, img_file)
        )
        # Copy HTML file
        shutil.copy2(
            os.path.join(source_dir, html_file),
            os.path.join(train_dir, html_file)
        )
    
    for img_file, html_file in test_pairs:
        # Copy image file
        shutil.copy2(
            os.path.join(source_dir, img_file),
            os.path.join(test_dir, img_file)
        )
        # Copy HTML file
        shutil.copy2(
            os.path.join(source_dir, html_file),
            os.path.join(test_dir, html_file)
        )
    
    print(f"Total pairs: {len(file_pairs)}")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Testing pairs: {len(test_pairs)}")

if __name__ == "__main__":
    # Define paths
    source_dir = "kaggle_dataset/kaggle_dataset"
    train_dir = "kaggle_dataset/train"
    test_dir = "kaggle_dataset/test"
    
    # Split the dataset
    split_dataset(source_dir, train_dir, test_dir) 