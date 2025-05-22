#!/usr/bin/env python
import os
import glob
import sys
sys.path.append('compiler/classes')  # Add the classes directory to Python path
from classes.Compiler import *
import kagglehub

# Download latest version
path = kagglehub.dataset_download("vikramtiwari/pix2code")

print("Path to dataset files:", path)

# Path to the dataset directory
DATASET_DIR = '/root/.cache/kagglehub/datasets/vikramtiwari/pix2code/versions/1/web/all_data'
# Path to the DSL mapping file - now relative to compiler directory
DSL_PATH = '../assets/web-dsl-mapping.json'

def compile_all_gui_files():
    # Initialize the compiler
    compiler = Compiler(DSL_PATH)
    
    # Get all .gui files in the dataset directory
    gui_files = glob.glob(os.path.join(DATASET_DIR, '*.gui'))
    total_files = len(gui_files)
    
    print(f"Found {total_files} GUI files to process")
    
    # Process each file
    for i, gui_file in enumerate(gui_files, 1):
        try:
            # Get the base name without extension
            base_name = os.path.splitext(gui_file)[0]
            # Create the output HTML file path
            html_file = f"{base_name}.html"
            
            print(f"Processing file {i}/{total_files}: {os.path.basename(gui_file)}")
            
            # Compile the GUI file to HTML
            compiler.compile(gui_file, html_file)
            
        except Exception as e:
            print(f"Error processing {gui_file}: {str(e)}")
    
    print("\nCompilation complete!")

if __name__ == "__main__":
    compile_all_gui_files() 