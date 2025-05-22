#!/usr/bin/env python
from __future__ import print_function
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import sys
import os
import glob
from os.path import basename
from classes.Utils import *
from classes.Compiler import *

if __name__ == "__main__":
    argv = sys.argv[1:]
    
    if len(argv) == 0:
        print("Error: not enough argument supplied:")
        print("web-compiler-final.py <directory_path>")
        exit(0)
    
    directory = argv[0]
    
    FILL_WITH_RANDOM_TEXT = True
    TEXT_PLACE_HOLDER = "[]"
    
    # Update path to DSL mapping file
    dsl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "web-dsl-mapping.json")
    
    # Get ALL GUI files (remove [:3] to process all files)
    gui_files = glob.glob(os.path.join(directory, "*.gui"))
    
    print(f"Found {len(gui_files)} GUI files to process...\n")
    
    # ABSOLUTE PATH for output folder
    output_folder = "/root/image_to_html/processed_files"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    def render_content_with_text(key, value):
        if FILL_WITH_RANDOM_TEXT:
            if key.find("btn") != -1:
                value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text())
            elif key.find("title") != -1:
                value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=5, space_number=0))
            elif key.find("text") != -1:
                value = value.replace(TEXT_PLACE_HOLDER,
                                    Utils.get_random_text(length_text=56, space_number=7, with_upper_case=False))
        return value
    
    # Process each file with a NEW compiler instance
    successful = 0
    failed = 0
    
    for i, gui_file in enumerate(gui_files):
        print(f"[{i+1}/{len(gui_files)}] Processing: {basename(gui_file)}")
        
        # Create a NEW compiler for each file
        compiler = Compiler(dsl_path)
        
        file_uid = basename(gui_file)[:basename(gui_file).find(".")]
        
        input_file_path = gui_file
        output_file_path = os.path.join(output_folder, f"{file_uid}.html")
        
        try:
            compiler.compile(input_file_path, output_file_path, rendering_function=render_content_with_text)
            print(f"  ✓ Success")
            successful += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully processed: {successful} files")
    if failed > 0:
        print(f"❌ Failed: {failed} files")
    print(f"📁 All HTML files saved in: {output_folder}")
    print(f"{'='*60}")