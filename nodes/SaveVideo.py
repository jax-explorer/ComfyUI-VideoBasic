import hashlib
import os
import shutil

import folder_paths

class VideoBasicSaveVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING",)
            },
        }

    CATEGORY = "VideoBasic"
    OUTPUT_NODE = True


    RETURN_TYPES = ()
    FUNCTION = "save"

    def save(self, file_path):
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        
        # Handle relative paths, convert them to absolute paths
        if not os.path.isabs(file_path):
            file_path = os.path.join(output_dir, file_path)
        else:
            file_path = os.path.abspath(file_path)

        print(f"file_path is {file_path}, output_dir is {output_dir}")
        
        # Check if the path is in the output directory and the file exists
        if file_path.startswith(output_dir) and os.path.exists(file_path):
            # Get the path relative to output_dir
            rel_path = os.path.relpath(file_path, output_dir)
            # Separate filename and subfolder path
            subfolder = os.path.dirname(rel_path)
            filename = os.path.basename(file_path)
        elif os.path.exists(file_path):
            # File exists but not in the output directory, need to copy to output directory
            filename = os.path.basename(file_path)
            subfolder = ""  # Default save to output root directory
            
            # Create destination path
            dest_path = os.path.join(output_dir, filename)
            
            # Copy file to output directory
            shutil.copy2(file_path, dest_path)
            print(f"Copied file from {file_path} to {dest_path}")
            
            # Update file_path to the new path
            file_path = dest_path
        else:
            filename = None
            subfolder = ""
            
        results = list()
        results.append({
                "filename": filename,
                "subfolder": subfolder,
                "type": "output"
        })
        print(f"videos {results}")
        return { "ui": { "videos": results } }