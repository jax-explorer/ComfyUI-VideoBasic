import hashlib
import os

import folder_paths

class VideoBasicSaveVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"tooltip": "filename"})
            },
        }

    CATEGORY = "VideoBasic"
    OUTPUT_NODE = True


    RETURN_TYPES = ()
    FUNCTION = "save"

    def save(self, file_path):
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        
        # 处理相对路径，将其转换为绝对路径
        if not os.path.isabs(file_path):
            file_path = os.path.join(output_dir, file_path)
        else:
            file_path = os.path.abspath(file_path)

        print(f"file_path is {file_path}, output_dir is {output_dir}")
        
        # 判断路径是否在输出目录中且文件存在
        if file_path.startswith(output_dir) and os.path.exists(file_path):
            # 获取相对于output_dir的路径
            rel_path = os.path.relpath(file_path, output_dir)
            # 分离文件名和子文件夹路径
            subfolder = os.path.dirname(rel_path)
            filename = os.path.basename(file_path)
        else:
            filename = None
            subfolder = ""
            
        results = list()
        results.append({
                "filename": filename,
                "subfolder": subfolder,
                "type": "output"
        })
        return { "ui": { "file": results } }