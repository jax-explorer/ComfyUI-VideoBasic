import os
import subprocess
import tempfile
import folder_paths


class VideoBasicMergeVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path_1": ("STRING",),
                "video_path_2": ("STRING",),
            },
            "optional": {
                "video_path_3": ("STRING", {"default": ""}),
                "video_path_4": ("STRING", {"default": ""}),
                "video_path_5": ("STRING", {"default": ""}),
                "video_path_6": ("STRING", {"default": ""}),
                "video_path_7": ("STRING", {"default": ""}),
                "video_path_8": ("STRING", {"default": ""}),
                "video_path_9": ("STRING", {"default": ""}),
                "video_path_10": ("STRING", {"default": ""}),
                "output_filename": ("STRING", {"default": "merged_video.mp4"}),
                "use_gpu": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "VideoBasic"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_video_path",)

    FUNCTION = "merge_videos"

    def merge_videos(self, video_path_1, video_path_2, video_path_3="", video_path_4="", 
                    video_path_5="", video_path_6="", video_path_7="", video_path_8="", 
                    video_path_9="", video_path_10="", output_filename="merged_video.mp4",
                    use_gpu=False):
        
        # 收集所有非空的视频路径
        video_paths = [video_path_1, video_path_2]
        optional_paths = [video_path_3, video_path_4, video_path_5, video_path_6, 
                         video_path_7, video_path_8, video_path_9, video_path_10]
        
        for path in optional_paths:
            if path and path.strip():
                video_paths.append(path)
        
        print(f"Merging {len(video_paths)} videos using FFmpeg: {video_paths}")
        
        # 验证所有视频文件是否存在
        for i, path in enumerate(video_paths):
            if not os.path.exists(path):
                raise ValueError(f"Video file {i+1} not found: {path}")
        
        # 获取输出目录
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, output_filename)
        
        # 检查FFmpeg是否可用
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("FFmpeg not found. Please install FFmpeg to use this node.")
        
        # 使用FFmpeg的concat filter进行合并
        # 这种方法比逐帧处理快得多，因为它直接在编码层面操作
        try:
            # 方法1: 使用concat filter (适用于不同格式的视频)
            # 构建filter_complex参数
            inputs = []
            filter_parts = []
            
            for i, video_path in enumerate(video_paths):
                inputs.extend(['-i', video_path])
                filter_parts.append(f'[{i}:v][{i}:a]')
            
            # 构建concat filter
            concat_filter = f"{''.join(filter_parts)}concat=n={len(video_paths)}:v=1:a=1[outv][outa]"
            
            # 构建完整的FFmpeg命令
            cmd = ['ffmpeg', '-y'] + inputs + [
                '-filter_complex', concat_filter,
                '-map', '[outv]',
                '-map', '[outa]',
                '-c:v', 'libx264',  # 使用H.264编码
                '-preset', 'medium',  # 平衡速度和质量
                '-crf', '23',  # 质量设置
                '-c:a', 'aac',  # 音频编码
                '-b:a', '128k',  # 音频比特率
                output_path
            ]
            
            # 如果启用GPU加速
            if use_gpu:
                # 尝试使用NVIDIA GPU加速
                cmd[cmd.index('-c:v')+1] = 'h264_nvenc'
                cmd.extend(['-preset', 'fast'])
                print("Using GPU acceleration (NVIDIA)")
            
            print(f"Running FFmpeg command: {' '.join(cmd)}")
            
            # 执行FFmpeg命令
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                # 如果GPU加速失败，回退到CPU
                if use_gpu:
                    print("GPU acceleration failed, falling back to CPU...")
                    cmd[cmd.index('h264_nvenc')] = 'libx264'
                    # 移除GPU特定参数
                    if '-preset' in cmd and 'fast' in cmd:
                        preset_idx = cmd.index('-preset')
                        cmd[preset_idx+1] = 'medium'
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            print(f"Successfully merged {len(video_paths)} videos to: {output_path}")
            return (output_path,)
            
        except Exception as e:
            # 如果concat filter失败，尝试使用文件列表方法
            print(f"Concat filter failed ({e}), trying file list method...")
            return self._merge_with_filelist(video_paths, output_path, use_gpu)

    def _merge_with_filelist(self, video_paths, output_path, use_gpu=False):
        """使用文件列表方法合并视频（适用于相同格式的视频）"""
        
        # 创建临时文件列表
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            filelist_path = f.name
            for video_path in video_paths:
                # 转义文件路径中的特殊字符
                escaped_path = video_path.replace("'", "'\"'\"'")
                f.write(f"file '{escaped_path}'\n")
        
        try:
            # 使用concat demuxer
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', filelist_path,
                '-c', 'copy',  # 直接复制，不重新编码（最快）
                output_path
            ]
            
            print(f"Running FFmpeg with file list: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"File list method failed: {result.stderr}")
                # 如果直接复制失败，尝试重新编码
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', filelist_path,
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-c:a', 'aac',
                    output_path
                ]
                
                if use_gpu:
                    cmd[cmd.index('-c:v')+1] = 'h264_nvenc'
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            return (output_path,)
            
        finally:
            # 清理临时文件
            try:
                os.unlink(filelist_path)
            except:
                pass

    @classmethod
    def VALIDATE_INPUTS(s, video_path_1, video_path_2, **kwargs):
        # 验证必需的视频文件
        if not os.path.exists(video_path_1):
            return f"Video file 1 not found: {video_path_1}"
        if not os.path.exists(video_path_2):
            return f"Video file 2 not found: {video_path_2}"
        
        # 验证可选的视频文件
        optional_paths = [
            kwargs.get('video_path_3', ''),
            kwargs.get('video_path_4', ''),
            kwargs.get('video_path_5', ''),
            kwargs.get('video_path_6', ''),
            kwargs.get('video_path_7', ''),
            kwargs.get('video_path_8', ''),
            kwargs.get('video_path_9', ''),
            kwargs.get('video_path_10', ''),
        ]
        
        for i, path in enumerate(optional_paths, 3):
            if path and path.strip() and not os.path.exists(path):
                return f"Video file {i} not found: {path}"
        
        return True
