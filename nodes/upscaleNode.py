from comfy import model_management
import comfy.utils
import torch
import cv2
import os
import numpy as np
from tqdm import tqdm


class VideoUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "upscale_model": ("UPSCALE_MODEL",),
                    "video_path": ("STRING", {"default": ""}),
                    "batch_size": ("INT", {"default": 8, "min": 1, "max": 1000, "step": 1}),
                    "output_path": ("STRING", {"default": "upscaled_video.mp4"}),
                }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "upscale_video"

    CATEGORY = "VideoBasic"

    def upscale_video(self, upscale_model, video_path, batch_size, output_path):
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算上采样后的分辨率
        new_width = int(width * upscale_model.scale)
        new_height = int(height * upscale_model.scale)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
        
        device = model_management.get_torch_device()
        
        # 估计所需内存
        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (width * height * 3) * 4 * max(upscale_model.scale, 1.0) * 384.0 * batch_size
        model_management.free_memory(memory_required, device)
        
        # 将模型移至设备
        upscale_model.to(device)
        
        # 设置瓦片参数
        tile = 512
        overlap = 32
        
        # 批量处理帧
        frames_processed = 0
        
        try:
            with tqdm(total=total_frames) as pbar:
                while frames_processed < total_frames:
                    # 读取一批帧
                    batch_frames = []
                    for _ in range(min(batch_size, total_frames - frames_processed)):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # 转换为RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # 归一化到0-1
                        frame = frame.astype(np.float32) / 255.0
                        batch_frames.append(frame)
                    
                    if not batch_frames:
                        break
                    
                    # 转换为PyTorch张量
                    batch_tensor = torch.from_numpy(np.stack(batch_frames)).permute(0, 3, 1, 2).to(device)
                    
                    # 上采样处理
                    upscaled_batch = None
                    oom = True
                    current_tile = tile
                    
                    while oom:
                        try:
                            steps = batch_tensor.shape[0] * comfy.utils.get_tiled_scale_steps(
                                batch_tensor.shape[3], batch_tensor.shape[2], 
                                tile_x=current_tile, tile_y=current_tile, overlap=overlap
                            )
                            pbar_upscale = comfy.utils.ProgressBar(steps)
                            
                            # 逐帧处理批次
                            upscaled_frames = []
                            for i in range(batch_tensor.shape[0]):
                                frame = batch_tensor[i:i+1]
                                upscaled = comfy.utils.tiled_scale(
                                    frame, lambda a: upscale_model(a), 
                                    tile_x=current_tile, tile_y=current_tile, 
                                    overlap=overlap, upscale_amount=upscale_model.scale, 
                                    pbar=pbar_upscale
                                )
                                upscaled_frames.append(upscaled)
                            
                            upscaled_batch = torch.cat(upscaled_frames, dim=0)
                            oom = False
                        except model_management.OOM_EXCEPTION as e:
                            current_tile //= 2
                            if current_tile < 128:
                                raise e
                    
                    # 转换回numpy并写入视频
                    upscaled_batch = torch.clamp(upscaled_batch.permute(0, 2, 3, 1), min=0, max=1.0)
                    upscaled_batch = upscaled_batch.cpu().numpy()
                    
                    for frame in upscaled_batch:
                        # 转换为BGR并缩放到0-255
                        frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)
                    
                    frames_processed += len(batch_frames)
                    pbar.update(len(batch_frames))
        
        finally:
            # 清理资源
            upscale_model.to("cpu")
            cap.release()
            out.release()
        
        print(f"Video upscaling complete. Output saved to: {output_path}")
        return (output_path,)