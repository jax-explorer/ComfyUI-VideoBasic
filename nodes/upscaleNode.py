from comfy import model_management
import comfy.utils
import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
import folder_paths


class VideoUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "upscale_model": ("UPSCALE_MODEL",),
                    "video_path": ("STRING", {"default": ""}),
                    "batch_size": ("INT", {"default": 8, "min": 1, "max": 1000, "step": 1}),
                }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "upscale_video"

    CATEGORY = "VideoBasic"

    def upscale_video(self, upscale_model, video_path, batch_size):
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Get temporary directory
        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        video_filename = os.path.basename(video_path)
        output_filename = f"upscaled_{video_filename}"
        if not output_filename.lower().endswith('.mp4'):
            output_filename += '.mp4'
        output_path = os.path.join(output_dir, output_filename)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video information
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate upsampled resolution
        new_width = int(width * upscale_model.scale)
        new_height = int(height * upscale_model.scale)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
        
        device = model_management.get_torch_device()
        
        # Estimate required memory
        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (width * height * 3) * 4 * max(upscale_model.scale, 1.0) * 384.0 * batch_size
        model_management.free_memory(memory_required, device)
        
        # Move model to device
        upscale_model.to(device)
        
        # Set tile parameters
        tile = 512
        overlap = 32
        
        # Process frames in batches
        frames_processed = 0
        
        try:
            with tqdm(total=total_frames) as pbar:
                while frames_processed < total_frames:
                    # Read a batch of frames
                    batch_frames = []
                    for _ in range(min(batch_size, total_frames - frames_processed)):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Convert to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Normalize to 0-1
                        frame = frame.astype(np.float32) / 255.0
                        batch_frames.append(frame)
                    
                    if not batch_frames:
                        break
                    
                    # Convert to PyTorch tensor
                    batch_tensor = torch.from_numpy(np.stack(batch_frames)).permute(0, 3, 1, 2).to(device)
                    
                    # Upsampling process
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
                            
                            # Process frames in batch one by one
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
                    
                    # Convert back to numpy and write to video
                    upscaled_batch = torch.clamp(upscaled_batch.permute(0, 2, 3, 1), min=0, max=1.0)
                    upscaled_batch = upscaled_batch.cpu().numpy()
                    
                    for frame in upscaled_batch:
                        # Convert to BGR and scale to 0-255
                        frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)
                    
                    frames_processed += len(batch_frames)
                    pbar.update(len(batch_frames))
        
        finally:
            # Clean up resources
            upscale_model.to("cpu")
            cap.release()
            out.release()
        
        print(f"Video upscaling complete. Output saved to: {output_path}")
        return (output_path,)