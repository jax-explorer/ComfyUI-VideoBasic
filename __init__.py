


from .nodes.SaveVideo import VideoBasicSaveVideo
from .nodes.VideoUpload import VideoBasicUploadVideo
from .nodes.upscaleNode import VideoUpscaleWithModel


NODE_CLASS_MAPPINGS = { 
    "VideoBasicLoadVideo": VideoBasicUploadVideo,
    "VideoBasicVideoSave": VideoBasicSaveVideo,
    "VideoBasicVideoUpscaleWithModel": VideoUpscaleWithModel,
}

NODE_DISPLAY_NAME_MAPPINGS = { 
    "VideoBasicLoadVideo": "VideoBasic Load Video",
    "VideoBasicVideoSave": "VideoBasic Save Video",
    "VideoBasicVideoUpscaleWithModel": "VideoBasic Video Upscale with Model",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]