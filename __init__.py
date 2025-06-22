from .nodes.SaveVideo import VideoBasicSaveVideo
from .nodes.VideoUpload import VideoBasicUploadVideo
from .nodes.upscaleNode import VideoUpscaleWithModel
from .nodes.MergeVideo import VideoBasicMergeVideo


NODE_CLASS_MAPPINGS = { 
    "VideoBasicLoadVideo": VideoBasicUploadVideo,
    "VideoBasicVideoSave": VideoBasicSaveVideo,
    "VideoBasicVideoUpscaleWithModel": VideoUpscaleWithModel,
    "VideoBasicMergeVideo": VideoBasicMergeVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = { 
    "VideoBasicLoadVideo": "VideoBasic Load Video",
    "VideoBasicVideoSave": "VideoBasic Save Video",
    "VideoBasicVideoUpscaleWithModel": "VideoBasic Video Upscale with Model",
    "VideoBasicMergeVideo": "VideoBasic Merge Video",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]