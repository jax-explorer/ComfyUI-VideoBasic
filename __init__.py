


from .nodes.SaveVideo import VideoBasicSaveVideo
from .nodes.VideoUpload import VideoBasicUploadVideo


NODE_CLASS_MAPPINGS = { 
    "VideoBasicLoadVideo": VideoBasicUploadVideo,
    "VideoBasicVideoSave": VideoBasicSaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = { 
    "VideoBasicLoadVideo": "VideoBasic Load Video",
    "VideoBasicVideoSave": "VideoBasic Save Video",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]