import sys, os
sys.path.append(os.path.dirname(__file__))
from .node import (ResetModelPatcherCalculateWeight,
                   ApplyDenoiseMaskScheduler,
                   DetailTransferAdd,
                   DetailTransferLatentAdd,
                   DynamicImageResize)


NODE_CLASS_MAPPINGS = {
    "ResetModelPatcherCalculateWeight": ResetModelPatcherCalculateWeight,
    "ApplyDenoiseMaskScheduler": ApplyDenoiseMaskScheduler,
    "DetailTransferAdd":DetailTransferAdd,
    "DetailTransferLatentAdd":DetailTransferLatentAdd,
    "DynamicImageResize":DynamicImageResize
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ResetModelPatcherCalculateWeight": "Reset injected model patcher (middlek)",
    "ApplyDenoiseMaskScheduler": "Apply denoise mask scheduler (middlek)",
    "DetailTransferAdd":"Detail transfer mode:add (middlek)",
    "DetailTransferLatentAdd": "Detail transfer latent mode:add (middlek)",
    "DynamicImageResize":"Dynamic image resize (middlek)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
