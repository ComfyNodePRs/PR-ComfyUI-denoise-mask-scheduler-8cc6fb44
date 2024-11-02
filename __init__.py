import sys, os
sys.path.append(os.path.dirname(__file__))
from .node import (ApplyDenoiseMaskSchedulerWithSigma,
                   ApplyDenoiseMaskSchedulerWithStep,
                   DynamicImageResize,
                   )


NODE_CLASS_MAPPINGS = {
    "ApplyDenoiseMaskSchedulerWithSigma": ApplyDenoiseMaskSchedulerWithSigma,
    "ApplyDenoiseMaskSchedulerWithStep": ApplyDenoiseMaskSchedulerWithStep,
    "DynamicImageResize":DynamicImageResize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyDenoiseMaskSchedulerWithSigma": "Apply Denoise Mask Scheduler (use sigma)",
    "ApplyDenoiseMaskSchedulerWithStep": "Apply Denoise Mask Scheduler (use step)",
    "DynamicImageResize":"Dynamic image resize",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
