import sys
from .utils import dynamic_resize
from denoise_mask_scheduler import DenoiseMaskScheduler
from advanced_sampler import KSamplerX0Inpaint, inject_ksamplerx0inpaint_call
import logging

def inject_ksampler():
    # KSamplerX0Inpaint의 __call__ 메서드를 수정된 버전으로 교체
    original_ksampler_call_fn = KSamplerX0Inpaint.__call__
    KSamplerX0Inpaint.__call__ = inject_ksamplerx0inpaint_call(original_ksampler_call_fn)
    logging.info("\033[94m[Apply denoise-mask-scheduler] KSamplerX0Inpaint.__call__ is injected to inject_ksamplerx0inpaint_call\033[0m")

class ApplyDenoiseMaskSchedulerWithSigma:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model":("MODEL", ),
                             "start_sigma": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 15.0, "step": 0.05}),
                             "end_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 15.0, "step": 0.05}),
                             "mask_alpha": ("FLOAT", {"default":1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                             "scheduler_type": (["skip"], {"default":"skip"}),
                             },
                }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "denoise-mask-scheduler"

    def run(self, model, start_sigma, end_sigma, mask_alpha, scheduler_type):
        if scheduler_type == "skip":
            scheduler_type = "skip_with_sigma"
        else:
            raise ValueError(f"{scheduler_type} is not supported for denoise mask scheduler.")
        
        denoise_mask_scheduler = DenoiseMaskScheduler(
            scheduler_type=scheduler_type,
            start_sigma=start_sigma,
            end_sigma=end_sigma,
            mask_alpha=mask_alpha,
        )

        if hasattr(model, "model_options"):
            model.model_options["denoise_mask_function"] = denoise_mask_scheduler.get_denoise_mask
        else:
            raise TypeError(f"'{type(model)}' is not a ComfyUI ModelPatcher type.")
        
        inject_ksampler()
        
        return (model, )

class ApplyDenoiseMaskSchedulerWithStep:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model":("MODEL", ),
                             "start_step": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                             "end_step": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                             "mask_alpha": ("FLOAT", {"default":1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                             "scheduler_type": (["skip"], {"default":"skip"}),
                             },
                }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "denoise-mask-scheduler"

    def run(self, model, start_step, end_step, mask_alpha, scheduler_type):
        if scheduler_type == "skip":
            scheduler_type = "skip_with_step"
        else:
            raise ValueError(f"{scheduler_type} is not supported for denoise mask scheduler.")
        
        denoise_mask_scheduler = DenoiseMaskScheduler(
            scheduler_type=scheduler_type,
            start_step=start_step,
            end_step=end_step,
            mask_alpha=mask_alpha,
        )

        if hasattr(model, "model_options"):
            model.model_options["denoise_mask_function"] = denoise_mask_scheduler.get_denoise_mask
        else:
            raise TypeError(f"'{type(model)}' is not a ComfyUI ModelPatcher type.")

        return (model, )

# 동적 이미지 리사이즈
class DynamicImageResize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE",),
                    "max_pixels": ("INT", {"default": 1024*1024, "min": 500, "max": sys.maxsize, "step": 1}),
                    "min_pixels": ("INT", {"default": 512*512, "min": 500, "max": sys.maxsize, "step": 1}),
                    }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dynamic_image_resize"

    def dynamic_image_resize(self, image, max_pixels:int, min_pixels:int):
        resized_image = dynamic_resize(image, max_pixels=max_pixels, min_pixels=min_pixels)
        return (resized_image, )
