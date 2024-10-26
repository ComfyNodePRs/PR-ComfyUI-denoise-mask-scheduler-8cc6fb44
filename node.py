import sys
import logging

from comfy import model_management
from comfy.model_patcher import ModelPatcher
from comfy.samplers import KSamplerX0Inpaint

from .advanced_sampler import inject_ksamplerx0inpaint_call
from .utils import (simple_resize, add_detail_transfer, dynamic_resize)

# ModelPatcher의 calculate_weight 메서드를 초기화하는 클래스
class ResetModelPatcherCalculateWeight:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model":("MODEL", ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "reset_moodelpatcher_weight"
    CATEGORY = "productfix"

    def reset_moodelpatcher_weight(self, model:ModelPatcher):
        # 원래의 calculate_weight 메서드로 복원
        if hasattr(model, "original_calculate_weight"):
            model.calculate_weight = ModelPatcher.original_calculate_weight
            ModelPatcher.calculate_weight = ModelPatcher.original_calculate_weight
        return (model, )

from denoise_mask_scheduler import DenoiseMaskScheduler
# 잠재 공간에 이미지를 주입하는 클래스
class ApplyDenoiseMaskScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model":("MODEL", ),
                             "latents": ("LATENT", ),
                             "start_sigma": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 15.0, "step": 0.05}),
                             "end_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 15.0, "step": 0.05}),
                             "mask_alpha": ("FLOAT", {"default":1.0, "min": 0.0, "max": 1.0, "step": 0.1})
                             },
                "optional": {
                    "inject_latents": ("LATENT", {"default":None}),
                    "remain_injected":([True, False], {"default":True})
                    },
                }
    RETURN_TYPES = ("MODEL", "LATENT",)
    FUNCTION = "apply_latent_injection"
    CATEGORY = "productfix"

    def apply_latent_injection(self, model, latents, start_sigma, end_sigma, mask_alpha, inject_latents=None, remain_injected=True):

        device = model_management.get_torch_device()
        dtype = model_management.VAE_DTYPES[0]
        
        # KSamplerX0Inpaint의 __call__ 메서드를 수정된 버전으로 교체
        original_ksampler_call_fn = KSamplerX0Inpaint.__call__
        KSamplerX0Inpaint.__call__ = inject_ksamplerx0inpaint_call(original_ksampler_call_fn)
        logging.info("\033[94m[middlek latent injection] KSamplerX0Inpaint.__call__ is injected to inject_ksamplerx0inpaint_call\033[0m")

        # 잠재 공간 및 마스크 준비
        # (데이터 형식 변환 및 디바이스 이동)

        b, c, h, w = latents["samples"].shape

        if inject_latents is not None:
            if isinstance(inject_latents, dict):
                inject_latents = inject_latents["samples"]
        
            inject_latents = inject_latents.to(device=device, dtype=dtype)
            _, _, ih, iw = inject_latents.shape

            if ih != h or iw != w:
                raise ValueError("inject_latents size is not match with latents")
        
        denoise_mask_scheduler = DenoiseMaskScheduler(
            start_sigma=start_sigma,
            end_sigma=end_sigma,
            mask_alpha=mask_alpha,
        )

        # 모델 옵션에 latent injection 파라미터 추가
        if hasattr(model, "model_options"):
            model.model_options["denoise_mask_scheduler_opts"] = {"inject_latents":inject_latents, "remain_injected":remain_injected}
            model.model_options["denoise_mask_function"] = denoise_mask_scheduler.get_denoise_mask
        
        return (model, latents, )

# 디테일 전송을 수행하는 클래스 (이미지 도메인)
class DetailTransferAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "target": ("IMAGE", ),
                    "source": ("IMAGE", ),
                    "blur": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.01}),
                    "blend_ratio": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,  "round": 0.001}),
                },
            "optional": {
                "mask": ("MASK", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detail_transfer_add"
    CATEGORY = "productfix"

    def detail_transfer_add(self, target, source, blur, blend_ratio, mask=None):
        output_image = add_detail_transfer(target, source, blur, blend_ratio, mask)
        return (output_image, )

# 디테일 전송을 수행하는 클래스 (잠재 공간 도메인)
class DetailTransferLatentAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "target": ("LATENT", ),
                    "source": ("LATENT", ),
                    "blur": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.01}),
                    "blend_ratio": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,  "round": 0.001}),
                },
            "optional": {
                "mask": ("MASK", ),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "detail_transfer_add"
    CATEGORY = "productfix"

    def detail_transfer_add(self, target, source, blur, blend_ratio, mask=None):
        # 잠재 공간을 이미지 형식으로 변환
        if type(target) == dict:
            target = target["samples"]
        if type(source) == dict:
            source = source["samples"]
        
        target = target.permute(0,2,3,1)
        source = source.permute(0,2,3,1)
        
        # 디테일 전송 수행
        output_image = add_detail_transfer(target, source, blur, blend_ratio, mask)
        
        # 결과를 다시 잠재 공간 형식으로 변환
        output_latent = output_image.permute(0,3,1,2)

        return (output_latent, )
    
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
    CATEGORY = "productfix"

    def dynamic_image_resize(self, image, max_pixels:int, min_pixels:int):
        resized_image = dynamic_resize(image, max_pixels=max_pixels, min_pixels=min_pixels)
        return (resized_image, )