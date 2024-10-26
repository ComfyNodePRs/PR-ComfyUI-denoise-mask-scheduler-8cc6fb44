import logging
from functools import wraps
from comfy.samplers import KSamplerX0Inpaint
import torch
def inject_ksamplerx0inpaint_call(original_ksampler_call_fn):
    @wraps(original_ksampler_call_fn)
    def wrapper(self, x, sigma, denoise_mask, model_options={}, seed=None):
        
        # latent injection 옵션 확인
        denoise_mask_scheduler_opts = model_options.get("denoise_mask_scheduler_opts", None)

        if denoise_mask_scheduler_opts is not None:
            inject_latents = denoise_mask_scheduler_opts["inject_latents"]
            remain_injected = denoise_mask_scheduler_opts["remain_injected"]

        if denoise_mask is not None:
            # 사용자 정의 마스크 함수 적용 (있는 경우)
            if "denoise_mask_function" in model_options:
                denoise_mask, sigma = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + self.inner_model.inner_model.model_sampling.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1)), self.noise, self.latent_image) * latent_mask

        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        
        # print(torch.nonzero(denoise_mask==1.0).numel(), sigma)
        
        if denoise_mask_scheduler_opts is not None:
            # remain injected 옵션에 따라 원래 함수로 복구
            if remain_injected == False and self.sigmas[-2].item() >= sigma.item():
                KSamplerX0Inpaint.__call__ = original_ksampler_call_fn
                logging.info("\033[94m[middlek latent injection] KSamplerX0Inpaint.__call__ return to original_ksampler_call_fn\033[0m")
                
        return out
    return wrapper
