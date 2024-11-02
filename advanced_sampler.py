import logging
from functools import wraps
from comfy.samplers import KSamplerX0Inpaint

def inject_ksamplerx0inpaint_call(original_ksampler_call_fn):
    @wraps(original_ksampler_call_fn)
    def wrapper(self, x, sigma, denoise_mask, model_options={}, seed=None):

        if denoise_mask is not None:
            # 사용자 정의 마스크 함수 적용 (있는 경우)
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + self.inner_model.inner_model.model_sampling.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1)), self.noise, self.latent_image) * latent_mask

        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        
        if self.sigmas[-2].item() >= sigma.item():
            KSamplerX0Inpaint.__call__ = original_ksampler_call_fn
            logging.info("\033[94m[Apply denoise-mask-scheduler] KSamplerX0Inpaint.__call__ return to original_ksampler_call_fn\033[0m")
                
        return out
    return wrapper
