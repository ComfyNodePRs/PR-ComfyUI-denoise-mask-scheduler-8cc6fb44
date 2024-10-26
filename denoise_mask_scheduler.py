import torch

class DenoiseMaskScheduler:
    def __init__(self,
                 kernel_size:int=7, 
                 start_sigma:float=10.0, 
                 end_sigma:float=0.5, 
                 mask_alpha:float=1.0,
                 scheduler_type:str="constant"
                ):
        self.kernel_size = kernel_size
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma
        self.mask_alpha = mask_alpha
        self.scheduler_type = scheduler_type
        
        self.device = None
        self.dtype = None
        self.kernel = None

        self.kernel_padding = int((kernel_size-1)/2)
        self.schedule = []
        
    def dilate_mask(self, mask:torch.Tensor):
        mask_clone = mask.clone()
        dilated_mask = torch.functional.F.conv2d(mask_clone, self.kernel, padding=self.kernel_padding).squeeze(0)
        dilated_mask = torch.clamp(dilated_mask, 0, 1.0)
        del mask_clone
        return dilated_mask
    
    def make_constant_schedule(self, sigmas:torch.Tensor):
        for sigma in sigmas:
            if self.start_sigma < sigma:
                self.schedule.append((-1, sigma))
            elif self.end_sigma < sigma <= self.start_sigma:
                self.schedule.append((self.mask_alpha, sigma))
            elif sigma <= self.end_sigma:
                self.schedule.append((-1, sigma))
            else:
                pass

    def set_variables(self, denoise_mask:torch.Tensor, sigmas:torch.Tensor):
        self.device = denoise_mask.device
        self.dtype = denoise_mask.dtype
        self.kernel = torch.ones((4, 4, self.kernel_size, self.kernel_size), device=self.device, dtype=self.dtype)

        if self.scheduler_type == "constant":
            self.make_constant_schedule(sigmas)
        
    def get_denoise_mask(self, sigma:torch.Tensor, denoise_mask:torch.Tensor, extra_options:dict={}):
        sigmas = extra_options.get("sigmas")

        if len(self.schedule) == 0:
            self.set_variables(denoise_mask, sigmas)

        schedule_idx = (torch.abs(sigmas - sigma)).argmin().item()
        mask_alpha, new_sigma = self.schedule[schedule_idx]

        clone_denoise_mask = denoise_mask.clone()

        if mask_alpha == -1:
            clone_denoise_mask = torch.ones_like(clone_denoise_mask, device=self.device, dtype=self.dtype)
        else:
            clone_denoise_mask = clone_denoise_mask * mask_alpha
        
        if len(self.schedule)-2 <= schedule_idx:
            self.reset()

        return clone_denoise_mask, new_sigma.unsqueeze(0)
    
    def reset(self):
        self.schedule = []
        self.call_count = 0