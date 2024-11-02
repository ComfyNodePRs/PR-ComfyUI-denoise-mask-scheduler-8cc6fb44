import torch

class DenoiseMaskScheduler:
    def __init__(self,
                 start_sigma:float=15.0, 
                 end_sigma:float=0.5, 
                 start_step:int=0,
                 end_step:int=30,
                 mask_alpha:float=1.0,
                 scheduler_type:str="skip_with_step",
                 kernel_size:int=7,
                ):
        self.kernel_size = kernel_size
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma
        self.start_step = start_step
        self.end_step = end_step
        self.mask_alpha = mask_alpha
        self.scheduler_type = scheduler_type
        
        self.device = None
        self.dtype = None
        self.kernel = None

        self.kernel_padding = int((kernel_size-1)/2)
        self.schedule = []
        
    def get_dilate_mask(self, mask:torch.Tensor):
        mask_clone = mask.clone()
        dilated_mask = torch.functional.F.conv2d(mask_clone, self.kernel, padding=self.kernel_padding).squeeze(0)
        dilated_mask = torch.clamp(dilated_mask, 0, 1.0)
        del mask_clone
        return dilated_mask
    
    def get_ones_mask(self, mask:torch.Tensor):
        return torch.ones_like(mask, device=self.device, dtype=self.dtype)

    def get_zeros_mask(self, mask:torch.Tensor):
        return torch.zeros_like(mask, device=self.device, dtype=self.dtype)

    def make_skip_schedule(self, sigmas:torch.Tensor, denoise_mask:torch.Tensor, use_sigma=False):
        if use_sigma == True:
            for step, sigma in enumerate(sigmas):
                if self.start_sigma < sigma:
                    self.schedule.append(self.get_ones_mask(denoise_mask))
                elif self.end_sigma < sigma <= self.start_sigma:
                    self.schedule.append(self.mask_alpha * denoise_mask)
                elif sigma <= self.end_sigma:
                    self.schedule.append(self.get_ones_mask(denoise_mask))
                else:
                    pass
        else:
            for step, sigma in enumerate(sigmas):
                if step < self.start_step:
                    self.schedule.append(self.get_ones_mask(denoise_mask))
                elif self.start_step <= step <= self.end_step:
                    self.schedule.append(self.mask_alpha * denoise_mask)
                elif self.end_step < step:
                    self.schedule.append(self.get_ones_mask(denoise_mask))
                else:
                    pass

    def set_variables(self, denoise_mask:torch.Tensor, sigmas:torch.Tensor):
        self.device = denoise_mask.device
        self.dtype = denoise_mask.dtype
        self.kernel = torch.ones((4, 4, self.kernel_size, self.kernel_size), device=self.device, dtype=self.dtype)

        if self.scheduler_type == "skip_with_step":
            self.make_skip_schedule(sigmas, denoise_mask, use_sigma=False)
        elif self.scheduler_type == "skip_with_sigma":
            self.make_skip_schedule(sigmas, denoise_mask, use_sigma=True)
        else:
            Warning(f"Unknown {self.scheduler_type}. denoise mask schedluer is ignored.")
        
    def get_denoise_mask(self, sigma:torch.Tensor, denoise_mask:torch.Tensor, extra_options:dict={}):
        sigmas = extra_options.get("sigmas")

        if len(self.schedule) == 0:
            self.set_variables(denoise_mask, sigmas)
            if len(self.schedule) == 0:
                Warning(f"Denoise mask schedule length is '{len(self.schedule)}'. original denoise will be used.")
                return denoise_mask

        schedule_idx = (torch.abs(sigmas - sigma)).argmin().item()
        
        return self.schedule[schedule_idx]
    
    def reset(self):
        self.schedule = []
        self.call_count = 0
