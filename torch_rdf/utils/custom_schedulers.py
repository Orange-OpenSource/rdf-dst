import torch.optim.lr_scheduler as lr_scheduler

class LinearWarmupScheduler(lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warm-up phase
            lr_scale = self.last_epoch / self.warmup_steps
        else:
            # Linear decay phase
            lr_scale = 1 - (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return [base_lr * lr_scale for base_lr in self.base_lrs]


#TODO: Try other schedulers

