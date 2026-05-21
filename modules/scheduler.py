class LRScheduler:
    def __init__(self, optimizer, initial_lr, total_iterations, decay_iterations, min_lr=None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.total_iterations = total_iterations
        self.decay_iterations = decay_iterations
        self.constant_iterations = total_iterations - decay_iterations
        self.min_lr = min_lr if min_lr is not None else 0
        self.current_step = 0
    
    def step(self):
        if self.current_step < self.constant_iterations:
            lr = self.initial_lr
        
        else:
            decay_ratio = (self.current_step - self.constant_iterations) / self.decay_iterations
            lr = max(self.min_lr, self.initial_lr * (1 - decay_ratio))

        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        self.current_step += 1