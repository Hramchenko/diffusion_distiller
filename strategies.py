import torch

class TrainingStrategy:

    def init(self, student_diffusion, student_lr, total_steps):
        raise Exception()

    def zero_grad(self):
        raise Exception()

    def step(self):
        raise Exception()

    def stop(self, N, max_iter):
        return N > max_iter

class StrategyOneCycle(TrainingStrategy):

    def init(self, student_diffusion, student_lr, total_steps):
        self.student_optimizer = torch.optim.SGD(student_diffusion.net_.parameters(), lr=student_lr, weight_decay=min(1e-5, 0.5*student_lr/10))
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.student_optimizer, max_lr=student_lr, total_steps=total_steps + 2)

    def zero_grad(self):
        self.student_optimizer.zero_grad()

    def step(self):
        self.student_optimizer.step()
        self.scheduler.step()

class StrategyConstantLR(TrainingStrategy):

    def init(self, student_diffusion, student_lr, total_steps):
        self.student_optimizer = torch.optim.AdamW(student_diffusion.net_.parameters(), lr=student_lr)

    def zero_grad(self):
        self.student_optimizer.zero_grad()

    def step(self):
        self.student_optimizer.step()

class StrategyLinearLR(TrainingStrategy):

    def init(self, student_diffusion, student_lr, total_steps):
        self.student_optimizer = torch.optim.AdamW(student_diffusion.net_.parameters(), lr=student_lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.student_optimizer, start_factor=1, end_factor=0, total_iters=total_steps)

    def zero_grad(self):
        self.student_optimizer.zero_grad()

    def step(self):
        self.student_optimizer.step()
        self.scheduler.step()

class StrategyCosineAnnel(TrainingStrategy):

    def init(self, student_diffusion, student_lr, total_steps):
        self.student_lr = student_lr
        self.eta_min = 0.01
        self.student_optimizer = torch.optim.SGD(student_diffusion.net_.parameters(), lr=student_lr, weight_decay=min(1e-5, student_lr/10))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.student_optimizer, T_0=100, T_mult=2, eta_min=self.eta_min, last_epoch=-1)

    def zero_grad(self):
        self.student_optimizer.zero_grad()

    def step(self):
        self.student_optimizer.step()
        self.scheduler.step()

    def stop(self, N, max_iter):
        return (self.scheduler.get_last_lr()[0] < self.student_lr * self.eta_min * 3) and (N > max_iter)
