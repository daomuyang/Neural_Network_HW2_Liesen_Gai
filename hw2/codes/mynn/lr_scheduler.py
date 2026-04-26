from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0


class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        """
        Multi-step learning rate scheduler.
        Args:
            optimizer: 优化器实例
            milestones: 学习率衰减的步数列表（如 [800, 2400, 4000]）
            gamma: 学习率衰减系数
        """
        super().__init__(optimizer)
        self.milestones = milestones  # 必须是升序排列的整数列表
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        # 当当前步数到达里程碑时，衰减学习率
        if self.step_count in self.milestones:
            self.optimizer.init_lr *= self.gamma


class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma=0.99) -> None:
        """
        Exponential learning rate scheduler.
        Args:
            optimizer: 优化器实例
            gamma: 每步的学习率衰减系数
        """
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        # 每一步都按指数衰减学习率：lr = lr * gamma
        self.optimizer.init_lr *= self.gamma