from abc import abstractmethod
import numpy as np

class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass

class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    layer.params[key] -= self.init_lr * layer.grads[key]

class Momentum(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self._init_momentum_cache()

    def _init_momentum_cache(self):
        for layer in self.model.layers:
            if layer.optimizable:
                layer.velocity = {}
                for key in layer.params.keys():
                    layer.velocity[key] = np.zeros_like(layer.params[key])

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    layer.velocity[key] = self.mu * layer.velocity[key] - self.init_lr * layer.grads[key]
                    layer.params[key] += layer.velocity[key]