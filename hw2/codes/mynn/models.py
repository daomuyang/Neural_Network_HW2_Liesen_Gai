from .op import *
import pickle

class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    def forward(self):
        pass

    def backward(self):
        pass


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)


class Model_MLP(Layer):
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func
        self.layers = []

        if size_list is not None and act_func is not None:
            for i in range(len(size_list)-1):
                layer = Linear(size_list[i], size_list[i+1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                else:
                    layer.weight_decay = False
                    layer.weight_decay_lambda = 0.0
                self.layers.append(layer)
                if i < len(size_list)-2:
                    self.layers.append(ReLU())

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        g = grad
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def save_model(self, path):
        params = []
        for layer in self.layers:
            if layer.optimizable:
                params.append({
                    "W": layer.W,
                    "b": layer.b
                })
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
        idx = 0
        for layer in self.layers:
            if layer.optimizable and idx < len(params):
                layer.W = params[idx]["W"]
                layer.b = params[idx]["b"]
                idx += 1


class Model_CNN(Layer):
    def __init__(self):
        super().__init__()
        self.layers = [
            conv2D(1,8,3,1,1),
            ReLU(),
            conv2D(8,16,3,1,0),
            ReLU(),
            Flatten(),
            Linear(16*26*26, 128),
            ReLU(),
            Linear(128,10)
        ]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        g = grad
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def save_model(self, path):
        params = []
        for layer in self.layers:
            if layer.optimizable:
                params.append({
                    "W": layer.W,
                    "b": layer.b
                })
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
        idx = 0
        for layer in self.layers:
            if layer.optimizable and idx < len(params):
                layer.W = params[idx]["W"]
                layer.b = params[idx]["b"]
                idx += 1
            