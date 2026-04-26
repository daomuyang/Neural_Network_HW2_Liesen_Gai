from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # 初始化参数：W (in_dim, out_dim), b (1, out_dim)
        self.W = initialize_method(size=(in_dim, out_dim)) * 0.01
        self.b = np.zeros((1, out_dim))  
        self.grads = {'W' : None, 'b' : None}
        self.input = None 

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay 
        self.weight_decay_lambda = weight_decay_lambda 
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X  # 保存输入用于反向传播
        # 前向计算：Y = X @ W + b
        output = np.dot(X, self.W) + self.b
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # 计算 dL/dW = X.T @ grad
        self.grads['W'] = np.dot(self.input.T, grad)
        # 计算 dL/db = sum(grad, axis=0) (batch维度求和)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        
        # 加上L2正则化项的梯度
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        
        # 计算 dL/dX = grad @ W.T，传递给前一层
        grad_input = np.dot(grad, self.W.T)
        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He初始化
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (kernel_size*kernel_size*in_channels))
        self.b = np.zeros((1, out_channels, 1, 1))
        self.grads = {'W': None, 'b': None}
        self.input = None
        
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        N, C, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        
        # 计算输出尺寸
        H_out = (H + 2*p - k) // s + 1
        W_out = (W + 2*p - k) // s + 1
        
        # 1. 填充输入
        X_pad = np.pad(X, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        
        # 2. 初始化列空间矩阵
        cols = np.zeros((N, C, k, k, H_out, W_out))
        
        # 3. 填充列空间
        for h in range(k):
            for w in range(k):
                cols[:, :, h, w, :, :] = X_pad[:, :, h:h+H_out*s:s, w:w+W_out*s:s]
        
        # 4. 重塑成二维矩阵
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N*H_out*W_out, -1)
        W_col = self.W.reshape(self.out_channels, -1)
        
        # 5. 矩阵乘法
        out = cols @ W_col.T + self.b.reshape(1, -1)
        
        # 6. 重塑回输出形状
        out = out.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)
        
        # 保存中间结果
        self.cols = cols
        self.X_pad = X_pad
        self.N = N
        self.H_out = H_out
        self.W_out = W_out
        
        return out

    def backward(self, grads):
        N, C, H, W = self.input.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        H_out = self.H_out
        W_out = self.W_out
        
        # 1. 计算b的梯度
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True)
        
        # 2. 计算W的梯度
        grads_reshaped = grads.transpose(0, 2, 3, 1).reshape(N*H_out*W_out, -1)
        self.grads['W'] = grads_reshaped.T @ self.cols
        self.grads['W'] = self.grads['W'].reshape(self.W.shape)
        
        # 3. 计算输入的梯度
        W_col = self.W.reshape(self.out_channels, -1)
        dX_col = grads_reshaped @ W_col
        
        # 4. col2im：还原回输入空间
        dX_col = dX_col.reshape(N, H_out, W_out, C, k, k).transpose(0, 3, 4, 5, 1, 2)
        dX_pad = np.zeros_like(self.X_pad)
        
        for h in range(k):
            for w in range(k):
                dX_pad[:, :, h:h+H_out*s:s, w:w+W_out*s:s] += dX_col[:, :, h, w, :, :]
        
        # 5. 去掉填充
        if p > 0:
            dX = dX_pad[:, :, p:-p, p:-p]
        else:
            dX = dX_pad
        
        # 6. L2正则化
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        
        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = True
        self.softmax_pred = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.labels = labels
        batch_size = predicts.shape[0]
        
        if self.has_softmax:
            # 先做Softmax
            self.softmax_pred = softmax(predicts)
        else:
            self.softmax_pred = predicts
        
        # 取对应label的概率，加1e-8防止log(0)
        correct_log_probs = -np.log(self.softmax_pred[np.arange(batch_size), labels] + 1e-8)
        # 平均loss
        loss = np.mean(correct_log_probs)
        
        return loss
    
    def backward(self):
        batch_size = self.softmax_pred.shape[0]
        
        # 梯度计算：(softmax_pred - one_hot_labels) / batch_size
        self.grads = self.softmax_pred.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads /= batch_size
        
        # Then send the grads to model for back propagation
        if self.model is not None:
            self.model.backward(self.grads)
        
        return self.grads

    def cancel_soft_max(self):
        self.has_softmax = False
        return self


class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, model, lambda_l2=1e-4):
        super().__init__()
        self.model = model
        self.lambda_l2 = lambda_l2
        self.optimizable = False

    def forward(self):
        # 计算L2正则化项：0.5 * lambda * sum(W^2)
        reg_loss = 0.0
        for layer in self.model.layers:
            if hasattr(layer, 'W'):
                reg_loss += 0.5 * self.lambda_l2 * np.sum(layer.W ** 2)
        return reg_loss

    def backward(self):
        # L2正则化的梯度：lambda * W
        for layer in self.model.layers:
            if hasattr(layer, 'W') and layer.grads['W'] is not None:
                layer.grads['W'] += self.lambda_l2 * layer.W
        return None
       

def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition