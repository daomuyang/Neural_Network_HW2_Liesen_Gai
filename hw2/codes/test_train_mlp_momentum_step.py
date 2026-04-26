import mynn as nn
from draw_tools.plot import plot
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import os

np.random.seed(309)

train_images_path = './dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = './dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28) / 255.0

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# 模型
linear_model = nn.models.Model_MLP([784, 600, 10], 'ReLU', None)

optimizer = nn.optimizer.Momentum(init_lr=0.1, model=linear_model, mu=0.9)

loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model)
scheduler = nn.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.8)
runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

# 训练
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], 
             num_epochs=40, log_iters=50, save_dir='./saved_models',save_name='mlp_momentum_step.pickle')

# 绘图
_, axes = plt.subplots(1, 2, figsize=(12, 5))
plot(runner, axes)
os.makedirs('./figs', exist_ok=True)
plt.savefig('./figs/mlp_momentum_step.png', dpi=300, bbox_inches='tight')
plt.show()