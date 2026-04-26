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
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28) / 255.0

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 标准划分
valid_imgs = train_imgs[50000:]
valid_labs = train_labs[50000:]
train_imgs = train_imgs[:50000]
train_labs = train_labs[:50000]

# ✅ 创建模型
model = nn.models.Model_CNN()

# ✅ 👇 关键：手动给所有可优化层加 L2，不改 models.py！
for layer in model.layers:
    if layer.optimizable:
        layer.weight_decay = True
        layer.weight_decay_lambda = 1e-4

# ✅ 其他参数和基线完全一致
optimizer = nn.optimizer.SGD(init_lr=0.1, model=model)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model)

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=128)

runner.train(
    [train_imgs, train_labs],
    [valid_imgs, valid_labs],
    num_epochs=40,
    log_iters=50,
    save_dir='./saved_models',
    save_name='cnn_l2_1e-4.pickle'
)

_, axes = plt.subplots(1, 2, figsize=(12, 5))
plot(runner, axes)
os.makedirs('./figs', exist_ok=True)
plt.savefig('./figs/cnn_l2_1e-4.png', dpi=300, bbox_inches='tight')
plt.show()