# 神经网络HW2盖烈森

## 环境配置
```bash
# 安装依赖
pip install numpy==1.26.0 matplotlib==3.8.0 seaborn==0.13.0
```

## 数据集准备
本实验使用MNIST数据集，**未随代码上传**，如需运行请放置于 `codes/dataset/MNIST/` 目录下，确保包含以下文件：
```
codes/dataset/MNIST/
├── t10k-images-idx3-ubyte.gz
├── t10k-labels-idx1-ubyte.gz
├── train-images-idx3-ubyte.gz
└── train-labels-idx1-ubyte.gz
```

## 运行实验
所有实验脚本均位于 `codes/` 目录下，直接运行即可：

### MLP 实验
```bash
cd codes

# 1. MLP基线
python test_train_mlp.py

# 2. MLP + Momentum
python test_train_mlp_momentum.py

# 3. MLP + StepLR
python test_train_mlp_step.py

# 4. MLP + ExponentialLR
python test_train_mlp_exp.py

# 5. MLP + Momentum + StepLR
python test_train_mlp_momentum_step.py

# 6. MLP + 早停
python test_train_mlp_earlystop.py

# 7. MLP + L2正则化
python test_train_mlp_l2_1e-4.py
```

### CNN 实验
```bash
cd codes

# 8. CNN基线
python test_train_cnn_baseline.py

# 9. CNN + Momentum
python test_train_cnn_momentum.py

# 10. CNN + L2正则化
python test_train_cnn_l2_1e-4.py
```

## 测试模型
```bash
cd codes
python test_model.py
```
该代码中采用交互式设计，可以自行选择想要测试效果的实验

## 生成可视化
### 1. 权重热力图
```bash
cd codes
python weight_visualization.py
```
该代码中采用交互式设计，可以自行选择想要生成图片的实验
生成的图片将保存至 `codes/figs/` 目录。

### 2. 混淆矩阵
```bash
cd codes
python confusion_matrix.py
```
该代码中采用交互式设计，可以自行选择想要生成图片的实验
生成的图片将保存至 `codes/figs/` 目录。

## 模型
本实验训练好的模型**未随代码上传**，请从ModelScope下载（链接详见实验报告），并放置于 `codes/saved_models/` 目录下；最优模型放置于 `codes/best_models/` 目录下
