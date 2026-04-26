import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===================== 所有实验配置 =====================
EXPERIMENTS = {
    1: {"name": "MLP_baseline", "model_type": "mlp", "path": "./saved_models/mlp_baseline.pickle"},
    2: {"name": "SGD+Momentum", "model_type": "mlp", "path": "./saved_models/mlp_momentum.pickle"},
    3: {"name": "SGD+StepLR", "model_type": "mlp", "path": "./saved_models/mlp_step.pickle"},
    4: {"name": "SGD+ExponentialLR", "model_type": "mlp", "path": "./saved_models/mlp_exp.pickle"},
    5: {"name": "Momentum+StepLR", "model_type": "mlp", "path": "./saved_models/mlp_momentum_step.pickle"},
    6: {"name": "MLP_earlystop(patience=15)", "model_type": "mlp", "path": "./saved_models/mlp_earlystop.pickle"},
    7: {"name": "MLP_l2_1e-4", "model_type": "mlp", "path": "./saved_models/mlp_l2_1e-4.pickle"},
    8: {"name": "CNN_baseline", "model_type": "cnn", "path": "./saved_models/cnn_baseline.pickle"},
    9: {"name": "CNN_momentum", "model_type": "cnn", "path": "./saved_models/cnn_momentum.pickle"},
    10: {"name": "CNN_l2_1e-4", "model_type": "cnn", "path": "./saved_models/cnn_l2_1e-4.pickle"}
}

# ===================== 打印实验目录 =====================
print("="*50)
print("📋 实验列表（输入序号即可生成混淆矩阵）")
print("="*50)
for idx, exp in EXPERIMENTS.items():
    print(f"{idx:2d}. {exp['name']}")
print("="*50)

# ===================== 输入实验序号 =====================
while True:
    try:
        exp_id = int(input("\n请输入要生成混淆矩阵的实验序号（输入0退出）："))
        if exp_id == 0:
            print("👋 退出")
            break
        if exp_id not in EXPERIMENTS:
            print("❌ 序号不存在，请重新输入")
            continue
        
        exp = EXPERIMENTS[exp_id]
        print(f"\n🚀 正在生成：{exp['name']} 的混淆矩阵")
        
        # ===================== 自动加载对应模型 =====================
        if exp["model_type"] == "mlp":
            model = nn.models.Model_MLP(
                size_list=[784, 600, 10],
                act_func="ReLU",
                lambda_list=None
            )
        else:
            model = nn.models.Model_CNN()
        
        model.load_model(exp["path"])
        
        # ===================== 加载测试集 =====================
        with gzip.open("./dataset/MNIST/t10k-images-idx3-ubyte.gz", "rb") as f:
            magic, n, r, c = unpack(">4I", f.read(16))
            imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, -1) / 255.0
        
        # 自动调整输入形状
        if exp["model_type"] == "cnn":
            imgs = imgs.reshape(-1, 1, 28, 28)
        
        with gzip.open("./dataset/MNIST/t10k-labels-idx1-ubyte.gz", "rb") as f:
            magic, n = unpack(">2I", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        # ===================== 预测并生成混淆矩阵 =====================
        logits = model(imgs)
        preds = np.argmax(logits, axis=-1)
        
        # 计算混淆矩阵
        cm = np.zeros((10, 10), dtype=int)
        for true, pred in zip(labels, preds):
            cm[true][pred] += 1
        
        # ===================== 绘制混淆矩阵 =====================
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[str(i) for i in range(10)],
                    yticklabels=[str(i) for i in range(10)])
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'{exp["name"]} - Confusion Matrix', fontsize=14)
        
        # ===================== 保存图片 =====================
        os.makedirs('./figs', exist_ok=True)
        save_name = f"./figs/{exp_id}_{exp['name'].replace('(', '').replace(')', '').replace('=', '')}_confusion.png"
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"✅ 混淆矩阵已保存到：{save_name}")
        plt.close()
        
    except ValueError:
        print("❌ 请输入数字序号")
    except ImportError:
        print("❌ 缺少seaborn库，请运行：pip install seaborn")
    except Exception as e:
        print(f"❌ 生成失败：{e}")