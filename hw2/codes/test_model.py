import mynn as nn
import numpy as np
from struct import unpack
import gzip

# ===================== 所有实验配置（和你的表格完全一致）=====================
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
print("📋 实验列表（输入序号即可测试）")
print("="*50)
for idx, exp in EXPERIMENTS.items():
    print(f"{idx:2d}. {exp['name']}")
print("="*50)

# ===================== 输入实验序号 =====================
while True:
    try:
        exp_id = int(input("\n请输入要测试的实验序号（输入0退出）："))
        if exp_id == 0:
            print("👋 退出测试")
            break
        if exp_id not in EXPERIMENTS:
            print("❌ 序号不存在，请重新输入")
            continue
        
        exp = EXPERIMENTS[exp_id]
        print(f"\n🚀 正在测试：{exp['name']}")
        
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
        
        # ===================== 测试并输出结果 =====================
        logits = model(imgs)
        acc = nn.metric.accuracy(logits, labels)
        print(f"✅ {exp['name']} 测试集准确率: {acc:.4f}")
        
    except ValueError:
        print("❌ 请输入数字序号")
    except Exception as e:
        print(f"❌ 测试失败：{e}")