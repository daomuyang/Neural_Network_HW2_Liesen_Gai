import mynn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== 所有实验配置 =====================
EXPERIMENTS = {
    1: {"name": "MLP_baseline", "model_type": "mlp", "path": "./saved_models/mlp_baseline.pickle"},
    2: {"name": "SGD+Momentum", "model_type": "mlp", "path": "./saved_models/mlp_momentum.pickle"},
    3: {"name": "SGD+StepLR", "model_type": "mlp", "path": "./saved_models/mlp_step.pickle"},
    4: {"name": "SGD+ExponentialLR", "model_type": "mlp", "path": "./saved_models/mlp_exp.pickle"},
    5: {"name": "Momentum+StepLR", "model_type": "mlp", "path": "./saved_models/mlp_momentum_step.pickle"},
    6: {"name": "MLP_earlystop(patience=5)", "model_type": "mlp", "path": "./saved_models/mlp_earlystop.pickle"},
    7: {"name": "MLP_l2_1e-4", "model_type": "mlp", "path": "./saved_models/mlp_l2_1e-4.pickle"},
    8: {"name": "CNN_baseline", "model_type": "cnn", "path": "./saved_models/cnn_baseline.pickle"},
    9: {"name": "CNN_momentum", "model_type": "cnn", "path": "./saved_models/cnn_momentum.pickle"},
    10: {"name": "CNN_l2_1e-4", "model_type": "cnn", "path": "./saved_models/cnn_l2_1e-4.pickle"}
}

# ===================== 打印实验目录 =====================
print("="*50)
print("📋 实验列表（输入序号即可生成权重图）")
print("="*50)
for idx, exp in EXPERIMENTS.items():
    print(f"{idx:2d}. {exp['name']}")
print("="*50)

# ===================== 输入实验序号 =====================
while True:
    try:
        exp_id = int(input("\n请输入要生成权重图的实验序号（输入0退出）："))
        if exp_id == 0:
            print("👋 退出可视化")
            break
        if exp_id not in EXPERIMENTS:
            print("❌ 序号不存在，请重新输入")
            continue
        
        exp = EXPERIMENTS[exp_id]
        print(f"\n🚀 正在生成：{exp['name']} 的权重图")
        
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
        
        # ===================== 自动提取权重 + 自动正确绘图 =====================
        plt.figure(figsize=(10, 4))
        
        if exp["model_type"] == "cnn":
            # CNN：画第一个卷积层权重 (8,1,3,3)
            W = model.layers[0].W
            weight_map = W.reshape(-1, 3)
            plt.imshow(weight_map, cmap="viridis", aspect="auto")
            plt.title(f"{exp['name']} - Conv1 Weight Heatmap", fontsize=14)
            plt.ylabel("Conv Filters")
            plt.xlabel("3x3 Kernel Weights")
        else:
            # MLP：画第二层 → 输出层权重 (600,10)
            W = model.layers[2].W
            plt.imshow(W.T, cmap="viridis", aspect="auto")
            plt.title(f"{exp['name']} - Layer 2 Weight (Hidden → Output)", fontsize=14)
            plt.ylabel("Output Class (0-9)")
            plt.yticks(range(10))
        
        # ===================== 通用样式 =====================
        plt.colorbar(label="Weight Value")
        plt.xticks([])
        os.makedirs('./figs', exist_ok=True)
        
        # ===================== 自动保存 =====================
        save_name = f"./figs/{exp_id}_{exp['name'].replace('(', '').replace(')', '').replace('=', '')}_weight.png"
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"✅ 权重图已保存到：{save_name}")
        plt.close()
        
    except ValueError:
        print("❌ 请输入数字序号")
    except Exception as e:
        print(f"❌ 生成失败：{e}")