import matplotlib.pyplot as plt
import numpy as np

# 指数平滑：只给验证集用
def smooth(points, factor=0.8):
    if len(points) == 0:
        return []
    smoothed = []
    last = points[0]
    for p in points:
        smoothed_val = last * factor + (1 - factor) * p
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot(runner, axes):
    # 颜色
    train_color = "#E3A937"   # 黄色
    val_color   = "#8B6B3C"  # 棕色

    # ======================== 损失图 ========================
    # 1. 原始训练曲线（细线）
    axes[0].plot(runner.train_loss, color=train_color, linewidth=1, label="Train Loss")

    # 2. 验证集趋势线（棕色粗虚线！！！你要的）
    val_loss_x = np.linspace(0, len(runner.train_loss), len(runner.dev_loss))
    axes[0].plot(
        val_loss_x, smooth(runner.dev_loss),
        color=val_color, linestyle="--", linewidth=3, label="Val Loss (Trend)"
    )

    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ======================== 准确率图 ========================
    # 1. 原始训练曲线（细线）
    axes[1].plot(runner.train_scores, color=train_color, linewidth=1, label="Train Acc")

    # 2. 验证集趋势线（棕色粗虚线！！！你要的）
    val_acc_x = np.linspace(0, len(runner.train_scores), len(runner.dev_scores))
    axes[1].plot(
        val_acc_x, smooth(runner.dev_scores),
        color=val_color, linestyle="--", linewidth=3, label="Val Acc (Trend)"
    )

    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()