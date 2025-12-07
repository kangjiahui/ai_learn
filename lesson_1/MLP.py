"""
interactive_mlp.py

交互式多层感知机演示：
- 输入层2个神经元
- 隐藏层3个神经元，ReLU激活
- 输出层2个神经元
- 梯度下降训练
- 随机生成50个训练数据
- 每轮迭代可视化预测 vs 真值，鼠标点击继续
"""
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 配置
# -----------------------------
N_SAMPLES = 50
INPUT_DIM = 2
HIDDEN_DIM = 3
OUTPUT_DIM = 2
LR = 0.1
EPOCHS = 20
np.random.seed(0)

# # -----------------------------
# # 生成随机训练数据
# # -----------------------------
# X = np.random.uniform(-5,5,(N_SAMPLES,INPUT_DIM))
# true_W = np.array([[1.5, -2.0],[0.5, 1.0]])
# true_b = np.array([0.5,-1.0])
# y = X @ true_W.T + true_b + np.random.randn(N_SAMPLES,OUTPUT_DIM)*0.5

# -----------------------------
# 生成非线性训练数据（二元二次多项式）
# -----------------------------
X = np.random.uniform(-3,3,(N_SAMPLES,2))
x1, x2 = X[:,0], X[:,1]

y1 = 1.0*x1**2 -2.0*x1*x2 + 0.5*x2**2 + 0.5 + np.random.randn(N_SAMPLES)*0.2
y2 = -1.5*x1**2 + 0.5*x1*x2 + 2.0*x2**2 -1.0 + np.random.randn(N_SAMPLES)*0.2
y = np.vstack([y1, y2]).T  # shape (N_SAMPLES,2)

# -----------------------------
# 初始化权重
# -----------------------------
W1 = np.random.randn(HIDDEN_DIM, INPUT_DIM)
b1 = np.zeros(HIDDEN_DIM)
W2 = np.random.randn(OUTPUT_DIM, HIDDEN_DIM)
b2 = np.zeros(OUTPUT_DIM)

# -----------------------------
# 激活函数
# -----------------------------
def relu(x):
    return np.maximum(0,x)

def relu_grad(x):
    return (x>0).astype(float)

# -----------------------------
# 训练与可视化
# -----------------------------
plt.ion()
for epoch in range(1,EPOCHS+1):
    # 前向传播
    Z1 = X @ W1.T + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2.T + b2
    y_pred = Z2

    # 损失
    loss = np.mean((y_pred - y)**2)

    # 反向传播
    dZ2 = 2*(y_pred - y)/N_SAMPLES
    dW2 = dZ2.T @ A1
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ W2
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = dZ1.T @ X
    db1 = np.sum(dZ1, axis=0)

    # 参数更新
    W2 -= LR * dW2
    b2 -= LR * db2
    W1 -= LR * dW1
    b1 -= LR * db1

    # 打印信息
    print(f"Epoch {epoch}: loss={loss:.4f}")

    # 可视化
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    for i in range(OUTPUT_DIM):
        axs[i].scatter(range(N_SAMPLES), y[:,i], label='y_true')
        axs[i].scatter(range(N_SAMPLES), y_pred[:,i], label='y_pred')
        axs[i].set_title(f'Output neuron {i+1}')
        axs[i].legend()
    plt.suptitle(f"Epoch {epoch} | loss={loss:.4f}")
    plt.draw()
    print("点击图窗继续下一轮迭代...")
    try:
        plt.ginput(1)
    except Exception:
        pass
    plt.close()

plt.ioff()
print("训练完成，预测示例:")
for i in range(5):
    print(f"X={X[i]} -> y_true={y[i]} -> y_pred={y_pred[i]}")