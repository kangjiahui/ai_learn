import numpy as np
import matplotlib.pyplot as plt


# 自动生成二维样本配置
N_SAMPLES = 50        # 样本数量
TRUE_W = 2.5          # 真实斜率
TRUE_B = -1.0         # 真实截距
NOISE_STD = 1.0       # 噪声标准差

# 梯度下降算法超参数配置
LR = 0.1             # 学习率
EPOCHS = 20           # 迭代轮数

np.random.seed(0)

# 生成训练样本
X = np.random.uniform(-5, 5, size=N_SAMPLES)
y = TRUE_W * X + TRUE_B + np.random.randn(N_SAMPLES) * NOISE_STD

# 可视化函数
def plot_line(X, y, w, b, epoch=None, loss=None):
    plt.figure(figsize=(6,6))
    plt.scatter(X, y, label='Data')
    x_line = np.linspace(X.min()-1, X.max()+1, 200)
    y_line = w * x_line + b
    plt.plot(x_line, y_line, 'r', label='Fitted line')
    title = f"Linear Regression | w={w:.3f}, b={b:.3f}"
    if epoch is not None:
        title += f" | Epoch {epoch}"
    if loss is not None:
        title += f" | Loss={loss:.2f}"
    plt.title(title)
    plt.legend()
    plt.draw()
    print("点击图窗继续...")
    try:
        plt.ginput(1)
    except Exception:
        pass
    plt.close()

# 方法 1: 最小二乘法解析解
def OLS():
    print("使用最小二乘法（解二元一次方程）求解参数...")    
    Sx  = np.sum(X)
    Sy  = np.sum(y)
    Sxx = np.sum(X * X)
    Sxy = np.sum(X * y)
    # 解二元一次方程组：
    # w*Sx + b*N = Sy
    # w*Sxx + b*Sx = Sxy
    w = (N_SAMPLES * Sxy - Sx * Sy) / (Sxx * N_SAMPLES - Sx * Sx)
    b = (Sy - w * Sx) / N_SAMPLES

    print(f"解析解结果: w={w:.4f}, b={b:.4f}")
    plot_line(X, y, w, b, epoch=0, loss=np.mean((y - (w*X + b))**2))

# 方法 2: 梯度下降
def GD():
    print("使用梯度下降求解参数...")
    w = np.random.randn()
    b = np.random.randn()
    for epoch in range(1, EPOCHS+1):
        y_pred = w*X + b
        loss = np.mean((y_pred - y)**2)
        dw = 2 * np.mean((y_pred - y) * X)
        db = 2 * np.mean(y_pred - y)
        w -= LR * dw
        b -= LR * db
        print(f"Epoch {epoch}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")
        plot_line(X, y, w, b, epoch=epoch, loss=loss)

#main 函数
if __name__ == "__main__":
    # OLS()
    GD()
