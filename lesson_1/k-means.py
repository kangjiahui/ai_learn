import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 生成示例数据
# --------------------------
np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 2) + np.array([0, 0]),
    np.random.randn(50, 2) + np.array([5, 5]),
    np.random.randn(50, 2) + np.array([0, 5])
])

K = 3
max_iter = 10

# 初始化质心
indices = np.random.choice(len(X), K, replace=False)
centroids = X[indices]

history_centroids = [centroids.copy()]
history_labels = []

def assign_labels(X, centroids):
    distances = np.linalg.norm(X[:, None] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, K):
    return np.array([X[labels == i].mean(axis=0) for i in range(K)])

# 运行 K-means，记录每一步
for _ in range(max_iter):
    labels = assign_labels(X, centroids)
    history_labels.append(labels.copy())
    new_centroids = update_centroids(X, labels, K)
    history_centroids.append(new_centroids.copy())
    if np.allclose(new_centroids, centroids):
        break
    centroids = new_centroids

# --------------------------
# 手动逐步显示图像：点击鼠标继续
# --------------------------
plt.ion()  # 打开交互模式
fig, ax = plt.subplots(figsize=(6, 6))

for i in range(len(history_labels)):
    ax.clear()
    labels = history_labels[i]
    centroids = history_centroids[i]

    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3)

    ax.set_title(f"K-means Iteration {i}")
    ax.set_xlim(-4, 10)
    ax.set_ylim(-4, 10)

    plt.draw()

    print(f"等待点击继续：Iteration {i}")
    plt.ginput(1)  # 等待 1 次鼠标点击
    print("继续...")

plt.ioff()
plt.show()