import numpy as np
import matplotlib.pyplot as plt


# 配置
NUM_CLASSES = 4       # 类别数
N_PER_CLASS = 20      # 每类样本数
K = 5                 # KNN k值
N_TEST = 6            # 测试点数量
COV = 0.8             # 生成点的方差

np.random.seed(0)


# 随机生成 NUM_CLASSES 簇数据
means = [(-5 + i*3, -5 + i*3) for i in range(NUM_CLASSES)]
X = np.vstack([np.random.randn(N_PER_CLASS, 2) * COV + m for m in means])
y = np.hstack([[i] * N_PER_CLASS for i in range(NUM_CLASSES)])

# 打乱
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# 切割训练集和测试集
test_indices = np.random.choice(len(X), N_TEST, replace=False)
X_test = X[test_indices]
y_test = y[test_indices]

train_mask = np.ones(len(X), dtype=bool)
train_mask[test_indices] = False
X_train = X[train_mask]
y_train = y[train_mask]

print(f"Generated dataset: {len(X)} points, class counts:", {i: int(sum(y==i)) for i in np.unique(y)})

# KNN 类
class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict_one(self, x):
        distances = np.linalg.norm(self.X - x, axis=1)
        k_idx = np.argsort(distances)[:self.k]
        k_labels = self.y[k_idx]
        counts = np.bincount(k_labels)
        pred = counts.argmax()
        return pred, k_idx, distances[k_idx]


# 显示图像
knn = KNN(k=K)
knn.fit(X_train, y_train)

plt.ion()
fig, ax = plt.subplots(figsize=(6,6))

for i, (xt, yt) in enumerate(zip(X_test, y_test)):
    ax.clear()
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='tab10', edgecolor='k', s=40)
    ax.scatter(xt[0], xt[1], c='none', edgecolor='r', s=200, linewidth=2, marker='o', label='test point')
    pred, k_idx, dists = knn.predict_one(xt)
    ax.scatter(X_train[k_idx,0], X_train[k_idx,1], facecolors='none', edgecolor='r', s=150, linewidth=2, marker='s', label='neighbors')
    ax.set_title(f"KNN test {i+1}/{len(X_test)} | true={int(yt)} -> pred={int(pred)} (k={K})")
    ax.legend()
    plt.draw()

    print(f"\nKNN Test point {i+1}/{len(X_test)}: coords={xt}, true={int(yt)}, predicted={int(pred)}")
    print("Nearest neighbors:")
    for idx_local, (idx_global, dist) in enumerate(zip(k_idx, dists)):
        print(f" neighbor {idx_local+1}: train_idx={int(idx_global)}, coord={X_train[int(idx_global)]}, dist={dist:.4f}, label={int(y_train[int(idx_global)])}")

    print("点击图窗继续到下一个测试点...")
    try:
        plt.ginput(1)
    except Exception:
        pass

plt.close(fig)
plt.ioff()