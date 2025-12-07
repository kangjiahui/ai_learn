"""
interactive_svm.py

交互式 One-vs-Rest 线性 SVM 演示：
- 自动生成 2D 数据
- 类别数可配置
- 每个 epoch 显示决策区域、权重方向、训练 loss 和 accuracy
- 鼠标点击继续
"""
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 配置
# -----------------------------
NUM_CLASSES = 2      # 类别数
N_PER_CLASS = 20     # 每类样本数
LR = 0.01            # 学习率
C = 1.0              # soft-margin 权重
EPOCHS = 15          # 训练轮数
COV = 0.8            # 生成点方差

np.random.seed(0)

# -----------------------------
# 生成数据
# -----------------------------
means = [(-5 + i*3, -5 + i*3) for i in range(NUM_CLASSES)]
X = np.vstack([np.random.randn(N_PER_CLASS, 2) * COV + m for m in means])
y = np.hstack([[i] * N_PER_CLASS for i in range(NUM_CLASSES)])
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

print(f"Generated dataset: {len(X)} points, class counts:", {i: int(sum(y==i)) for i in np.unique(y)})

# -----------------------------
# One-vs-Rest 线性 SVM
# -----------------------------
class OneVsRestLinearSVM:
    def __init__(self, n_classes, lr=0.01, C=1.0):
        self.n_classes = n_classes
        self.lr = lr
        self.C = C

    def fit(self, X, y, epochs=20):
        n_samples, n_features = X.shape
        self.W = np.zeros((self.n_classes, n_features))
        self.b = np.zeros(self.n_classes)
        for epoch in range(1, epochs+1):
            indices = np.random.permutation(n_samples)
            total_loss = 0.0
            for idx in indices:
                xi = X[idx]
                yi = y[idx]
                for c in range(self.n_classes):
                    y_bin = 1 if c == yi else -1
                    score = np.dot(self.W[c], xi) + self.b[c]
                    margin = y_bin * score
                    if margin >= 1:
                        dW = self.W[c]
                        db = 0.0
                        loss = 0.0
                    else:
                        dW = self.W[c] - self.C * y_bin * xi
                        db = -self.C * y_bin
                        loss = 1 - margin
                    self.W[c] -= self.lr * dW
                    self.b[c] -= self.lr * db
                    total_loss += loss
            # metrics
            preds = self.predict(X)
            acc = (preds==y).mean()
            weight_norms = np.linalg.norm(self.W, axis=1)
            print(f"Epoch {epoch}: loss={total_loss:.4f}, acc={acc*100:.2f}%, weight_norms={weight_norms.round(4)}")
            self._plot_decision_region(X, y, epoch, acc, total_loss)
        return

    def decision_function(self, X):
        return X.dot(self.W.T) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return np.argmax(scores, axis=1)

    def _plot_decision_region(self, X, y, epoch, acc, loss):
        x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
        y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
        xx, yy = np.meshgrid(np.linspace(x_min,x_max,200), np.linspace(y_min,y_max,200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid).reshape(xx.shape)
        plt.figure(figsize=(6,6))
        plt.contourf(xx, yy, Z, alpha=0.25)
        for cls in range(self.n_classes):
            plt.scatter(X[y==cls,0], X[y==cls,1], label=f"Class {cls}", edgecolor='k')
            # 画权重箭头
            w = self.W[cls]
            if np.linalg.norm(w)>1e-6:
                center = X[y==cls].mean(axis=0)
                plt.arrow(center[0], center[1], w[0], w[1], head_width=0.2, length_includes_head=True)
        plt.title(f"SVM One-vs-Rest (Epoch {epoch}) | acc={acc*100:.2f}% | loss={loss:.3f}")
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        plt.legend()
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        print("点击图窗继续到下一轮训练...")
        try:
            plt.ginput(1)
        except Exception:
            pass
        plt.close()

# -----------------------------
# 训练
# -----------------------------
svm = OneVsRestLinearSVM(NUM_CLASSES, LR, C)
svm.fit(X, y, epochs=EPOCHS)

# 最终决策面展示
plt.figure(figsize=(6,6))
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min,x_max,200), np.linspace(y_min,y_max,200))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.25)
for cls in range(NUM_CLASSES):
    plt.scatter(X[y==cls,0], X[y==cls,1], label=f"Class {cls}", edgecolor='k')
plt.title("Final SVM decision regions")
plt.legend()
plt.show()