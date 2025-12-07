import torch
import torch.linalg as LA

# 1. 准备数据（与之前相同）
x_data = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
y_data = torch.tensor([[3.0], [5.0], [7.0]], dtype=torch.float32)

# 2. 构建设计矩阵 X
# 对于 y = kx + b，我们需要在x前面加一列1来表示偏置项b
ones = torch.ones_like(x_data)
X = torch.cat([x_data, ones], dim=1)  # 形状: (3, 2)

print("设计矩阵 X:")
print(X)

# 3. 使用最小二乘法求解正规方程
# 正规方程: (X^T * X) * θ = X^T * y，其中 θ = [k, b]^T
# 解为: θ = (X^T * X)^(-1) * X^T * y

# 方法1: 手动计算
X_T = X.t()  # 转置
X_T_X = X_T @ X  # X^T * X
X_T_y = X_T @ y_data  # X^T * y

# 解方程: (X^T * X) * θ = X^T * y
theta = torch.linalg.solve(X_T_X, X_T_y)

# 方法2: 使用PyTorch的lstsq函数（更稳定）
theta_lstsq, _, _, _ = torch.linalg.lstsq(X, y_data)

print("\n最小二乘法结果:")
print(f"手动求解: k = {theta[0].item():.6f}, b = {theta[1].item():.6f}")
print(f"lstsq求解: k = {theta_lstsq[0].item():.6f}, b = {theta_lstsq[1].item():.6f}")

# 4. 验证结果
k, b = theta[0].item(), theta[1].item()
print(f"\n验证拟合直线: y = {k:.1f}x + {b:.1f}")

# 计算预测值
y_pred = k * x_data + b
print("\n预测结果对比:")
for i in range(len(x_data)):
    print(f"x={x_data[i].item():.0f}, 真实y={y_data[i].item():.0f}, 预测y={y_pred[i].item():.1f}")

# 5. 计算损失（应该为0，因为完美拟合）
loss = torch.mean((y_data - y_pred) ** 2)
print(f"\n最终损失: {loss.item():.8f}")