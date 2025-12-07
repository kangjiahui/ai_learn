import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据
# 输入特征 (房屋面积)
x_data = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
# 真实标签 (房价)
y_data = torch.tensor([[3.0], [5.0], [7.0]], dtype=torch.float32)

# 2. 定义线性模型 y = k*x + b
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度1，输出维度1 (自动包含权重k和偏置b)
    
    def forward(self, x):
        return self.linear(x)

# 3. 初始化模型
model = LinearModel()

# 4. 查看初始参数 (k和b)
print("初始参数:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.item()}")

# 5. 定义损失函数 (均方误差)
criterion = nn.MSELoss()

# 6. 定义优化器 (梯度下降)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 学习率0.01

# 7. 训练循环
num_epochs = 50  # 训练50轮
print("\n开始训练...")
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x_data)
    
    # 计算损失
    loss = criterion(y_pred, y_data)
    
    # 反向传播
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度 (PyTorch自动完成反向传播!)
    
    # 更新参数
    optimizer.step()       # 根据梯度更新k和b
    
    # 每10轮打印一次进度
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        # 打印当前参数
        params = list(model.parameters())
        k = params[0].data[0][0].item()
        b = params[1].data[0].item()
        print(f'    k = {k:.4f}, b = {b:.4f}')

# 8. 训练完成后查看最终参数
print("\n训练完成!")
final_params = list(model.parameters())
final_k = final_params[0].data[0][0].item()
final_b = final_params[1].data[0].item()
print(f"最终参数: k = {final_k:.4f}, b = {final_b:.4f}")

# 9. 测试模型
test_x = torch.tensor([[4.0]], dtype=torch.float32)
predicted_y = model(test_x)
print(f"\n预测结果: 当x=4时, y={predicted_y.item():.2f} (真实值应为9)")