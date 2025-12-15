# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 定义两层神经网络模型
class TwoLayerNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=50, output_size=10):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x

# 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

# 预测单张图片
def predict_single_image(model, image_path=None, image_tensor=None):
    model.eval()
    if image_path:
        # 使用 OpenCV 读取和预处理图片
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        # 调整大小到 28x28
        image = cv2.resize(image, (28, 28))
        # 反转颜色（黑底白字）
        image = 255 - image
        # 归一化到 [0,1]
        image = image.astype(np.float32) / 255.0
        # 进一步归一化到 [-1,1]（匹配训练时的 Normalize）
        image = (image - 0.5) / 0.5
        # 转换为张量，形状 (1, 1, 28, 28)
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    elif image_tensor is None:
        raise ValueError("必须提供 image_path 或 image_tensor")

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        probabilities = torch.softmax(output, dim=1)
    return predicted.item(), probabilities.squeeze().numpy()

def traine_model():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载MNIST数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化模型、损失函数、优化器
    model = TwoLayerNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, epochs=20)

    # 保存模型
    torch.save(model.state_dict(), 'two_layer_net.pth')
    print("模型已保存为 two_layer_net.pth")

def inference_model(image_path):
    # 初始化模型
    model = TwoLayerNet()
    model.load_state_dict(torch.load('two_layer_net.pth'))
    model.eval()  # 设置为评估模式

    # 预测
    predicted_class, probs = predict_single_image(model, image_path=image_path)
    print(f"预测标签: {predicted_class}")
    print(f"预测概率: {probs}")

    # 可视化
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted_class}")
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 训练
    # traine_model()

    # 推理
    inference_model('lesson_2/5.jpeg')    