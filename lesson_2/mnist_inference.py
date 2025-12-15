# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist
from common.functions import softmax
import cv2


network = TwoLayerNet(input_size=784, hidden_size=30, output_size=10)
network.load_params("lesson_2/params.pkl")

def _load_image_as_mnist(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError('Failed to read image with OpenCV: {}'.format(path))
    # resize to 28x28 using INTER_AREA for downsampling
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    # invert if background is light
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    arr = arr.reshape(1, 1, 28, 28)
    return arr


def predict_single(network, image, show=False):
    """Predict a single image using the provided TwoLayerNet instance.

    Args:
        network: TwoLayerNet 已加载好参数的实例
        image: 文件路径 (str) 或 numpy 数组 (H,W) 或 (1,H,W) 或 (1,1,H,W)
        show: 如果为 True，则用 matplotlib 显示预处理后的 28x28 图像并在标题中显示预测

    Returns:
        pred: 整数，预测的标签(0-9)
        probs: 1D numpy array, 各类的概率分布（长度10）
    """
    # 准备输入 x，形状 (1,1,28,28)
    if isinstance(image, str):
        x = _load_image_as_mnist(image)
    else:
        x = np.array(image, dtype=np.float32)
        # possible shapes: (28,28), (1,28,28), (1,1,28,28)
        if x.ndim == 2:
            x = x.reshape(1, 1, 28, 28)
        elif x.ndim == 3:
            if x.shape[0] == 1 and x.shape[1] == 28 and x.shape[2] == 28:
                x = x.reshape(1, 1, 28, 28)
            else:
                # maybe (28,28,1)
                x = x.transpose(2, 0, 1) if x.shape[2] == 1 else x
                if x.ndim == 3 and x.shape[0] == 1:
                    x = x.reshape(1, 1, 28, 28)
                else:
                    raise ValueError('Unsupported image array shape: {}'.format(image.shape))
        elif x.ndim == 4:
            # assume already (1,1,28,28) or (N,C,H,W)
            if x.shape[0] != 1:
                # take first sample
                x = x[0:1]

        # normalize if needed
        if x.max() > 1.0:
            x = x / 255.0
        # heuristic invert if background is light
        if x.mean() > 0.5:
            x = 1.0 - x

    # forward
    score = network.predict(x)  # shape (1,10)
    probs = softmax(score).flatten()
    pred = int(np.argmax(probs))

    if show:
        plt.figure()
        plt.imshow(x.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('pred: {} (p={:.3f})'.format(pred, probs[pred]))
        plt.axis('off')
        plt.show()

    return pred, probs


if __name__ == '__main__':
    image_path = "lesson_2/5.jpeg"
    pred, probs = predict_single(network, image_path, show=True)
    print('Prediction:', pred)
