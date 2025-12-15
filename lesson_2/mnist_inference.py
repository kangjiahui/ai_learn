# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
from common.functions import softmax
import cv2


network = TwoLayerNet(input_size=784, hidden_size=30, output_size=10)
network.load_params("params.pkl")

def _load_image_as_mnist(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError('Failed to read image with OpenCV: {}'.format(path))
    # resize to 28x28 using INTER_AREA for downsampling
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    # 由于mnist训练时是黑底白字，因此预测时若发现是白底黑字，就需要做个反转
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    arr = arr.reshape(1, 1, 28, 28)
    return arr


def predict_single(network, image, show=False):
    x = _load_image_as_mnist(image)

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
