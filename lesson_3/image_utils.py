import cv2
import numpy as np


def edge_detect(gray, method='canny', ksize=3, low_threshold=50, high_threshold=150):
    """
    通用边缘检测函数
    输入:
      gray: 灰度图 (uint8)
      method: 'canny','sobel','laplacian'
    返回:
      单通道 uint8 图像, 范围 0-255
    """
    if method == 'canny':
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return edges

    if method == 'sobel':
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        mag = np.sqrt(gx * gx + gy * gy)
        if np.max(mag) > 0:
            mag = (mag / np.max(mag) * 255).astype(np.uint8)
        else:
            mag = np.zeros_like(gray)
        return mag

    if method == 'laplacian':
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        edges = np.uint8(np.absolute(lap))
        return edges

    raise ValueError('Unknown method: ' + str(method))
