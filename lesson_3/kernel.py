import cv2
import numpy as np

# 读取灰度图
img = cv2.imread("lesson_3/clock.PNG", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found!")

# 统一尺寸，便于拼接显示
img = cv2.resize(img, (256, 256))

# 均值滤波
kernel_mean = np.ones((3, 3), dtype=np.float32) / 9

# Sobel X（竖直边缘）
kernel_sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Sobel Y（水平边缘）
kernel_sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# 随机卷积核（模拟“未训练的 CNN 卷积核”）
kernel_random = np.random.randn(3, 3).astype(np.float32)

# 卷积运算
blur = cv2.filter2D(img, -1, kernel_mean)
sobel_x = cv2.filter2D(img, cv2.CV_32F, kernel_sobel_x)
sobel_y = cv2.filter2D(img, cv2.CV_32F, kernel_sobel_y)

# 边缘强度
edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)

# 随机核输出
random_out = cv2.filter2D(img, cv2.CV_32F, kernel_random)

# 归一化到 0~255
def normalize(img):
    img = np.abs(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

sobel_x = normalize(sobel_x)
sobel_y = normalize(sobel_y)
edge_mag = normalize(edge_mag)
random_out = normalize(random_out)

# 合并显示
row1 = np.hstack([img, blur, sobel_x])
row2 = np.hstack([sobel_y, edge_mag, random_out])
canvas = np.vstack([row1, row2])

# 添加标题文字
labels = [
    ("Original", (10, 20)),
    ("Mean Blur", (266, 20)),
    ("Sobel X", (522, 20)),
    ("Sobel Y", (10, 276)),
    ("Edge Mag", (266, 276)),
    ("Random Kernel", (522, 276)),
]

for text, (x, y) in labels:
    cv2.putText(
        canvas, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 1, cv2.LINE_AA
    )

# 显示结果
cv2.imshow("Traditional Image Filters Demo", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()