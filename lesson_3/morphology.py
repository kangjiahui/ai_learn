import cv2
import numpy as np
from image_utils import edge_detect


# 读取灰度图
img = cv2.imread("lesson_3/clock.PNG", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found!")

# 先做边缘检测（可改 method 为 'canny','sobel',...）
edges = edge_detect(img, method='canny', low_threshold=50, high_threshold=150)

_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 定义形态学运算的核大小
kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 开运算：先腐蚀再膨胀
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_morph)

# 闭运算：先膨胀再腐蚀
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_morph)

# 合并显示：原图 | 边缘 | 开运算 | 闭运算
row1 = np.hstack([img, binary, opening, closing])
canvas = np.vstack([row1])

# 为了方便放上白色文字，转为 BGR
canvas_color = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

# 添加标题文字
labels = [
    ("Original", (10, 20)),
    ("Binary", (266, 20)),
    ("Opening", (522, 20)),
    ("Closing", (778, 20)),
]

for text, (x, y) in labels:
    cv2.putText(canvas_color, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

# 显示结果
cv2.imshow("Morphology with Edge Preprocessing", canvas_color)
cv2.waitKey(0)
cv2.destroyAllWindows()