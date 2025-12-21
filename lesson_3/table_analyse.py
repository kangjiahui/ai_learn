import cv2
import numpy as np

def show(title, img, cmap=None):
    cv2.imshow(title, img)
    cv2.waitKey(0)

def detect_table_cells_debug(image_path, show_steps=True):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if show_steps:
        show("1. Original Image", img)
        show("2. Grayscale", gray, cmap="gray")

    # 二值化
    binary = cv2.adaptiveThreshold(
        ~gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)

    if show_steps:
        show("3. Binary Image", binary, cmap="gray")

    h, w = binary.shape

    # 横线检测
    horizontal = binary.copy()
    h_kernel_len = max(1, w // 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    horizontal = cv2.erode(horizontal, h_kernel)
    horizontal = cv2.dilate(horizontal, h_kernel)

    if show_steps:
        show("4. Horizontal Lines", horizontal, cmap="gray")

    # 竖线检测
    vertical = binary.copy()
    v_kernel_len = max(1, h // 30)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    vertical = cv2.erode(vertical, v_kernel)
    vertical = cv2.dilate(vertical, v_kernel)

    if show_steps:
        show("5. Vertical Lines", vertical, cmap="gray")

    # 合并横线和竖线
    table_mask = cv2.add(horizontal, vertical)

    if show_steps:
        show("6. Table Grid Mask", table_mask, cmap="gray")

    # 轮廓提取
    contours, _ = cv2.findContours(
        table_mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cells = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        if cw < 20 or ch < 20:
            continue
        if cw > 0.9 * w and ch > 0.9 * h:
            continue

        cells.append((x, y, cw, ch))

    cells = sorted(cells, key=lambda b: (b[1], b[0]))

    # 可视化最终结果
    vis = img.copy()
    for (x, y, cw, ch) in cells:
        cv2.rectangle(vis, (x, y), (x + cw, y + ch), (0, 255, 0), 2)

    if show_steps:
        show("7. Detected Cells", vis)

    return cells


if __name__ == "__main__":
    image_path = "lesson_3/line_table.PNG"

cells = detect_table_cells_debug(image_path, show_steps=True)

print("Detected cells:")
for c in cells:
    print(c)