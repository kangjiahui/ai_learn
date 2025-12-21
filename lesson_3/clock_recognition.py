import cv2
import numpy as np
import math
from image_utils import edge_detect


def detect_clock_face(gray):
    # 使用 HoughCircles 检测表盘圆
    img_blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
    # 尝试多个参数以提高鲁棒性
    attempts = [
        dict(dp=1.2, minDist=100, param1=50, param2=30, minRadius=30, maxRadius=0),
        dict(dp=1.0, minDist=80, param1=50, param2=28, minRadius=20, maxRadius=0),
        dict(dp=1.5, minDist=120, param1=60, param2=40, minRadius=30, maxRadius=0),
    ]
    circles = None
    for p in attempts:
        circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, **p)
        if circles is not None:
            break
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))
    candidates = circles[0, :]
    # 选择中心最接近图像中心的圆（更稳健，防止表盘被部分遮挡）
    h, w = gray.shape[:2]
    img_cx, img_cy = w // 2, h // 2
    best = min(candidates, key=lambda c: (int(c[0]) - img_cx) ** 2 + (int(c[1]) - img_cy) ** 2)
    x, y, r = best
    return (int(x), int(y), int(r))


def lines_from_edges(edges, radius):
    """HoughLinesP 提取直线，带参数回退。输入为二值图（白色为前景）。"""
    minLen = max(10, int(radius * 0.20))
    attempts = [50, 40, 30, 20]
    lines = None
    for thr in attempts:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=thr,
                                minLineLength=minLen, maxLineGap=10)
        if lines is not None:
            break
    if lines is None:
        return []
    return [l[0] for l in lines]


def point_line_distance(px, py, x1, y1, x2, y2):
    # 点到线段距离
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    dot = A * C + B * D
    len_sq = C * C + D * D
    param = dot / len_sq if len_sq != 0 else -1
    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    dx = px - xx
    dy = py - yy
    return math.hypot(dx, dy)


def sector_pixel_voting(center, radius, binary_img, n_sectors=60, debug=False):
    """在原始二值图上按扇形像素投票。返回每个扇区的像素计数。"""
    cx, cy = center
    h, w = binary_img.shape[:2]
    votes = np.zeros(n_sectors, dtype=float)
    for i in range(n_sectors):
        ang1 = (i * 360.0 / n_sectors)
        ang2 = ((i + 1) * 360.0 / n_sectors)
        # 角度到弧度并计算端点
        a1 = math.radians(ang1)
        a2 = math.radians(ang2)
        x1 = int(cx + math.sin(a1) * radius)
        y1 = int(cy - math.cos(a1) * radius)
        x2 = int(cx + math.sin(a2) * radius)
        y2 = int(cy - math.cos(a2) * radius)
        pts = np.array([[cx, cy], [x1, y1], [x2, y2]], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        # 统计区域内的白色像素（假设指针为白）
        count = int(cv2.countNonZero(cv2.bitwise_and(binary_img, mask)))
        votes[i] = count
        if debug and i % max(1, n_sectors // 12) == 0:
            print(f"sector {i}: ang [{ang1:.1f},{ang2:.1f}) count={count}")
    return votes


def sector_voting(center, radius, lines, n_sectors=360, debug=False):
    """基于直线段的扇区投票。返回 (votes, line_infos)。"""
    cx, cy = center
    votes = np.zeros(n_sectors, dtype=float)
    line_infos = []
    if debug:
        print(f"sector_voting: n_lines={len(lines)}, n_sectors={n_sectors}, radius={radius}")
    for (x1, y1, x2, y2) in lines:
        length = math.hypot(x2 - x1, y2 - y1)
        dist = min(point_line_distance(cx, cy, x1, y1, x2, y2),
                   math.hypot(x1 - cx, y1 - cy), math.hypot(x2 - cx, y2 - cy))
        if dist > radius * 0.6:
            continue
        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)
        if d1 > d2:
            ex, ey = x1, y1
        else:
            ex, ey = x2, y2
        angle_rad = math.atan2(cy - ey, ex - cx)
        angle_deg = (math.degrees(angle_rad) + 360) % 360
        angle_clock = (90 - angle_deg) % 360
        sector = int((angle_clock % 360.0) * n_sectors / 360.0) % n_sectors
        weight = length / (1.0 + max(0.0, dist))
        votes[sector] += weight
        info = {'line': (x1, y1, x2, y2), 'angle': angle_clock, 'length': length, 'dist': dist, 'weight': weight}
        line_infos.append(info)
        if debug:
            print(f"  line {(x1,y1,x2,y2)} len={length:.1f} dist={dist:.1f} sector={sector} weight={weight:.3f}")
    return votes, line_infos


def find_peaks(votes, k=2, min_separation=1):
    # 返回 k 个峰值扇区索引（基于 votes 的索引）
    peaks = []
    votes_copy = votes.copy()
    for _ in range(k):
        idx = int(np.argmax(votes_copy))
        val = votes_copy[idx]
        if val <= 0:
            break
        peaks.append(idx)
        # 抑制邻域
        sep = int(min_separation)
        low = max(0, idx - sep)
        high = min(len(votes_copy) - 1, idx + sep)
        votes_copy[low:high + 1] = 0
    return peaks


def angle_to_time(angle_deg_hour, angle_deg_minute):
    # angle 0 为 12 点，顺时针增大
    minute = int(round(angle_deg_minute * 60.0 / 360.0)) % 60
    hour = int(round(angle_deg_hour * 12.0 / 360.0)) % 12
    return hour, minute


def detect_hands_and_time(image_path, debug=True):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    # 缩放到合适大小以提高速度
    scale = 512.0 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 保留二值化图像用于霍夫直线识别
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 如果背景为白色而指针为黑色，将二值图反转，使指针为前景色白色
    if np.mean(binary) > 127:
        binary_hough = cv2.bitwise_not(binary)
    else:
        binary_hough = binary.copy()
    # 形态学去噪，保留主要指针结构
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_hough = cv2.morphologyEx(binary_hough, cv2.MORPH_OPEN, kernel_small)

    # Canny 边缘检测
    edges = edge_detect(gray, method='canny', low_threshold=50, high_threshold=150)
    if debug:
        try:
            cv2.imshow('Edges (Canny)', edges)
            cv2.waitKey(0)
            cv2.imshow('Binary (Otsu)', binary)
            cv2.waitKey(0)
            cv2.imshow('Binary for Hough (processed)', binary_hough)
            cv2.waitKey(0)
            cv2.destroyWindow('Edges (Canny)')
            cv2.destroyWindow('Binary (Otsu)')
            cv2.destroyWindow('Binary for Hough (processed)')
        except Exception:
            pass

    # 找表盘圆心和半径
    circ = detect_clock_face(gray)
    if circ is None:
        # 如果未检测到圆，使用图像中心和最小边长为半径
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
        radius = min(cx, cy) - 10
    else:
        cx, cy, radius = circ
    print(f"Image size: {img.shape}, center: ({cx},{cy}), radius: {radius})")

    # 在原始二值图上，用检测出的圆形分割60个扇形，按前景像素对每个扇形进行投票
    N_SECTORS = 60
    votes_pixels = sector_pixel_voting((cx, cy), radius, binary_hough, n_sectors=N_SECTORS, debug=debug)

    # 用 HoughLinesP 提取直线并做基于直线的扇区投票
    lines = lines_from_edges(binary_hough, radius)
    if debug:
        print(f"Detected {len(lines)} Hough lines")
    votes_lines, line_infos = sector_voting((cx, cy), radius, lines, n_sectors=N_SECTORS, debug=debug)

    # 归一化并合并像素投票与直线投票（权重可调）
    vp = votes_pixels.astype(float)
    vl = votes_lines.astype(float)
    if vp.max() > 0:
        vp = vp / vp.max()
    if vl.max() > 0:
        vl = vl / vl.max()
    weight_pixels = 1.0
    weight_lines = 1.0
    combined = weight_pixels * vp + weight_lines * vl

    # min_separation: 12 degrees -> 转换为扇区数
    min_sep_sectors = max(1, int(round(12 * N_SECTORS / 360.0)))
    peaks = find_peaks(combined, k=3, min_separation=min_sep_sectors)
    # 展示投票结果（曲线）
    if debug:
        try:
            import matplotlib.pyplot as plt
            import io
            fig = plt.figure(figsize=(8,3))
            x = np.arange(N_SECTORS)
            plt.plot(x, votes_pixels, marker='o', label='pixels')
            plt.plot(x, votes_lines, marker='x', label='lines')
            plt.plot(x, combined, marker='s', label='combined')
            plt.legend()
            plt.title('Sector votes (pixels | lines | combined)')
            plt.xlabel('Sector')
            plt.ylabel('Normalized vote')
            plt.grid(True)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            arr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if arr is not None:
                cv2.imshow('Sector votes (pixels|lines|combined)', arr)
                cv2.waitKey(0)
                cv2.destroyWindow('Sector votes (pixels|lines|combined)')
        except Exception:
            pass
    print(f"Peaks found: {peaks}")
    if len(peaks) == 0:
        return img, (None, None), (cx, cy, radius)

    # 选择票数最高的两个扇区：最高为分针，次高为时针
    peak_scores = []
    for p in peaks:
        window = 5
        idxs = [(p + i) % N_SECTORS for i in range(-window, window + 1)]
        score = combined[idxs].sum() if isinstance(combined, np.ndarray) else sum(combined[i] for i in idxs)
        peak_scores.append((p, score))
    peak_scores = sorted(peak_scores, key=lambda x: x[1], reverse=True)
    chosen = [p for p, s in peak_scores[:2]]
    if len(chosen) == 1:
        chosen.append((chosen[0] + N_SECTORS // 2) % N_SECTORS)
    minute_sector = chosen[0]
    hour_sector = chosen[1]

    # 将扇区索引转为角度后计算时间
    hour_angle = (hour_sector * 360.0 / N_SECTORS) % 360
    minute_angle = (minute_sector * 360.0 / N_SECTORS) % 360
    hour, minute = angle_to_time(hour_angle, minute_angle)

    # 可视化：画圆并标出两个指针方向
    vis = img.copy()
    cv2.circle(vis, (cx, cy), int(radius), (0, 255, 0), 2)
    # 画出假定的指针方向线段
    def draw_hand(vis, center, angle_deg, length_ratio, color=(0, 0, 255), thickness=3):
        cx, cy = center
        ang_rad = math.radians(angle_deg)
        x = int(cx + math.sin(ang_rad) * radius * length_ratio)
        y = int(cy - math.cos(ang_rad) * radius * length_ratio)
        cv2.line(vis, (cx, cy), (x, y), color, thickness)
        return (x, y)

    draw_hand(vis, (cx, cy), minute_angle, 0.95, color=(0, 0, 255), thickness=3)
    draw_hand(vis, (cx, cy), hour_angle, 0.6, color=(255, 0, 0), thickness=4)

    cv2.putText(vis, f"Estimated: {hour:02d}:{minute:02d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    if debug:
        # 显示边缘与 Hough 直线用于调试，并展示结果
        debug_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for (x1, y1, x2, y2) in lines:
            cv2.line(debug_vis, (x1, y1), (x2, y2), (0, 128, 255), 1)
        try:
            combined = np.hstack([cv2.resize(vis, (debug_vis.shape[1], debug_vis.shape[0])), debug_vis])
            cv2.imshow('Clock detection (result | edges+lines)', combined)
            cv2.waitKey(0)
            cv2.destroyWindow('Clock detection (result | edges+lines)')
        except Exception:
            pass

    return vis, (hour, minute), (cx, cy, radius)


if __name__ == '__main__':
    img_path = 'lesson_3/clock.PNG'
    vis, time_est, circ = detect_hands_and_time(img_path, debug=True)
    h, m = time_est
    if h is not None:
        print(f'Estimated time: {h:02d}:{m:02d}')
    else:
        print('No hands detected')
