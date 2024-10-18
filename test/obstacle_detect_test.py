import numpy as np
import matplotlib.pyplot as plt

# 檢查點是否在矩形內
def is_point_in_rectangle(point, rect_bottom_left, rect_top_right):
    return (rect_bottom_left[0] <= point[0] <= rect_top_right[0]) and \
           (rect_bottom_left[1] <= point[1] <= rect_top_right[1])

# 計算兩線段是否相交
def are_segments_intersecting(p1, p2, q1, q2):
    def orientation(p, q, r):
        return np.sign((q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]))

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # 檢查一般情況下是否相交
    if o1 != o2 and o3 != o4:
        return True
    
    return False

# 檢查線段是否與矩形相交
def is_segment_overlapping_rectangle(segment_start, segment_end, rect_bottom_left, rect_top_right):
    # 檢查線段的端點是否在矩形內
    if is_point_in_rectangle(segment_start, rect_bottom_left, rect_top_right) or \
       is_point_in_rectangle(segment_end, rect_bottom_left, rect_top_right):
        return True
    
    # 定義矩形的四個頂點
    rect_top_left = np.array([rect_bottom_left[0], rect_top_right[1]])
    rect_bottom_right = np.array([rect_top_right[0], rect_bottom_left[1]])

    # 定義矩形的四條邊
    rectangle_edges = [
        (rect_bottom_left, rect_top_left),       # 左邊
        (rect_top_left, rect_top_right),         # 上邊
        (rect_top_right, rect_bottom_right),     # 右邊
        (rect_bottom_right, rect_bottom_left)    # 下邊
    ]
    
    # 檢查線段是否與矩形的任何一條邊相交
    for edge_start, edge_end in rectangle_edges:
        if are_segments_intersecting(segment_start, segment_end, edge_start, edge_end):
            return True
    
    return False

# 畫圖
def plot_path_and_rectangle(segment_start, segment_end, rect_bottom_left, rect_top_right):
    fig, ax = plt.subplots()

    # 畫線段
    ax.plot([segment_start[0], segment_end[0]], [segment_start[1], segment_end[1]], 'b-', label='Line Segment')

    # 畫矩形
    rect = plt.Rectangle(rect_bottom_left, rect_top_right[0] - rect_bottom_left[0], rect_top_right[1] - rect_bottom_left[1], 
                         color='r', alpha=0.5, label='Obstacle')
    ax.add_artist(rect)

    # 設置圖形範圍
    ax.set_xlim(min(segment_start[0], segment_end[0], rect_bottom_left[0]) - 1,
                max(segment_start[0], segment_end[0], rect_top_right[0]) + 1)
    ax.set_ylim(min(segment_start[1], segment_end[1], rect_bottom_left[1]) - 1,
                max(segment_start[1], segment_end[1], rect_top_right[1]) + 1)
    
    # 標示
    ax.set_aspect('equal', 'box')
    ax.legend()
    ax.set_title('Path and Rectangular Obstacle')
    plt.grid(True)
    plt.show()

# 測試
segment_start = np.array([0, 0])
segment_end = np.array([10, 10])
rect_bottom_left = np.array([4, 4])
rect_top_right = np.array([8, 8])

# 檢查線段是否與矩形障礙物重疊，並繪圖
overlapping = is_segment_overlapping_rectangle(segment_start, segment_end, rect_bottom_left, rect_top_right)

if overlapping:
    print("線段與矩形障礙物重疊")
else:
    print("線段與矩形障礙物不重疊")

# 畫出結果
plot_path_and_rectangle(segment_start, segment_end, rect_bottom_left, rect_top_right)
