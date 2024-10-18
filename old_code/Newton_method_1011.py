'''
1. In this program, the variable `safety_size` is defined as the space surrounding the obstacle 
that the path should avoid.
2. Objective function 計算新舊路徑點的距離平方和（不包括起始和終點）、新路徑點之間的距離平方和、新點與碰撞起始點、終點的距離平方和。
3. 判斷路徑上的點以及線段是否與障礙物相交之方法尚未確認。
   (兩種方案:1.Bool只判斷是否在障礙物範圍內 2.利用點/線之間的數學關係式計算出Danger factor數值)
4. 針對新的避障路徑點位置，使用牛頓法取得最佳位置，使路徑最短、平滑且避開障礙物。
'''

import numpy as np
import matplotlib.pyplot as plt
import time

# 定義環境配置
grid_width = 11
grid_height = 11
start = np.array([0, 0]) # 起點
goal = np.array([10, 10]) # 終點
# 使用矩形的左下角和右上角座標來定義障礙物
obstacles = [
    (np.array([0.5, 1.5]), np.array([2.5, 2.5])),  # 第一個矩形障礙物
    (np.array([4.5, 3.5]), np.array([5.5, 5.5]))  # 第二個矩形障礙物
]
safety_size = 0.1  # 障礙物周圍安全距離
waypoint_distance = 0.8  # 初始路徑上點的距離

# 生成初始路徑
def generate_initial_path(start, goal, distance):
    # 生成從起點到終點的路徑，並在路徑上添加點
    num_points = int(np.linalg.norm(goal - start) / distance)
    initial_path = np.linspace(start, goal, num=num_points + 1)
    return initial_path

# 計算路徑的總長度
def calculate_path_length(path):
    return np.sum([np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))])

# 可視化
def visualize_grid(ax, start, goal, obstacles, positions=None, path=None, new_points=None, original_path=None, closest_points=None):
    ax.cla()  # 清除當前的圖表，保留窗口
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')  # 起點
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')  # 終點

    for bottom_left, top_right in obstacles:
        rect = plt.Rectangle(bottom_left, top_right[0] - bottom_left[0], top_right[1] - bottom_left[1], color='gray')
        ax.add_patch(rect)  # 畫出矩形障礙物

    circle_radius = 0.1  # 這裡設定一個圓的半徑

    # 畫出原始路徑的點並用線連接它們
    if original_path is not None:
        # 畫出點
        for point in original_path:
            circle = plt.Circle(point, radius=circle_radius, color='c', fill=True)
            ax.add_patch(circle)
        # 連接點之間的線
        ax.plot(original_path[:, 0], original_path[:, 1], 'c-', label='Original Path')

    # 當前位置的圓形，將 positions 列表轉換成 NumPy 陣列
    if positions is not None:
        positions = np.array(positions)  # 將 positions 列表轉換成 NumPy 陣列
        for pos in positions:
            circle = plt.Circle(pos, radius=circle_radius, color='b', fill=True)
            ax.add_patch(circle)
        # 連接點之間的線
        ax.plot(positions[:, 0], positions[:, 1], 'b-', label='New Path')

    # 新生成點的圓形
    if new_points is not None and len(new_points) > 0:
        for point in new_points:
            circle = plt.Circle(point, radius=circle_radius, color='purple', fill=True)
            ax.add_patch(circle)

    # 路徑的圓形
    if path is not None:
        path = np.array(path)
        for pos in path:
            circle = plt.Circle(pos, radius=circle_radius, color='b', fill=True)
            ax.add_patch(circle)

    # 碰撞起始和終點的圓形
    if closest_points is not None and len(closest_points) > 0:
        for point in closest_points:
            circle = plt.Circle(point, radius=circle_radius, color='orange', fill=True, label='Closest Points')
            ax.add_patch(circle)

    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_title("Newton's Method Path Planning Optimization")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)
    plt.draw()
    plt.pause(0.1)

# 讓圓點沿著路徑移動
def move_along_path(path, start, goal):
    fig, ax = plt.subplots()
    positions = []
    for point in path:
        positions.append(point)
        visualize_grid(ax, start, goal, obstacles, positions=positions, original_path=path)
    plt.show()

# Main
original_path = generate_initial_path(start, goal, waypoint_distance)  # 生成初始路徑
move_along_path(original_path, start, goal)  # 讓圓點沿著初始路徑移動
