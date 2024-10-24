import numpy as np
import matplotlib.pyplot as plt

# 可視化
def visualize_grid(ax, start, goal, obstacles, circle_radius=None, positions=None, original_path=None, path=None, new_points=None, closest_points=None):
    ax.cla()  # 清除當前的圖表，保留窗口
    ax.plot(start[0], start[1], 'go', markersize=3, label='Start')  # 起點
    ax.plot(goal[0], goal[1], 'ro', markersize=3, label='Goal')  # 終點

    # 畫出環境中的障礙物
    for bottom_left, top_right in obstacles:
        rect = plt.Rectangle(bottom_left, top_right[0] - bottom_left[0], top_right[1] - bottom_left[1], color='gray')
        ax.add_patch(rect)  # 畫出矩形障礙物

    # 畫出原始路徑的點並用線連接它們
    if original_path is not None:
        # 畫出點
        for point in original_path:
            circle = plt.Circle(point, radius=circle_radius, color='c', fill=True)
            ax.add_patch(circle)
        # 連接點之間的線
        ax.plot(original_path[:, 0], original_path[:, 1], 'c-', label='Original Path')

    # 當前位置點位的圓形，將 positions 列表轉換成 NumPy 陣列
    if positions is not None:
        positions = np.array(positions)  # 將 positions 列表轉換成 NumPy 陣列
        for i, pos in enumerate(positions):
            ax.plot(positions[0], positions[1], 'bo', label='Recent Point' if i == 0 else "")

    # 新生成的避障路徑
    if new_points is not None:
        new_points = np.array(new_points)  # 將 new_points 轉換為 NumPy 陣列
        for i, point in enumerate(new_points):  # 遍歷 new_points 中的每個點
            ax.plot(point[0], point[1], 'mo', markersize=5, label='Avoidance Path' if i == 0 else "")
    
    # 路徑的圓形
    if path is not None:
        path = np.array(path)
        for pos in path:
            circle = plt.Circle(pos, radius=circle_radius, color='k', fill=True)
            ax.add_patch(circle)
        ax.plot(path[:, 0], path[:, 1], 'k-', label='full_path_traveled')

    # 碰撞起始、終點、碰撞點位的圓形
    if closest_points is not None and len(closest_points) > 0:
        for point in closest_points:
            ax.plot(point[0], point[1], color='orange')

    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_title("Newton's Method Path Planning Optimization")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)
    plt.draw()
    plt.pause(0.1)