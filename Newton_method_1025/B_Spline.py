import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Use B-Spline to smooth the path.
def b_spline(final_path):
    # 將final_path轉換為唯一的控制點並去除重複的點
    control_points = np.array(final_path)
    
    # 確保控制點數量足夠
    if control_points.shape[0] < 4:
        raise ValueError("Not enough unique control points to perform B-spline interpolation.")
    
    # B-樣條插值
    tck, u = splprep(control_points.T, s=0)

    # 在插值曲線中插入更多的點
    u_fine = np.linspace(0, 1, 1000)
    x_fine, y_fine = splev(u_fine, tck)

    # 計算插值路徑的長度
    b_spline_path_length = 0
    for i in range(1, len(x_fine)):
        segment_length = np.sqrt((x_fine[i] - x_fine[i - 1])**2 + (y_fine[i] - y_fine[i - 1])**2)
        b_spline_path_length += segment_length

    print(f"The total length of the B-spline path is: {b_spline_path_length}")

    # 在曲線中等距取樣15個點
    num_points = control_points.shape[0] * 2
    u_sampled = np.linspace(0, 1, num_points)
    x_sampled, y_sampled = splev(u_sampled, tck)
    
    # Visualization
    obstacles = [
        (np.array([0.5, 1.5]), np.array([2.5, 2.5])), # 第一個矩形障礙物
        (np.array([4.5, 3.5]), np.array([5.5, 5.5]))  # 第二個矩形障礙物
    ]
    start = np.array([0, 0])
    goal = np.array([10, 10])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points')
    ax.plot(x_sampled, y_sampled, 'go', label='Sampled Points')
    ax.plot(x_fine, y_fine, 'b-', label='Interpolated Curve', linewidth=2)
    ax.plot(start[0], start[1], 'go', markersize=3, label='Start')  # 起點
    ax.plot(goal[0], goal[1], 'ro', markersize=3, label='Goal')  # 終點
    # 畫出矩形障礙物
    for bottom_left, top_right in obstacles:
        rect = plt.Rectangle(bottom_left, top_right[0] - bottom_left[0], top_right[1] - bottom_left[1], color='gray')
        ax.add_patch(rect)  # 這裡改成使用 `ax.add_patch`
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Newton's Method Path Planning Optimization with B-Spline")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)
    plt.show()