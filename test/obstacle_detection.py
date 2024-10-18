import numpy as np
import matplotlib.pyplot as plt

# 定義環境配置
grid_width = 11
grid_height = 11
start = np.array([0, 0])  # 起點
goal = np.array([10, 10])  # 終點
# 使用矩形的左下角和右上角座標來定義障礙物
obstacles = [
    (np.array([0.5, 1.5]), np.array([2.5, 2.5])),  # 第一個矩形障礙物
    (np.array([4.5, 3.5]), np.array([5.5, 5.5]))  # 第二個矩形障礙物
]
safety_size = 0.2  # 障礙物周圍安全距離
circle_radius = 0.1  # 點位的半徑(mobile robot大小)
waypoint_distance = 0.8  # 初始路徑上點的距離

# 生成初始路徑
def generate_initial_path(start, goal, distance):
    num_points = int(np.linalg.norm(goal - start) / distance)
    initial_path = np.linspace(start, goal, num=num_points + 1)
    return initial_path

# 計算路徑的總長度
def calculate_path_length(path):
    return np.sum([np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path))])

# 判斷「點位」是否在擴展過的矩形內(障礙物含安全距離)
def point_in_obstacle(point, bottom_left, top_right, safety_size):
    expanded_bottom_left = bottom_left - np.array([safety_size, safety_size])
    expanded_top_right = top_right + np.array([safety_size, safety_size])
    return (expanded_bottom_left[0] <= point[0] <= expanded_top_right[0] and
            expanded_bottom_left[1] <= point[1] <= expanded_top_right[1])

# 判斷「線段」是否穿過擴展過的矩形範圍（障礙物含安全距離）
def line_intersects_obstacle(p1, p2, bottom_left, top_right, safety_size):
    expanded_bottom_left = bottom_left - np.array([safety_size, safety_size])
    expanded_top_right = top_right + np.array([safety_size, safety_size])
    rect_edges = [
        (expanded_bottom_left, np.array([expanded_top_right[0], expanded_bottom_left[1]])),  # 下邊
        (expanded_bottom_left, np.array([expanded_bottom_left[0], expanded_top_right[1]])),  # 左邊
        (np.array([expanded_top_right[0], expanded_bottom_left[1]]), expanded_top_right),    # 右邊
        (np.array([expanded_bottom_left[0], expanded_top_right[1]]), expanded_top_right)     # 上邊
    ]
    
    for edge_start, edge_end in rect_edges:
        if line_segments_intersect(p1, p2, edge_start, edge_end):
            return True
    return False

# 幾何計算：檢測兩線段是否相交
def line_segments_intersect(p1, p2, q1, q2):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # 共線
        return 1 if val > 0 else 2  # 順時針或逆時針

    def on_segment(p, q, r):
        return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
                min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, p2, q2): return True
    if o3 == 0 and on_segment(q1, q2, p1): return True
    if o4 == 0 and on_segment(q1, q2, p2): return True

    return False

# 將經過障礙物上的點位儲存成避障前原始路徑
def path_before_obstacle_avoidance(path, obstacle, safety_size):
    bottom_left, top_right = obstacle
    points_within_obstacle = []

    for p in path:
        if point_in_obstacle(p, bottom_left, top_right, safety_size):
            points_within_obstacle.append(p)

    if not points_within_obstacle:
        return []

    first_obstacle_point = points_within_obstacle[0]
    last_obstacle_point = points_within_obstacle[-1]

    first_idx = np.where(np.all(path == first_obstacle_point, axis=1))[0][0]
    last_idx = np.where(np.all(path == last_obstacle_point, axis=1))[0][0]

    closest_points = []
    if first_idx > 0:
        closest_points.append(path[first_idx - 1])  # 碰撞起始點的前一點
    closest_points.extend(points_within_obstacle)  # 障礙物內的點
    if last_idx < len(path) - 1:
        closest_points.append(path[last_idx + 1])  # 碰撞終點的下一點

    return closest_points  # 返回碰撞起始點、終點和障礙物內的點

# 將初始路徑上的碰撞點偏移出障礙物(含安全距離)的範圍行程新路徑，作為避障路徑最佳化的Initial Guess。
def generate_new_path(path, obstacles, safety_size, offset_distance):
    new_paths = []  # 用於儲存所有障礙物的避障路徑

    for obstacle in obstacles:
        offset_directions = []  # 儲存每個障礙物的偏移方向

        closest_points = path_before_obstacle_avoidance(path, obstacle, safety_size)

        if closest_points:  # 確保有碰撞點
            tangent_vector = closest_points[-1] - closest_points[0]
            tangent_vector /= np.linalg.norm(tangent_vector)  # 標準化切線向量

            normal_vector_left = np.array([-tangent_vector[1], tangent_vector[0]])
            normal_vector_right = np.array([tangent_vector[1], -tangent_vector[0]])

            for point in closest_points:
                closest_obstacle_edge_distance = float('inf')
                chosen_normal_vector = None
                
                bottom_left, top_right = obstacle
                distances = [
                    (bottom_left[0] - safety_size) - point[0],  # 左邊
                    point[0] - (top_right[0] + safety_size),    # 右邊
                    (bottom_left[1] - safety_size) - point[1],  # 下邊
                    point[1] - (top_right[1] + safety_size)     # 上邊
                ]

                min_distance = min(distances)
                if abs(min_distance) < abs(closest_obstacle_edge_distance):
                    closest_obstacle_edge_distance = min_distance
                    if min_distance == distances[0]:  # 左邊
                        chosen_normal_vector = normal_vector_right  # 向右偏移
                    elif min_distance == distances[1]:  # 右邊
                        chosen_normal_vector = normal_vector_left  # 向左偏移
                    elif min_distance == distances[2]:  # 下邊
                        chosen_normal_vector = normal_vector_left  # 向上偏移
                    elif min_distance == distances[3]:  # 上邊
                        chosen_normal_vector = normal_vector_right  # 向下偏移

                offset_directions.append(tuple(chosen_normal_vector))  # 轉換為元組

            # 找到出現最多的偏移方向
            most_common_direction = max(set(offset_directions), key=offset_directions.count)

            new_path = []
            for point in closest_points:
                new_point = point + np.array(most_common_direction) * offset_distance
                new_path.append(new_point)

            new_paths.append(np.array(new_path))

    return new_paths  # 返回所有障礙物的新路徑

# 可視化
def visualize_grid(ax, start, goal, obstacles, circle_radius=circle_radius, positions=None, paths=None):
    # 畫出環境中的障礙物
    for bottom_left, top_right in obstacles:
        rect = plt.Rectangle(bottom_left, top_right[0] - bottom_left[0], top_right[1] - bottom_left[1], color='red', alpha=0.5)
        ax.add_patch(rect)
    
    # 畫出起點和終點
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'bo', markersize=10, label='Goal')

    # 畫出原始路徑
    if positions is not None:
        ax.plot(positions[:, 0], positions[:, 1], 'g--', label='Original Path')

    # 畫出避障路徑
    if paths is not None:
        for path in paths:
            ax.plot(path[:, 0], path[:, 1], 'b-', label='Avoidance Path')

    ax.set_xlim(-1, grid_width + 1)
    ax.set_ylim(-1, grid_height + 1)
    ax.set_xticks(np.arange(-1, grid_width + 1, 1))
    ax.set_yticks(np.arange(-1, grid_height + 1, 1))
    ax.grid()
    ax.set_aspect('equal')
    ax.legend()

# 主程序
initial_path = generate_initial_path(start, goal, waypoint_distance)  # 生成初始路徑
new_paths = generate_new_path(initial_path, obstacles, safety_size, offset_distance=1.0)

# 繪圖
fig, ax = plt.subplots(figsize=(8, 8))
visualize_grid(ax, start, goal, obstacles, positions=initial_path, paths=new_paths)
plt.title('Obstacle Avoidance Path Planning')
plt.show()
