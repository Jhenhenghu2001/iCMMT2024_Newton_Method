# This code is from Newton_method_1015.py
'''
1. In this program, the variable `safety_size` is defined as the space surrounding the obstacle 
that the path should avoid.
2. Objective function 計算新舊路徑點的距離平方和（不包括起始和終點）、新路徑點之間的距離平方和、新點與碰撞起始點、終點的距離平方和。
3. 判斷路徑上的點以及線段是否與障礙物相交之方法尚未確認。
   (兩種方案:1.Bool只判斷是否在障礙物範圍內 2.利用點/線之間的數學關係式計算出Danger factor數值)
4. 針對新的避障路徑點位置，使用牛頓法取得最佳位置，使路徑最短、平滑且避開障礙物。
5. 若要將mobile robot的大小納入碰撞檢測的考量 則點位碰撞檢測需要改成以下程式
def point_in_obstacle(point, bottom_left, top_right, safety_size, circle_radius):
    # 增加障礙物周圍的安全距離，同時考慮點位的半徑
    expanded_bottom_left = bottom_left - np.array([safety_size + circle_radius, safety_size + circle_radius])
    expanded_top_right = top_right + np.array([safety_size + circle_radius, safety_size + circle_radius])
    # 檢查點是否在擴展過的矩形內
    return (expanded_bottom_left[0] <= point[0] <= expanded_top_right[0] and
            expanded_bottom_left[1] <= point[1] <= expanded_top_right[1])
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
safety_size = 0.2  # 障礙物周圍安全距離
circle_radius = 0.1  # 點位的半徑(mobile robot大小)
waypoint_distance = 0.7 # 初始路徑上點的距離

#####

# 生成初始路徑
def generate_initial_path(start, goal, distance):
    # 生成從起點到終點的路徑，並在路徑上添加點
    num_points = int(np.linalg.norm(goal - start) / distance)
    initial_path = np.linspace(start, goal, num=num_points + 1)
    return initial_path

# 計算路徑的總長度
def calculate_path_length(path):
    return np.sum([np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))])

#####

# 判斷「點位」是否在擴展過的矩形內(障礙物含安全距離)
def point_in_obstacle(point, bottom_left, top_right, safety_size):
    # 增加障礙物周圍的安全距離，同時考慮點位的半徑
    expanded_bottom_left = bottom_left - np.array([safety_size, safety_size])
    expanded_top_right = top_right + np.array([safety_size, safety_size])
    
    # 檢查點是否在擴展過的矩形內
    return (expanded_bottom_left[0] <= point[0] <= expanded_top_right[0] and
            expanded_bottom_left[1] <= point[1] <= expanded_top_right[1])

# 判斷「線段」是否穿過擴展過的矩形範圍（障礙物含安全距離）
def line_intersects_obstacle(p1, p2, bottom_left, top_right, safety_size):
    # 增加障礙物周圍的安全距離
    expanded_bottom_left = bottom_left - np.array([safety_size, safety_size])
    expanded_top_right = top_right + np.array([safety_size, safety_size])
    
    # 定義擴展後的矩形的四條邊的端點
    rect_edges = [
        (expanded_bottom_left, np.array([expanded_top_right[0], expanded_bottom_left[1]])),  # 下邊
        (expanded_bottom_left, np.array([expanded_bottom_left[0], expanded_top_right[1]])),  # 左邊
        (np.array([expanded_top_right[0], expanded_bottom_left[1]]), expanded_top_right),    # 右邊
        (np.array([expanded_bottom_left[0], expanded_top_right[1]]), expanded_top_right)     # 上邊
    ]
    
    # 檢查線段是否與矩形的任意一條邊相交
    for edge_start, edge_end in rect_edges:
        if line_segments_intersect(p1, p2, edge_start, edge_end):
            return True  # 如果相交，則返回True
    return False  # 否則返回False

# 幾何計算：檢測兩線段是否相交
def line_segments_intersect(p1, p2, q1, q2):
    def orientation(p, q, r):
        """計算p-q-r三點的方向"""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0  # 共線
        return 1 if val > 0 else 2  # 順時針或逆時針

    def on_segment(p, q, r):
        """檢查點r是否在p-q線段上"""
        return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
                min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))

    # 找出四個點之間的方向
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # 一般情況下線段相交
    if o1 != o2 and o3 != o4:
        return True

    # 特殊情況：三點共線且點在對方線段上
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, p2, q2): return True
    if o3 == 0 and on_segment(q1, q2, p1): return True
    if o4 == 0 and on_segment(q1, q2, p2): return True

    return False  # 沒有相交

#####

# 將經過障礙物上的點位儲存成避障前原始路徑
def path_before_obstacle_avoidance(path, obstacle, safety_size):
    bottom_left, top_right = obstacle  # 解壓障礙物的左下角和右上角座標
    points_within_obstacle = []  # 用於儲存在障礙物內的點

    # 檢查路徑上每個點是否在擴展過的障礙物區域內
    for p in path:
        if point_in_obstacle(p, bottom_left, top_right, safety_size):
            points_within_obstacle.append(p)

    if not points_within_obstacle:
        return []  # 如果沒有點在障礙物內，返回空列表

    # 找到最早和最晚穿過障礙物的點
    first_obstacle_point = points_within_obstacle[0]
    last_obstacle_point = points_within_obstacle[-1]

    # 找到這些點在路徑中的索引
    first_idx = np.where(np.all(path == first_obstacle_point, axis=1))[0][0]
    last_idx = np.where(np.all(path == last_obstacle_point, axis=1))[0][0]

    # 獲取與這些障礙物點相鄰的點
    closest_points = []
    if first_idx > 0:
        closest_points.append(path[first_idx - 1]) # 碰撞起始點的前一點
    closest_points.extend(points_within_obstacle)  # 障礙物內的點
    if last_idx < len(path) - 1:
        closest_points.append(path[last_idx + 1])  # 碰撞終點的下一點

    return closest_points  # 返回碰撞起始點、終點和障礙物內的點

#####

#  將初始路徑上的碰撞點偏移出障礙物(含安全距離)的範圍行程新路徑，作為避障路徑最佳化的Initial Guess。
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
                    (bottom_left[0] - safety_size) - point[0],  # 左側邊緣
                    point[0] - (top_right[0] + safety_size),    # 右側邊緣
                    (bottom_left[1] - safety_size) - point[1],  # 下方邊緣
                    point[1] - (top_right[1] + safety_size)     # 上方邊緣
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
            for point in closest_points[1:-1]: # 排除首尾點位，只考慮中間的碰撞點進行偏移
                new_point = point + np.array(most_common_direction) * offset_distance
                new_path.append(new_point)

            new_paths.append(np.array(new_path))

    return new_paths  # 返回所有障礙物的新路徑

#####

# Obstacle avoidance path planning optimization using Newton's method.

# 定義目標函數
def objective_function(new_points, closest_points):
    w_old = 0.5  # 原始路徑點的權重
    w_new = 0.5  # 新路徑點的權重
    w_transition = 0.5  # 路徑過渡點之間的權重

    distance_to_original = 0
    distance_between_new_points = 0
    transition_distances = 0
    
    # 計算新舊路徑點的距離平方和（不包括起始和終點）
    for new_p, old_p in zip(new_points, closest_points[1:-1]):  # 只對內部點進行計算
        distance_to_original += (np.linalg.norm(new_p - old_p) ** 2) * w_old
    
    # 計算新路徑點之間的距離平方和
    for i in range(1, len(new_points)):
        distance_between_new_points += (np.linalg.norm(new_points[i] - new_points[i - 1]) ** 2) * w_new
    
    # 計算新點與碰撞起始點、終點的距離平方和
    transition_distances += (np.linalg.norm(new_points[0] - closest_points[0]) ** 2) * w_transition  # 第一個新點到碰撞起始點
    transition_distances += (np.linalg.norm(new_points[-1] - closest_points[-1]) ** 2) * w_transition  # 最後一個新點到碰撞終點
    
    return distance_to_original + distance_between_new_points + transition_distances  # 返回總的距離作為目標函數值

# initial guess 有2(x, y)*3(中際點數量)=6個變數值
def Hessian_caculate():
    
    return

#####

# 可視化
def visualize_grid(ax, start, goal, obstacles, circle_radius=circle_radius, positions=None, path=None, new_points=None, original_path=None, closest_points=None):
    ax.cla()  # 清除當前的圖表，保留窗口
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')  # 起點
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')  # 終點

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

    # 當前位置的圓形，將 positions 列表轉換成 NumPy 陣列
    if positions is not None:
        positions = np.array(positions)  # 將 positions 列表轉換成 NumPy 陣列
        for pos in positions:
            circle = plt.Circle(pos, radius=circle_radius, color='b', fill=True)
            ax.add_patch(circle)
        # 連接點之間的線
        ax.plot(positions[:, 0], positions[:, 1], 'b-', label='New Path')

    # 新生成的避障路徑
    if new_points is not None and len(new_points) > 0:
        new_points = np.array(new_points)
        for point in new_points:
            ax.plot(point[:, 0], point[:, 1], 'm-o', label='Avoidance Path')

    # 路徑的圓形
    if path is not None:
        path = np.array(path)
        for pos in path:
            circle = plt.Circle(pos, radius=circle_radius, color='b', fill=True)
            ax.add_patch(circle)

    # 碰撞起始、終點、碰撞點位的圓形
    if closest_points is not None and len(closest_points) > 0:
        for point in closest_points:
            circle = plt.Circle(point, radius=circle_radius, color='orange', fill=True)
            ax.add_patch(circle)

    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_title("Newton's Method Path Planning Optimization")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)
    plt.draw()
    plt.pause(0.1)

#####

# 讓圓點沿著路徑移動
def move_along_path(path, start, goal):
    fig, ax = plt.subplots()
    positions = []
    for point in path:
        positions.append(point)
        visualize_grid(ax, start, goal, obstacles, circle_radius=circle_radius, positions=positions, original_path=path)
    plt.show()

# Main
original_path = generate_initial_path(start, goal, waypoint_distance)  # 生成初始路徑

# 紀錄所有障礙物的碰撞點
all_closest_points = []

# 檢查路徑是否經過障礙物
for obstacle in obstacles:
    closest_points = path_before_obstacle_avoidance(original_path, obstacle, safety_size)
    if closest_points:
        all_closest_points.extend(closest_points)  # 儲存碰撞點

# Generate new path based on obstacle avoidance
new_path = generate_new_path(all_closest_points, obstacles, safety_size, offset_distance=1.0)

# Optimize new path based on obstacle avoidance with using Newton's method
# print("new path", new_path)
new_path_length = objective_function(new_path, all_closest_points)

# 使用 visualize_grid 函數進行可視化，並將碰撞點傳入
fig, ax = plt.subplots()
positions = []

for point in original_path:
    positions.append(point)
    visualize_grid(ax, start, goal, obstacles, circle_radius=circle_radius, 
                   positions=positions, original_path=original_path, 
                   closest_points=all_closest_points, new_points=new_path)

plt.show()

####
# 測試程式

def test_line_intersects_obstacle():
    # 定義障礙物的左下角和右上角
    obstacle_bottom_left = np.array([3.0, 3.0])
    obstacle_top_right = np.array([5.0, 5.0])
    safety_size = 0.5  # 障礙物的安全距離

    # 測試用的線段 (p1, p2) 和預期結果
    test_cases = [
        (np.array([1.0, 1.0]), np.array([6.0, 6.0]), True),   # 斜線穿過障礙物，應該相交
        (np.array([0.0, 0.0]), np.array([2.0, 2.0]), False),  # 線段遠離障礙物，應該不相交
        (np.array([4.0, 4.0]), np.array([6.0, 4.0]), True),   # 線段經過障礙物右邊，應該相交
        (np.array([2.0, 4.0]), np.array([4.0, 2.0]), False),  # 線段靠近但不穿過障礙物，應該不相交
        (np.array([2.0, 4.0]), np.array([4.0, 6.0]), True)    # 線段在障礙物上方相交
    ]

    # 執行每個測試案例
    for i, (p1, p2, expected) in enumerate(test_cases):
        result = line_intersects_obstacle(p1, p2, obstacle_bottom_left, obstacle_top_right, safety_size)
        print(f"Test case {i + 1}: {'Passed' if result == expected else 'Failed'} (Expected {expected}, Got {result})")

# 呼叫測試函數
test_line_intersects_obstacle()

# 確認line_intersects_obstacle是可以正常執行的