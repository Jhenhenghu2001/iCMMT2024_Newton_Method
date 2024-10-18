'''
1. In this program, the variable `safety_size` is defined as the space surrounding the obstacle 
that the path should avoid.
2. Objective function 計算新舊路徑點的距離平方和（不包括起始和終點）、新路徑點之間的距離平方和、新點與碰撞起始點、終點的距離平方和。
3. 判斷路徑上的點以及線段是否與障礙物相交之方法。(採用以下方法1)
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
def point_in_obstacle(point, obstacle, safety_size):
    bottom_left, top_right = obstacle
    # 增加障礙物周圍的安全距離，同時考慮點位的半徑
    expanded_bottom_left = bottom_left - np.array([safety_size, safety_size])
    expanded_top_right = top_right + np.array([safety_size, safety_size])
    
    # 檢查點是否在擴展過的矩形內
    return (expanded_bottom_left[0] <= point[0] <= expanded_top_right[0] and
            expanded_bottom_left[1] <= point[1] <= expanded_top_right[1])

# 判斷「線段」是否穿過擴展過的矩形範圍（障礙物含安全距離）

def line_intersects_any_obstacle(p1, p2, obstacles, safety_size):
    # 遍歷每個障礙物
    for obstacle in obstacles:
        if line_intersects_obstacle(p1, p2, obstacle, safety_size):
            return True  # 如果相交，則返回True
    return False  # 否則返回False

def line_intersects_obstacle(p1, p2, obstacle, safety_size):
    bottom_left, top_right = obstacle
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
        
        # If val is an array (NumPy array), check conditions properly
        if isinstance(val, np.ndarray):
            if np.all(val > 0):
                return 1  # Clockwise
            elif np.all(val < 0):
                return 2  # Counterclockwise
            else:
                return 0  # Collinear
        else:
            return 1 if val > 0 else (2 if val < 0 else 0)

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
    points_within_obstacle = []  # 用於儲存在障礙物內的點

    # 檢查路徑上每個點是否在擴展過的障礙物區域內
    for p in path:
        if point_in_obstacle(p, obstacle, safety_size):
            points_within_obstacle.append(p)

    if not points_within_obstacle:
        return []  # 如果沒有點在障礙物內，返回空列表

    # 找到最早和最晚穿過障礙物的點
    first_obstacle_point = points_within_obstacle[0]
    last_obstacle_point = points_within_obstacle[-1]

    # 若想要將每個經過障礙物的點位都做偏移，則把以下兩行程式註解即可。
    # 若只有一個點經過障礙物，則直接返回該點，不做處理
    if len(points_within_obstacle) == 1:
        return [first_obstacle_point]
    # 以下兩行程式主要是只取經過障礙物的起始、中間、結束這三個點位以減少計算量。
    # 計算初始和結束點的平均值作為中間點
    middle_point = (first_obstacle_point + last_obstacle_point) / 2
    # 更新points_within_obstacle，保留初始點、中間點、結束點
    points_within_obstacle = [first_obstacle_point, middle_point, last_obstacle_point]

    # 找到這些點在路徑中的索引
    first_idx = np.where(np.all(path == first_obstacle_point, axis=1))[0][0]
    last_idx = np.where(np.all(path == last_obstacle_point, axis=1))[0][0]

    # 獲取與這些障礙物點相鄰的點
    closest_points = []
    if first_idx > 0:
        closest_points.append(path[first_idx - 1])  # 碰撞起始點的前一點
    closest_points.extend(points_within_obstacle)  # 初始點、中間點和結束點
    if last_idx < len(path) - 1:
        closest_points.append(path[last_idx + 1])  # 碰撞終點的下一點

    return closest_points  # 返回碰撞起始點、中間點、終點和障礙物內的點

#####

#  將初始路徑上的碰撞點偏移出障礙物(含安全距離)的範圍行程新路徑，作為避障路徑最佳化的Initial Guess。
# def generate_new_path(path, obstacles, safety_size, offset_distance):
#     new_paths = []  # 用於儲存所有障礙物的避障路徑

#     for obstacle in obstacles:
#         offset_directions = []  # 儲存每個障礙物的偏移方向

#         closest_points = path_before_obstacle_avoidance(path, obstacle, safety_size)

#         if closest_points:  # 確保有碰撞點
#             tangent_vector = closest_points[-1] - closest_points[0]
#             tangent_vector /= np.linalg.norm(tangent_vector)  # 標準化切線向量

#             normal_vector_left = np.array([-tangent_vector[1], tangent_vector[0]])
#             normal_vector_right = np.array([tangent_vector[1], -tangent_vector[0]])

#             for point in closest_points:
#                 closest_obstacle_edge_distance = float('inf')
#                 chosen_normal_vector = None
                
#                 bottom_left, top_right = obstacle
#                 distances = [
#                     (bottom_left[0] - safety_size) - point[0],  # 左側邊緣
#                     point[0] - (top_right[0] + safety_size),    # 右側邊緣
#                     (bottom_left[1] - safety_size) - point[1],  # 下方邊緣
#                     point[1] - (top_right[1] + safety_size)     # 上方邊緣
#                 ]

#                 min_distance = min(distances)
#                 if abs(min_distance) < abs(closest_obstacle_edge_distance):
#                     closest_obstacle_edge_distance = min_distance
#                     if min_distance == distances[0]:  # 左邊
#                         chosen_normal_vector = normal_vector_right  # 向右偏移
#                     elif min_distance == distances[1]:  # 右邊
#                         chosen_normal_vector = normal_vector_left  # 向左偏移
#                     elif min_distance == distances[2]:  # 下邊
#                         chosen_normal_vector = normal_vector_left  # 向上偏移
#                     elif min_distance == distances[3]:  # 上邊
#                         chosen_normal_vector = normal_vector_right  # 向下偏移

#                 offset_directions.append(tuple(chosen_normal_vector))  # 轉換為元組

#             # 找到出現最多的偏移方向
#             most_common_direction = max(set(offset_directions), key=offset_directions.count)

#             new_path = []
#             for point in closest_points[1:-1]: # 排除首尾點位，只考慮中間的碰撞點進行偏移
#                 new_point = point + np.array(most_common_direction) * offset_distance
#                 new_path.append(new_point)

#             new_paths.append(np.array(new_path))

#     return new_paths  # 返回所有障礙物的新路徑

def generate_new_path(path, obstacles, safety_size, offset_distance):
    new_paths = []  # 用於儲存所有障礙物的避障路徑

    offset_directions = []  # 儲存每個障礙物的偏移方向

    closest_points = path_before_obstacle_avoidance(path, obstacles, safety_size)

    if closest_points:  # 確保有碰撞點
        tangent_vector = closest_points[-1] - closest_points[0]
        tangent_vector /= np.linalg.norm(tangent_vector)  # 標準化切線向量

        normal_vector_left = np.array([-tangent_vector[1], tangent_vector[0]])
        normal_vector_right = np.array([tangent_vector[1], -tangent_vector[0]])

        for point in closest_points:
            closest_obstacle_edge_distance = float('inf')
            chosen_normal_vector = None
            
            bottom_left, top_right = obstacles
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

def objective_function(X, X_ori, W):
    # 解包，獲取內部的數組
    X_array = X[0]

    X1, X2, X3 = [X_array[0]], [X_array[1]], [X_array[2]]  # 分解 X 為 X1*, X2*, X3*
    XT, X1_ori, X2_ori, X3_ori, XH = X_ori[0], X_ori[1], X_ori[2], X_ori[3], X_ori[4] # 分解 X_ori 為 XT, X1_ori, X2_ori, X3_ori, XH
    w1, w2, w3, wH, wT, w21, w32 = W[0], W[1], W[2], W[3], W[4], W[5], W[6] # 權重

    # 計算目標函數
    f = (w1 * np.linalg.norm(X1 - X1_ori)**2 +
         w2 * np.linalg.norm(X2 - X2_ori)**2 +
         w3 * np.linalg.norm(X3 - X3_ori)**2 +
         wH * np.linalg.norm(X1 - XH)**2 +
         wT * np.linalg.norm(X3 - XT)**2 +
         w21 * np.linalg.norm(X2 - X1)**2 +
         w32 * np.linalg.norm(X3 - X2)**2)
    
    return f

def gradient(X, X_ori, W):
    # 解包，獲取內部的數組
    X_array = X[0]
    
    X1, X2, X3 = [X_array[0]], [X_array[1]], [X_array[2]]  # 分解 X 為 X1*, X2*, X3*
    XT, X1_ori, X2_ori, X3_ori, XH = X_ori[0], X_ori[1], X_ori[2], X_ori[3], X_ori[4] # 分解 X_ori 為 XT, X1_ori, X2_ori, X3_ori, XH
    w1, w2, w3, wH, wT, w21, w32 = W[0], W[1], W[2], W[3], W[4], W[5], W[6] # 權重

    # 計算各點的梯度
    grad_X1 = (2 * w1 * (X1 - X1_ori) +
               2 * wH * (X1 - XH) +
               2 * w21 * (X1 - X2))
    
    grad_X2 = (2 * w2 * (X2 - X2_ori) +
               2 * w21 * (X2 - X1) +
               2 * w32 * (X2 - X3))
    
    grad_X3 = (2 * w3 * (X3 - X3_ori) +
               2 * wT * (X3 - XT) +
               2 * w32 * (X3 - X2))
    
    # 將梯度組合為一個向量
    grad = np.array([grad_X1, grad_X2, grad_X3])
    
    return grad

def hessian(W):
    w1, w2, w3, wH, wT, w21, w32 = W[0], W[1], W[2], W[3], W[4], W[5], W[6] # 權重

    # 單位矩陣 (2x2)，因為每個點都有 (x, y) 兩個維度
    I = np.eye(2)

    # 計算對角線 Hessian，這些是二階導數 f''(X1), f''(X2), f''(X3)
    H11 = 2 * (w1 + wH + w21) * I  # 對 X1* 的二階導數
    H22 = 2 * (w2 + w21 + w32) * I # 對 X2* 的二階導數
    H33 = 2 * (w3 + wT + w32) * I  # 對 X3* 的二階導數

    # 計算交叉項 Hessian，這些是二階交叉導數 f''(X1, X2), f''(X2, X3)
    H12 = H21 = -2 * w21 * I  # 對 X1* 和 X2* 的交叉二階導數
    H23 = H32 = -2 * w32 * I  # 對 X2* 和 X3* 的交叉二階導數

    # 組成 Hessian 矩陣，總共 6x6 大小，將每個 2x2 的子矩陣放入總矩陣中
    H = np.block([[H11, H12, np.zeros((2, 2))],
                  [H21, H22, H23],
                  [np.zeros((2, 2)), H32, H33]])

    return H

def inverse_hessian(W):
    H = hessian(W)  # 計算 Hessian 矩陣
    det = np.linalg.det(H)  # 計算行列式
    
    if det == 0:
        raise ValueError("Hessian 矩陣是奇異矩陣，不可逆。")
    
    H_inv = np.linalg.inv(H)  # 計算 Hessian 的逆矩陣
    return H_inv

def newton_method(X_init, X_ori, W, max_iter=1000, tol=1e-3):
    """
    使用多維牛頓法進行最佳化，直到滿足收斂條件或達到最大迭代次數。
    
    參數：
    - X_init: 初始點 [X1*, X2*, X3*] 的向量
    - X_ori: 原始點，包括 XT, X1_ori, X2_ori, X3_ori, XH
    - W: 權重向量

    限制條件：
    - max_iter: 最大迭代次數 (預設 1000)
    - tol: 收斂誤差門檻值 (預設 1e-3)
    
    回傳：
    - X_opt: 最佳化後的點位 [X1*, X2*, X3*]
    - k: 總迭代次數
    """
    
    X = X_init  # 初始化 X
    error = np.inf  # 初始化誤差，設為無窮大
    k = 0  # 初始化迭代次數
    
    while error > tol and k < max_iter:
        k += 1
        
        # 計算梯度和 Hessian 的逆矩陣
        grad = gradient(X, X_ori, W)
        H_inv = inverse_hessian(W)
        
        # 更新 X(k+1) = X(k) - H^(-1) * grad
        X_new = X - np.dot(H_inv, grad.flatten())
        
        # 計算誤差：這裡可以用梯度的範數來表示
        error = np.linalg.norm(grad)
        
        # 更新 X
        X = X_new
    
    return X, k    

#####

# def find_closest_obstacle(current_position, obstacle_list, visited_obstacles):
#     """
#     找出最近的未經過的矩形障礙物。
    
#     參數：
#     current_position (np.array): 當前點位的位置 [x, y]。
#     obstacle_list (list): 包含矩形障礙物的列表，每個障礙物是 (np.array([x1, y1]), np.array([x2, y2]))。
#     visited_obstacles (list): 已經經過的障礙物列表，形式與 obstacle_list 相同。
    
#     返回：
#     tuple: 最近的障礙物，形式是 (np.array([x1, y1]), np.array([x2, y2]))。
#     """
#     min_distance = float('inf')
#     closest_obstacle = None

#     for obstacle in obstacle_list:
#         bottom_left, top_right = obstacle
        
#         # 如果這個障礙物已經走過，則跳過
#         if any(np.array_equal(bottom_left, visited_obstacle[0]) and 
#                np.array_equal(top_right, visited_obstacle[1]) for visited_obstacle in visited_obstacles):
#             continue
        
#         # 計算當前點到矩形的最短距離
#         distance = point_to_rectangle_distance(current_position, bottom_left, top_right)
        
#         # 更新最近的障礙物
#         if distance < min_distance:
#             min_distance = distance
#             closest_obstacle = obstacle

#     return closest_obstacle

def find_closest_obstacle(current_position, obstacle_list, visited_obstacles):
    """
    找出最近的未經過的矩形障礙物。
    
    參數：
    current_position (np.array): 當前點位的位置 [x, y]。
    obstacle_list (list): 包含矩形障礙物的列表，每個障礙物是 (np.array([x1, y1]), np.array([x2, y2]))。
    visited_obstacles (list): 已經經過的障礙物列表，形式與 obstacle_list 相同。
    
    返回：
    tuple: 最近的未經過障礙物 (np.array([x1, y1]), np.array([x2, y2]))。
    """
    
    closest_obstacle = None
    min_distance = float('inf')
    
    for obstacle in obstacle_list:
        if obstacle in visited_obstacles:
            continue  # 跳過已經訪問過的障礙物
        
        # 計算當前位置到障礙物中心的距離
        bottom_left, top_right = obstacle
        obstacle_center = (bottom_left + top_right) / 2
        distance = np.linalg.norm(current_position - obstacle_center)
        
        # 更新最近的障礙物和最小距離
        if distance < min_distance:
            min_distance = distance
            closest_obstacle = obstacle
    
    return closest_obstacle

# def point_to_rectangle_distance(point, bottom_left, top_right):
#     """
#     計算點到矩形的最短距離。
    
#     參數：
#     point (np.array): 點的座標 [x, y]。
#     bottom_left (np.array): 矩形的左下角座標 [x1, y1]。
#     top_right (np.array): 矩形的右上角座標 [x2, y2]。
    
#     返回：
#     float: 點到矩形的最短距離。
#     """
#     x, y = point
#     x1, y1 = bottom_left
#     x2, y2 = top_right

#     # 如果點在矩形的範圍內，最短距離是0
#     if x1 <= x <= x2 and y1 <= y <= y2:
#         return 0

#     # 計算點到矩形的最短距離
#     dx = max(x1 - x, 0, x - x2)
#     dy = max(y1 - y, 0, y - y2)
    
#     return np.sqrt(dx * dx + dy * dy)

#####

# 可視化
def visualize_grid(ax, start, goal, obstacles, circle_radius=None, positions=None, path=None, new_points=None, original_path=None, closest_points=None):
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
            ax.plot(positions[0], positions[1], 'b-o', label='Recent Point')

    # 新生成的避障路徑
    if new_points is not None and len(new_points) > 0:
        new_points = np.array(new_points)
        for point in new_points:
            ax.plot(point[0], point[1], 'm-o', label='Avoidance Path')

    # 路徑的圓形
    if path is not None:
        path = np.array(path)
        for pos in path:
            circle = plt.Circle(pos, radius=circle_radius, color='b', fill=True)
            ax.add_patch(circle)

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
    plt.pause(1)

#####

# Main
def main():
    fig, ax = plt.subplots()
    # 環境參數初始化
    grid_size = (11, 11)  # Grid的大小
    obstacles = [
        (np.array([0.5, 1.5]), np.array([2.5, 2.5])), # 第一個矩形障礙物
        (np.array([4.5, 3.5]), np.array([5.5, 5.5]))  # 第二個矩形障礙物
    ]  # 障礙物列表
    safety_distance = 0.2  # 安全距離
    waypoint_distance = 0.5  # 每個路徑點之間的距離
    circle_radius = 0.1  # 點位的半徑(mobile robot大小)
    # 初始化牛頓法參數
    W = [1, 1, 1, 1, 1, 1, 1] # Objective function各項距離的權重(XT, X1_ori, X2_ori, X3_ori, XH)

    # 產生初始路徑
    start = np.array([0, 0])
    goal = np.array([10, 10])
    initial_path = generate_initial_path(start, goal, waypoint_distance)

    # 建立點P並初始化位置
    current_position_index = 0
    current_position = initial_path[current_position_index]
    full_path_traveled = [current_position]  # 紀錄點P走過的完整路徑
    visited_obstacles = []  # 初始化已經經過的障礙物列表

    # 視覺化初始化
    visualize_grid(ax, start, goal, obstacles, circle_radius=circle_radius, positions=current_position, original_path=initial_path, new_points=full_path_traveled)

    # 主流程
    while np.linalg.norm(current_position - goal) > 0.1:
        # 獲取下一個目標點
        next_position = initial_path[current_position_index + 1]

        # 檢查當前點到下一個點的線段是否經過障礙物或安全區域
        if line_intersects_any_obstacle(current_position, next_position, obstacles, safety_distance):
            # 找到最近的障礙物，並準備進行避障
            closest_obstacle = find_closest_obstacle(current_position, obstacles, visited_obstacles)
            
            # 如果找到了最近的障礙物，則將其加入 visited_obstacles
            if closest_obstacle is not None:
                visited_obstacles.append(closest_obstacle)

            # 找到初始路徑中所有經過障礙物的點
            points_near_obstacle = path_before_obstacle_avoidance(initial_path, closest_obstacle, safety_distance)

            # 根據避障策略生成新的路徑，將避障點設為 initial guess
            new_path = generate_new_path(points_near_obstacle, closest_obstacle, safety_distance, offset_distance=1.0)

            # 使用牛頓法進行路徑最佳化
            optimized_path = newton_method(new_path, initial_path, W)

            # 確認新路徑是否避開障礙物
            X_opt_final = [initial_path[0]] + optimized_path + [initial_path[-1]]
            for i in range(len(X_opt_final) - 1):
                if line_intersects_obstacle(X_opt_final[i], X_opt_final[i+1], obstacles, safety_distance):
                    # 若新路徑有經過障礙物，將當前解作為 initial guess 重新最佳化
                    optimized_path = newton_method(X_opt_final, initial_path, W)
                    X_opt_final = [initial_path[0]] + optimized_path + [initial_path[-1]]
                    break  # 重新檢測路徑

            # 將當前位置沿著新的路徑移動，更新current_position
            initial_path = X_opt_final  # 更新路徑
            current_position_index = 0  # 重新從起點開始走
        else:
            # 沒有碰到障礙物，更新當前位置
            current_position = next_position
            current_position_index += 1
            full_path_traveled.append(current_position)

        # 檢查是否到達終點
        if np.linalg.norm(current_position - goal) <= 0.1:
            print("到達終點!")
            break

        # 更新視覺化
        visualize_grid(ax, start, goal, obstacles, circle_radius=circle_radius, positions=current_position, original_path=initial_path, closest_points=full_path_traveled)
    plt.show()
main()