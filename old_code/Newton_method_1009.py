'''
1. In this program, the variable `safety_size` is defined as the space surrounding the obstacle 
that the path should avoid.
2. Objective function 計算新舊路徑點的距離平方和（不包括起始和終點）、新路徑點之間的距離平方和、新點與碰撞起始點、終點的距離平方和。
'''

import numpy as np
import matplotlib.pyplot as plt
import time  

# 定義環境配置
grid_width = 11
grid_height = 11
start = np.array([0, 0])
goal = np.array([10, 10])
# 使用矩形的左下角和右上角座標來定義障礙物
obstacles = [
    (np.array([0.5, 1.5]), np.array([2.5, 2.5])),  # 第一個矩形障礙物
    (np.array([4.5, 3.5]), np.array([5.5, 5.5]))  # 第二個矩形障礙物
]
safety_size = 0.1
waypoint_distance = 0.8  # 路徑上點的距離

# 檢查點是否在障礙物的正方形區域內
def is_point_in_square(point, obstacle, safety_size):
    square_bottom_left = obstacle[0]
    square_top_right = obstacle[1]
    return (square_bottom_left[0] - safety_size <= point[0] <= square_top_right[0] + safety_size and
            square_bottom_left[1] - safety_size <= point[1] <= square_top_right[1] + safety_size)

# 生成新的點，位於障礙物邊緣之外
def generate_new_point_away_from_obstacle(closest_points, obstacle, safety_size):
    new_points = []
    # 提取不包括第一個和最後一個點的中間點
    for point in closest_points[1:-1]:  # 跳過第一個和最後一個點
        direction = point - obstacle
        distance = np.linalg.norm(direction)
        if distance == 0:
            new_points.append(point)  # 若點正好在障礙物中心，返回原點
        else:
            # 將點從障礙物邊緣偏移一定距離(距離障礙物邊緣加上safety_size的距離)
            new_point = obstacle + (direction / distance) * (safety_size) 
            new_points.append(new_point)
    return new_points  # 返回新生成的點位

# 定義目標函數
def objective_function(new_points, closest_points):
    """
    new_points: 由 generate_new_point_away_from_obstacle 生成的新點
    closest_points: find_closest_points_to_obstacle 回傳的 closest_points 包含碰撞起始和終點(皆為原始路徑上的點位)
    weights: 權重，包含 w_old, w_new, w_transition 等
    """
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

# 定義約束函數，檢查與障礙物的距離
def constraint_function(x, obstacles, safety_size):
    for obs in obstacles:
        if is_point_in_square(x, obs, safety_size):
            return False  # 若距離小於安全距離，則返回 False
    return True  # 若所有障礙物都安全，返回 True

# 找到原始路徑中與障礙物重疊或接觸的點，並返回這些點和兩側相鄰點
def find_closest_points_to_obstacle(path, obstacle, safety_size):
    # 找出路徑中穿過障礙物或接觸到障礙物邊緣的點
    points_within_obstacle = []  # 建立空列表來儲存結果
    for p in path:  # 對路徑中的每個點進行迴圈
        if is_point_in_square(p, obstacle , safety_size):  # 如果該點在障礙物範圍內
            points_within_obstacle.append(p)  # 將該點加入結果列表

    print(f"Points within obstacle: {points_within_obstacle}")

    if not points_within_obstacle:
        return []  # 如果沒有穿過障礙物的點，返回空列表

    # 找到最早和最晚穿過障礙物的點
    first_obstacle_point = points_within_obstacle[0]
    last_obstacle_point = points_within_obstacle[-1]

    # 找到這些障礙物點在原始路徑中的索引
    first_idx = np.where(np.all(path == first_obstacle_point, axis=1))[0][0]
    last_idx = np.where(np.all(path == last_obstacle_point, axis=1))[0][0]

    # 獲取相鄰點
    closest_points = []

    #### 如果碰撞起始點和碰撞終點距離障礙物太近，可能需要再各多加一個點。####
    # 如果第一個障礙物點前面有相鄰點，取它作為碰撞起始點
    if first_idx > 0:
        closest_points.append(path[first_idx - 1])  # 相鄰的前一點作為碰撞起始點

    # 加入所有穿過障礙物的點
    closest_points.extend(points_within_obstacle)

    # 如果最後一個障礙物點後面有相鄰點，取它作為碰撞終點
    if last_idx < len(path) - 1:
        closest_points.append(path[last_idx + 1])  # 相鄰的下一點作為碰撞終點

    return closest_points  # 返回結果，包括碰撞起始點、終點和障礙物內的點

# 可視化
def visualize_grid(ax, start, goal, obstacles, positions=None, path=None, new_points=None, original_path=None, closest_points=None):
    ax.cla()  # 清除當前的圖表，保留窗口
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')  # 起點
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')  # 終點
    
    for bottom_left, top_right in obstacles:
        rect = plt.Rectangle(bottom_left, top_right[0] - bottom_left[0], top_right[1] - bottom_left[1], color='gray')
        ax.add_patch(rect)  # 畫出矩形障礙物

    circle_radius = 0.2  # 這裡設定一個圓的半徑

    # 畫出原始路徑的點作為圓形
    if original_path is not None:
        for point in original_path:
            circle = plt.Circle(point, radius=circle_radius, color='c', fill=True, label='Original Path')
            ax.add_patch(circle)

    # 當前位置的圓形
    if positions is not None:
        for pos in positions:
            circle = plt.Circle(pos, radius=circle_radius, color='b', fill=True, label='Current Position')
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

    # 新生成點的圓形
    if new_points is not None and len(new_points) > 0:
        for point in new_points:
            circle = plt.Circle(point, radius=circle_radius, color='purple', fill=True, label='New Points')
            ax.add_patch(circle)

    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_title('Dragonfly Path Planning Optimization')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)
    plt.draw()
    plt.pause(0.1)


# 生成初始路徑
def generate_initial_path(start, goal, distance):
    # 生成從起點到終點的路徑，並在路徑上添加點
    num_points = int(np.linalg.norm(goal - start) / distance)
    initial_path = np.linspace(start, goal, num=num_points + 1)
    # print(f"Initial Path: {initial_path}")
    return initial_path

# 計算路徑的總長度
def calculate_path_length(path):
    return np.sum([np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))])

# 牛頓法實現
def newton_method(start, goal, obstacles, safety_size, initial_guess, original_path, max_iter=10, tol=0.2):
    x = np.array(initial_guess)  # 將初始猜測轉換為 numpy 陣列
    positions = []  # 記錄每次的當前位置
    path = []      # 記錄已走的路徑
    new_points = []  # 記錄新生成的點
    fig, ax = plt.subplots()  # 初始化圖表
    visualize_grid(ax, start, goal, obstacles, original_path=original_path)  # 初始可視化
    
    # 計算開始的時間
    start_time = time.time()  # 開始計時

    step_size = 0.3  # 步長
    stuck_counter = 0  # 停滯計數器
    stuck_threshold = 2  # 停滯阈值

    for j in range(max_iter):  # 增加迭代計數器 j
        near_obstacle = False
        closest_points = []

        # 檢查是否接近障礙物
        for obs in obstacles:
            if np.linalg.norm(x - obs) < safety_size:
                near_obstacle = True
                print("Approaching an obstacle, adjusting path...")

                # 獲取原始路徑中經過障礙物(含安全距離)的點位
                closest_points = find_closest_points_to_obstacle(original_path, obs, safety_size)
                
                # 將每個經過障礙物的原始路徑點生成一個新的點，位於障礙物邊緣之外
                for point in closest_points:
                    new_point = generate_new_point_away_from_obstacle(point, obs, safety_size)
                    if constraint_function(new_point, obs, safety_size):  # 確保新點安全
                        new_points.append(new_point)  # 記錄每個新生成的點
                        positions.append(new_point)  # 記錄當前位置
                        path.append(new_point)  # 記錄路徑
                x = path[-1]  # 更新為最後一個新生成的位置
                break
        
        # 如果沒有接近障礙物，沿著原始路徑前進
        if not near_obstacle:
            grad = (x - goal) / np.linalg.norm(x - goal)
            new_position = x - grad * step_size  # 計算新的位置
            
            # 確保新位置安全
            if constraint_function(new_position, obs, safety_size):
                positions.append(x.copy())  # 記錄當前位置
                path.append(x.copy())  # 記錄路徑
                x = new_position  # 更新位置
                stuck_counter = 0  # 重置停滯計數器
            else:
                print("New position is not safe, trying to adjust...")
                stuck_counter += 1  # 增加停滯計數器
                if stuck_counter > stuck_threshold:  # 如果停滯次數超過阈值
                    # 在當前位置周圍隨機擾動
                    x += np.random.uniform(-1, 1, size=x.shape)
                    stuck_counter = 0  # 重置停滯計數器

        # Convert positions to a consistent format before visualization
        positions_array = np.array(positions)
        path_array = np.array(path)
        new_points_array = np.array(new_points) if new_points else np.empty((0, 2))

        # 更新可視化，包括最近的四個點
        visualize_grid(ax, start, goal, obstacles, positions=positions_array, path=path_array, new_points=new_points_array, original_path=original_path, closest_points=closest_points)

        # 印出每次迭代的位置
        print(f"Iteration {j + 1}, Current position: {x}")

        # 停止條件：如果距離目標點足夠接近，或已達到最大迭代次數
        if np.linalg.norm(x - goal) < tol:
            print("Goal reached!")
            break

    # 計算結束的時間
    end_time = time.time()  # 結束計時

    # 計算最終路徑長度
    path_length = calculate_path_length(np.array(path))
    
    # 計算總運算時間
    total_time = end_time - start_time
    print(f"Final Path Length: {path_length:.4f} units")
    print(f"Total Computation Time: {total_time:.4f} seconds")
    
    return x

# 使用範例
initial_guess = start.copy()  # 將初始猜測位置設置為起點
original_path = generate_initial_path(start, goal, waypoint_distance)  # 生成初始路
final_position = newton_method(start, goal, obstacles, safety_size, initial_guess, original_path)
print("Final Position:", final_position)

plt.show()