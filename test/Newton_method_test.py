import numpy as np
import matplotlib.pyplot as plt
import time  

# 定義環境配置
grid_width = 11
grid_height = 11
start = np.array([0, 0])
goal = np.array([10, 10])
obstacles = [np.array([5, 5]), np.array([2, 2]), np.array([5, 4]), np.array([1, 2])]
obstacle_size = 1
safety_size = 0.1
waypoint_distance = 0.8  # 路徑上點的距離
weight_old = 0.5  # 舊點距離的權重
weight_new = 0.5  # 新點距離的權重

# 檢查點是否在障礙物的正方形區域內
def is_point_in_square(point, square_center, size, tolerance=0.8):
    half_size = size / 2
    return (square_center[0] - half_size <= point[0] <= square_center[0] + half_size + tolerance and
            square_center[1] - half_size <= point[1] <= square_center[1] + half_size + tolerance)

# 生成新的點，位於障礙物邊緣之外
def generate_new_point_away_from_obstacle(point, obstacle, obstacle_size):
    # 計算到障礙物中心的向量
    direction = point - obstacle
    distance = np.linalg.norm(direction)
    if distance == 0:
        return point  # 若點正好在障礙物中心，返回原點
    # 計算一個新的點，距離障礙物邊緣加上safety_size的距離
    new_point = obstacle + (direction / distance) * (obstacle_size+ safety_size)
    return new_point

# 定義目標函數
def objective_function(original_path, new_points):
    # 目標是最小化原始路徑點的距離，以及新點之間的距離
    distance_to_original = np.sum([np.linalg.norm(new_p - p) ** 2 * weight_old for new_p, p in zip(new_points, original_path)])
    distance_between_new_points = np.sum([np.linalg.norm(new_points[i] - new_points[i-1]) ** 2 * weight_new for i in range(1, len(new_points))])
    return distance_to_original + distance_between_new_points  # 返回目標函數值

# 定義約束函數，檢查與障礙物的距離
def constraint_function(x, obstacles, obstacle_size, safety_size):
    for obs in obstacles:
        if is_point_in_square(x, obs, obstacle_size + safety_size):
            return False  # 若距離小於安全距離，則返回 False
    return True  # 若所有障礙物都安全，返回 True

# 找到原始路徑中最接近障礙物的點，並返回最近的四個點
def find_closest_points_to_obstacle(path, obstacle, obstacle_size, num_points=5):
    # 找出路徑中穿過障礙物的點
    points_within_obstacle = []  # 建立空列表來儲存結果
    for p in path:  # 對路徑中的每個點進行迴圈
        if is_point_in_square(p, obstacle, obstacle_size):  # 如果該點在障礙物範圍內
            points_within_obstacle.append(p)  # 將該點加入結果列表
    print(f"Points within obstacle: {points_within_obstacle}")
    if len(points_within_obstacle) >= num_points:
        # 如果穿過障礙物的點足夠，直接返回最近的四個點
        distances = [np.linalg.norm(p - obstacle) for p in points_within_obstacle]
        sorted_indices = np.argsort(distances)  # 按距離排序
        closest_points = [points_within_obstacle[i] for i in sorted_indices[:num_points]]
    else:
        # 如果穿過障礙物的點不足四個，先取穿過障礙物的點
        closest_points = points_within_obstacle
        
        # 尋找相鄰的點，補充到四個點
        if len(points_within_obstacle) > 0:
            # 找到最靠近障礙物的點，並找其相鄰的點
            first_obstacle_point = points_within_obstacle[0]
            last_obstacle_point = points_within_obstacle[-1]
            
            # 找到這些障礙物點在原始路徑中的索引
            first_idx = np.where(np.all(path == first_obstacle_point, axis=1))[0][0]
            last_idx = np.where(np.all(path == last_obstacle_point, axis=1))[0][0]

            # 檢查相鄰點
            adjacent_candidates = []
            if first_idx > 0:
                adjacent_candidates.append(path[first_idx - 1])  # 取左邊相鄰點
            if last_idx < len(path) - 1:
                adjacent_candidates.append(path[last_idx + 1])  # 取右邊相鄰點

            # 計算這些相鄰點到障礙物邊緣的距離，並選擇距離最近的點
            def distance_to_obstacle_edge(point, obstacle, obstacle_size):
                # 計算點到障礙物邊緣的距離
                return max(0, np.linalg.norm(point - obstacle) - obstacle_size / 2)

            # 找到與障礙物邊緣距離最近的相鄰點
            if adjacent_candidates:
                distances_to_edge = [distance_to_obstacle_edge(p, obstacle, obstacle_size) for p in adjacent_candidates]
                closest_adjacent_point = adjacent_candidates[np.argmin(distances_to_edge)]
                closest_points.append(closest_adjacent_point)

    return closest_points

# 可視化
def visualize_grid(ax, start, goal, obstacles, positions=None, path=None, new_points=None, original_path=None, closest_points=None):
    ax.cla()  # 清除當前的圖表，保留窗口
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')  # 起點
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')  # 終點
    for obstacle in obstacles:
        square = plt.Rectangle((obstacle[0] - obstacle_size / 2, obstacle[1] - obstacle_size / 2), obstacle_size, obstacle_size, color='gray')
        ax.add_patch(square)  # 畫出障礙物
    if original_path is not None:
        # 繪製原始路徑
        ax.plot(original_path[:, 0], original_path[:, 1], 'c--', linewidth=2, label='Original Path')  # 原始路徑
        ax.plot(original_path[:, 0], original_path[:, 1], 'co', markersize=5)  # 原始路徑上的點
    if positions is not None:
        ax.plot(positions[:, 0], positions[:, 1], 'bo', markersize=5, label='Current Position')  # 當前位置
    if path is not None:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Path')  # 路徑
        ax.plot(path[:, 0], path[:, 1], 'bo', markersize=5)
    if closest_points is not None and len(closest_points) > 0:
        closest_points_array = np.array(closest_points)
        ax.plot(closest_points_array[:, 0], closest_points_array[:, 1], 'orange', marker='o', linestyle='None', markersize=8, label='Closest Points')  # 標示最接近障礙物的五個點
    if new_points is not None and len(new_points) > 0:
        new_points_array = np.array(new_points)
        ax.plot(new_points_array[:, 0], new_points_array[:, 1], 'purple', marker='o', linestyle='None', markersize=6, label='New Points')  # 標示新生成的點
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_title('Dragonfly Path Planning Optimization')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
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
def newton_method(start, goal, obstacles, obstacle_size, safety_size, initial_guess, original_path, max_iter=10, tol=0.2):
    x = np.array(initial_guess)  # 將初始猜測轉換為 numpy 陣列
    positions = []  # 記錄每次的當前位置
    path = []      # 記錄已走的路徑
    new_points = []  # 記錄新生成的點
    fig, ax = plt.subplots()  # 初始化圖表
    visualize_grid(ax, start, goal, obstacles, original_path=original_path)  # 初始可視化
    
    # 計算開始的時間
    start_time = time.time()  # 開始計時

    step_size = 0.5  # 步長
    stuck_counter = 0  # 停滯計數器
    stuck_threshold = 2  # 停滯阈值

    for j in range(max_iter):  # 增加迭代計數器 j
        near_obstacle = False
        closest_points = []

        # 檢查是否接近障礙物
        for obs in obstacles:
            if np.linalg.norm(x - obs) < safety_size + obstacle_size:
                near_obstacle = True
                print("Approaching an obstacle, adjusting path...")

                # 獲取原始路徑中距離障礙物最近的四個點
                closest_points = find_closest_points_to_obstacle(original_path, obs, obstacle_size, num_points=5)
                
                # 將每個經過障礙物的原始路徑點生成一個新的點，位於障礙物邊緣之外
                for point in closest_points:
                    new_point = generate_new_point_away_from_obstacle(point, obs, obstacle_size)
                    if constraint_function(new_point, obstacles, obstacle_size, safety_size):  # 確保新點安全
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
            if constraint_function(new_position, obstacles, obstacle_size, safety_size):
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
final_position = newton_method(start, goal, obstacles, obstacle_size, safety_size, initial_guess, original_path)
print("Final Position:", final_position)

plt.show()