import numpy as np
import matplotlib.pyplot as plt
import time
from Obstacle_detection import point_in_obstacle, line_intersects_any_obstacle, find_closest_obstacle
from GUI import visualize_grid
from Path import generate_initial_path, path_before_obstacle_avoidance, generate_new_path, calculate_path_length
from Newton_opt import newton_method

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
# 建立初始路徑的複製列表以避免行徑時因避障而導致修改到initial_path的路徑。
# 這個列表可以任意修改裡面的點位，把新的避障路徑替換掉原本路徑上經過障礙物的點位。
origin_path = initial_path.copy() 

# 建立點P並初始化位置
current_position_index = 0
current_position = initial_path[current_position_index]
full_path_traveled = [current_position]  # 紀錄點P走過的完整路徑
visited_obstacles = []  # 初始化已經經過的障礙物列表

# 視覺化初始化
visualize_grid(ax, start, goal, obstacles, circle_radius=circle_radius, positions=current_position, original_path=initial_path)

# 主流程
while np.linalg.norm(current_position - goal) > 0.1:
    
    # 獲取下一個目標點
    next_position = origin_path[current_position_index + 1]

    # 檢查當前點到下一個點的線段是否經過障礙物或安全區域
    if line_intersects_any_obstacle(current_position, next_position, obstacles, safety_distance):
        # 找到最近的障礙物，並準備進行避障
        closest_obstacle = find_closest_obstacle(current_position, obstacles, visited_obstacles)
        
        # 如果找到了最近的障礙物，則將其加入 visited_obstacles
        if closest_obstacle is not None:
            visited_obstacles.append(closest_obstacle)

        # 找到初始路徑中所有經過最近障礙物的點
        points_on_obstacle = path_before_obstacle_avoidance(origin_path, closest_obstacle, safety_distance)

        # 根據避障策略生成新的路徑，將避障點設為 initial guess
        new_path, indices_of_closest_points = generate_new_path(points_on_obstacle, closest_obstacle, safety_distance, offset_distance=1.0)

        # 使用牛頓法進行路徑最佳化
        optimized_path, iter_num = newton_method(new_path, points_on_obstacle, W)

        # 確認新路徑是否避開障礙物
        X_opt_final = [points_on_obstacle[0]] + list(optimized_path) + [points_on_obstacle[-1]]
        for i in range(len(X_opt_final) - 1):
            if line_intersects_any_obstacle(X_opt_final[i], X_opt_final[i+1], obstacles, safety_distance):
                # 若新路徑有經過障礙物，將當前解(optimized_path)作為 initial guess 重新最佳化
                optimized_path, iter_num = newton_method(optimized_path, points_on_obstacle, W)
                X_opt_final = [points_on_obstacle[0]] + list(optimized_path) + [points_on_obstacle[-1]]
        
        # 以上程式結束後所得到的X_opt_final是最佳化路徑結果(包含避障起始點和避障結束點)
        # origin_path 


        # # 將當前位置沿著新的路徑移動，更新current_position
        # origin_path = X_opt_final  # 更新路徑
        # # current_position_index = 0  # 重新從起點開始走
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