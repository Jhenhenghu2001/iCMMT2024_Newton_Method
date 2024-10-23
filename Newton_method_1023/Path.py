import numpy as np
from Obstacle_detection import point_in_obstacle

# 生成初始路徑
def generate_initial_path(start, goal, distance):
    # 生成從起點到終點的路徑，並在路徑上添加點
    num_points = int(np.linalg.norm(goal - start) / distance)
    initial_path = np.linspace(start, goal, num=num_points + 1)
    return initial_path

# 計算路徑的總長度
def calculate_path_length(path):
    return np.sum([np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))])

# 將經過障礙物上的點位儲存成避障前原始路徑
def path_before_obstacle_avoidance(path, obstacle, safety_size):
    points_within_obstacle = []  # 用於儲存在障礙物內的點
    indices_of_closest_points = []  # 用於儲存 closest_points 的索引值

    # 檢查路徑上每個點是否在擴展過的障礙物區域內
    for p in path:
        if point_in_obstacle(p, obstacle, safety_size):
            points_within_obstacle.append(p)
    # print('path', path)
    if not points_within_obstacle:
        return [], []  # 如果沒有點在障礙物內，返回空列表和空索引

    # 找到最早和最晚穿過障礙物的點
    first_obstacle_point = points_within_obstacle[0]
    last_obstacle_point = points_within_obstacle[-1]

    # 若只有一個點經過障礙物，則直接返回該點，不做處理
    if len(points_within_obstacle) == 1:
        first_idx = np.where(np.all(path == first_obstacle_point, axis=1))[0][0]
        return [first_obstacle_point], [first_idx]  # 返回點和其索引
    
    # 計算初始和結束點的平均值作為中間點
    middle_point = (first_obstacle_point + last_obstacle_point) / 2
    # 更新points_within_obstacle，保留初始點、中間點、結束點
    points_within_obstacle = [first_obstacle_point, middle_point, last_obstacle_point]

    # 找到這些點在路徑中的索引
    first_idx = np.where(np.all(path == first_obstacle_point, axis=1))[0][0]
    last_idx = np.where(np.all(path == last_obstacle_point, axis=1))[0][0]

    # 將 first_idx 和 last_idx 之間的所有點的索引加入到 indices_of_closest_points
    indices_of_closest_points.extend(range(first_idx, last_idx + 1))

    # 獲取與這些障礙物點相鄰的點
    closest_points = []
    if first_idx > 0:
        closest_points.append(path[first_idx - 1])  # 碰撞起始點的前一點
        indices_of_closest_points.insert(0, first_idx - 1)  # 儲存相鄰索引
    closest_points.extend(points_within_obstacle)  # 初始點、中間點和結束點
    if last_idx < len(path) - 1:
        closest_points.append(path[last_idx + 1])  # 碰撞終點的下一點
        indices_of_closest_points.append(last_idx + 1)  # 儲存相鄰索引
    
    return closest_points, indices_of_closest_points  # 返回所有障礙物點及其相鄰點的索引

#  將初始路徑上的碰撞點偏移出障礙物(含安全距離)的範圍行程新路徑，作為避障路徑最佳化的Initial Guess。
def generate_new_path(closest_points, obstacles, safety_size, offset_distance):
    new_paths = []  # 用於儲存所有障礙物的避障路徑
    offset_directions = []  # 儲存每個障礙物的偏移方向

    if closest_points:  # 確保有碰撞點
        tangent_vector = closest_points[-1] - closest_points[0]
        tangent_vector /= np.linalg.norm(tangent_vector)  # 標準化切線向量

        normal_vector_left = np.array([-tangent_vector[1], tangent_vector[0]])
        normal_vector_right = np.array([tangent_vector[1], -tangent_vector[0]])

        # 排除首尾點位，只考慮中間的點來判斷偏移方向
        for point in closest_points: # 排除第一個和最後一個點
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
        # print('offset_directions', offset_directions)
        most_common_direction = max(set(offset_directions), key=offset_directions.count)

        new_path = []
        for point in closest_points[1:-1]: # 排除首尾點位，只考慮中間的碰撞點進行偏移
            new_point = point + np.array(most_common_direction) * offset_distance
            new_path.append(new_point)

        # new_paths.append(np.array(new_path))
        new_paths.append(new_path)

    return new_paths  # 返回經過障礙物的碰撞點偏移過後的新路徑(不含未經過障礙物的碰撞起始和結束點位)

def update_origin_path(origin_path, X_opt_final, indices_of_closest_points):
    # 0. 複製原始路徑陣列到一個新陣列使用
    new_origin_path = origin_path.copy()
    # 1. 將原始路徑新陣列刪除掉經過障礙物點位(含後面全部點位)
    end = len(origin_path)
    start = indices_of_closest_points[1]
    new_index_list = [] # 建立需要刪除的索引位置清單
    for i in range(start, end):
        new_index_list.append(i)
        i+=1
    for i in sorted(new_index_list, reverse=True):
        new_origin_path = np.delete(new_origin_path, i, axis=0)
    # 2. 將原始路徑新陣列後方加上最佳化之後的新路徑點位
    # 先把X_opt_final轉換形狀(1, 3, 2)to(3, 2)
    X_opt_final = X_opt_final.reshape(3, 2)
    new_origin_path = np.vstack((new_origin_path, X_opt_final))
    # 3. 將原先後方不在障礙物範圍的點位重新加入到原始路徑新陣列
    new_origin_path = np.vstack((new_origin_path, origin_path[indices_of_closest_points[-1]:]))

    return new_origin_path