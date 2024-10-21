import numpy as np

# 判斷「點位」是否在擴展過的矩形內(障礙物含安全距離)
def point_in_obstacle(point, obstacle, safety_size):
    bottom_left, top_right = obstacle
    # 增加障礙物周圍的安全距離，同時考慮點位的半徑
    expanded_bottom_left = bottom_left - np.array([safety_size, safety_size])
    expanded_top_right = top_right + np.array([safety_size, safety_size])
    # 檢查點是否在擴展過的矩形內
    value = (expanded_bottom_left[0] <= point[0] <= expanded_top_right[0] and
            expanded_bottom_left[1] <= point[1] <= expanded_top_right[1])
    return value

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

# 尋找最接近當前點位的障礙物
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
    # print(type(obstacle_list))
    # print(type(visited_obstacles))
    closest_obstacle = None
    min_distance = float('inf')
    
    for obstacle in obstacle_list:
        if any(np.array_equal(obstacle, visited) for visited in visited_obstacles):
            continue  # 跳過已經訪問過的障礙物
        # 計算當前位置到障礙物中心的距離
        print('obs ', obstacle)
        bottom_left, top_right = obstacle
        obstacle_center = (bottom_left + top_right) / 2
        distance = np.linalg.norm(current_position - obstacle_center)
        
        # 更新最近的障礙物和最小距離
        if distance < min_distance:
            min_distance = distance
            closest_obstacle = obstacle
    return closest_obstacle