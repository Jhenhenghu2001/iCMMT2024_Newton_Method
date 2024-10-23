import numpy as np
from GUI import visualize_grid
from Path import generate_initial_path
import matplotlib.pyplot as plt

def generate_new_path(closest_points, obstacles, safety_size, offset_distance):
    new_paths = []  # 用於儲存所有障礙物的避障路徑
    offset_directions = []  # 儲存每個障礙物的偏移方向

    if closest_points.size > 0:  # 確保有碰撞點
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

    return new_paths  # 返回經過障礙物的碰撞點偏移過後的新路徑(不含位經過障礙物的碰撞起始和結束點位)

fig, ax = plt.subplots()
start = np.array([0, 0])
goal = np.array([10, 10])
circle_radius = 0.1
waypoint_distance = 0.3
initial_path = generate_initial_path(start, goal, waypoint_distance)
closest_points = np.array([[1.27659574, 1.27659574], [1.4893617, 1.4893617], [2.0212766, 2.0212766], [2.55319149, 2.55319149], [2.76595745, 2.76595745]])
obs = np.array([[0.5, 1.5], [2.5, 2.5]])
obstacles = [
    (np.array([0.5, 1.5]), np.array([2.5, 2.5])), # 第一個矩形障礙物
    (np.array([4.5, 3.5]), np.array([5.5, 5.5]))  # 第二個矩形障礙物
]  # 障礙物列表
safety_size = 0.2
new_path = generate_new_path(closest_points, obs, safety_size, offset_distance=1.0)
print('new_path', new_path)
# new_path [[array([2.19646848, 0.78225492]), array([2.72838338, 1.31416982]), array([3.26029827, 1.84608471])]]
visualize_grid(ax, start, goal, obstacles, circle_radius=circle_radius, original_path=initial_path, new_points=new_path)