import numpy as np
import matplotlib.pyplot as plt
import time  

obstacles = [
    (np.array([0.5, 1.5]), np.array([2.5, 2.5])),  # 第一個矩形障礙物
    (np.array([4.5, 3.5]), np.array([5.5, 5.5]))  # 第二個矩形障礙物
] 

# 檢查點是否在障礙物的正方形區域內
def is_point_in_square(point, obstacle , safety_size):
    square_bottom_left = obstacle[0]
    square_top_right = obstacle[1]
    return (square_bottom_left[0] - safety_size <= point[0] <= square_top_right[0] + safety_size and
            square_bottom_left[1] - safety_size <= point[1] <= square_top_right[1] + safety_size)

point = np.array([2.0, 2.0])
safety_size = 0.3

# 檢查點是否在任何一個障礙物範圍內
point_in_obstacle = False  # 預設為 False

for obs in obstacles:
    if is_point_in_square(point, obs, safety_size):
        point_in_obstacle = True  # 如果點在障礙物內，設置為 True
        break  # 找到一個障礙物後，可以直接退出迴圈

if point_in_obstacle:
    print("Point is inside an obstacle.")
else:
    print("Point is outside all obstacles.")