# import numpy as np
# import matplotlib.pyplot as plt
# import time  

# obstacles = [
#     (np.array([0.5, 1.5]), np.array([2.5, 2.5])),  # 第一個矩形障礙物
#     (np.array([4.5, 3.5]), np.array([5.5, 5.5]))  # 第二個矩形障礙物
# ] 

# # 檢查點是否在障礙物的正方形區域內
# def is_point_in_square(point, obstacle , safety_size):
#     square_bottom_left = obstacle[0]
#     square_top_right = obstacle[1]
#     return (square_bottom_left[0] - safety_size <= point[0] <= square_top_right[0] + safety_size and
#             square_bottom_left[1] - safety_size <= point[1] <= square_top_right[1] + safety_size)

# point = np.array([2.0, 2.0])
# safety_size = 0.3

# # 檢查點是否在任何一個障礙物範圍內
# point_in_obstacle = False  # 預設為 False

# for obs in obstacles:
#     if is_point_in_square(point, obs, safety_size):
#         point_in_obstacle = True  # 如果點在障礙物內，設置為 True
#         break  # 找到一個障礙物後，可以直接退出迴圈

# if point_in_obstacle:
#     print("Point is inside an obstacle.")
# else:
#     print("Point is outside all obstacles.")

# Draw Danger Factor

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定義網格範圍
x = np.linspace(0, 1000, 200)  # X 軸範圍
y = np.linspace(0, 1000, 200)  # Y 軸範圍
X, Y = np.meshgrid(x, y)

# 定義矩形範圍，以 (500,500) 為中心
rect_x_min = 450  # 矩形X的最小值
rect_x_max = 550  # 矩形X的最大值
rect_y_min = 450  # 矩形Y的最小值
rect_y_max = 550  # 矩形Y的最大值

# 初始化危險因子矩陣
danger_factor = np.zeros(X.shape)

# 矩形內的危險因子設為1
inside_rect = (X >= rect_x_min) & (X <= rect_x_max) & (Y >= rect_y_min) & (Y <= rect_y_max)
danger_factor[inside_rect] = 1

# 定義矩形外的危險因子隨距離遞減
for i in range(len(x)):
    for j in range(len(y)):
        if not inside_rect[i, j]:
            # 計算到最近矩形邊界的距離
            dx = max(rect_x_min - X[i, j], 0, X[i, j] - rect_x_max)
            dy = max(rect_y_min - Y[i, j], 0, Y[i, j] - rect_y_max)
            dist_to_rect = np.sqrt(dx**2 + dy**2)
            
            # 危險因子根據距離遞減
            danger_factor[i, j] = max(1 - dist_to_rect / 50, 0)  # 50 為調整危險因子衰減的參數

# 繪製立體圖
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, danger_factor, cmap='jet')

# 設置軸標籤
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Danger Factor')
# plt.title('危險因子三維即視圖')

# 顯示顏色條
plt.colorbar(ax.plot_surface(X, Y, danger_factor, cmap='jet'), ax=ax, shrink=0.5, aspect=5)

# 顯示圖像
plt.show()
