import numpy as np
import matplotlib.pyplot as plt

# 定義目標函數
def objective_function(X, X_ori, W):
    X1, X2, X3 = X[0], X[1], X[2]  # 解包 X 為 X1*, X2*, X3*
    XT, X1_ori, X2_ori, X3_ori, XH = X_ori[0], X_ori[1], X_ori[2], X_ori[3], X_ori[4]  # 解包 X_ori 為 XT, X1_ori, X2_ori, X3_ori, XH
    w1, w2, w3, wH, wT, w21, w32 = W  # 權重

    # 計算目標函數值
    f = (w1 * np.linalg.norm(X1 - X1_ori)**2 +
         w2 * np.linalg.norm(X2 - X2_ori)**2 +
         w3 * np.linalg.norm(X3 - X3_ori)**2 +
         wH * np.linalg.norm(X1 - XH)**2 +
         wT * np.linalg.norm(X3 - XT)**2 +
         w21 * np.linalg.norm(X2 - X1)**2 +
         w32 * np.linalg.norm(X3 - X2)**2)
    return f

# 使用 finite difference 計算梯度
def gradient_finite_difference(f, X, X_ori, W, delta=0.01):
    grad = np.zeros_like(X)
    for i in range(len(X)):
        X_forward = np.copy(X)
        X_forward[i] += delta
        grad[i] = (f(X_forward, X_ori, W) - f(X, X_ori, W)) / delta
    return grad

# 使用 finite difference 計算 Hessian 矩陣
def hessian_finite_difference(f, X, X_ori, W, delta=0.01):
    n = len(X)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            X_ij = np.copy(X)
            X_ij[i] += delta
            X_ij[j] += delta
            f_ij = f(X_ij, X_ori, W)
            
            X_i = np.copy(X)
            X_i[i] += delta
            f_i = f(X_i, X_ori, W)
            
            X_j = np.copy(X)
            X_j[j] += delta
            f_j = f(X_j, X_ori, W)
            
            f_0 = f(X, X_ori, W)
            hessian[i, j] = (f_ij - f_i - f_j + f_0) / (delta**2)
    return hessian

# 牛頓法最佳化
def newton_method(X_init, X_ori, W, obstacle, safety_size, tol=1e-3, delta=0.01):
    X = np.copy(X_init)
    iter_num = 0
    error = 100

    while error > tol :
        iter_num += 1
        
        # 計算梯度和 Hessian
        grad = gradient_finite_difference(objective_function, X, X_ori, W, delta)
        hess = hessian_finite_difference(objective_function, X, X_ori, W, delta)
        
        # 逆 Hessian 矩陣
        H_inv = np.linalg.inv(hess)
        
        # 更新 X
        X_new = X - np.dot(H_inv, grad)
        
        # 計算誤差
        error = np.linalg.norm(X_new - X)
        print(f'Iter = {iter_num}, X = {X}, error = {error}')

        # 更新 X
        X = X_new
    X = np.flip(X, axis=0) # 將更新結果轉回原始形狀後，所有矩陣內部元素順序會顛倒，因此要反轉回來。
    return X, iter_num

# 計算危險因子 Danger Factor
def danger_factor(X, obstacle, safety_size=0.2, object_radius=0.1):
    # X 是物體的中心座標 (假設 X[0], X[1] 是物體的 X 和 Y 位置)
    # obstacle 由左下角和右上角座標組成 [[x_min, y_min], [x_max, y_max]]
    
    obj_x, obj_y = X[0], X[1]
    obs_x_min, obs_y_min = obstacle[0][0], obstacle[0][1]
    obs_x_max, obs_y_max = obstacle[1][0], obstacle[1][1]

    # 計算物體與矩形障礙物邊界的最小距離，考慮安全距離
    dx = np.maximum(obs_x_min - (obj_x + object_radius), (obj_x - object_radius) - obs_x_max)
    dy = np.maximum(obs_y_min - (obj_y + object_radius), (obj_y - object_radius) - obs_y_max)
    
    dist_to_obstacle = np.sqrt(dx**2 + dy**2)

    # 危險因子根據距離遞減，危險因子範圍為 [0, 1]
    # 當距離小於 safety_size 時，危險因子最大為 1
    if np.all(dist_to_obstacle <= safety_size):
        return 1
    elif np.all(dist_to_obstacle >= safety_size + 0.5):
        return 0
    else:
        return max(1 - (np.min(dist_to_obstacle) - safety_size) / 0.5, 0)

def newton_method_with_partial_freeze(X_init, X_ori, W, obstacle, safety_size=0.2, object_radius=0.1, tol=1e-3, delta=0.001):
    X = np.copy(X_init)
    iter_num = 0
    error = 100
    freeze_points = np.zeros(len(X), dtype=bool)  # 初始化所有點為未凍結

    while error > tol:
        iter_num += 1
        X_new = np.copy(X)

        # 對每個點計算梯度、Hessian 及更新
        for i in range(len(X)):
            if not freeze_points[i]:  # 若該點未凍結，則進行優化更新
                # 不再單獨處理每個點的資料，依然傳遞完整的 X 集合
                grad = gradient_finite_difference(objective_function, X, X_ori, W, delta)
                hess = hessian_finite_difference(objective_function, X, X_ori, W, delta)

                # 逆 Hessian 矩陣
                H_inv = np.linalg.inv(hess)

                # 更新 X[i]
                X_new[i] = X[i] - np.dot(H_inv, grad)[i].flatten()  # 只更新 X[i]

                # 計算危險因子
                df = danger_factor(X_new[i], obstacle, safety_size, object_radius)

                # 如果危險因子過高，凍結該點
                if df >= 0.5:
                    freeze_points[i] = True
                    X_new[i] = X[i]  # 恢復成原來的點，停止更新
                    print(f"Point {i} frozen. Danger factor = {df}")

        # 計算誤差（只計算未凍結點的誤差）
        error = np.linalg.norm(X_new[~freeze_points] - X[~freeze_points])
        print(f'Iter = {iter_num}, X = {X}, error = {error}')

        # 更新 X
        X = X_new

        # 如果所有點都凍結，則終止迭代
        if np.all(freeze_points):
            print("All points are frozen. Stopping optimization.")
            break
    
    return X, iter_num

# 輸入參數內容設定
X_init = np.array([[2.13567821, 0.72146465], 
                   [2.6713925, 1.25717893], 
                   [3.20710678, 1.79289322]])  # initial guess
X_origin = np.array([[1.07142857, 1.07142857], 
                     [1.42857143, 1.42857143], 
                     [1.96428571, 1.96428571], 
                     [2.5, 2.5], 
                     [2.85714286, 2.85714286]])  # XT, X1_ori, X2_ori, X3_ori, XH
W = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # weights

# 定義障礙物及其安全距離
obstacle = np.array([[0.5, 1.5], [2.5, 2.5]])  # 左下角和右上角座標
safety_size = 0.3  # 障礙物周圍安全範圍，例如 r=0.1 的圓形物件對應的安全距離

# 執行牛頓法並顯示輸出結果
# print('X_init type', type(X_init))
# print('X_init size', np.shape(X_init))
# optimal_X, iter_num = newton_method(X_init, X_origin, W, obstacle, safety_size)
# print(f"Optimal solution: {optimal_X}")
# print(f"Converged in {iter_num} iterations.")
# optimal_X_2, iter_num_2 = newton_method_with_danger_check(X_init, X_origin, W, obstacle)
optimal_X_2, iter_num_2 = newton_method_with_partial_freeze(X_init, X_origin, W, obstacle)
print(f"Optimal solution 2: {optimal_X_2}")
print(f"Converged in {iter_num_2} iterations.")

#############################

# 可視化
def visualize_grid(ax, start, goal, obstacles, circle_radius=None, positions=None, original_path=None, path=None, new_points=None, closest_points=None):
    ax.cla()  # 清除當前的圖表，保留窗口
    ax.plot(start[0], start[1], 'go', markersize=5, label='Start')  # 起點
    ax.plot(goal[0], goal[1], 'ro', markersize=5, label='Goal')  # 終點

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

    # 當前位置點位的圓形，將 positions 列表轉換成 NumPy 陣列
    if positions is not None:
        positions = np.array(positions)  # 將 positions 列表轉換成 NumPy 陣列
        for pos in positions:
            ax.plot(positions[0], positions[1], 'bo', label='Recent Point')

    # 新生成的避障路徑
    if new_points is not None:
        new_points = np.array(new_points)  # 將 new_points 轉換為 NumPy 陣列
        for i, point in enumerate(new_points):  # 遍歷 new_points 中的每個點
            ax.plot(point[0], point[1], 'm*', label='Avoidance Path' if i == 0 else "")
    
    # 路徑的圓形
    if path is not None:
        path = np.array(path)
        for pos in path:
            circle = plt.Circle(pos, radius=circle_radius, color='k', fill=True)
            ax.add_patch(circle)
        ax.plot(path[:, 0], path[:, 1], 'k-', label='full_path_traveled')

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

# 生成初始路徑
def generate_initial_path(start, goal, distance):
    # 生成從起點到終點的路徑，並在路徑上添加點
    num_points = int(np.linalg.norm(goal - start) / distance)
    initial_path = np.linspace(start, goal, num=num_points + 1)
    return initial_path

# Visualization
fig, ax = plt.subplots()
start = np.array([0, 0])
goal = np.array([10, 10])
circle_radius = 0.1
waypoint_distance = 0.3
initial_path = generate_initial_path(start, goal, waypoint_distance)
obstacles = [
    (np.array([0.5, 1.5]), np.array([2.5, 2.5])), # 第一個矩形障礙物(左下角座標, 右上角座標)
    (np.array([4.5, 3.5]), np.array([5.5, 5.5]))  # 第二個矩形障礙物(左下角座標, 右上角座標)
]
visualize_grid(ax, start, goal, obstacles, circle_radius=circle_radius, original_path=initial_path, new_points=optimal_X_2)
plt.show()

#######################

# Not use
# 修正牛頓法的主程式，加入危險因子檢查
# def newton_method_with_danger_check(X_init, X_ori, W, obstacle, safety_size=0.2, object_radius=0.1, tol=1e-3, delta=0.001):
#     X = np.copy(X_init)
#     iter_num = 0
#     error = 100

#     while error > tol :
#         iter_num += 1
        
#         # 計算梯度和 Hessian
#         grad = gradient_finite_difference(objective_function, X, X_ori, W, delta)
#         hess = hessian_finite_difference(objective_function, X, X_ori, W, delta)
        
#         # 逆 Hessian 矩陣
#         H_inv = np.linalg.inv(hess)
        
#         # 更新 X
#         X_new = X - np.dot(H_inv, grad)
        
#         # 計算危險因子
#         df = danger_factor(X_new, obstacle, safety_size, object_radius)
        
#         # 假設危險因子大於或等於 0.5，則停止優化（碰撞區間）
#         if df >= 0.5:
#             print(f"Warning: Danger factor too high! DF = {df}")
#             break
        
#         # 計算誤差
#         error = np.linalg.norm(X_new - X)
#         print(f'Iter = {iter_num}, X = {X}, error = {error}, Danger Factor = {df}')

#         # 更新 X
#         X = X_new
    
#     return X, iter_num
