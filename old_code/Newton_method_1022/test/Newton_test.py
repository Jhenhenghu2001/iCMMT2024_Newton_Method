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

# 計算梯度 (使用有限差分法)
def gradient(f, X, X_ori, W, h=1e-4):
    grad = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(2):  # X 是一個 (n, 2) 的數組，計算每個座標 (x, y) 的梯度
            e = np.zeros_like(X)
            e[i][j] = 1  # 單位向量 e_j，擾動第 i 個點的第 j 個座標
            grad[i, j] = (f(X + h * e, X_ori, W) - f(X, X_ori, W)) / h  # 使用前向差分法
    return grad

# 計算 Hessian 矩陣 (使用有限差分法)
def hessian(f, X, X_ori, W, h=1e-4):
    n, m = X.shape  # n 為變量數，m = 2 (每個變量有兩個座標)
    H = np.zeros((n, m, n, m))  # 初始化 Hessian 矩陣為 4 維數組
    for i in range(n):
        for j in range(m):
            for k in range(n):
                for l in range(m):
                    e_ij = np.zeros_like(X)
                    e_ij[i, j] = 1  # 單位向量 e_ij，擾動第 i 個點的第 j 個座標
                    e_kl = np.zeros_like(X)
                    e_kl[k, l] = 1  # 單位向量 e_kl，擾動第 k 個點的第 l 個座標
                    # 進行擾動來計算 Hessian
                    f_forward_ij_kl = f(X + h * e_ij + h * e_kl, X_ori, W)
                    f_forward_ij = f(X + h * e_ij, X_ori, W)
                    f_forward_kl = f(X + h * e_kl, X_ori, W)
                    f_current = f(X, X_ori, W)
                    # 計算二階導數的有限差分公式
                    H[i, j, k, l] = (f_forward_ij_kl - f_forward_ij - f_forward_kl + f_current) / (h ** 2)
    return H

#  計算H^(-1)
def inverse_hessian(H):
    det = np.linalg.det(H) # 計算行列式
    if det == 0:
        raise ValueError("Hessian 矩陣是奇異矩陣，不可逆。")
    H_inv = np.linalg.inv(H)
    return H_inv

# 牛頓法更新步驟
def newton_method(X_init, X_ori, W, tol=1e-3, max_iter=100):
    X = np.array(X_init, dtype=float)
    iter_num = 0
    X_prev = X 
    error = 100
    iter_num = 0
    max_iter=int(max_iter)
    for i in range(max_iter):
        iter_num += 1
        grad = gradient(objective_function, X, X_ori, W)
        grad_reshaped = grad.flatten()  # 展開梯度為一維向量 (也可寫成grad.reshape(-1))
        H = hessian(objective_function, X, X_ori, W)
        H_reshaped = H.reshape(len(X) * 2, len(X) * 2)  # 將 Hessian 轉換為 2D 矩陣
        H_inverse = inverse_hessian(H_reshaped) # H^(-1)
        # 更新X(i+1) = X(i) - H^(-1) * grad(X(i))
        X_new = np.array(X).reshape(-1) - np.dot(H_inverse, grad_reshaped)
        # 將更新結果轉回原始形狀
        X_new = X_new.reshape(X.shape)
        # 計算誤差：取當次迭代的位置與前一次迭代的位置的範數
        error = np.linalg.norm(X_new-X_prev)
        # 更新 X_prev 為當前的X_new，以便下一次計算誤差
        X_prev = X_new
        # 更新 X
        X = X_new
        # Converged Check: 判斷是否達到限制條件，若符合則回傳當前結果做為最佳解
        if error < tol:
            break
    X = np.flip(X, axis=1) # 將更新結果轉回原始形狀後，所有矩陣內部元素順序會顛倒，因此要反轉回來。
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
W = [1, 1, 1, 1, 1, 1, 1]  # weights

# 定義障礙物及其安全距離
obstacle = np.array([[0.5, 1.5], [2.5, 2.5]])  # 左下角和右上角座標
safety_size = 0.3  # 障礙物周圍安全範圍，例如 r=0.1 的圓形物件對應的安全距離

# 執行牛頓法並顯示輸出結果
optimal_X, iter_num = newton_method(X_init, X_origin, W, obstacle, safety_size)
print(f"Optimal solution: {optimal_X}")
print(f"Converged in {iter_num} iterations.")

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
visualize_grid(ax, start, goal, obstacles, circle_radius=circle_radius, original_path=initial_path, new_points=optimal_X)
plt.show()

#############################

# 淘汰程式
# 牛頓法更新步驟(這段程式是可以使用的，只是不包含使用Danger factor當作其中一項限制條件)
# def newton_method(X_init, X_ori, W, tol=1e-3, max_iter=100):
#     X = np.array(X_init, dtype=float)
#     print('Initial Guess ', X)
#     iter_num = 0
#     X_prev = X 
#     error = 100
#     for i in range(max_iter):
#         grad = gradient(objective_function, X, X_ori, W)
#         # print('grad', grad)
#         H = hessian(objective_function, X, X_ori, W)
#         # print('hessian', H)
#         H_reshaped = H.reshape(len(X) * 2, len(X) * 2)  # 將 Hessian 轉換為 2D 矩陣
#         grad_reshaped = grad.flatten()  # 展開梯度為一維向量
#         delta_X = np.linalg.solve(H_reshaped, -grad_reshaped)  # 進行牛頓法的更新步驟
#         delta_X = delta_X.reshape(X.shape)  # 將更新結果轉回原始形狀
#         X = X + delta_X
#         print('New X', X)
#         # 計算誤差：取當次迭代的位置與前一次迭代的位置的範數
#         error = np.linalg.norm(X-X_prev)
#         # 更新 X_prev 為當前的X，以便下一次計算誤差
#         X_prev = X
#         # Converged Check: 判斷是否達到限制條件，若符合則回傳當前結果做為最佳解
#         if np.linalg.norm(error) < tol:
#             break
#     iter_num = i+1
#     return X, iter_num

# 這版程式已經有包含Danger factor但尚未將其當成限制條件，但有設定當其中一點已經接近danger factor的閥值就凍結
# 繼續最佳化其他還距離障礙物較遠的點位(danger factor還很小、未到達設定之最大閥值)
# def newton_method(X_init, X_ori, W, obstacle, safety_size, tol=1e-3, max_iter=100):
#     X = np.array(X_init, dtype=float)
#     # print('Initial Guess ', X)
#     iter_num = 0
#     X_prev = X 
#     error = 100
#     frozen_points = np.zeros(len(X), dtype=bool)  # 用來追蹤哪些點已達到閥值

#     for i in range(max_iter):
#         grad = gradient(objective_function, X, X_ori, W)
#         H = hessian(objective_function, X, X_ori, W)
#         H_reshaped = H.reshape(len(X) * 2, len(X) * 2)  # 將 Hessian 轉換為 2D 矩陣
#         grad_reshaped = grad.flatten()  # 展開梯度為一維向量
        
#         # 若某點危險因子過高，則將其對應的梯度設為 0 以凍結該點
#         for idx, point in enumerate(X):
#             danger_factor = calculate_danger_factor(point, obstacle, safety_size)
#             if danger_factor >= 0.4:
#                 frozen_points[idx] = True
#                 grad_reshaped[idx * 2: (idx + 1) * 2] = 0  # 將對應點的梯度設為 0
        
#         delta_X = np.linalg.solve(H_reshaped, -grad_reshaped)  # 進行牛頓法的更新步驟
#         delta_X = delta_X.reshape(X.shape)  # 將更新結果轉回原始形狀

#         # 只更新未凍結的點
#         for idx, is_frozen in enumerate(frozen_points):
#             if not is_frozen:
#                 X[idx] = X[idx] + delta_X[idx]
        
#         # 檢查是否所有點都達到了閥值或誤差已經收斂
#         if all(frozen_points) and np.linalg.norm(X - X_prev) < tol:
#             break
        
#         # 更新 X_prev 為當前的X，以便下一次計算誤差
#         X_prev = X

#     iter_num = i + 1
#     return X, iter_num

# 這版程式除了包含Danger Factor另外加入project_out_of_obstacle把最佳化之後進入障礙物範圍內的點位移出去
# 但目前尚未有成效。
# 牛頓法更新步驟
# def newton_method(X_init, X_ori, W, obstacle, safety_size, tol=1e-3, max_iter=100):
#     X = np.array(X_init, dtype=float)
#     iter_num = 0
#     X_prev = X
#     frozen_points = np.zeros(len(X), dtype=bool)  # 用來追蹤哪些點已達到閥值

#     for i in range(max_iter):
#         grad = gradient(objective_function, X, X_ori, W)
#         H = hessian(objective_function, X, X_ori, W)
#         H_reshaped = H.reshape(len(X) * 2, len(X) * 2)  # 將 Hessian 轉換為 2D 矩陣
#         grad_reshaped = grad.flatten()  # 展開梯度為一維向量

#         # 若某點危險因子過高，則將其對應的梯度設為 0 以凍結該點
#         for idx, point in enumerate(X):
#             danger_factor = calculate_danger_factor(point, obstacle, safety_size)
#             if danger_factor >= 0.4:
#                 frozen_points[idx] = True
#                 grad_reshaped[idx * 2: (idx + 1) * 2] = 0  # 將對應點的梯度設為 0

#         delta_X = np.linalg.solve(H_reshaped, -grad_reshaped)  # 進行牛頓法的更新步驟
#         delta_X = delta_X.reshape(X.shape)  # 將更新結果轉回原始形狀

#         # 只更新未凍結的點
#         for idx, is_frozen in enumerate(frozen_points):
#             if not is_frozen:
#                 X[idx] = X[idx] + delta_X[idx]
#                 # 檢查是否進入障礙物範圍，如果是則進行修正
#                 if calculate_danger_factor(X[idx], obstacle, safety_size) >= 0.4:
#                     X[idx] = project_out_of_obstacle(X[idx], obstacle, safety_size)

#         # 檢查是否所有點都達到了閥值且誤差已經收斂
#         if all(frozen_points) and np.linalg.norm(X - X_prev) < tol:
#             break

#         # 更新 X_prev 為當前的X，以便下一次計算誤差
#         X_prev = X

#     iter_num = i + 1
#     return X, iter_num



# def newton_method(X_init, X_ori, W, obstacle, safety_size, tol=1e-3, max_iter=100):
#     X = np.array(X_init, dtype=float)
#     iter_num = 0
#     frozen_points = np.zeros(len(X), dtype=bool)  # 用來追蹤哪些點已達到閥值

#     for i in range(max_iter):
#         grad = gradient(objective_function, X, X_ori, W)
#         H = hessian(objective_function, X, X_ori, W)
#         H_reshaped = H.reshape(len(X) * 2, len(X) * 2)  # 將 Hessian 轉換為 2D 矩陣
#         grad_reshaped = grad.flatten()  # 展開梯度為一維向量

#         # 若某點危險因子過高，則將其對應的梯度設為 0 以凍結該點
#         for idx, point in enumerate(X):
#             danger_factor = calculate_danger_factor(point, obstacle, safety_size)
#             if danger_factor >= 0.4:
#                 frozen_points[idx] = True
#                 # 將點移到障礙物外
#                 X[idx] = project_out_of_obstacle(X[idx], obstacle, safety_size)
#                 # 點位移出後重新計算梯度與 Hessian，允許該點繼續參與優化
#                 grad = gradient(objective_function, X, X_ori, W)
#                 H = hessian(objective_function, X, X_ori, W)
#                 H_reshaped = H.reshape(len(X) * 2, len(X) * 2)
#                 grad_reshaped = grad.flatten()

#         delta_X = np.linalg.solve(H_reshaped, -grad_reshaped)  # 進行牛頓法的更新步驟
#         delta_X = delta_X.reshape(X.shape)  # 將更新結果轉回原始形狀

#         # 只更新未凍結的點
#         for idx, is_frozen in enumerate(frozen_points):
#             if not is_frozen:
#                 X[idx] = X[idx] + delta_X[idx]
#                 # 再次檢查是否進入障礙物範圍，如果是則進行修正並重新檢查
#                 if calculate_danger_factor(X[idx], obstacle, safety_size) >= 0.4:
#                     X[idx] = project_out_of_obstacle(X[idx], obstacle, safety_size)
#                     # 再次重新計算 Hessian 和梯度，讓最佳化可以繼續
#                     grad = gradient(objective_function, X, X_ori, W)
#                     H = hessian(objective_function, X, X_ori, W)
#                     H_reshaped = H.reshape(len(X) * 2, len(X) * 2)
#                     grad_reshaped = grad.flatten()

#         delta_X = np.linalg.solve(H_reshaped, -grad_reshaped)  # 進行牛頓法的更新步驟
#         delta_X = delta_X.reshape(X.shape)  # 將更新結果轉回原始形狀

#         # 檢查是否所有點都已收斂或凍結，且更新幅度小於容許誤差
#         if all(frozen_points) or np.linalg.norm(delta_X) < tol:
#             break

#     iter_num = i + 1
#     return delta_X, iter_num

# # 計算危險因子
# def calculate_danger_factor(point, obstacle, safety_size):
#     bottom_left, top_right = obstacle
#     x, y = point

#     # 計算物件與障礙物最近邊界的距離
#     dx = max(bottom_left[0] - x, 0, x - top_right[0])
#     dy = max(bottom_left[1] - y, 0, y - top_right[1])
#     distance_to_obstacle = np.sqrt(dx**2 + dy**2)
    
#     # 計算危險因子
#     if distance_to_obstacle < safety_size:
#         danger_factor = 1.0  # 危險因子為1表示在安全距離內
#     else:
#         danger_factor = 0.5 * (2 * 0.1 + 0.1) / distance_to_obstacle  # 依據距離遞減的危險因子
#     return danger_factor

# # 邊界修正函數(當有最佳化點位進入障礙物(含安全範圍)內時，需要將其重新移出範圍外)
# def project_out_of_obstacle(point, obstacle, safety_size):
#     bottom_left, top_right = obstacle
#     x, y = point

#     # 判斷點是否在障礙物的邊界範圍內（包含安全距離）
#     if (bottom_left[0] - safety_size <= x <= top_right[0] + safety_size and
#         bottom_left[1] - safety_size <= y <= top_right[1] + safety_size):
        
#         # 點在障礙物或其安全範圍內，根據最短距離將點移出
#         if x < bottom_left[0]:
#             x = bottom_left[0] - safety_size  # 移到障礙物左邊
#         elif x > top_right[0]:
#             x = top_right[0] + safety_size  # 移到障礙物右邊
        
#         if y < bottom_left[1]:
#             y = bottom_left[1] - safety_size  # 移到障礙物下方
#         elif y > top_right[1]:
#             y = top_right[1] + safety_size  # 移到障礙物上方

#     return np.array([x, y])