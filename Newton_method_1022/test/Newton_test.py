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

# 牛頓法更新步驟
def newton_method(X_init, X_ori, W, tol=1e-3, max_iter=100):
    X = np.array(X_init, dtype=float)
    iter_num = 0
    X_prev = X 
    error = 100
    for i in range(max_iter):
        grad = gradient(objective_function, X, X_ori, W)
        print('grad', grad)
        H = hessian(objective_function, X, X_ori, W)
        print('hessian', H)
        H_reshaped = H.reshape(len(X) * 2, len(X) * 2)  # 將 Hessian 轉換為 2D 矩陣
        grad_reshaped = grad.flatten()  # 展開梯度為一維向量
        delta_X = np.linalg.solve(H_reshaped, -grad_reshaped)  # 進行牛頓法的更新步驟
        delta_X = delta_X.reshape(X.shape)  # 將更新結果轉回原始形狀
        X = X + delta_X
        print('New X', X)
        # 計算誤差：取當次迭代的位置與前一次迭代的位置的範數
        error = np.linalg.norm(X-X_prev)
        # 更新 X_prev 為當前的X，以便下一次計算誤差
        X_prev = X
        # Converged Check: 判斷是否達到限制條件，若符合則回傳當前結果做為最佳解
        if np.linalg.norm(error) < tol:
            break
    iter_num = i+1
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

# 執行牛頓法並顯示輸出結果
optimal_X, iter_num = newton_method(X_init, X_origin, W)
print(f"Optimal solution: {optimal_X}")
print(f"Converged in {iter_num} iterations.")

#############################

# 可視化
def visualize_grid(ax, start, goal, obstacles, circle_radius=None, positions=None, original_path=None, path=None, new_points=None, closest_points=None):
    ax.cla()  # 清除當前的圖表，保留窗口
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')  # 起點
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')  # 終點

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
safety_size = 0.2
initial_path = generate_initial_path(start, goal, waypoint_distance)
obstacles = [
    (np.array([0.5, 1.5]), np.array([2.5, 2.5])), # 第一個矩形障礙物(左下角座標, 右上角座標)
    (np.array([4.5, 3.5]), np.array([5.5, 5.5]))  # 第二個矩形障礙物(左下角座標, 右上角座標)
]
visualize_grid(ax, start, goal, obstacles, circle_radius=circle_radius, original_path=initial_path, new_points=X_init)
plt.show()