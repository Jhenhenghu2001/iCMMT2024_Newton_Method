import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
# from GUI import visualize_grid
# from Path import generate_initial_path



# 定義變數與參數
x1, y1 = sp.symbols('x1 y1')  # X1*
x2, y2 = sp.symbols('x2 y2')  # X2*
x3, y3 = sp.symbols('x3 y3')  # X3*

# 已知參數
x1_ori, y1_ori = sp.symbols('x1_ori y1_ori')
x2_ori, y2_ori = sp.symbols('x2_ori y2_ori')
x3_ori, y3_ori = sp.symbols('x3_ori y3_ori')
xH, yH = sp.symbols('xH yH')
xT, yT = sp.symbols('xT yT')

# 權重參數
w1, w2, w3, wH, wT, w21, w32 = sp.symbols('w1 w2 w3 wH wT w21 w32')

# 目標函數
def objective_function(X, W):
    x1, y1, x2, y2, x3, y3 = X
    w1, w2, w3, wH, wT, w21, w32 = W[0], W[1], W[2], W[3], W[4], W[5], W[6]
    f = (w1 * sp.sqrt((x1 - x1_ori)**2 + (y1 - y1_ori)**2) +
         w2 * sp.sqrt((x2 - x2_ori)**2 + (y2 - y2_ori)**2) +
         w3 * sp.sqrt((x3 - x3_ori)**2 + (y3 - y3_ori)**2) +
         wH * sp.sqrt((x1 - xH)**2 + (y1 - yH)**2) +
         wT * sp.sqrt((x3 - xT)**2 + (y3 - yT)**2) +
         w21 * sp.sqrt((x2 - x1)**2 + (y2 - y1)**2) +
         w32 * sp.sqrt((x3 - x2)**2 + (y3 - y2)**2))
    return f

# 計算梯度 (使用有限差分法) 
# 使用forward finite difference計算梯度
def gradient(f, X, h=1e-4):
    grad = np.zeros(len(X))
    for j in range(len(X)):
        e_j = np.zeros(len(X))  # 單位向量，只有第 j 個分量是 1
        e_j[j] = 1
        grad[j] = (f(X + h * e_j) - f(X)) / h  # 使用前向差分法計算第 j 個分量的梯度
    return grad

# 計算 Hessian 矩陣 (使用有限差分法) 
# 使用forward finite difference計算Hessian matrix中的梯度
def hessian(f, X, h=1e-4):
    n = len(X)  # X 是變量向量
    H = np.zeros((n, n))  # 初始化 Hessian 矩陣
    for i in range(n):
        for j in range(n):
            e_i = np.zeros(n)  # 單位向量 e_i
            e_j = np.zeros(n)  # 單位向量 e_j
            e_i[i] = 1  # 對應變數 x_i 的單位向量
            e_j[j] = 1  # 對應變數 x_j 的單位向量
            
            # 進行擾動來計算 Hessian
            f_forward_ij = f(X + h * e_i + h * e_j)  # f(X + h * e_i + h * e_j)
            f_forward_i = f(X + h * e_i)             # f(X + h * e_i)
            f_forward_j = f(X + h * e_j)             # f(X + h * e_j)
            f_current = f(X)                         # f(X)
            
            # 計算二階導數的有限差分公式
            H[i, j] = (f_forward_ij - f_forward_i - f_forward_j + f_current) / (h ** 2)
    
    return H

# 牛頓法更新步驟
def newton_method(X_init, tol=1e-3, max_iter=100):
    X = np.array(X_init, dtype=float)
    for i in range(max_iter):
        grad = gradient(objective_function, X)
        H = hessian(objective_function, X)
        delta_X = np.linalg.solve(H, -grad)  # 進行牛頓法的更新步驟
        X = X + delta_X
        if np.linalg.norm(delta_X) < tol:
            print(f"Converged in {i+1} iterations.")
            break
    return X

# 測試
X_init = np.array([[[2.13567821, 0.72146465], 
                           [2.6713925, 1.25717893], 
                           [3.20710678, 1.79289322]]]) # initial guess
X_origin = np.array([[1.07142857, 1.07142857], 
                               [1.42857143, 1.42857143], 
                               [1.96428571, 1.96428571], 
                               [2.5, 2.5], 
                               [2.85714286, 2.85714286]]) # objective function variables
W = [1, 1, 1, 1, 1, 1, 1] # weights
optimal_X = newton_method(X_init)
print(f"Optimal solution: {optimal_X}")

# ============================================================

# fig, ax = plt.subplots()
# start = np.array([0, 0])
# goal = np.array([10, 10])
# circle_radius = 0.1
# waypoint_distance = 0.3
# safety_size = 0.2
# # initial_path = generate_initial_path(start, goal, waypoint_distance)
# new_path =  np.array([[[2.13567821, 0.72146465], [2.6713925,  1.25717893], [3.20710678, 1.79289322]]])       
# points_on_obstacle = np.array([[1.07142857, 1.07142857], [1.42857143, 1.42857143], [1.96428571, 1.96428571], 
#                       [2.5, 2.5], [2.85714286, 2.85714286]])
# obstacles = [
#     (np.array([0.5, 1.5]), np.array([2.5, 2.5])), # 第一個矩形障礙物(左下角座標, 右上角座標)
#     (np.array([4.5, 3.5]), np.array([5.5, 5.5]))  # 第二個矩形障礙物(左下角座標, 右上角座標)
# ]  # 障礙物列表
# W = [1, 1, 1, 1, 1, 1, 1]
# obs = [np.array([0.5, 1.5]), np.array([2.5, 2.5])]
# X_opt_final, iter_num = newton_method(new_path, points_on_obstacle, W, safety_size, obstacles=obstacles)
# print('X_opt_final ', X_opt_final)
# visualize_grid(ax, start, goal, obstacles, circle_radius=circle_radius, original_path=initial_path, new_points=X_opt_final)
# plt.show()
