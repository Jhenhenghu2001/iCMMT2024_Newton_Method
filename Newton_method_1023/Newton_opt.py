import numpy as np

# Obstacle avoidance path planning optimization using Newton's method.
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