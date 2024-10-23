import numpy as np

# Obstacle avoidance path planning optimization using Newton's method.

def newton_method(X_init, X_ori, W, tol=1e-3):
    """
    使用多維牛頓法進行最佳化，直到滿足收斂條件或達到最大迭代次數。
    
    參數：
    - X_init: 初始點 [X1*, X2*, X3*] 的向量
    - X_ori: 原始點，包括 XT, X1_ori, X2_ori, X3_ori, XH
    - W: 權重向量

    限制條件：
    - tol: 收斂誤差門檻值 (預設 1e-3)
    
    回傳：
    - X_opt: 最佳化後的點位 [X1*, X2*, X3*]
    - k: 總迭代次數
    """

    X = X_init  # 初始化 X
    error = 100  # 初始化誤差
    k = 0  # 初始化迭代次數
    X_prev = X  # 記錄前一次的 X
    # print('initial X ', X_init)
    while error > tol:  # 當 error 大於 tol
        k += 1
        
        # 計算梯度和 Hessian 的逆矩陣
        grad = gradient(X, X_ori, W)
        H_inv = inverse_hessian(W)
        
        # 調整梯度形狀
        grad_flat = grad.reshape(-1)  # 展開梯度為一維向量
        # 更新 X(k+1) = X(k) - H^(-1) * grad
        X_new = np.array(X).reshape(-1) - np.dot(H_inv, grad_flat)
        # 將 X_new 重新調整回原來的形狀
        X_new = X_new.reshape(1, 3, 2)
        
        # 計算誤差：取當次迭代的位置與前一次迭代的位置的范數
        error = np.linalg.norm(X_new - X_prev)
        
        # 更新 X_prev 為當前的 X，以便下一次計算誤差
        X_prev = X_new
        
        # 更新 X
        X = X_new
        # print('X', X)
    # print('k', k)
    # print('error', error)
    # print('before flip X', X)
    X = np.flip(X, axis=1)  # 調回原先矩陣的形狀後所有內部矩陣順序會顛倒，因此要反轉回來
    # print('after flip X', X)
    return X, k

def objective_function(X, X_ori, W):
    # 解包，獲取內部的數組
    X_array = X[0]

    X1, X2, X3 = [X_array[0]], [X_array[1]], [X_array[2]]  # 分解 X 為 X1*, X2*, X3*
    XT, X1_ori, X2_ori, X3_ori, XH = X_ori[0], X_ori[1], X_ori[2], X_ori[3], X_ori[4] # 分解 X_ori 為 XT, X1_ori, X2_ori, X3_ori, XH
    w1, w2, w3, wH, wT, w21, w32 = W[0], W[1], W[2], W[3], W[4], W[5], W[6] # 權重

    # 計算目標函數
    f = (w1 * np.linalg.norm(X1 - X1_ori)**2 +
         w2 * np.linalg.norm(X2 - X2_ori)**2 +
         w3 * np.linalg.norm(X3 - X3_ori)**2 +
         wH * np.linalg.norm(X1 - XH)**2 +
         wT * np.linalg.norm(X3 - XT)**2 +
         w21 * np.linalg.norm(X2 - X1)**2 +
         w32 * np.linalg.norm(X3 - X2)**2)
    
    return f

def gradient(X, X_ori, W):
    # 解包，獲取內部的數組
    X_array = X[0]
    
    X1, X2, X3 = [X_array[0]], [X_array[1]], [X_array[2]]  # 分解 X 為 X1*, X2*, X3*
    XT, X1_ori, X2_ori, X3_ori, XH = X_ori[0], X_ori[1], X_ori[2], X_ori[3], X_ori[4] # 分解 X_ori 為 XT, X1_ori, X2_ori, X3_ori, XH
    w1, w2, w3, wH, wT, w21, w32 = W[0], W[1], W[2], W[3], W[4], W[5], W[6] # 權重

    # 計算各點的梯度
    grad_X1 = (2 * w1 * (X1 - X1_ori) +
               2 * wH * (X1 - XH) +
               2 * w21 * (np.array(X1) - np.array(X2)))
    
    grad_X2 = (2 * w2 * (X2 - X2_ori) +
               2 * w21 * (np.array(X2) - np.array(X1)) +
               2 * w32 * (np.array(X2) - np.array(X3)))
    
    grad_X3 = (2 * w3 * (X3 - X3_ori) +
               2 * wT * (np.array(X3) - np.array(XT)) +
               2 * w32 * (np.array(X3) - np.array(X2)))
    
    # 將梯度組合為一個向量
    grad = np.array([grad_X1, grad_X2, grad_X3])
    
    return grad

def hessian(W):
    w1, w2, w3, wH, wT, w21, w32 = W[0], W[1], W[2], W[3], W[4], W[5], W[6] # 權重

    # 單位矩陣 (2x2)，因為每個點都有 (x, y) 兩個維度
    I = np.eye(2)

    # 計算對角線 Hessian，這些是二階導數 f''(X1), f''(X2), f''(X3)
    H11 = 2 * (w1 + wH + w21) * I  # 對 X1* 的二階導數
    H22 = 2 * (w2 + w21 + w32) * I # 對 X2* 的二階導數
    H33 = 2 * (w3 + wT + w32) * I  # 對 X3* 的二階導數

    # 計算交叉項 Hessian，這些是二階交叉導數 f''(X1, X2), f''(X2, X3)
    H12 = H21 = -2 * w21 * I  # 對 X1* 和 X2* 的交叉二階導數
    H23 = H32 = -2 * w32 * I  # 對 X2* 和 X3* 的交叉二階導數

    # 組成 Hessian 矩陣，總共 6x6 大小，將每個 2x2 的子矩陣放入總矩陣中
    H = np.block([[H11, H12, np.zeros((2, 2))],
                  [H21, H22, H23],
                  [np.zeros((2, 2)), H32, H33]])

    return H

def inverse_hessian(W):
    H = hessian(W)  # 計算 Hessian 矩陣
    det = np.linalg.det(H)  # 計算行列式
    
    if det == 0:
        raise ValueError("Hessian 矩陣是奇異矩陣，不可逆。")
    
    H_inv = np.linalg.inv(H)  # 計算 Hessian 的逆矩陣
    return H_inv

########################

# 淘汰程式

# def newton_method(X_init, X_ori, W, max_iter=5, tol=1e-3):
#     """
#     使用多維牛頓法進行最佳化，直到滿足收斂條件或達到最大迭代次數。
    
#     參數：
#     - X_init: 初始點 [X1*, X2*, X3*] 的向量
#     - X_ori: 原始點，包括 XT, X1_ori, X2_ori, X3_ori, XH
#     - W: 權重向量

#     限制條件：
#     - max_iter: 最大迭代次數 (預設 5)
#     - tol: 收斂誤差門檻值 (預設 1e-3)
    
#     回傳：
#     - X_opt: 最佳化後的點位 [X1*, X2*, X3*]
#     - k: 總迭代次數
#     """

#     X = X_init  # 初始化 X
#     error = np.inf  # 初始化誤差，設為無窮大
#     k = 0  # 初始化迭代次數
#     # print('initial X ', X_init)
#     while error > tol or k < max_iter:
#         k += 1
        
#         # 計算梯度和 Hessian 的逆矩陣
#         grad = gradient(X, X_ori, W)
#         H_inv = inverse_hessian(W)
        
#         # 調整梯度形狀
#         grad_flat = grad.reshape(-1)  # 展開梯度為一維向量
#         # 更新 X(k+1) = X(k) - H^(-1) * grad
#         X_new = np.array(X).reshape(-1) - np.dot(H_inv, grad_flat)
#         # 將 X_new 重新調整回原來的形狀
#         X_new = X_new.reshape(1, 3, 2)
#         # 計算誤差：取前一次迭代的位置與當次迭代的位置的范數
#         error = np.linalg.norm()

#         # 更新 X
#         X = X_new
#         # print('X', X)
#     # print('k', k)
#     # print('before flip X', X)
#     X = np.flip(X, axis=1) # 調回原先矩陣的形狀後所有內部矩陣順序會顛倒，因此要反轉回來
#     # print('after flip X', X)
#     return X, k