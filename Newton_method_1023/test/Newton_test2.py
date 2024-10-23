import numpy as np

# 定義目標函數
def objective_function(X):
    x1, x2 = X[0], X[1]  # 解包 X 為 x1, x2
    # 計算目標函數值
    f = x1**4 + 10*x2**4 - 5*x1 + 6*x2  # 使用 ** 來表示次方運算
    return f

def newton_method():
    error = 100
    X_init = np.array([5.0, 5.0])  # 初始猜測，轉換為 numpy array
    X = X_init
    iter_num = 0
    delta = 0.01
    print('Initial guess = ', X_init)
    
    while error > 10**-3:
        iter_num += 1
        f_x = objective_function(X)
        f_dx = objective_function(X + delta * np.array([1, 0]))  # 轉換為 numpy array 並進行運算
        f_dy = objective_function(X + delta * np.array([0, 1]))
        f_ndx = objective_function(X - delta * np.array([1, 0]))
        f_ndy = objective_function(X - delta * np.array([0, 1]))
        f_dxdy = objective_function(X + delta * np.array([1, 1]))

        # 使用正確的運算符號 ** 來代替 ^
        dfx = (f_dx - f_x) / delta
        dfy = (f_dy - f_x) / delta
        dfxx = (f_dx - 2 * f_x + f_ndx) / (delta**2)
        dfxy = (f_dxdy - f_dx - f_dy + f_x) / (delta**2)
        dfyy = (f_dy - 2 * f_x + f_ndy) / (delta**2)

        grad = np.array([dfx, dfy])
        H = np.array([[dfxx, dfxy], [dfxy, dfyy]])

        # 計算 Hessian 的逆矩陣並更新 X
        H_inv = np.linalg.inv(H)
        X_new = X - np.dot(H_inv, grad)

        # 計算誤差
        error = np.linalg.norm(X_new - X)
        print('Iter = ', iter_num, ', X = ', X, ', error = ', error)

        # 更新 X 和迭代次數
        X = X_new
    return X, iter_num

# 執行牛頓法
Opt_X, iter_num = newton_method()
print('Optimal solution = ', Opt_X, ', Total iterations = ', iter_num)
