import numpy as np

######## 方法一 ########

origin_path = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
X_opt_final = [5, 30, 20, 30, 10]
indices_of_closest_points = [5, 6, 7, 8, 9, 10]

def update_origin_path(origin_path, X_opt_final, indices_of_closest_points):

    # 計算差值
    len_difference = len(X_opt_final) - len(indices_of_closest_points)

    # 1. 刪除 origin_path 中的點
    # 由於刪除點位會改變索引，因此我們還是從後往前刪除
    for idx in sorted(indices_of_closest_points, reverse=True):
        del origin_path[idx]

    # 2. 插入 X_opt_final 的新點位
    # 如果 len_difference > 0, 需要往前插入更多點
    insert_position = indices_of_closest_points[0]  # 插入的起始位置

    for i, point in enumerate(X_opt_final):
        origin_path.insert(insert_position + i, point)

    # 3. 若 X_opt_final 長度超過 indices_of_closest_points 的長度
    #    則需要將後續的點位「推前」移動
    if len_difference > 0:
        # 從插入位置之後的點位往前移動
        for _ in range(len_difference):
            origin_path.pop(insert_position + len(X_opt_final))
    return origin_path
new_origin_path = update_origin_path(origin_path, X_opt_final, indices_of_closest_points)
# print('new_origin_path', new_origin_path)

######## 方法二 ######## (使用此方法)

origin_path = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
X_opt_final = np.array([[20, 20], [30, 30], [40, 40]])
indices_of_closest_points = [3, 4, 5, 6, 7, 8]
print('X_opt_final', X_opt_final)
# print(origin_path[indices_of_closest_points[0]])

def update_origin_path_v2(origin_path, X_opt_final, indices_of_closest_points):

    new_origin_path = origin_path.copy()
    new_origin_path = np.delete(new_origin_path, new_origin_path[indices_of_closest_points[1]:], axis=0)
    new_origin_path = np.vstack((new_origin_path, X_opt_final))
    new_origin_path = np.vstack((new_origin_path, origin_path[indices_of_closest_points[-1]:]))
    
    return new_origin_path

new_origin_path_v2 = update_origin_path_v2(origin_path, X_opt_final, indices_of_closest_points)
print('new_origin_path', new_origin_path_v2)