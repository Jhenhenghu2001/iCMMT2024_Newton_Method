import numpy as np

# ######## 方法一 ########

# origin_path = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# X_opt_final = [5, 30, 20, 30, 10]
# indices_of_closest_points = [5, 6, 7, 8, 9, 10]

# def update_origin_path(origin_path, X_opt_final, indices_of_closest_points):

#     # 計算差值
#     len_difference = len(X_opt_final) - len(indices_of_closest_points)

#     # 1. 刪除 origin_path 中的點
#     # 由於刪除點位會改變索引，因此我們還是從後往前刪除
#     for idx in sorted(indices_of_closest_points, reverse=True):
#         del origin_path[idx]

#     # 2. 插入 X_opt_final 的新點位
#     # 如果 len_difference > 0, 需要往前插入更多點
#     insert_position = indices_of_closest_points[0]  # 插入的起始位置

#     for i, point in enumerate(X_opt_final):
#         origin_path.insert(insert_position + i, point)

#     # 3. 若 X_opt_final 長度超過 indices_of_closest_points 的長度
#     #    則需要將後續的點位「推前」移動
#     if len_difference > 0:
#         # 從插入位置之後的點位往前移動
#         for _ in range(len_difference):
#             origin_path.pop(insert_position + len(X_opt_final))
#     return origin_path
# new_origin_path = update_origin_path(origin_path, X_opt_final, indices_of_closest_points)
# # print('new_origin_path', new_origin_path)

# ######## 方法二 ######## 

# origin_path = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
# X_opt_final = np.array([[20, 20], [30, 30], [40, 40]])
# indices_of_closest_points = [3, 4, 5, 6, 7, 8]
# # print('X_opt_final', X_opt_final)
# # print(origin_path[indices_of_closest_points[0]])

# def update_origin_path_v2(origin_path, X_opt_final, indices_of_closest_points):

#     new_origin_path = origin_path.copy()
#     new_origin_path = np.delete(new_origin_path, new_origin_path[indices_of_closest_points[1]:], axis=0)
#     new_origin_path = np.vstack((new_origin_path, X_opt_final))
#     new_origin_path = np.vstack((new_origin_path, origin_path[indices_of_closest_points[-1]:]))
    
#     return new_origin_path

# new_origin_path_v2 = update_origin_path_v2(origin_path, X_opt_final, indices_of_closest_points)
# # print('new_origin_path', new_origin_path_v2)

######## 方法三 ######## (使用此方法)

origin_path = np.array([[ 0.       ,   0.        ], [ 0.35714286 , 0.35714286],[ 0.71428571 , 0.71428571],[ 1.07142857 , 1.07142857],[ 1.42857143 , 1.42857143],
            [ 1.78571429 , 1.78571429],[ 2.14285714 , 2.14285714],[ 2.5        , 2.5       ],[ 2.85714286 , 2.85714286],[ 3.21428571 , 3.21428571],
            [ 3.57142857 , 3.57142857], [ 3.92857143 , 3.92857143],[ 4.28571429 , 4.28571429],[ 4.64285714 , 4.64285714],[ 5.         , 5.        ],
            [ 5.35714286 , 5.35714286],[ 5.71428571 , 5.71428571],[ 6.07142857 , 6.07142857],[ 6.42857143 , 6.42857143],[ 6.78571429 , 6.78571429],
            [ 7.14285714 , 7.14285714],[ 7.5        , 7.5       ],[ 7.85714286 , 7.85714286],[ 8.21428571 , 8.21428571],[ 8.57142857 , 8.57142857],
            [ 8.92857143 , 8.92857143],[ 9.28571429 , 9.28571429],[ 9.64285714 , 9.64285714],[10.         ,10.        ]])
X_opt_final = np.array([[1.8452381 , 1.8452381 ], [1.96428571, 1.96428571],[2.08333333, 2.08333333]])
indices_of_closest_points = [3, 4, 5, 6, 7, 8]

def update_origin_path_v3(origin_path, X_opt_final, indices_of_closest_points):
    # print('origin_path', origin_path)
    # print('X_opt_final', X_opt_final)
    # print('indices_of_closest_points',indices_of_closest_points)
    print(np.shape(origin_path))
    print(np.shape(X_opt_final))
    print(np.shape(indices_of_closest_points))
    # 0. 複製原始路徑陣列到一個新陣列使用
    new_origin_path = origin_path.copy()
    # 1. 將原始路徑新陣列刪除掉經過障礙物點位(含後面全部點位)
    end = len(origin_path)
    start = indices_of_closest_points[1]
    new_index_list = [] # 建立需要刪除的索引位置清單
    for i in range(start, end):
        new_index_list.append(i)
        i+=1
    for i in sorted(new_index_list, reverse=True):
        new_origin_path = np.delete(new_origin_path, i, axis=0)
    # print('1. new_origin_path', new_origin_path)
    # 2. 將原始路徑新陣列後方加上最佳化之後的新路徑點位
    new_origin_path = np.vstack((new_origin_path, X_opt_final))
    # print('2. new_origin_path', new_origin_path)
    # 3. 將原先後方不在障礙物範圍的點位重新加入到原始路徑新陣列
    new_origin_path = np.vstack((new_origin_path, origin_path[indices_of_closest_points[-1]:]))
    # print('3. new_origin_path', new_origin_path)
    
    return new_origin_path

new_origin_path_v3 = update_origin_path_v3(origin_path, X_opt_final, indices_of_closest_points)
# print('new_origin_path', new_origin_path_v3)

## 方法三測試用程式
# end = len(origin_path)
# start = indices_of_closest_points[1]
# new_origin_path = origin_path.copy()
# new_index_list = []
# for i in range(start, end):
#         new_index_list.append(i)
# print('new_index_list',new_index_list)
# for i in sorted(new_index_list, reverse=True):
#         new_origin_path = np.delete(new_origin_path, i, axis=0)
# print('new_origin_path', new_origin_path)
##