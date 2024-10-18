# ReadMe

1. In this program, the variable `safety_size` is defined as the space surrounding the obstacle 
that the path should avoid.
2. Objective function 計算新舊路徑點的距離平方和（不包括起始和終點）、新路徑點之間的距離平方和、新點與碰撞起始點、終點的距離平方和。
3. 判斷路徑上的點以及線段是否與障礙物相交之方法。(採用以下方法1)
   - 方法1 : Bool只判斷是否在障礙物範圍內 
   - 方法2 : 利用點/線之間的數學關係式計算出Danger factor數值
4. 針對新的避障路徑點位置，使用牛頓法取得最佳位置，使路徑最短、平滑且避開障礙物。
5. 若要將mobile robot的大小納入碰撞檢測的考量 則點位碰撞檢測需要改成以下程式
    ```py
    def point_in_obstacle(point, bottom_left, top_right, safety_size, circle_radius):
        # 增加障礙物周圍的安全距離，同時考慮點位的半徑
        expanded_bottom_left = bottom_left - np.array([safety_size + circle_radius, safety_size + circle_radius])
        expanded_top_right = top_right + np.array([safety_size + circle_radius, safety_size + circle_radius])
        # 檢查點是否在擴展過的矩形內
        return (expanded_bottom_left[0] <= point[0] <= expanded_top_right[0] and
                expanded_bottom_left[1] <= point[1] <= expanded_top_right[1])
    ```
6. array_test的測試程式如下
    ```py
    import numpy as np

    origin_path = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    X_opt_final = [5, 30, 20, 30, 40, 50 , 60, 10]
    indices_of_closest_points = [5, 6, 7, 8, 9, 10]

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

    print('origin_path', origin_path)
    ```
7. 所有測試的檔案都是放在`test`資料夾底下。