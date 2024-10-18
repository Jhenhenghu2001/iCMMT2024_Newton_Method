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