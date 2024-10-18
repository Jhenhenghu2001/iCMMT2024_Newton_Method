import numpy as np
import matplotlib.pyplot as plt
import time

#  將初始路徑上的碰撞點偏移出障礙物(含安全距離)的範圍行程新路徑，作為避障路徑最佳化的Initial Guess。
def generate_new_path(path, obstacles, safety_size, offset_distance):
    new_path = [] # 用於儲存障礙物的避障路徑

    # Calculate the tangent vector from start to end
    tangent_vector = path[-1] - path[0]
    tangent_vector /= np.linalg.norm(tangent_vector)  # Normalize the tangent vector

    # Calculate the normal vector (rotate 90 degrees)
    normal_vector_left = np.array([-tangent_vector[1], tangent_vector[0]])
    normal_vector_right = np.array([tangent_vector[1], -tangent_vector[0]])

    # Loop through each point in the original path
    for point in path:
        closest_obstacle_edge_distance = float('inf')
        chosen_normal_vector = None
        
        # Find the closest obstacle edge
        for obstacle in obstacles:
            bottom_left, top_right = obstacle
            
            # Calculate distances to each edge of the obstacle considering the safety size
            distances = [
                (bottom_left[0] - safety_size) - point[0],  # left edge
                point[0] - (top_right[0] + safety_size),    # right edge
                (bottom_left[1] - safety_size) - point[1],  # bottom edge
                point[1] - (top_right[1] + safety_size)     # top edge
            ]
            
            # Find the minimum distance to an edge
            min_distance = min(distances)
            
            # Determine if this is the closest obstacle
            if abs(min_distance) < abs(closest_obstacle_edge_distance):
                closest_obstacle_edge_distance = min_distance
                # Choose normal based on which edge is closer
                if min_distance == distances[0]:  # Left edge
                    chosen_normal_vector = normal_vector_right  # Offset to the right
                elif min_distance == distances[1]:  # Right edge
                    chosen_normal_vector = normal_vector_left  # Offset to the left
                elif min_distance == distances[2]:  # Bottom edge
                    chosen_normal_vector = normal_vector_left  # Offset up
                elif min_distance == distances[3]:  # Top edge
                    chosen_normal_vector = normal_vector_right  # Offset down

        # Move the point outward by offset_distance along the chosen normal vector
        new_point = point + chosen_normal_vector * offset_distance
        new_path.append(new_point)

    return np.array(new_path)