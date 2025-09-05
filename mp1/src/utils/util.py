import os
import re
import cv2
import numpy as np
import xml.etree.ElementTree as ET


def euler_to_quaternion(r):
    (roll, pitch, yaw) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return [roll, pitch, yaw]


def point_in_rectangle(corners, point):
    x, y = point
    x_coords = [x for (x,_) in corners]
    y_coords = [y for (_,y) in corners]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return (x_min < x and x < x_max) and (y_min < y and y < y_max)


def get_transformation_matrix(target_frame_pose, from_world=False):
    p_w_t = target_frame_pose[:2]
    yaw = target_frame_pose[-1]
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    # homogenous transformation matrix
    if from_world:
        T = np.column_stack((R.T, -R.T @ p_w_t))
    else:
        T = np.column_stack((R, p_w_t))
    T = np.vstack((T, np.array([0,0,1])))
    return T


def perspective_transform(img, src):
    """
    Get bird's eye view from input image
    """
    height, width = img.shape[:2]
    dst = np.float32([(0,0), (0,height), (width, height), (width, 0)])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = np.linalg.inv(M)
    warped_img = cv2.warpPerspective(img, M, (width, height))
    return warped_img, M, Minv


def get_roads_info(world_fn: str):
    tree = ET.parse(world_fn)
    world_root = tree.getroot()
    roads_info = { }
    for model in world_root.findall('.//model'):
        model_name = model.get('name')
        if (model_name is not None) and ('road' in model_name.lower()):
            if model_name in roads_info.keys():
                continue
            pose_element = model.find(".//pose")
            pose_values = pose_element.text.strip().split()
            pose = [float(v) for v in pose_values]
            tmp = re.split(r"\s-\s|\s|_", model_name)
            size = int(tmp[3][:-1])
            shape = tmp[1].lower()
            local_corners, global_corners = [], []
            if shape == "curved":
                radius = size + 11/2
                frame_pos = pose[:2]
                local_corners = [(radius/2, radius/2), (-radius/2, -radius/2)]
                global_corners = [[frame_pos[0] + radius/2, frame_pos[1] + radius/2],
                            [frame_pos[0] - radius/2, frame_pos[1] - radius/2]]
            if shape == "straight":
                # from mesh analysis
                min_x, max_x = -size/2, size/2
                min_y, max_y = -15.5, -4.5
                local_corners = [[max_x, max_y], [min_x, min_y]]
                T = get_transformation_matrix(pose, from_world=False)
                for corner in local_corners:
                    local_corner_hom = np.array(corner+[1]).reshape(3,1)
                    global_corner = (T @ local_corner_hom)[:2].flatten().tolist()
                    global_corners.append(global_corner)
            # (min_x, max_y)
            principal_local_corner = [local_corners[1][0], local_corners[0][1]] 
            major_local_pose = (*principal_local_corner, -np.pi/2)
            A = get_transformation_matrix(major_local_pose, from_world=True)
            roads_info[model_name] = {
                "pose": [float(v) for v in pose_values],
                "size": size,
                "shape": shape,
                "corners": {
                    "local": local_corners,
                    "global": global_corners,
                },
                "lane_map_frame": A
            }
    return roads_info
