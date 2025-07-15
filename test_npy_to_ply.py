import os
import random
import numpy as np
import open3d as o3d

# 配置参数
dir_path = 'point_cloud_cache_fusion360'
output_dir = 'point_cloud_cache_fusion360_ply_test'
os.makedirs(output_dir, exist_ok=True)

# 获取所有npy文件
npy_files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]

# 随机抽取10个
sample_files = random.sample(npy_files, min(10, len(npy_files)))

for npy_file in sample_files:
    npy_path = os.path.join(dir_path, npy_file)
    ply_path = os.path.join(output_dir, npy_file.replace('.npy', '.ply'))
    points = np.load(npy_path)
    if points.ndim == 2 and points.shape[1] == 3:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"Converted {npy_file} -> {ply_path}")
    else:
        print(f"Skipped {npy_file}: shape {points.shape} is not (N, 3)")
