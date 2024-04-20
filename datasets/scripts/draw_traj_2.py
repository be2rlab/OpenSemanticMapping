import numpy as np
from matplotlib import pyplot as plt

import open3d as o3d

data = np.loadtxt("/home/sber20/dev/SBER/open_semantic_pipeline/datasets/data/Datasets/Gibson/Adrian/traj.txt")
data = np.array([data.reshape(4, 4) for data in data])
num_poses = 5 # data.shape[0]
trajectory = [np.eye(4) for _ in range(num_poses)]

for i in range(num_poses):
    trajectory[i][:3, 3] = data[i, :3, 3]
    trajectory[i][:3, :3] = data[i, :3, :3]

vis = o3d.visualization.Visualizer()
vis.create_window()

line_set = o3d.geometry.LineSet()

points = np.zeros((num_poses, 3))
lines = np.zeros((num_poses - 1, 2), dtype=np.int32)
colors = np.ones((num_poses, 3))

for i in range(num_poses):
    points[i, :] = trajectory[i][:3, 3]
    if i < num_poses - 1:
        lines[i, :] = [i, i + 1]

line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)

vis.add_geometry(line_set)

for i in range(num_poses):
    rotation_matrix = trajectory[i][:3, :3]
    
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=trajectory[i][:3, 3])
    frame.rotate(rotation_matrix, center=trajectory[i][:3, 3])
    
    vis.add_geometry(frame)

vis.run()
vis.destroy_window()
