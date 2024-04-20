import open3d as o3d
import os

# Path to the folder containing RGBD images
rgbd_folder = "/home/sber20/dev/SBER/open_semantic_pipeline/datasets/data/Datasets/Replica/office2/results"
# rgbd_folder = "/home/sber20/dev/SBER/open_semantic_pipeline/datasets/data/Datasets/Replica/office0/results"

# Read intrinsic parameters
fx = 600.0
fy = 600.0
cx = 599.5
cy = 339.5
width = 1200
height = 680
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Path to the trajectory file containing poses
trajectory_file = "/home/sber20/dev/SBER/open_semantic_pipeline/datasets/data/Datasets/Replica/office2/traj.txt"
# trajectory_file = "/home/sber20/dev/SBER/open_semantic_pipeline/datasets/data/Datasets/Replica/office0/traj.txt"

# Read poses from the trajectory file
poses = []  
with open(trajectory_file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        # print(line)
        pose_values = line.strip().split(' ')
        # print(pose_values)
        pose = [] 
        for value in pose_values:
            pose.append(float(value))
            
        poses.append(pose)

point_clouds = []  


# Process each RGBD image and corresponding pose
for file_name in sorted(os.listdir(rgbd_folder))[0:100]:
    if file_name.endswith(".png"):
        depth_path = os.path.join(rgbd_folder, file_name)
        color_path = os.path.join(rgbd_folder, file_name.replace("depth", "frame").replace(".png", ".jpg"))
        
        depth = o3d.io.read_image(depth_path)
        color = o3d.io.read_image(color_path)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False,depth_scale=6553.5)

        # Get the corresponding pose for this RGBD image
        index = int(file_name.split("depth")[1].split(".png")[0])  
        if index < len(poses):
            pose = poses[index]
            # print(pose)
            # Create a transformation matrix from the pose data
            transformation = [
                [pose[0], pose[1], pose[2], pose[3]],
                [pose[4], pose[5], pose[6], pose[7]],
                [pose[8], pose[9], pose[10], pose[11]],
                [0.0, 0.0, 0.0, 1.0]
            ]
            # print(transformation)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            pcd.transform(transformation)

            # o3d.visualization.draw_geometries([pcd])
            point_clouds.append(pcd)  

# Visualize all point clouds together in one window
o3d.visualization.draw_geometries(point_clouds)
