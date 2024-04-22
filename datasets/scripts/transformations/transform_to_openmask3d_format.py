import os
import shutil
import glob
import json




# Change this to specify the dataset you are working with (relative path from OpenSemanticMapping directory)
DATASET_PATH = "Datasets/Replica/room0"
 





ROOT_DIR = os.getcwd()
DATASET_DIR = os.path.join(ROOT_DIR, DATASET_PATH)
DATASET_NAME = os.path.basename(DATASET_DIR)
TRANSFORMED_DATASET_DIR = os.path.join(ROOT_DIR, "datasets", "data", "Datasets", "openmask3d_datasets", DATASET_NAME)



# Creating a directory for the transformed dataset
os.chdir(os.path.join(ROOT_DIR, "datasets", "data", "Datasets"))
if not os.path.exists("openmask3d_datasets"):
    os.mkdir("openmask3d_datasets")
os.chdir("openmask3d_datasets")
if not os.path.exists(DATASET_NAME):
    os.mkdir(DATASET_NAME)



# Making subdirectories 
os.chdir(DATASET_NAME)
os.mkdir("color")
os.mkdir("depth")
os.mkdir("pose")
os.mkdir("intrinsic")



# Copying point cloud
print("Copying point cloud...", end="")
os.chdir(os.path.dirname(DATASET_DIR))
shutil.copy(glob.glob(DATASET_NAME + "*" + ".ply")[0], TRANSFORMED_DATASET_DIR)
print("done")



# Copying frames
print("Copying frames...", end="")
os.chdir(os.path.join(DATASET_NAME, "results"))
frames = glob.glob("frame*")
frames.sort()

for i in range(len(frames)):
    shutil.copy(frames[i], os.path.join(TRANSFORMED_DATASET_DIR, "color", str(i) + os.path.splitext(frames[i])[1]))

print("done")




# Copying depths
print("Copying depths...", end="")
depth = glob.glob("depth*")
depth.sort()

for i in range(len(depth)):
    shutil.copy(depth[i], os.path.join(TRANSFORMED_DATASET_DIR, "depth", str(i) + os.path.splitext(depth[i])[1]))

print("done")




# Copying poses
print("Copying poses...", end="")
with open(os.path.join(DATASET_DIR, "traj.txt"), 'r') as traj_src:
    i = 0
    for pose in traj_src:
        pose = pose.split(" ")
        with open(os.path.join(TRANSFORMED_DATASET_DIR, "pose", str(i) + ".txt"), 'a') as traj_dst:
            for m in range(4):
                for n in range(4):
                    traj_dst.write(pose[m * 4 + n] + ' ')
                traj_dst.write('\n')
        i+=1
print("done")




# Copying intrinsics
print("Copying intrinsics...", end="")
with open(os.path.join(os.path.dirname(DATASET_DIR), "cam_params.json"), 'r') as intrinsics_src:
    data = json.load(intrinsics_src)
    fx = data["camera"]["fx"]
    fy = data["camera"]["fy"]
    cx = data["camera"]["cx"]
    cy = data["camera"]["cy"]
    with open(os.path.join(TRANSFORMED_DATASET_DIR, "intrinsic", "intrinsic_color.txt"), 'a') as intrinsic_dst:
        intrinsic_dst.write(str(fx) + " 0.000000 " + str(cx) + " 0.000000\n" +
                            "0.000000 " + str(fy) + " " + str(cy) + " 0.000000\n" +
                            "0.000000 0.000000 1.000000 0.000000\n" +
                            "0.000000 0.000000 0.000000 1.000000\n")
print("done")