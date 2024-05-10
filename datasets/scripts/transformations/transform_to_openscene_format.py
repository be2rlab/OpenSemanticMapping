import glob, os
import multiprocessing as mp
import numpy as np
import imageio
import cv2
import torch
from tqdm import tqdm
import json
import sys
import plyfile
import math


def make_intrinsic(fx, fy, mx, my):
    '''Create camera intrinsics.'''

    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    '''Adjust camera intrinsics.'''

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(
                    intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def process_one_scene_2d(fn):
    '''process one scene.'''

    # process RGB images
    img_name = fn.split('/')[-1]
    img_id = int(int(img_name.split('frame')[-1].split('.')[0])/sample_freq)
    img = imageio.v3.imread(fn)
    img = cv2.resize(img, img_dim, interpolation=cv2.INTER_LINEAR)
    imageio.imwrite(os.path.join(out_dir_color, str(img_id)+'.jpg'), img)

    # process depth images
    depth_name = img_name.replace('.jpg', '.png').replace('frame', 'depth')
    fn_depth = os.path.join(fn.split('frame')[0], depth_name)
    depth = imageio.v3.imread(fn_depth).astype(np.uint16)
    depth = cv2.resize(depth, img_dim, interpolation=cv2.INTER_LINEAR)
    imageio.imwrite(os.path.join(out_dir_depth, str(img_id)+'.png'), depth)

    #process poses
    np.savetxt(os.path.join(out_dir_pose, str(img_id)+'.txt'), pose_list[img_id])
    

def process_one_scene_3d(fn):
    '''process one scene.'''

    scene_name = fn.split('/')[-1].split('_mesh')[0]
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, -3:]) / 127.5 - 1

    # no GT labels are provided, set all to 255
    labels = 255*np.ones((coords.shape[0], ), dtype=np.int32)
    torch.save((coords, colors, labels),
            os.path.join(out_dir,  scene_name + '.pth'))
    print(fn)


#####################################
out_dir = os.path.join(sys.argv[2], "replica_2d")
in_path = sys.argv[1]
sample_freq = 1
#####################################

os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(os.path.dirname(in_path), "cam_params.json"), 'r') as intrinsics_src:
    data = json.load(intrinsics_src)
    fx = data["camera"]["fx"]
    fy = data["camera"]["fy"]
    cx = data["camera"]["cx"]
    cy = data["camera"]["cy"]
    w = data["camera"]["w"]
    h = data["camera"]["h"]

img_dim = (640, 360)
original_img_dim = (w, h)

intrinsics = make_intrinsic(fx, fy, cx, cy)

# save the intrinsic parameters of resized images
intrinsics = adjust_intrinsic(intrinsics, original_img_dim, img_dim)
np.savetxt(os.path.join(out_dir, 'intrinsics.txt'), intrinsics)

scene = os.path.basename(in_path)
in_path = os.path.dirname(in_path)

out_dir_color = os.path.join(out_dir, scene, 'color')
out_dir_depth = os.path.join(out_dir, scene, 'depth')
out_dir_pose = os.path.join(out_dir, scene, 'pose')
if not os.path.exists(out_dir_color):
    os.makedirs(out_dir_color)
if not os.path.exists(out_dir_depth):
    os.makedirs(out_dir_depth)
if not os.path.exists(out_dir_pose):
    os.makedirs(out_dir_pose)

# save the camera parameters to the folder
camera_dir = os.path.join(in_path,
        scene, 'traj.txt')
poses = np.loadtxt(camera_dir).reshape(-1, 4, 4)
pose_list = poses[::sample_freq]

files = glob.glob(os.path.join(in_path, scene, 'results', '*.jpg'))
files = sorted(files)
files = files[::sample_freq] # every 10 frames

process_one_scene_2d(files[0])

p = mp.Pool(processes=mp.cpu_count())
p.map(process_one_scene_2d, files)
p.close()


#####################################
out_dir = os.path.join(sys.argv[2], "replica_3d")
#####################################

os.makedirs(out_dir, exist_ok=True)

file = os.path.join(in_path, '{}_mesh.ply'.format(scene))

process_one_scene_3d(file)