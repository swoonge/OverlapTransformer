# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generate the overlap and orientation combined mapping file.
import sys
import os
import glob
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from tools.utils.utils import *
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import yaml
from com_overlap import com_overlap
from mpl_toolkits.mplot3d import Axes3D

from modules.overlap_transformer import featureExtracter

# load config ================================================================
config_filename = '../config/config.yml'
config = yaml.safe_load(open(config_filename))
test_weights = config["demo1_config"]["test_weights"]
# ============================================================================

# set args for the script
parser = argparse.ArgumentParser(description='Generate overlap and orientation combined mapping file')
parser.add_argument('--dataset_path', type=str, default='/media/vision/Seagate/DataSets/kitti/dataset/sequences/', help='path to the scan data')

def plot_two_poses(poses, poses_new):
    # poses와 poses_new에서 x, y, z 좌표 추출
    x_old, y_old, z_old = zip(*[pose[:3, 3] for pose in poses])
    x_new, y_new, z_new = zip(*[pose[:3, 3] for pose in poses_new])

    # 3D 플롯 생성
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 원래 포즈 시퀀스를 점과 선으로 표시
    ax.plot(x_old, y_old, z_old, 'r.-', label='Original Pose')  # 빨간색 점과 선
    # 변환된 포즈 시퀀스를 점과 선으로 표시
    ax.plot(x_new, y_new, z_new, 'b.-', label='New Pose')  # 파란색 점과 선

    # 축 레이블 설정
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 범례 표시
    plt.legend()
    # 플롯 표시
    plt.show()

def __main__():
  # load scan paths and poses
    args = parser.parse_args()
    sequence = "00"
    dataset_path = os.path.join(args.dataset_path, sequence)

    # load scan paths
    scan_paths = sorted(glob.glob(os.path.join(dataset_path, 'velodyne', '*.bin')))
    scans = []
    for scan_path in tqdm(scan_paths):
        scans.append(load_vertex(scan_path))

    # load calibrations
    calib_file = os.path.join(dataset_path, 'calib.txt')
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # load poses
    poses_file = os.path.join(dataset_path, 'poses.txt')
    poses = load_poses(poses_file)
    pose0_inv = np.linalg.inv(poses[0])

    # for KITTI dataset, we need to convert the provided poses
    # from the camera coordinate system into the LiDAR coordinate system
    poses_new = []

    for pose in poses:
        poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
    # plot_two_poses(poses, poses_new)
    poses = np.array(poses_new)

    # calculate range image
    range_images = []
    for i in range(len(scans)):
        current_points = scans[i]
        proj_range, _, _, _ = range_projection(current_points, fov_up=3, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50)
        range_images.append(proj_range)

    # calculate overlap
    ground_truth_mapping, current_range, reference_range_list = com_overlap(scan_paths, poses, frame_idx=0)


    # build model and load pretrained weights
    amodel = featureExtracter(channels=1, use_transformer=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amodel.to(device)
    print("Loading weights from ", test_weights)
    checkpoint = torch.load(test_weights)
    amodel.load_state_dict(checkpoint['state_dict'])
    amodel.eval()
    
    overlap_pos = round(ground_truth_mapping[1,-1]*100,2)
    overlap_neg = round(ground_truth_mapping[2,-1]*100,2)

    reference_range_pos = reference_range_list[10]
    reference_range_neg = reference_range_list[20]
    currentrange_neg_tensor = torch.from_numpy(current_range).unsqueeze(0)
    currentrange_neg_tensor = currentrange_neg_tensor.unsqueeze(0).cuda()
    reference_range_pos_tensor = torch.from_numpy(reference_range_pos).unsqueeze(0)
    reference_range_pos_tensor = reference_range_pos_tensor.unsqueeze(0).cuda()
    reference_range_neg_tensor = torch.from_numpy(reference_range_neg).unsqueeze(0)
    reference_range_neg_tensor = reference_range_neg_tensor.unsqueeze(0).cuda()

    # generate descriptors
    des_cur = amodel(currentrange_neg_tensor).cpu().detach().numpy()
    des_pos = amodel(reference_range_pos_tensor).cpu().detach().numpy()
    des_neg = amodel(reference_range_neg_tensor).cpu().detach().numpy()

    # calculate similarity
    dis_pos = np.linalg.norm(des_cur - des_pos)
    dis_neg = np.linalg.norm(des_cur - des_neg)
    sim_pos = round(1/(1+dis_pos),2)
    sim_neg = round(1/(1+dis_neg),2)

    plt.figure(figsize=(8,4))
    plt.subplot(311)
    plt.title("query: " + scan_paths[0])
    plt.imshow(current_range)
    plt.subplot(312)
    plt.title("positive reference: " + scan_paths[10] +  " - similarity: " + str(sim_pos))
    plt.imshow(reference_range_list[10])
    plt.subplot(313)
    plt.title("negative reference: " + scan_paths[20] +  " - similarity: " + str(sim_neg))
    plt.imshow(reference_range_list[20])
    plt.show()
    ## 실행시켜보면, 0번 scan으로 10, 20번 scan의 pose위치로 projection한 rangeimage를 볼 수 있다. 이거 문제 해결해야 함.

    

if __name__ == '__main__':
    __main__()
