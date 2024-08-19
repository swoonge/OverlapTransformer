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

class SLAMMatcherSimulator:
    def __init__(self, test_weights, threshold_distance, pose_distance_threshold, exclusion_range=50):
        self.descriptors = []
        self.poses = []
        self.scans = []
        self.range_images = []
        self.threshold_distance = threshold_distance
        self.pose_distance_threshold = pose_distance_threshold
        self.exclusion_range = exclusion_range
        self.matching_results = []
        self.set_overlap_transformer_model(test_weights)

    def set_overlap_transformer_model(self, test_weights):
        # build model and load pretrained weights
        self.amodel = featureExtracter(channels=1, use_transformer=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(device)
        print("Loading weights from ", test_weights)
        checkpoint = torch.load(test_weights)
        self.amodel.load_state_dict(checkpoint['state_dict'])
        self.amodel.eval()

    def add_frame(self, pose, scan):
        """
        새로운 frame(즉, 새로운 pose와 descriptor)을 추가하고, 매칭을 시도합니다.
        """
        self.poses.append(pose)
        self.scans.append(scan)
        proj_range, _, _, _ = range_projection(scan, fov_up=3, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50)
        self.range_images.append(proj_range)
        range_image_tensor = torch.from_numpy(proj_range).unsqueeze(0)
        range_image_tensor = range_image_tensor.unsqueeze(0).cuda()
        self.descriptors.append(self.amodel(range_image_tensor).cpu().detach().numpy())
        self.match_descriptor(len(self.descriptors) - 1)

    def calculate_descriptor_distance(self, descriptor1, descriptor2):
        # Calculate Euclidean distance
        return np.linalg.norm(descriptor1 - descriptor2)

    def calculate_pose_distance(self, pose1, pose2):
        # Calculate distance based on translation vector in 4x4 pose matrix
        translation1 = pose1[:3, 3]
        translation2 = pose2[:3, 3]
        return np.linalg.norm(translation1 - translation2)

    def match_descriptor(self, current_index):
        """
        주어진 index의 descriptor를 이전 descriptor들과 매칭 시도.
        최근 50개의 index는 제외.
        """
        if current_index < self.exclusion_range:
            return  # Exclude first frames where not enough history is available

        current_descriptor = self.descriptors[current_index]
        best_match_index = None
        best_match_distance = float('inf')

        for j in range(current_index - self.exclusion_range):
            distance = self.calculate_descriptor_distance(current_descriptor, self.descriptors[j])
            
            if distance < best_match_distance and distance < self.threshold_distance:
                best_match_distance = distance
                best_match_index = j

        if best_match_index is not None:
            pose_distance = self.calculate_pose_distance(self.poses[current_index], self.poses[best_match_index])
            if pose_distance < self.pose_distance_threshold:
                self.matching_results.append((current_index, best_match_index, best_match_distance, pose_distance, "Success"))
            else:
                self.matching_results.append((current_index, best_match_index, best_match_distance, pose_distance, "Failure"))

    def calculate_metrics(self):
        true_positives = sum(1 for result in self.matching_results if result[4] == "Success")
        false_negatives = sum(1 for result in self.matching_results if result[4] == "Failure")
        total_attempts = len(self.matching_results)
        
        success_rate = true_positives / total_attempts if total_attempts > 0 else 0
        failure_rate = false_negatives / total_attempts if total_attempts > 0 else 0

        precision = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "Total Matching Attempts": total_attempts,
            "True Positives (Successes)": true_positives,
            "False Negatives (Failures)": false_negatives,
            "Success Rate": success_rate,
            "Failure Rate": failure_rate,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score
        }

def __main__():
  # load scan paths and poses
    args = parser.parse_args()
    sequence = "00"
    dataset_path = os.path.join(args.dataset_path, sequence)

    # load scans
    print("Loading scans ...")
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

    # Example usage
    slam_matcher = SLAMMatcherSimulator(test_weights, threshold_distance=0.2, pose_distance_threshold=3.0)

    # Simulate SLAM loop with new frames being added
    print("Simulating SLAM Matching ...")
    for i in tqdm(range(len(scans))):
        slam_matcher.add_frame(poses[i], scans[i])

    metrics = slam_matcher.calculate_metrics()

    print("Real-time SLAM Matching Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
    

if __name__ == '__main__':
    __main__()
