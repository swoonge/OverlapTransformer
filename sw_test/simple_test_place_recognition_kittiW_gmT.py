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
from sklearn.metrics import roc_auc_score, precision_recall_curve
from scipy.spatial.distance import pdist, squareform

from modules.overlap_transformer import featureExtracter

# load config ================================================================
config_filename = '../config/config_gm.yml'
config = yaml.safe_load(open(config_filename))
test_weights = config["demo1_config"]["test_weights"]
# ============================================================================

# set args for the script
parser = argparse.ArgumentParser(description='Generate overlap and orientation combined mapping file')
parser.add_argument('--dataset_path', type=str, default='/media/vision/Data0/DataSets/gm_datasets/', help='path to the scan data')

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

def calculate_distance_matrix_fast(descriptors):
    # Calculate pairwise distances using pdist, then convert to a square form matrix
    distance_matrix = squareform(pdist(descriptors, 'euclidean'))
    return distance_matrix

def calculate_descriptor_distance(descriptor1, descriptor2):
    return np.linalg.norm(descriptor1 - descriptor2)

def calculate_pose_distance(pose1, pose2):
    translation1 = pose1[:3, 3]
    translation2 = pose2[:3, 3]
    return np.linalg.norm(translation1 - translation2)

def calculate_pose_distances(poses):
    n = len(poses)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = calculate_pose_distance(poses[i], poses[j])
            distances[j, i] = distances[i, j]
    return distances

def find_matching_poses(poses, descriptors, descriptor_threshold, pose_threshold, exclusion_range=300):
    n = len(poses)
    matching_results = []
    
    # 미리 pose와 descriptor들 끼리의 거리 계산
    print("Calculating poses and descriptors distances ...")
    pose_distances = calculate_pose_distances(poses)
    descriptor_distances = calculate_distance_matrix_fast(descriptors)
    
    print("Finding matching poses ...")

    for i in tqdm(range(exclusion_range, n)):
        revisit = []
        match_candidates = []
        matches = []
        for j in range(0, i - exclusion_range):
            if descriptor_distances[i, j] < descriptor_threshold:
                match_candidates.append((j, pose_distances[i, j], descriptor_distances[i, j]))
            if pose_distances[i, j] < pose_threshold[0]:
                revisit.append(j)
        match_candidates.sort(key=lambda x: x[2])
        match_candidates = np.array(match_candidates)

        if match_candidates.shape[0] > 0: # Positive Prediction 
            for candidate in match_candidates:
                #  matching된 j     gt에 있는 j
                if candidate[1] <= pose_threshold[0]:
                    # True Positive (TP): 매칭에 성공하고, 그 pose가 실제 매칭되어야 하는 경우
                    matches.append((i, candidate[0], candidate[1], candidate[2], "tp"))
                elif candidate[1] > pose_threshold[1]:
                    # False Positive (FP): 매칭에 성공했으나, 그 pose가 실제로 매칭되어야 하는 것이 아닌 경우
                    matches.append((i, candidate[0], candidate[1], candidate[2], "fp"))
            if not matches: # 매칭된 모두가 3~20m 사이에 있어 tp, fp 모두 안된 경우. 이 경우는 거의 없다.
                if revisit:
                    # False Negative (FN): 매칭에 실패했으나, 실제로 매칭해야 하는 것이 있는 경우
                    matches.append((i, -1, -1, -1, "fn"))
                else:
                    # True Negative (TN): 매칭에 실패하고, 실제로도 매칭되는 것이 없는 경우
                    matches.append((i, -1, -1, -1, "tn"))
        else: # Negative Prediction
            if revisit:
                # False Negative (FN): 매칭에 실패했으나, 실제로 매칭해야 하는 것이 있는 경우
                matches.append((i, -1, -1, -1, "fn"))
            else:
                # True Negative (TN): 매칭에 실패하고, 실제로도 매칭되는 것이 없는 경우
                matches.append((i, -1, -1, -1, "tn"))
            

        matching_results.append(matches)
    
    return matching_results

def calculate_metrics(matching_results, top_k=5):
    tp = 0  # True Positives
    tn = 0  # True Negatives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    total_attempts = len(matching_results)
    
    topk_tp = 0  # Top-K Recall 계산을 위한 변수
    
    for matches in matching_results:
        first_match = matches[0]  # 첫 번째 매칭 결과
        
        if first_match[4] == "tp":
            tp += 1
        elif first_match[4] == "tn":
            tn += 1
        elif first_match[4] == "fp":
            fp += 1
        elif first_match[4] == "fn":
            fn += 1
        
        # Top-K Recall 계산 (상위 K개의 매칭에서 적어도 하나가 True Positive일 경우 성공으로 간주)
        if any(match[4] == "tp" for match in matches[:top_k]):
            topk_tp += 1
    
    # 메트릭 계산
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total_attempts if total_attempts > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    topk_recall = topk_tp / (topk_tp + fn) if (topk_tp + fn) > 0 else 0
    
    return {
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "Precision": precision,
        "Recall (TPR)": recall,
        "F1-Score": f1_score,
        "Accuracy": accuracy,
        "False Positive Rate (FPR)": fpr,
        "Top-{} Recall".format(top_k): topk_recall
    }

def __main__(sequence):
    # load scan paths and poses
    args = parser.parse_args()
    matrics_total = {}
    for seq in sequence:
        dataset_path = os.path.join(args.dataset_path, seq)

        # build model and load pretrained weights
        amodel = featureExtracter(channels=1, use_transformer=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amodel.to(device)
        print("Loading weights from ", test_weights)
        checkpoint = torch.load(test_weights)
        amodel.load_state_dict(checkpoint['state_dict'])
        amodel.eval()

        if not os.path.exists("preprocessed_data_gm/descriptors_" + seq + ".npy"):
            # load scans
            print("Loading scans ...")
            scan_paths = sorted(glob.glob(os.path.join(dataset_path, 'velodyne', '*.bin')))
            scans = []
            for scan_path in tqdm(scan_paths):
                scans.append(load_vertex(scan_path))

            # calculate range image
            print("Calculating range images ...")
            range_images = []
            for scan in tqdm(scans):
                proj_range, _, _, _ = range_projection(scan, fov_up=3, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50)
                range_images.append(proj_range)

            # make discriptors for all range images
            print("Calculating descriptors ...")
            descriptors = []
            total_time = 0
            for range_image in tqdm(range_images):
                time_start = time.time()
                range_image_tensor = torch.from_numpy(range_image).unsqueeze(0)
                range_image_tensor = range_image_tensor.unsqueeze(0).cuda()
                descriptor = amodel(range_image_tensor).cpu().detach().numpy()
                time_end = time.time()
                total_time += time_end - time_start
                descriptors.append(np.squeeze(descriptor))
            mean_time = total_time / len(range_images)
            print(f"모델 평균 시간: {mean_time:.6f} 초")
            descriptors = np.array(descriptors)

            np.save("preprocessed_data_gm/descriptors_" + seq + ".npy", descriptors)
        else:
            print("Loading descriptors ...")
            descriptors = np.load("preprocessed_data_gm/descriptors_" + seq + ".npy")

        if not os.path.exists("preprocessed_data_gm/poses_" + seq + ".npy"):

            # load calibrations
            # calib_file = os.path.join(dataset_path, 'calib.txt')
            # T_cam_velo = load_calib(calib_file)
            # T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
            # T_velo_cam = np.linalg.inv(T_cam_velo)

            # load poses
            poses_file = os.path.join(dataset_path, 'poses.txt')
            poses = load_poses(poses_file)
            # for KITTI dataset, we need to convert the provided poses
            # from the camera coordinate system into the LiDAR coordinate system
            # poses_new = []
            # for pose in poses:
            #     poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
            plot_two_poses(poses, poses)
            np.save("preprocessed_data_gm/poses_" + seq + ".npy", poses)
        else:
            print("Loading poses ...")
            poses = np.load("preprocessed_data_gm/poses_" + seq + ".npy")

        # descriptor_threshold = 0.3  # descriptor 유사성 임계값
        # descriptor_thresholds = [0.3]  # descriptor 유사성 임계값
        # descriptor_thresholds = np.arange(0.01, 0.1, 0.025)
        descriptor_thresholds = np.arange(0.005, 0.01, 0.001)
        # descriptor_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]  # descriptor 유사성 임계값
        pose_threshold = [3.0, 20.0]  # 실제 pose 거리 임계값 3m, 20m

        f1_scores = []
        for distance_threshold in descriptor_thresholds:
            matching_results = find_matching_poses(poses, descriptors, distance_threshold, pose_threshold)
            metrics = calculate_metrics(matching_results, top_k=50)
            f1_scores.append(metrics["F1-Score"])
            matrics_total[seq + "_" + str(distance_threshold)] = metrics
            
    print("[Total Metrics]")
    for key, value in matrics_total.items():
        print(f"Sequence {key}:")
        for key2, value2 in value.items():
            print(f"\t{key2}: {value2:.3f}")

    print("[F1-Scores]")
    print(f"F1-Scores: {f1_scores}")
    print(f"F1-max: {max(f1_scores)}")

    # print("Matching Metrics :")
    # for key, value in metrics.items():
    #     print(f"{key}: {value:.3f}")
        

if __name__ == '__main__':
    # use sequences 03–10 for training, sequence 02 for validation, and sequence 00 for evaluation.
    # sequence = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    sequence = ["08_01"]
    
    __main__(sequence)
